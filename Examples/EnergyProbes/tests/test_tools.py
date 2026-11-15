"""Regression tests of the EnergyProbes tools and of the program's GPU-free modes.

    python3 -m pytest Examples/EnergyProbes/tests -q            # from the KTT root; about half a minute
    EP_BINARY=/path/to/EnergyProbesCuda python3 -m pytest ...   # another build (default: Build/x86_64_Release)

No GPU is needed. The data in tests/data are real runs of an RTX 3090 at 1200 MHz, made with the commands of the
README: the probe run (Probes.csv and its energy table), the counter run (Profile.csv), the validation run
(Validation_validation.csv) and a KTT output file of its counter pass. models/ has the model built from them and the
MT4G file. The tests of the program (--predict, --dry-run, option checks) are skipped when it is not built.
"""

import copy
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
DATA = os.path.join(HERE, "data")
MODEL = os.path.join(ROOT, "models", "rtx3090_1200mhz.model.json")
BINARY = os.environ.get("EP_BINARY", os.path.join(ROOT, "..", "..", "Build", "x86_64_Release", "EnergyProbesCuda"))
MT4G = os.path.join(ROOT, "models", "NVIDIA_GeForce_RTX_3090.json")
VALIDATION = os.path.join(DATA, "Validation_validation.csv")

sys.path.insert(0, TOOLS)
import counter_table  # noqa: E402
import energy_events  # noqa: E402
import energy_predictor  # noqa: E402

needs_binary = pytest.mark.skipif(not os.path.isfile(BINARY), reason="EnergyProbesCuda is not built")


def tool(name, *arguments, cwd=DATA, check=True):
    result = subprocess.run([sys.executable, os.path.join(TOOLS, name)] + list(arguments), cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    if check:
        assert result.returncode == 0, result.stdout
    return result


def program(*arguments):
    directory = os.path.dirname(os.path.abspath(BINARY))
    environment = dict(os.environ, LD_LIBRARY_PATH=directory + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    return subprocess.run([os.path.abspath(BINARY)] + list(arguments), cwd=directory, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


@pytest.fixture(scope="module")
def model():
    return energy_predictor.Model.load(MODEL)


@pytest.fixture(scope="module")
def validation():
    return energy_predictor.from_validation_csv(VALIDATION)


# ---- CSV reading --------------------------------------------------------------------------------------------------------

def test_read_rows_repairs_a_comma_in_the_note(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("probe,note,counter_a,counter_b\n"
                    "p1,plain,1,2\n"
                    "p2,25 of 68 SMs without 4 blocks, 11 sharing a stripe,3,4\n"
                    "p3,,5,\n")
    rows = counter_table.read_rows(str(path))
    assert [r["counter_a"] for r in rows] == ["1", "3", "5"]
    assert rows[1]["note"] == "25 of 68 SMs without 4 blocks, 11 sharing a stripe"
    assert rows[1]["counter_b"] == "4" and rows[2]["counter_b"] == ""


def test_reference_runs_have_no_shifted_rows():
    for name in ("Probes.csv", "Profile.csv", "Validation_validation.csv"):
        with open(os.path.join(DATA, name), newline="") as file:
            lengths = {len(fields) for fields in csv.reader(file)}
        assert len(lengths) == 1, name


# ---- Events -------------------------------------------------------------------------------------------------------------

def test_instruction_events_add_up_to_the_instructions(validation):
    # Without predicated-off lanes the instruction events are a partition of the warp instructions; with them the alu and
    # fma events are smaller (Coulomb's guarded moves: under half). So the sum never exceeds the total.
    for configurations in validation.values():
        for _, row, _, _, _ in configurations:
            counts = energy_events.events(row)
            total = energy_events.counter(row, "smsp__inst_executed.sum")
            instructions = sum(v for k, v in counts.items() if k.startswith("inst"))
            assert instructions <= total * 1.01  # counters come from several profiling passes
            assert instructions >= 0.1 * total


def test_busy_event_follows_the_dram_rate():
    counts = {"DRAM read sector": 1e9, "DRAM write sector": 0.0}
    busy = energy_events.add_busy(dict(counts), 1e9, 95.0)["DRAM busy ns"]  # 32 GB/s for 1 s
    assert busy == pytest.approx(1e9 * (1 - math.exp(-32 / 95.0)))
    assert energy_events.add_busy(dict(counts), 1e9)["DRAM busy ns"] == pytest.approx(busy)  # 95 GB/s is the default
    assert energy_events.add_busy(dict(counts), 0.0)["DRAM busy ns"] == 0.0


def test_every_counter_of_the_events_is_in_the_counter_table():
    energy_events.events({})  # value_of raises for a name that is not in the table


def test_cpp_predictor_knows_the_same_events():
    with open(os.path.join(ROOT, "EnergyModel.cpp")) as file:
        source = file.read()
    for name in energy_events.EVENTS:
        assert '"' + name + '"' in source, name


# ---- Fit and model file -------------------------------------------------------------------------------------------------

def test_the_model_file_is_what_the_reference_runs_give(tmp_path, model):
    """build_model.py on the reference runs reproduces the energies of the shipped model: a change in the fit, the events
    or the CSV reading shows here."""
    output = tmp_path / "rebuilt.model.json"
    result = tool("build_model.py", "--power", "Probes", "--counters", "Profile", "--calibration", "Validation",
                  "--output", str(output))
    rebuilt = energy_predictor.Model.load(str(output))
    for name, energy in model.pj.items():
        assert rebuilt.pj[name] == pytest.approx(energy, rel=1e-6, abs=1e-6), name
    assert rebuilt.baseline == model.baseline
    assert rebuilt.required == model.required
    assert rebuilt.uncertainty["dynamic_relative_sd"] == pytest.approx(model.uncertainty["dynamic_relative_sd"], rel=1e-6)
    assert len(rebuilt.data["probes"]) == len(model.data["probes"]) == 150
    assert "ordered right" in result.stdout
    # The file is written one entry per line.
    assert len(output.read_text().splitlines()) < 400


def test_energy_model_fit_and_scores():
    """The fit over the reference runs and its scores on the validation run."""
    result = tool("energy_model.py", "--power", "Probes", "--counters", "Profile", "--validation", "Validation")
    assert "Fit over 150 probes" in result.stdout and "no probe" in result.stdout
    line = next(l for l in result.stdout.splitlines() if "mean over the kernel runs" in l)
    scatter, bias, mape, regret = (float(part.split()[-2 if "regret" not in part else 2].rstrip("%"))
                                   for part in line.split(":", 1)[1].split(",")[:4])
    assert scatter == pytest.approx(5.5, abs=0.3)
    assert abs(bias) < 2 and mape < 3.0 and regret <= 1.0


def test_probes_at_one_duty_enter_with_their_row(tmp_path):
    """The dependent loads, the write-path and the misaligned probes run at duty 100 only: no fit, the row's energy.
    A probe that has two duties but no fit (a row withheld) stays out."""
    import contextlib
    import io
    import energy_model as m
    with contextlib.redirect_stdout(io.StringIO()):
        probes, _ = m.probe_table([os.path.join(DATA, "Probes")], [os.path.join(DATA, "Profile")])
    rows = {m.label_of(m.key_of(r)): r for r in m.probe_rows(os.path.join(DATA, "Probes"))}
    energies = {p.label: p.energy for p in probes}
    for label in ("DRAM chase-c16-w32 w4", "L2 st-off32 w16", "DRAM ld-sub-pairs-1words w4"):
        assert energies[label] == pytest.approx(float(rows[label]["pj_per_access"]))
    assert energies["L1 ld w16"] != pytest.approx(float(rows["L1 ld w16"]["pj_per_access"]), rel=1e-3)  # a slope
    assert "compute ffma" not in energies and "compute ffma3" in energies  # x * x + c stays out of the fit
    # Without its fit, a probe measured at two duties does not fall back to its duty-100 row.
    for name in ("Probes.csv", "Probes_energy_table.json"):
        shutil.copy(os.path.join(DATA, name), tmp_path / name)
    with open(tmp_path / "Probes_energy_table.json") as file:
        table = json.load(file)
    table["fits"] = [f for f in table["fits"] if not (f["operation"] == "ld" and f["level"] == "L1" and f["width_bytes"] == 16)]
    (tmp_path / "Probes_energy_table.json").write_text(json.dumps(table))
    with contextlib.redirect_stdout(io.StringIO()):
        fewer, _ = m.probe_table([str(tmp_path / "Probes")], [os.path.join(DATA, "Profile")])
    assert "L1 ld w16" not in {p.label for p in fewer} and len(fewer) == len(probes) - 1


def test_runs_must_have_the_same_counters():
    import energy_model as m
    rows = m.probe_rows(os.path.join(DATA, "Profile"))
    m.check_same_counters({"probes": rows, "validation": counter_table.read_rows(VALIDATION)})
    without = [{k: v for k, v in r.items() if k != "dram__sectors_read.sum"} for r in rows]
    with pytest.raises(SystemExit, match="dram__sectors_read"):
        m.check_same_counters({"probes": without, "validation": counter_table.read_rows(VALIDATION)})


def test_predictor_equals_energy_model(model, validation):
    import contextlib
    import io
    import energy_model as m
    with contextlib.redirect_stdout(io.StringIO()):
        probes, table = m.probe_table([os.path.join(DATA, "Probes")], [os.path.join(DATA, "Profile")])
        rows = m.load_validation(os.path.join(DATA, "Validation"), table)
    energies, _, _ = m.fit(probes)
    reference = {(row["kernel"], row["configuration"]): result[2] for row, result in zip(rows, m.predict(rows, energies))}
    compared = 0
    for kernel, configurations in validation.items():
        for label, row, seconds, _, _ in configurations:
            prediction = model.predict(row, seconds)
            assert prediction.dynamic_j == pytest.approx(reference[(kernel, label)], rel=1e-6)
            compared += 1
    assert compared == len(reference) > 100


def test_predictor_imports_without_numpy_and_scipy():
    code = ("import sys; sys.modules['numpy'] = None; sys.modules['scipy'] = None; "
            f"sys.path.insert(0, {TOOLS!r}); import energy_predictor; "
            f"m = energy_predictor.Model.load({MODEL!r}); print(len(m.pj))")
    result = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    assert result.returncode == 0, result.stdout


def test_model_file_is_checked(tmp_path, model):
    data = copy.deepcopy(model.data)
    data["version"] = energy_predictor.VERSION + 1
    with pytest.raises(energy_predictor.ModelError):
        energy_predictor.Model(data)
    data = copy.deepcopy(model.data)
    data["format"] = "something else"
    with pytest.raises(energy_predictor.ModelError):
        energy_predictor.Model(data)
    data = copy.deepcopy(model.data)
    data["events"]["an event of the future"] = {"pj": 1.0, "status": "fitted"}
    with pytest.raises(energy_predictor.ModelError):
        energy_predictor.Model(data)


# ---- Prediction ---------------------------------------------------------------------------------------------------------

def test_missing_counters_are_an_error_unless_lenient(model, validation):
    _, row, seconds, _, _ = next(iter(validation.values()))[0]
    broken = {k: v for k, v in row.items() if k != "dram__sectors_read.sum"}
    with pytest.raises(energy_predictor.ModelError, match="dram__sectors_read"):
        model.predict(broken, seconds)
    lenient = model.predict(broken, seconds, strict=False)
    assert any("dram__sectors_read" in w for w in lenient.warnings)
    assert lenient.dynamic_j < model.predict(row, seconds).dynamic_j


def test_prediction_parts(model, validation):
    _, row, seconds, temperature, _ = next(iter(validation.values()))[0]
    p = model.predict(row, seconds, temperature)
    assert p.energy_j == pytest.approx(p.baseline_j + p.dynamic_j)
    assert p.baseline_j == pytest.approx(model.baseline_w(temperature) * seconds)
    assert sum(p.shares.values()) == pytest.approx(1.0)
    assert 0 < p.sd_j < 0.1 * p.energy_j
    # An unknown temperature widens the uncertainty, a hotter GPU raises the baseline.
    assert model.predict(row, seconds).sd_j > p.sd_j
    assert model.baseline_w(80.0) > model.baseline_w(60.0)
    with pytest.raises(energy_predictor.ModelError):
        model.predict(row, 0.0)


def test_power_limit(model, validation):
    limited = energy_predictor.Model.load(MODEL)
    _, row, seconds, _, _ = next(iter(validation.values()))[0]
    free = model.predict(row, seconds)
    limited.power_limit_w = free.power_w + 10
    assert limited.predict(row, seconds).energy_j == pytest.approx(free.energy_j)
    limited.power_limit_w = 0.8 * free.power_w
    capped = limited.predict(row, seconds)
    assert capped.energy_j == pytest.approx(0.8 * free.energy_j)
    assert capped.energy_j == pytest.approx(capped.baseline_j + capped.dynamic_j)
    assert any("limit" in w for w in capped.warnings)


def test_clock_warning(model, validation):
    _, row, seconds, _, _ = next(iter(validation.values()))[0]
    assert not [w for w in model.predict(row, seconds, sm_clock_mhz=1200).warnings if "clock" in w]
    assert [w for w in model.predict(row, seconds, sm_clock_mhz=1900).warnings if "clock" in w]


def prediction(energy, dynamic, seconds, label=""):
    return energy_predictor.Prediction(energy_j=energy, sd_j=0.06 * dynamic, dynamic_j=dynamic, baseline_j=energy - dynamic,
                                       kernel_s=seconds, power_w=energy / seconds, events={}, shares={}, warnings=[],
                                       label=label)


def test_recommend(model):
    configurations = [prediction(10.0, 5.0, 1.0), prediction(10.2, 5.0, 0.8), prediction(14.0, 7.0, 0.5),
                      prediction(30.0, 20.0, 2.0)]
    choice = energy_predictor.recommend(model, configurations)
    assert choice.index == 0 and choice.contenders == [1] and not choice.confident
    # A wider band takes in more contenders, a band of 0 none.
    assert energy_predictor.recommend(model, configurations, z=0).confident
    assert set(energy_predictor.recommend(model, configurations, z=20).contenders) == {1, 2, 3}
    # Time: the fastest, without contenders. EDP weighs the time in.
    assert energy_predictor.recommend(model, configurations, objective="time").index == 2
    assert energy_predictor.recommend(model, configurations, objective="edp").index == 2
    # Only configurations within 70 % of the best time: 0 (1.0 s) is out, 1 (0.8 s) wins.
    limited = energy_predictor.recommend(model, configurations, max_slowdown=0.7)
    assert limited.index == 1 and math.isnan(limited.values[0]) and math.isnan(limited.values[3])
    # The fastest among those within 5 % of the lowest energy.
    assert energy_predictor.recommend(model, configurations, objective="time", max_energy_increase=0.05).index == 1
    assert choice.front == [2, 1, 0]
    with pytest.raises(ValueError):
        energy_predictor.recommend(model, [])


def test_the_choice_on_the_validation_run_is_good(model, validation):
    """On every kernel of the validation run the model's first choice uses at most 2.5 % more energy than the measured
    best, and the measured best is the choice or one of its contenders."""
    rows = counter_table.read_rows(VALIDATION)
    measured = {(r["kernel"], r["configuration"]): counter_table.number(r["energy_j"]) for r in rows}
    for kernel, configurations in validation.items():
        predictions = [model.predict(row, seconds, temperature, label=label)
                       for label, row, seconds, temperature, _ in configurations]
        choice = energy_predictor.recommend(model, predictions)
        energies = [measured[(kernel, p.label)] for p in predictions]
        assert energies[choice.index] <= 1.025 * min(energies), kernel
        assert energies.index(min(energies)) in [choice.index] + choice.contenders, kernel


def test_ktt_json_input_gives_the_same_choice(tmp_path, model, validation):
    """The same configurations as a KTT output file: counters and durations per computation."""
    kernel, configurations = next(iter(validation.items()))
    results = []
    for label, row, seconds, temperature, clock in configurations:
        counters = [{"Name": name, "Type": "Double", "Value": float(value)} for name, value in row.items()
                    if name in counter_table.ENTRIES and value not in ("", None)]
        results.append({"Status": "Ok", "KernelName": kernel,
                        "Configuration": [{"Name": part.split("=")[0], "Value": part.split("=")[1]}
                                          for part in label.split()],
                        "ComputationResults": [{"Duration": seconds * 1e6, "Temperature": temperature,
                                                "SMFrequency": clock, "ProfilingData": {"Counters": counters}}]})
    results.append({"Status": "ComputationFailed", "KernelName": kernel, "Configuration": [], "ComputationResults": []})
    path = tmp_path / "ktt.json"
    path.write_text(json.dumps({"Metadata": {"TimeUnit": "Microseconds"}, "Results": results}))
    loaded = energy_predictor.from_ktt_json(str(path))
    assert list(loaded) == [kernel] and len(loaded[kernel]) == len(configurations)
    from_json = [model.predict(c, s, t, label=l) for l, c, s, t, _ in loaded[kernel]]
    from_csv = [model.predict(c, s, t, label=l) for l, c, s, t, _ in configurations]
    for a, b in zip(from_json, from_csv):
        assert a.energy_j == pytest.approx(b.energy_j, rel=1e-9)
    assert energy_predictor.recommend(model, from_json).index == energy_predictor.recommend(model, from_csv).index
    result = tool("energy_predictor.py", "--model", MODEL, "--ktt", str(path))
    assert "use: " + from_csv[energy_predictor.recommend(model, from_csv).index].label in result.stdout


# ---- Safe recommendations, kernels compared, accounting (2026-11-13) ---------------------------------------------------

def test_no_admissible_configuration_is_an_error(model):
    """max_slowdown 0 admits only the fastest, max_energy_increase 0 only the cheapest: when they differ, nothing is left."""
    configurations = [prediction(10.0, 5.0, 1.0), prediction(12.0, 6.0, 0.5)]
    with pytest.raises(energy_predictor.ModelError, match="no configuration is admissible"):
        energy_predictor.recommend(model, configurations, max_slowdown=0.0, max_energy_increase=0.0)
    # One configuration that is both the fastest and the cheapest is admissible.
    assert energy_predictor.recommend(model, [prediction(10.0, 5.0, 0.5), prediction(12.0, 6.0, 1.0)],
                                      max_slowdown=0.0, max_energy_increase=0.0).index == 0


def test_negative_and_infinite_counters(model, validation):
    _, row, seconds, _, _ = next(iter(validation.values()))[0]
    zero = dict(row, **{"dram__sectors_write.sum": "0"})
    negative = dict(row, **{"dram__sectors_write.sum": "-5000"})
    # A negative value (CUPTI reports some) counts as 0, with a warning, also with strict.
    p = model.predict(negative, seconds)
    assert p.dynamic_j == pytest.approx(model.predict(zero, seconds).dynamic_j, rel=1e-12)
    assert any("negative" in w for w in p.warnings)
    infinite = dict(row, **{"dram__sectors_write.sum": "inf"})
    with pytest.raises(energy_predictor.ModelError, match="not finite"):
        model.predict(infinite, seconds)
    assert model.predict(infinite, seconds, strict=False).dynamic_j == pytest.approx(p.dynamic_j, rel=1e-12)


def test_model_file_keys_and_json(tmp_path, model):
    for key in energy_predictor.REQUIRED_KEYS:
        data = copy.deepcopy(model.data)
        entry = data
        for part in key[:-1]:
            entry = entry[part]
        del entry[key[-1]]
        with pytest.raises(energy_predictor.ModelError, match="missing " + "/".join(key)):
            energy_predictor.Model(data, "x.json")
    truncated = tmp_path / "truncated.model.json"
    truncated.write_text(open(MODEL).read()[:500])
    with pytest.raises(energy_predictor.ModelError, match="not JSON"):
        energy_predictor.Model.load(str(truncated))


def test_power_limit_uncertainty(model, validation):
    """At the limit the energy is the limit times the time, and its sd is how closely the GPU holds the limit."""
    limited = energy_predictor.Model.load(MODEL)
    _, row, seconds, temperature, _ = next(iter(validation.values()))[0]
    free = model.predict(row, seconds, temperature)
    limited.power_limit_w = 0.8 * free.power_w
    capped = limited.predict(row, seconds, temperature)
    assert capped.clamped and not free.clamped
    assert capped.sd_j == pytest.approx(energy_predictor.LIMIT_RELATIVE_SD * capped.energy_j)
    # The pair term uses the same sd: two capped configurations differ by their time only.
    other = limited.predict(row, 1.5 * seconds, temperature)
    assert limited.pair_sd(capped, other) == pytest.approx(math.hypot(capped.sd_j, other.sd_j))


def test_kernels_compared_with_the_wider_sd(model):
    """Two predictions 9 % apart: told apart within one kernel, contenders when they are two kernels (the bias between
    kernels widens the sd per prediction)."""
    within = [prediction(1.0, 0.5, 1.0)._replace(group="a"), prediction(1.09, 0.545, 1.0)._replace(group="a")]
    assert energy_predictor.recommend(model, within).confident
    between = [within[0], within[1]._replace(group="b")]
    choice = energy_predictor.recommend(model, between)
    assert choice.index == 0 and choice.contenders == [1]
    assert model.between_kernels_sd() == pytest.approx(model.uncertainty["dynamic_bias_sd_between_kernels"])
    assert 0.02 < model.between_kernels_sd() < 0.08


def test_kernel_scale(validation):
    """A scale measured on a few configurations multiplies the kernel's dynamic energy; against another kernel its bias
    is replaced by what is left of it, hypot(sd / sqrt(n), 2.4 %); a second scale from scaled predictions does not
    compound (it reads their unscaled dynamic energy)."""
    scaled = energy_predictor.Model.load(MODEL)
    _, row, seconds, temperature, _ = next(iter(validation.values()))[0]
    plain = scaled.predict(row, seconds, temperature, group="a")
    assert scaled.kernel_scale("a", [1.0, 2.0], [1.1, 2.2]) is None  # fewer than 3 measurements
    assert scaled.kernel_scale("a", [1.0, 2.0, 4.0], [1.1, 2.2, 4.4]) == pytest.approx(1.1)
    p = scaled.predict(row, seconds, temperature, group="a")
    assert p.dynamic_j == pytest.approx(1.1 * plain.dynamic_j) and p.baseline_j == pytest.approx(plain.baseline_j)
    other = scaled.predict(row, seconds, temperature, group="b")
    relative = scaled.uncertainty["dynamic_relative_sd"]
    residual = math.hypot(relative / math.sqrt(3), energy_predictor.SCALE_RUN_SD)
    assert scaled.independent_sd(p, other) == pytest.approx(math.hypot(relative, residual) * p.dynamic_j)
    assert scaled.independent_sd(other, p) == pytest.approx(math.hypot(relative, scaled.between_kernels_sd()) * other.dynamic_j)
    assert scaled.independent_sd(p, p) == pytest.approx(relative * p.dynamic_j)
    # Scaled predictions measured 1.1 x their unscaled energy: the scale stays 1.1, not 1.21.
    assert p.unscaled_dynamic_j == pytest.approx(plain.dynamic_j)
    assert scaled.kernel_scale("a", [p, p, p], [1.1 * plain.dynamic_j] * 3) == pytest.approx(1.1)


def test_accounting(validation):
    accounted = energy_predictor.Model.load(MODEL)
    _, row, seconds, temperature, _ = next(iter(validation.values()))[0]
    awake = accounted.predict(row, seconds, temperature)
    accounted.accounting = "above-idle"
    idle = accounted.predict(row, seconds, temperature)
    assert idle.baseline_j == pytest.approx(awake.baseline_j - accounted.idle_power_w() * seconds)
    assert accounted.idle_power_w() == pytest.approx(accounted.baseline["context_idle_w"])
    accounted.idle_w = 41.7
    assert accounted.predict(row, seconds, temperature).baseline_j == pytest.approx(awake.baseline_j - 41.7 * seconds)
    accounted.accounting = "dynamic"
    dynamic = accounted.predict(row, seconds, temperature)
    assert dynamic.baseline_j == 0 and dynamic.energy_j == pytest.approx(awake.dynamic_j)
    # Without a baseline its error does not enter; the power is the GPU's either way.
    assert dynamic.sd_j == pytest.approx(dynamic.dynamic_sd_j) and dynamic.power_w == pytest.approx(awake.power_w)
    accounted.accounting = "something"
    with pytest.raises(energy_predictor.ModelError):
        accounted.predict(row, seconds)


def test_model_file_provenance(model):
    assert set(model.data["tools"]) == {"commit", "events_sha256"}
    for key in ("driver_version", "cuda_driver_version", "uuid", "power_limit_w"):
        assert model.device[key]
    assert model.data["uncertainty"]["calibration"]["left_out_off_clock"] == 2
    assert set(model.data["options"]) == {"busy_scale_gbs"}
    assert model.optional == list(energy_predictor.OPTIONAL_COUNTERS)


def test_aliases_come_from_the_model_file(model, validation):
    """A counter under another chip's name (RTX 5090: dram__sectors_op_read) is read through the file's aliases."""
    _, row, seconds, _, _ = next(iter(validation.values()))[0]
    alias = next(a for a, c in model.aliases.items() if c == "dram__sectors_read.sum")
    renamed = {(alias if k == "dram__sectors_read.sum" else k): v for k, v in row.items()}
    assert model.predict(renamed, seconds).dynamic_j == pytest.approx(model.predict(row, seconds).dynamic_j, rel=1e-12)


def test_device_warning(model, validation):
    _, row, seconds, _, _ = next(iter(validation.values()))[0]
    assert not model.predict(row, seconds, device_name="NVIDIA GeForce RTX 3090").warnings
    assert any("calibrated on" in w for w in model.predict(row, seconds, device_name="NVIDIA A100").warnings)


@needs_binary
def test_cpp_accounting_limits_and_errors(tmp_path, model, validation):
    """The C++ predictor gives the Python numbers under every accounting and refuses what Python refuses."""
    prefix = str(tmp_path / "cpp")
    for accounting in ("above-idle", "dynamic"):
        result = program("--predict", VALIDATION, "--model-file", MODEL, "--accounting", accounting, "--output", prefix)
        assert result.returncode == 0, result.stdout
        accounted = energy_predictor.Model.load(MODEL)
        accounted.accounting = accounting
        rows = {(r["kernel"], r["configuration"]): r for r in counter_table.read_rows(prefix + "_predicted.csv")}
        for kernel, configurations in validation.items():
            predictions = [accounted.predict(c, s, t, f, l, group=kernel) for l, c, s, t, f in configurations]
            choice = energy_predictor.recommend(accounted, predictions)
            for i, p in enumerate(predictions):
                r = rows[(kernel, p.label)]
                assert float(r["predicted_j"]) == pytest.approx(p.energy_j, rel=1e-9)
                assert float(r["predicted_sd_j"]) == pytest.approx(p.sd_j, rel=1e-9, abs=1e-15)
                assert r["accounting"] == accounting
                assert r["choice"] == ("best" if i == choice.index else "contender" if i in choice.contenders else "")
    # Both limits at 0: kernels whose fastest is not their cheapest have no admissible configuration.
    none = program("--predict", VALIDATION, "--model-file", MODEL, "--max-slowdown", "0", "--max-energy-increase", "0",
                   "--output", prefix)
    assert none.returncode == 1 and "no configuration is admissible" in none.stdout
    # A truncated file and a file without a key are named.
    truncated = tmp_path / "truncated.model.json"
    truncated.write_text(open(MODEL).read()[:500])
    result = program("--predict", VALIDATION, "--model-file", str(truncated), "--output", prefix)
    assert result.returncode == 1 and "not JSON" in result.stdout
    data = copy.deepcopy(model.data)
    del data["baseline"]["mean_w"]
    keyless = tmp_path / "keyless.model.json"
    keyless.write_text(json.dumps(data))
    result = program("--predict", VALIDATION, "--model-file", str(keyless), "--output", prefix)
    assert result.returncode == 1 and "missing baseline/mean_w" in result.stdout


@needs_binary
def test_minimal_tuner_example_builds(tmp_path):
    """The README's minimal program (example/MinimalTuner.cpp) compiles and links with EnergyModel and EnergyTuner."""
    ktt = os.path.abspath(os.path.join(ROOT, "..", ".."))
    release = os.path.dirname(os.path.abspath(BINARY))
    cuda = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    output = tmp_path / "MinimalTuner"
    command = ["g++", "-std=c++17", "-O1", "-I" + os.path.join(ktt, "Source"), "-I" + os.path.join(ktt, "Libraries", "Json-3.9.1"),
               "-I" + os.path.join(cuda, "include"), os.path.join(ROOT, "example", "MinimalTuner.cpp"),
               os.path.join(ROOT, "EnergyModel.cpp"), os.path.join(ROOT, "EnergyTuner.cpp"),
               "-L" + release, "-lktt", "-o", str(output)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    assert output.exists()


# ---- check_model.py -----------------------------------------------------------------------------------------------------

def edited_run(tmp_path, name, change):
    """A copy of the reference power run with every probe row changed by change(row)."""
    rows = counter_table.read_rows(os.path.join(DATA, "Probes.csv"))
    path = tmp_path / (name + ".csv")
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            if row["level"] != "baseline" and row["pj_per_access"]:
                change(row)
            writer.writerow(row)
    return str(tmp_path / name)


def test_check_model_passes_on_its_own_calibration_run():
    result = tool("check_model.py", "--model", MODEL, "--run", "Probes")
    assert "PASS" in result.stdout


def test_check_model_finds_a_common_factor(tmp_path):
    def more_energy(row):
        row["pj_per_access"] = repr(1.2 * float(row["pj_per_access"]))
    run = edited_run(tmp_path, "Scaled", more_energy)
    written = tmp_path / "scaled.model.json"
    result = tool("check_model.py", "--model", MODEL, "--run", run, "--write", str(written))
    assert "SCALED" in result.stdout
    scaled, original = energy_predictor.Model.load(str(written)), energy_predictor.Model.load(MODEL)
    assert scaled.pj["DRAM read sector"] == pytest.approx(1.2 * original.pj["DRAM read sector"], rel=1e-3)
    assert scaled.data["adjusted"]["energy_factor"] == pytest.approx(1.2, rel=1e-3)


def test_check_model_fails_on_another_speed(tmp_path):
    def slower(row):
        row["active_s"] = repr(1.3 * float(row["active_s"]))
    result = tool("check_model.py", "--model", MODEL, "--run", edited_run(tmp_path, "Slow", slower), check=False)
    assert result.returncode == 1 and "FAIL" in result.stdout


def test_check_model_fails_without_a_constant_clock(tmp_path):
    def other_clock(row):
        row["clock_difference_pct"] = "-12"
    result = tool("check_model.py", "--model", MODEL, "--run", edited_run(tmp_path, "Unlocked", other_clock), check=False)
    assert result.returncode == 1 and "clock is not constant" in result.stdout


def edited_rows(tmp_path, source, name, change):
    """A copy of a run's CSV in tests/data with change(row) applied to every row."""
    rows = counter_table.read_rows(os.path.join(DATA, source + ".csv"))
    path = tmp_path / (name + ".csv")
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            change(row)
            writer.writerow(row)
    return str(tmp_path / name)


def test_check_model_judges_the_baseline(tmp_path):
    """Probes as in the calibration, sleeps 5 % higher: BASELINE, and --write takes this GPU's baseline."""
    def higher_sleep(row):
        if row["level"] == "baseline" and row["probe"].startswith("sleep-"):
            row["power_w"] = repr(1.05 * float(row["power_w"]))
    run = edited_rows(tmp_path, "Probes", "Warm", higher_sleep)
    written = tmp_path / "baseline.model.json"
    result = tool("check_model.py", "--model", MODEL, "--run", run, "--write", str(written))
    assert "BASELINE" in result.stdout and "PASS" not in result.stdout
    adjusted = energy_predictor.Model.load(str(written))
    assert adjusted.baseline["mean_w"] == pytest.approx(1.05 * energy_predictor.Model.load(MODEL).baseline["mean_w"],
                                                        rel=0.01)


def test_check_model_compares_counts(tmp_path):
    result = tool("check_model.py", "--model", MODEL, "--run", "Probes", "--counters", "Profile")
    assert "PASS" in result.stdout and "0 move" in result.stdout

    def doubled(row):
        if row.get("dram__sectors_read.sum"):
            row["dram__sectors_read.sum"] = repr(2 * float(row["dram__sectors_read.sum"]))
    run = edited_rows(tmp_path, "Profile", "Doubled", doubled)
    result = tool("check_model.py", "--model", MODEL, "--run", "Probes", "--counters", run, check=False)
    assert result.returncode == 1 and "event counts differ" in result.stdout and "DRAM read sector" in result.stdout


# ---- check_run.py -------------------------------------------------------------------------------------------------------

def test_check_run_passes_on_the_reference_runs():
    result = tool("check_run.py", "Probes", "--counters", "Profile")
    assert "0 warnings, 0 failed" in result.stdout and "locality, write hit their level" in result.stdout


def test_check_run_finds_a_throttled_row(tmp_path):
    def throttled(row):
        if row["operation"] == "ld" and row["level"] == "DRAM" and row["width_bytes"] == "16" and row["duty_pct"] == "100":
            row["clock_difference_pct"] = "-9.5"
    run = edited_rows(tmp_path, "Probes", "Throttled", throttled)
    shutil.copy(os.path.join(DATA, "Probes_energy_table.json"), run + "_energy_table.json")
    result = tool("check_run.py", run, check=False)
    assert result.returncode == 1 and "FAIL  SM clock of every row" in result.stdout


# ---- The program without a GPU ------------------------------------------------------------------------------------------

@needs_binary
def test_cpp_predictor_equals_python(tmp_path, model, validation):
    prefix = str(tmp_path / "cpp")
    result = program("--predict", VALIDATION, "--model-file", MODEL, "--output", prefix)
    assert result.returncode == 0, result.stdout
    rows = counter_table.read_rows(prefix + "_predicted.csv")
    python = {}
    for kernel, configurations in validation.items():
        predictions = [model.predict(row, seconds, temperature, clock, label)
                       for label, row, seconds, temperature, clock in configurations]
        choice = energy_predictor.recommend(model, predictions)
        for i, p in enumerate(predictions):
            python[(kernel, p.label)] = (p, "best" if i == choice.index else "contender" if i in choice.contenders else "")
    assert len(rows) == len(python)
    for row in rows:
        p, mark = python[(row["kernel"], row["configuration"])]
        assert float(row["predicted_j"]) == pytest.approx(p.energy_j, rel=1e-9)
        assert float(row["predicted_dynamic_j"]) == pytest.approx(p.dynamic_j, rel=1e-9)
        assert float(row["predicted_sd_j"]) == pytest.approx(p.sd_j, rel=1e-9)
        assert row["choice"] == mark


@needs_binary
def test_cpp_predictor_objectives_and_errors(tmp_path):
    prefix = str(tmp_path / "cpp")
    for objective in ("edp", "ed2p", "time"):
        result = program("--predict", VALIDATION, "--model-file", MODEL, "--objective", objective, "--output", prefix)
        assert result.returncode == 0 and "objective " + objective in result.stdout
    assert program("--predict", VALIDATION, "--output", prefix).returncode == 1          # no model
    assert program("--predict", VALIDATION, "--model-file", MT4G, "--output", prefix).returncode == 1  # not a model
    assert program("--predict", VALIDATION, "--model-file", MODEL, "--objective", "speed").returncode == 1
    # A counter the model needs is missing: an error per kernel, or a warning with --lenient.
    rows = counter_table.read_rows(VALIDATION)
    broken = tmp_path / "broken_validation.csv"
    with open(broken, "w", newline="") as file:
        names = [name for name in rows[0] if name != "dram__sectors_read.sum"]
        writer = csv.DictWriter(file, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    strict = program("--predict", str(broken), "--model-file", MODEL, "--output", prefix)
    assert strict.returncode == 1 and "dram__sectors_read" in strict.stdout
    lenient = program("--predict", str(broken), "--model-file", MODEL, "--output", prefix, "--lenient")
    assert lenient.returncode == 0 and "WARNING" in lenient.stdout


@needs_binary
def test_dry_run_plans():
    default = program("--dry-run", "--mt4g", MT4G)
    assert default.returncode == 0 and "242 configurations (12 of them baselines)" in default.stdout
    # The plan is the run of the reference data: one row per configuration (and three idle readings).
    assert len(counter_table.read_rows(os.path.join(DATA, "Probes.csv"))) == 242 + 3
    quick = program("--dry-run", "--mt4g", MT4G, "--quick")
    assert quick.returncode == 0 and "39 configurations" in quick.stdout
    profile = program("--dry-run", "--mt4g", MT4G, "--profile")
    assert profile.returncode == 0 and "163 configurations" in profile.stdout
    assert len(counter_table.read_rows(os.path.join(DATA, "Profile.csv"))) == 163
    # An explicit option wins over a preset wherever it stands.
    assert "at duty 50 " in program("--dry-run", "--mt4g", MT4G, "--duty", "50", "--quick").stdout
    only = program("--dry-run", "--mt4g", MT4G, "--only", "latency,banks")
    assert only.returncode == 0 and "27 configurations (3 of them baselines)" in only.stdout


@needs_binary
@pytest.mark.parametrize("arguments, message", [
    (["--duty", "abc"], "--duty"),
    (["--duty", "0"], "--duty"),
    (["--only", "texture"], "--only"),
    (["--block-size", "100"], "--block-size"),
    (["--max-power-diff", "-1"], "--max-power-diff"),
    (["--unroll", "6"], "--unroll"),
    (["--validate", "all", "--profile"], "--profile"),
    (["--tune", "gemm"], "--model-file"),
    (["--tune", "gemm", "--model-file", "x", "--tune-measure", "some"], "--tune-measure"),
    (["--tune", "mixed", "--model-file", "x"], "--tune"),
    (["--tune", "gemm", "--model-file", "x", "--accounting", "idle"], "--accounting"),
    (["--tune", "gemm", "--model-file", "x", "--idle-w", "40"], "--idle-w"),
    (["--tune", "gemm", "--model-file", "x", "--max-energy-increase", "-1"], "--max-energy-increase"),
    (["--no-such-option"], "Unknown option"),
])
def test_bad_options_are_refused(arguments, message):
    result = program("--dry-run", "--mt4g", MT4G, *arguments)
    assert result.returncode == 1 and message in result.stdout


@needs_binary
def test_analyze_reproduces_the_energy_table(tmp_path):
    """--analyze on the reference run's CSV gives the energies of its energy table again."""
    shutil.copy(os.path.join(DATA, "Probes.csv"), tmp_path / "Probes.csv")
    prefix = str(tmp_path / "again")
    result = program("--analyze", str(tmp_path / "Probes"), "--mt4g", MT4G, "--output", prefix)
    assert result.returncode == 0, result.stdout[-2000:]
    with open(prefix + "_energy_table.json") as file:
        again = json.load(file)
    with open(os.path.join(DATA, "Probes_energy_table.json")) as file:
        reference = json.load(file)
    key = lambda f: (f["level"], f["operation"], f["width_bytes"])  # noqa: E731
    fits = {key(f): f["pj_per_access"] for f in reference["fits"]}
    assert len(again["fits"]) == len(fits)
    for fit in again["fits"]:
        assert fit["pj_per_access"] == pytest.approx(fits[key(fit)], rel=1e-6)
    assert again["baseline"]["slope_w_per_c"] == pytest.approx(reference["baseline"]["slope_w_per_c"], rel=1e-6)
    assert again["idle"][0]["power_w"] == pytest.approx(reference["idle"][0]["power_w"], rel=1e-6)
    # The rows come out as they went in (energies recomputed).
    before = counter_table.read_rows(os.path.join(DATA, "Probes.csv"))
    after = counter_table.read_rows(prefix + ".csv")
    assert len(after) == len(before)
    for a, b in zip(before, after):
        assert a["operation"] == b["operation"] and a["note"] == b["note"]
        if a["pj_per_access"]:
            assert float(b["pj_per_access"]) == pytest.approx(float(a["pj_per_access"]), rel=1e-6)


# ---- SASS (slow: about 3 min; EP_SLOW=1) --------------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("EP_SLOW") or not shutil.which("nvcc", path=os.environ.get("PATH", "")
                                                                         + os.pathsep + "/usr/local/cuda/bin"),
                    reason="set EP_SLOW=1 (needs a CUDA toolkit, no GPU)")
def test_sass_checks():
    result = tool("check_sass.py", "--arch", "75,86,90", check=False)
    assert result.returncode == 0 and "all checks passed" in result.stdout, result.stdout[-3000:]


# ---- A real KTT output file ---------------------------------------------------------------------------------------------

def test_real_ktt_output_file(model, validation):
    """Ktt_gemm_counters.json is what KTT saved for the counter pass of the validation run (6 of its Gemm
    configurations): the predictor reads it, and the dynamic energies are those from the run's CSV, which has the same
    counters (the launch times differ: profiled launches take longer, and the time enters through the DRAM busy event)."""
    kernels = energy_predictor.from_ktt_json(os.path.join(DATA, "Ktt_gemm_counters.json"))
    assert list(kernels) == ["Gemm"] and len(kernels["Gemm"]) == 6  # the KTT kernel's name
    from_csv = {label: model.predict(row, seconds, strict=False) for label, row, seconds, _, _ in validation["gemm"]}
    for label, counters, seconds, temperature, clock in kernels["Gemm"]:
        p = model.predict(counters, seconds, temperature, clock, label)
        assert p.dynamic_j == pytest.approx(from_csv[label].dynamic_j, rel=0.05)
    result = tool("energy_predictor.py", "--model", MODEL, "--ktt", os.path.join(DATA, "Ktt_gemm_counters.json"))
    assert "use: " in result.stdout


@needs_binary
def test_cpp_kernel_scale_equals_python(tmp_path, validation):
    """KernelScale in C++ and kernel_scale in Python: the same scale, the same scaled dynamic energy and the same pair sd
    against another kernel (the residual bias of a scaled kernel), and no compounding on a second call."""
    ktt = os.path.abspath(os.path.join(ROOT, "..", ".."))
    kernels = list(validation)
    (la, ca, sa, ta, _), (lb, cb, sb, tb, _) = validation[kernels[0]][0], validation[kernels[1]][0]
    files = []
    for name, counters in (("a", ca), ("b", cb)):
        path = tmp_path / (name + ".txt")
        values = {k: counter_table.number(v) for k, v in counters.items()}
        path.write_text("".join(f"{k} {v!r}\n" for k, v in values.items() if not math.isnan(v)))
        files.append(str(path))
    source = tmp_path / "scale.cpp"
    source.write_text(r'''
#include <cstdio>
#include <fstream>
#include "EnergyModel.h"
CounterValues Read(const char* path) { CounterValues c; std::ifstream f(path); std::string n; double v; while (f >> n >> v) c[n] = v; return c; }
int main(int, char** argv)
{
    EnergyModel model(argv[1]);
    const CounterValues a = Read(argv[2]), b = Read(argv[3]);
    const double sa = std::atof(argv[4]), ta = std::atof(argv[5]), sb = std::atof(argv[6]), tb = std::atof(argv[7]);
    const EnergyPrediction plain = model.Predict(a, sa, ta, NAN, false, "a");
    std::vector<EnergyPrediction> three(3, plain);
    std::vector<double> measured(3, 1.1 * plain.dynamicJ);
    const double s1 = model.KernelScale("a", three, measured);
    const EnergyPrediction scaled = model.Predict(a, sa, ta, NAN, false, "a");
    std::vector<EnergyPrediction> again(3, scaled);
    const double s2 = model.KernelScale("a", again, measured);
    const EnergyPrediction other = model.Predict(b, sb, tb, NAN, false, "b");
    std::printf("%.17g %.17g %.17g %.17g\n", s1, s2, scaled.dynamicJ, model.PairSd(scaled, other));
}
''')
    output = tmp_path / "scale"
    command = ["g++", "-std=c++17", "-O1", "-I" + ROOT, "-I" + os.path.join(ktt, "Libraries", "Json-3.9.1"), str(source),
               os.path.join(ROOT, "EnergyModel.cpp"), "-o", str(output)]
    built = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert built.returncode == 0, built.stdout
    result = subprocess.run([str(output), MODEL] + files + [repr(sa), repr(ta), repr(sb), repr(tb)],
                            stdout=subprocess.PIPE, universal_newlines=True)
    s1, s2, dynamic, pair = (float(x) for x in result.stdout.split())
    python = energy_predictor.Model.load(MODEL)
    plain = python.predict(ca, sa, ta, strict=False, group="a")
    assert python.kernel_scale("a", [plain] * 3, [1.1 * plain.dynamic_j] * 3) == pytest.approx(s1, rel=1e-9)
    scaled = python.predict(ca, sa, ta, strict=False, group="a")
    assert python.kernel_scale("a", [scaled] * 3, [1.1 * plain.dynamic_j] * 3) == pytest.approx(s2, rel=1e-9)
    assert s1 == pytest.approx(1.1, rel=1e-9) and s2 == pytest.approx(1.1, rel=1e-9)
    other = python.predict(cb, sb, tb, strict=False, group="b")
    assert scaled.dynamic_j == pytest.approx(dynamic, rel=1e-9)
    assert python.pair_sd(scaled, other) == pytest.approx(pair, rel=1e-9)
