"""Regression tests of the EnergyProbes tools and of the program's GPU-free modes.

    python3 -m pytest Examples/EnergyProbes/tests -q            # from the KTT root; about half a minute
    EP_BINARY=/path/to/EnergyProbesCuda python3 -m pytest ...   # another build (default: Build/x86_64_Release)

No GPU is needed. The data in tests/data are real runs of an RTX 3090 at 1200 MHz, made with the commands of the
README: the probe run (Probes.csv and its energy table), the counter run (Profile.csv), the validation run
(Validation_validation.csv) and a KTT output file of its counter pass. models/ has the model built from them and the
MT4G file. The tests of the program (--predict, --dry-run, option checks) are skipped when it is not built.
"""

import collections
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

# The runs the shipped model is built from (since 2026-11-17, model 2d90caf0): the probes' power runs and the counter runs
# Profile2 (the whole calibration plan) and LanesProfile2, both with smsp__thread_inst_executed.sum.
REFERENCE_RUNS = ("--power", "Probes,Tensor1,LatSweep1,Rows1,Lanes1", "--counters", "Profile2,LanesProfile2")


def test_the_model_file_is_what_the_reference_runs_give(tmp_path, model):
    """build_model.py on the reference runs reproduces the energies of the shipped model: a change in the fit, the events
    or the CSV reading shows here."""
    output = tmp_path / "rebuilt.model.json"
    result = tool("build_model.py", *REFERENCE_RUNS, "--calibration", "AllMany2Dev2", "--output", str(output))
    rebuilt = energy_predictor.Model.load(str(output))
    for name, energy in model.pj.items():
        assert rebuilt.pj[name] == pytest.approx(energy, rel=1e-6, abs=1e-6), name
    assert rebuilt.baseline == model.baseline
    assert rebuilt.required == model.required
    assert rebuilt.uncertainty["dynamic_relative_sd"] == pytest.approx(model.uncertainty["dynamic_relative_sd"], rel=1e-6)
    # 150 of the calibration run, the IMMA probe, 16 more latency probes, the 4 rows probes and the 4 lanes probes
    # (2026-11-17; the counts of all of them from Profile2 and LanesProfile2, which have the thread counter)
    assert len(rebuilt.data["probes"]) == len(model.data["probes"]) == 175
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
        probes, table = m.probe_table([os.path.join(DATA, r) for r in REFERENCE_RUNS[1].split(",")],
                                      [os.path.join(DATA, r) for r in REFERENCE_RUNS[3].split(",")])
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
    """Two predictions 10 % apart: told apart within one kernel, contenders when they are two kernels (the bias between
    kernels widens the sd per prediction)."""
    within = [prediction(1.0, 0.5, 1.0)._replace(group="a"), prediction(1.10, 0.55, 1.0)._replace(group="a")]
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
    assert model.data["uncertainty"]["calibration"]["left_out_off_clock"] == 3
    assert set(model.data["options"]) == {"busy_scale_gbs", "busy_scale_source", "busy_scale_scan"}
    assert model.data["options"]["busy_scale_gbs"] == 95.0 and "inside this GPU's interval" in \
        model.data["options"]["busy_scale_source"]
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
    assert default.returncode == 0 and "265 configurations (13 of them baselines)" in default.stdout
    # The plan is the run of the reference data: one row per configuration (and three idle readings), but for the IMMA
    # probe, which joined the compute group after it (2026-11-16; its rows are in Tensor1 and ComputeProfile1), and the
    # 16 latency probes of the steady-rate sweep and the 4 rows probes (2026-11-17; their rows are in runs of their own).
    assert len(counter_table.read_rows(os.path.join(DATA, "Probes.csv"))) == 265 - 20 - 1 - 2 + 3
    quick = program("--dry-run", "--mt4g", MT4G, "--quick")
    assert quick.returncode == 0 and "40 configurations" in quick.stdout
    profile = program("--dry-run", "--mt4g", MT4G, "--profile")
    assert profile.returncode == 0 and "185 configurations" in profile.stdout
    # (and the rows group's sleep baseline)
    assert len(counter_table.read_rows(os.path.join(DATA, "Profile.csv"))) == 185 - 20 - 1 - 1
    # An explicit option wins over a preset wherever it stands.
    assert "at duty 50 " in program("--dry-run", "--mt4g", MT4G, "--duty", "50", "--quick").stdout
    only = program("--dry-run", "--mt4g", MT4G, "--only", "latency,banks")
    assert only.returncode == 0 and "43 configurations (3 of them baselines)" in only.stdout


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
    (["--validate", "heldout6"], "--validate"),
    (["--validate", "reduction,sortreduce"], "cannot run together"),
    (["--validate", "gemm", "--validate-data", "zeros"], "--validate-data"),
    (["--validate", "atfconv", "--validate-data", "ones"], "--validate-data"),
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


# ---- Newer architectures (handoff/NEWER_ARCHITECTURES.md, 2026-11-16) ---------------------------------------------------

TENSOR_HFMA2 = "smsp__inst_executed_pipe_tensor_op_hmma_type_hfma2.sum"
TENSOR_DMMA = "smsp__inst_executed_pipe_tensor_op_dmma.sum"


def with_tensor_extras(row, hfma2=1000.0, dmma=500.0):
    """A copy of a counter row as GA100/GH100 would report it: HFMA2.MMA and DMMA counted on the tensor pipe too."""
    changed = dict(row)
    changed[TENSOR_HFMA2] = str(hfma2)
    changed[TENSOR_DMMA] = str(dmma)
    for name in ("smsp__inst_executed_pipe_tensor.sum", "smsp__inst_executed.sum"):  # the total counts them too
        changed[name] = str(float(row.get(name) or 0) + hfma2 + dmma)
    return changed


def test_tensor_pipe_extras_are_priced_as_fp16_and_fp64(validation):
    """HFMA2.MMA on the tensor pipe is fp16 work, DMMA on it FP64 (as on GA102, where DMMA is part of the fp64 pipe);
    "inst tensor" keeps only the matrix products, and the other events do not move."""
    _, row, _, _, _ = next(iter(validation.values()))[0]
    before, after = energy_events.events(row), energy_events.events(with_tensor_extras(row))
    assert after["inst tensor"] == pytest.approx(before["inst tensor"])
    assert after["inst fp16"] == pytest.approx(before["inst fp16"] + 1000)
    assert after["inst fp64"] == pytest.approx(before["inst fp64"] + 500)
    assert after["inst other"] == pytest.approx(before["inst other"])
    for name in energy_events.EVENTS:
        if name not in ("inst fp16", "inst fp64"):
            assert after[name] == pytest.approx(before[name]), name


def test_optional_counters_are_the_same_in_python_and_cpp():
    with open(os.path.join(ROOT, "EnergyModel.cpp")) as file:
        source = file.read()
    block = source.split("DefaultOptional = {", 1)[1].split("};", 1)[0]
    assert re.findall(r'"([^"]+)"', block) == list(energy_predictor.OPTIONAL_COUNTERS)
    for name in (TENSOR_HFMA2, TENSOR_DMMA, "smsp__inst_executed_pipe_tensor_subpipe_dmma.sum"):
        assert counter_table.ENTRIES[name].get("optional") and name in energy_predictor.OPTIONAL_COUNTERS


@needs_binary
def test_cpp_prices_the_tensor_pipe_extras_like_python(tmp_path, model):
    rows = counter_table.read_rows(VALIDATION)
    edited = tmp_path / "Tensor_validation.csv"
    names = list(rows[0]) + [TENSOR_HFMA2, TENSOR_DMMA]
    with open(edited, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(with_tensor_extras(row) for row in rows)
    prefix = str(tmp_path / "cpp")
    result = program("--predict", str(edited), "--model-file", MODEL, "--output", prefix)
    assert result.returncode == 0, result.stdout
    python = {}
    for kernel, configurations in energy_predictor.from_validation_csv(str(edited)).items():
        for label, row, seconds, temperature, clock in configurations:
            python[(kernel, label)] = model.predict(row, seconds, temperature, clock, label).dynamic_j
    cpp = counter_table.read_rows(prefix + "_predicted.csv")
    assert len(cpp) == len(python) > 100
    for row in cpp:
        assert float(row["predicted_dynamic_j"]) == pytest.approx(python[(row["kernel"], row["configuration"])], rel=1e-9)


def test_check_chip_metrics(tmp_path):
    """tools/check_chip_metrics.py on listings like `ncu --query-metrics --chip X`: a chip with every counter passes; one
    without the DRAM sectors (GB10, unified LPDDR5x) fails and names the events; counters above a chip's compute
    capability (LDGSTS on Turing) are not expected."""
    def listing(path, drop=()):
        names = {name.split(".", 1)[0] for name in counter_table.NAMES} - set(drop)
        path.write_text("Device TU102 (no GPU)\n" + "".join(f"{n}   Counter   # something\n" for n in sorted(names)))
        return str(path)
    full = tool("check_chip_metrics.py", "--metrics", listing(tmp_path / "ga102.txt"))
    assert "compute capability 8.6" in full.stdout and "MISSING" not in full.stdout
    no_dram = tool("check_chip_metrics.py", "--metrics", listing(tmp_path / "gb10b.txt", ("dram__sectors_read",
                   "dram__sectors_write", "dram__sectors_op_read", "dram__sectors_op_write")), check=False)
    assert no_dram.returncode == 1 and "DRAM read sector" in no_dram.stdout
    turing = tool("check_chip_metrics.py", "--metrics", listing(tmp_path / "tu102.txt", ("smsp__inst_executed_op_ldgsts",
                  "smsp__inst_executed_pipe_tensor_op_dmma", "smsp__inst_executed_pipe_tensor_op_hmma_type_hfma2")))
    assert "compute capability 7.5" in turing.stdout and "ldgsts" not in turing.stdout


def test_sass_check_plans_the_tensor_probes_per_architecture():
    sys.path.insert(0, TOOLS)
    import check_sass
    assert [check_sass.planned(13, a) for a in (75, 80, 86, 90, 100)] == [False, True, True, True, True]
    assert [check_sass.planned(16, a) for a in (86, 89, 90, 100, 120, 121)] == [False, True, False, False, True, True]
    assert set(check_sass.TENSOR_OPCODES) == {13, 14, 15, 16}


def test_tensor_probes_stay_out_of_the_fit():
    import energy_model as m
    tensor = {"compute dmma", "compute hmma-sparse", "compute qmma-fp8"}
    assert m.CHECK_PROBES == tensor and m.IMMA_PROBE == "compute imma"
    with open(os.path.join(ROOT, "Common.cpp")) as file:
        source = file.read()
    for label in tensor | {m.IMMA_PROBE}:
        assert '"' + label.split(" ", 1)[1] + '"' in source, label


@needs_binary
def test_dry_run_tensor_group():
    """The tensor group is optional: not in the default plan, and on the RTX 3090 (sm_86) without FP8."""
    assert "tensor" not in program("--dry-run", "--mt4g", MT4G).stdout.split("Plan", 1)[1].split("configurations (")[0]
    tensor = program("--dry-run", "--mt4g", MT4G, "--only", "tensor")
    assert tensor.returncode == 0 and "6 configurations (2 of them baselines)" in tensor.stdout


def test_counter_table_marks_newer_counters():
    """Counters of newer chips carry the lowest compute capability that has them, so that an older GPU is not asked for
    them (ModelCounters in CounterTable.cpp) and check_chip_metrics.py does not expect them."""
    for name in ("smsp__inst_executed_pipe_tc.sum", "smsp__inst_executed_pipe_tmem.sum",
                 "smsp__inst_executed_pipe_tensor_op_gmma.sum", "smsp__inst_executed_pipe_tma.sum",
                 "lrc__ilc_output_sectors.sum"):
        assert counter_table.ENTRIES[name]["min_cc"] >= 9.0 and "use" in counter_table.ENTRIES[name], name
    with open(os.path.join(ROOT, "CounterTable.cpp")) as file:
        assert 'entry.contains("min_cc")' in file.read()


TENSOR_IMMA = "smsp__inst_executed_pipe_tensor_op_imma.sum"


def test_imma_has_a_price_of_its_own(model):
    """The IMMA probe prices "inst tensor imma" (2026-11-16): 3.32 against 2.44 nJ per warp instruction for HMMA on the
    RTX 3090, so the model's prices differ by about that much."""
    assert model.pj["inst tensor imma"] / model.pj["inst tensor"] == pytest.approx(3.32 / 2.44, rel=0.03)
    labels = [p["label"] for p in model.data["probes"]]
    assert "compute imma" in labels and "compute dmma" not in labels


@needs_binary
def test_model_files_without_the_imma_event_price_imma_as_tensor(tmp_path, model):
    """A model file from before 2026-11-16 has no "inst tensor imma": both predictors price IMMA with "inst tensor", as
    the model was fitted then; and with the event, C++ and Python agree on a run that counts IMMA."""
    data = json.loads(open(MODEL).read())
    del data["events"]["inst tensor imma"]
    old = tmp_path / "old.model.json"
    old.write_text(json.dumps(data))
    old_model = energy_predictor.Model.load(str(old))
    assert old_model.pj["inst tensor imma"] == old_model.pj["inst tensor"]
    rows = counter_table.read_rows(VALIDATION)
    edited = tmp_path / "Imma_validation.csv"
    with open(edited, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) + [TENSOR_IMMA], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            changed = dict(row)
            changed[TENSOR_IMMA] = "2000"
            for name in ("smsp__inst_executed_pipe_tensor.sum", "smsp__inst_executed.sum"):
                changed[name] = str(float(row.get(name) or 0) + 2000)
            writer.writerow(changed)
    configurations = energy_predictor.from_validation_csv(str(edited))
    for path, priced in ((MODEL, model), (str(old), old_model)):
        prefix = str(tmp_path / ("cpp_old" if priced is old_model else "cpp_new"))
        result = program("--predict", str(edited), "--model-file", path, "--output", prefix)
        assert result.returncode == 0, result.stdout
        python = {(kernel, label): priced.predict(row, seconds, temperature, clock, label).dynamic_j
                  for kernel, entries in configurations.items() for label, row, seconds, temperature, clock in entries}
        for row in counter_table.read_rows(prefix + "_predicted.csv"):
            assert float(row["predicted_dynamic_j"]) == pytest.approx(python[(row["kernel"], row["configuration"])], rel=1e-9)


# ---- The RTX 2080 Ti (Turing, sm_75; lean program of 2026-11-14, run 2026-11-15/16) ------------------------------------
TURING = os.path.join(DATA, "rtx2080ti")
TURING_MODEL = os.path.join(ROOT, "models", "rtx2080ti_1200mhz.model.json")


def test_rtx2080ti_model_file_is_what_its_runs_give(tmp_path):
    """build_model.py on the 2080 Ti's runs reproduces its shipped model. Turing has no IMMA probe (m16n8k32 needs
    sm_80), so IMMA is priced as an HMMA warp instruction, not as free."""
    output = tmp_path / "rebuilt.model.json"
    result = tool("build_model.py", "--power", "Probes", "--counters", "Profile", "--calibration", "Validation",
                  "--output", str(output), cwd=TURING)
    shipped = energy_predictor.Model.load(TURING_MODEL)
    rebuilt = energy_predictor.Model.load(str(output))
    for name, energy in shipped.pj.items():
        assert rebuilt.pj[name] == pytest.approx(energy, rel=1e-6, abs=1e-6), name
    assert rebuilt.uncertainty["dynamic_relative_sd"] == pytest.approx(shipped.uncertainty["dynamic_relative_sd"], rel=1e-6)
    assert shipped.device["compute_capability"] == "7.5" and len(shipped.data["probes"]) == 150
    assert shipped.data["events"]["inst tensor imma"]["status"] == "as inst tensor"
    assert shipped.pj["inst tensor imma"] == shipped.pj["inst tensor"] > 0
    assert "inst tensor imma" not in shipped.unpriced
    assert "priced as inst tensor" in result.stdout


def test_check_run_on_the_rtx2080ti_runs():
    """The latency probes pass although CUPTI counted their requests and sectors in different replays (0.96 requests,
    4.04 sectors per access), and Turing's DRAM over-fetch for isolated sectors is a warning. The one failure is the
    placement of the 16-byte write probes, which these runs made with the kernels before 2026-11-16."""
    result = tool("check_run.py", "Probes", "--counters", "Profile", cwd=TURING, check=False)
    assert "PASS  latency probes" in result.stdout
    assert "WARN  DRAM sectors per L2 read sector <= 1" in result.stdout
    assert "1 failed" in result.stdout and "FAIL  blocks placed one stripe each" in result.stdout


def test_check_model_tells_the_gpus_apart():
    assert "PASS" in tool("check_model.py", "--model", TURING_MODEL, "--run", "Probes", cwd=TURING).stdout
    result = tool("check_model.py", "--model", MODEL, "--run", "Probes", cwd=TURING, check=False)
    assert result.returncode != 0 and "FAIL" in result.stdout


@needs_binary
def test_cpp_predictor_equals_python_on_the_rtx2080ti(tmp_path):
    turing = energy_predictor.Model.load(TURING_MODEL)
    validation = os.path.join(TURING, "Validation_validation.csv")
    prefix = str(tmp_path / "cpp")
    result = program("--predict", validation, "--model-file", TURING_MODEL, "--output", prefix)
    assert result.returncode == 0, result.stdout
    python = {(kernel, label): turing.predict(row, seconds, temperature, clock, label)
              for kernel, entries in energy_predictor.from_validation_csv(validation).items()
              for label, row, seconds, temperature, clock in entries}
    rows = counter_table.read_rows(prefix + "_predicted.csv")
    assert len(rows) == len(python) == 206
    for row in rows:
        p = python[(row["kernel"], row["configuration"])]
        assert float(row["predicted_j"]) == pytest.approx(p.energy_j, rel=1e-9)
        assert float(row["predicted_sd_j"]) == pytest.approx(p.sd_j, rel=1e-9)


# ---- The grid walk of probeScatter and probePartial (Kimi review P1, 2026-11-16) ----------------------------------------
SECTOR = 32
L2_FOOTPRINT = 2 << 20        # the planner's L2 footprint for the RTX 3090's MT4G file
L1_BYTES = 108288             # its L1 size; an SM must walk at least 4x this at the L2 footprint
AFFECTED_SMS = (64, 80, 144)  # where the grid walk keeps an SM within 4x the L1


def walk_sectors(kernel, sms, steps=None, blocks_per_sm=4, block_size=256, footprint=L2_FOOTPRINT):
    """The sectors every counter of the grid touches, by step, with the address arithmetic of kernels/Scatter.cu (stream:
    stride 1, runBits 0, 16-byte elements, 2 threads per sector) and kernels/Partial.cu (stride 1, a sector per thread):
    an array [step, counter], the counters of a block being consecutive."""
    import numpy as np
    blocks = sms * blocks_per_sm
    per_block = block_size * 16 // SECTOR if kernel == "scatter" else block_size
    step = blocks * per_block
    mask = footprint // SECTOR - 1
    start = np.arange(blocks * per_block, dtype=np.int64)
    steps = steps or (mask + 1) // math.gcd(step, mask + 1)
    return (start[None, :] + step * np.arange(steps, dtype=np.int64)[:, None]) & mask, per_block


def walk_per_sm(kernel, sms):
    """Bytes of the footprint the 4 blocks of SM 0 walk over a whole cycle (blocks placed round-robin, block b on SM
    b mod SMs), simulated, and by the planner's formula (GridWalkPerSm in ProbePlan.cpp)."""
    import numpy as np
    sectors, per_block = walk_sectors(kernel, sms)
    mine = np.concatenate([np.arange(b * per_block, (b + 1) * per_block) for b in range(0, 4 * sms, sms)])
    simulated = len(np.unique(sectors[:, mine])) * SECTOR
    total = L2_FOOTPRINT // SECTOR
    classes = max(math.gcd(4 * sms * per_block, total) // per_block, 1)
    formula = min(L2_FOOTPRINT, len({i * sms % classes for i in range(4)}) * (L2_FOOTPRINT // classes))
    return simulated, formula


@pytest.mark.parametrize("sms", [64, 68, 80, 82, 132, 144])
def test_grid_walk_per_sm_matches_the_planner(sms):
    """The data an SM walks at the L2 footprint, simulated from the kernels' addresses, equals the planner's formula; it
    is at least 4x the L1 on 68, 82 and 132 SMs (the 2080 Ti and the 3090 measured it), and below on 64, 80 and 144,
    where the planner leaves these groups' L2 rows out."""
    for kernel in ("scatter", "partial"):
        simulated, formula = walk_per_sm(kernel, sms)
        assert simulated == formula, (kernel, simulated, formula)
        assert (simulated >= 4 * L1_BYTES) == (sms not in AFFECTED_SMS), (kernel, sms, simulated)


@pytest.mark.parametrize("kernel", ["scatter", "partial"])
@pytest.mark.parametrize("sms", [64, 80, 82])
def test_grid_walk_stays_coalesced(kernel, sms):
    """In every step the counters of a warp are consecutive sectors starting on a 128-byte line (Scatter: 16 sectors of 2
    threads = 4 lines; Partial: 32 sectors = 8 lines), and the grid touches every sector as evenly as it can: once where
    it has fewer counters than the footprint has sectors (Scatter), else at most twice (Partial at the L2 footprint:
    83968 counters on 65536 sectors on the RTX 3090)."""
    import numpy as np
    sectors, per_block = walk_sectors(kernel, sms, steps=64)
    total = L2_FOOTPRINT // SECTOR
    for row in sectors:
        assert np.bincount(row, minlength=total).max() == -(-len(row) // total)
    per_warp = 16 if kernel == "scatter" else 32
    warps = sectors.reshape(sectors.shape[0], -1, per_warp)
    assert np.all(warps[:, :, 0] % 4 == 0)
    assert np.all(np.diff(warps, axis=2) == 1)


@needs_binary
def test_the_planner_leaves_out_l2_rows_that_would_measure_l1(tmp_path):
    with open(MT4G) as file:
        description = json.load(file)
    for sms in (64, 68, 80, 82, 132, 144):
        description["compute"]["multiProcessorCount"] = sms
        path = tmp_path / f"sm{sms}.json"
        path.write_text(json.dumps(description))
        result = program("--mt4g", str(path), "--dry-run")
        assert result.returncode == 0, result.stdout
        assert "Footprints: L1 32 KiB, L2 2 MiB" in result.stdout and "L1 size: 108288 B" in result.stdout
        affected = sms in AFFECTED_SMS
        assert ("their L2 rows would measure L1" in result.stdout) == affected, sms
        # 28 configurations: the L2 rows of locality (8), sub-sector (8) and write-path (12).
        assert f"{237 if affected else 265} configurations" in result.stdout, sms


# ---- One clock filter for every score (Kimi review P2, 2026-11-16) ------------------------------------------------------
def test_scores_leave_out_off_clock_rows_everywhere():
    """A configuration that ran 10 % below the model's clock, cheapest and fastest by far, decides both regrets when it
    is scored, and none when the fixed-clock filter (the default, as in build_model.py) leaves it out."""
    sys.path.insert(0, TOOLS)
    import energy_model
    probes, table = energy_model.probe_table([os.path.join(DATA, "Probes"), os.path.join(DATA, "Tensor1")],
                                             [os.path.join(DATA, "Profile"), os.path.join(DATA, "ComputeProfile1")])
    energies, _, _ = energy_model.fit(probes)
    rows = [r for r in energy_model.load_validation(os.path.join(DATA, "Validation"), table) if r["kernel"] == "nbody"]
    on_clock = [r for r in rows if abs(float(r["sm_clock_mhz"]) / 1200 - 1) <= energy_model.CLOCK_TOLERANCE]
    assert len(on_clock) == len(rows) >= 10
    throttled = copy.deepcopy(rows)
    victim = throttled[0]
    victim["sm_clock_mhz"] = "1080"
    victim["energy_j"] = 0.5 * min(r["energy_j"] for r in rows)   # measured best by far
    victim["kernel_s"] = str(0.5 * min(float(r["kernel_s"]) for r in rows))  # and the fastest: time-only picks it
    kept, off = energy_model.at_model_clock(throttled, 1200)
    assert off == [victim] and len(kept) == len(rows) - 1
    with_all = energy_model.score({"V": throttled}, energies)[0]
    fixed = energy_model.score({"V": throttled}, energies, clock_mhz=1200)[0]
    reference = energy_model.score({"V": rows[1:]}, energies)[0]
    assert with_all[2] == len(rows) and fixed[2] == len(rows) - 1
    assert fixed[3:] == pytest.approx(reference[3:])               # scatter, bias, MAPE, regret, time-only regret
    assert with_all[7] == 0.0 and with_all[6] > 0.5                # time-only picks the victim; the model misses it


def test_energy_model_reports_its_scope():
    result = tool("energy_model.py", "--power", "Probes,Tensor1", "--counters", "Profile,ComputeProfile1",
                  "--validation", "Validation")
    assert "Scope: fixed clock, 1200 MHz +- 2 %: 204 of 206 configurations; left out off the clock: Validation gemm 2" \
        in result.stdout
    everything = tool("energy_model.py", "--power", "Probes,Tensor1", "--counters", "Profile,ComputeProfile1",
                      "--validation", "Validation", "--all-clocks")
    assert "Scope: all clocks" in everything.stdout


# ---- The DRAM busy scale on each GPU (Kimi review P3, 2026-11-16) -------------------------------------------------------
def test_busy_scale_scan_on_both_gpus():
    """The probes need the busy event on both GPUs and bound its scale from above (about 200 and 320 GB/s), not from
    below; the RTX 3090's 95 GB/s lies inside both intervals, so the shipped models keep it and say why."""
    sys.path.insert(0, TOOLS)
    import energy_model
    for power, counters, high in ((["Probes", "Tensor1"], ["Profile", "ComputeProfile1"], 202.0),
                                  (["rtx2080ti/Probes"], ["rtx2080ti/Profile"], 320.0)):
        scale, source, result = energy_model.busy_scale_choice([os.path.join(DATA, p) for p in power],
                                                               [os.path.join(DATA, c) for c in counters])
        assert result["needed"] and result["interval"] == (None, high) and 40 <= result["best"] <= 70
        assert scale == 95.0 and "inside this GPU's interval" in source
    shipped = json.loads(open(TURING_MODEL).read())["options"]
    assert shipped["busy_scale_scan"]["interval_gbs"] == [None, 320.0]


def test_busy_scale_outside_the_interval_is_replaced(tmp_path):
    """An inherited scale outside this GPU's interval is not kept: the model uses the scan's best and says so."""
    sys.path.insert(0, TOOLS)
    import energy_model
    scale, source, result = energy_model.busy_scale_choice([os.path.join(DATA, "Probes")], [os.path.join(DATA, "Profile")],
                                                           inherited=2000.0)
    assert scale == result["best"] and "fitted on these probes" in source
    result = tool("build_model.py", "--power", "Probes", "--counters", "Profile", "--busy-scale", "0",
                  "--output", str(tmp_path / "m.json"), check=False)
    assert result.returncode != 0 and "positive" in result.stdout


@pytest.mark.parametrize("value", [0, -5, "95", True])
def test_predictors_refuse_a_bad_busy_scale(tmp_path, value):
    data = json.loads(open(MODEL).read())
    data["options"]["busy_scale_gbs"] = value
    path = tmp_path / "bad.model.json"
    path.write_text(json.dumps(data))
    with pytest.raises(energy_predictor.ModelError, match="busy_scale_gbs"):
        energy_predictor.Model.load(str(path))
    if os.path.isfile(BINARY):
        result = program("--predict", VALIDATION, "--model-file", str(path), "--output", str(tmp_path / "p"))
        assert result.returncode != 0 and "busy_scale_gbs" in result.stdout


def test_fit_busy_reports_the_scan():
    result = tool("fit_busy.py", "--power", "Probes", "--counters", "Profile")
    assert "95 % interval <10 to 202 GB/s" in result.stdout and "build_model.py uses 95 GB/s" in result.stdout
    assert "Cross-check, dependent loads alone: 4 DRAM chase rows" in result.stdout


# ---- Another GPU type is an explicit choice (Kimi review P5) ------------------------------------------------------------
def test_python_predictor_refuses_another_gpus_ktt_output(tmp_path):
    source = os.path.join(DATA, "Ktt_gemm_counters.json")
    data = json.loads(open(source).read())
    text = json.dumps(data).replace("NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 2080 Ti")
    assert text != json.dumps(data)
    other = tmp_path / "other.json"
    other.write_text(text)
    result = tool("energy_predictor.py", "--model", MODEL, "--ktt", str(other), check=False)
    assert result.returncode != 0 and "--allow-uncalibrated-device" in result.stdout
    allowed = tool("energy_predictor.py", "--model", MODEL, "--ktt", str(other), "--allow-uncalibrated-device")
    assert "use:" in allowed.stdout
    assert "use:" in tool("energy_predictor.py", "--model", MODEL, "--ktt", source).stdout


@needs_binary
def test_tune_option_for_another_gpu_is_known():
    result = program("--help")
    assert "--allow-uncalibrated-device" in result.stdout


# ---- Identifiability and the bootstrap (Kimi review P6) -----------------------------------------------------------------
def test_bootstrap_and_identifiability_report():
    result = tool("energy_model.py", "--power", "Probes,Tensor1", "--counters", "Profile,ComputeProfile1",
                  "--validation", "Validation", "--bootstrap", "40")
    assert "Identifiability: condition number" in result.stdout and "L2 read miss request / DRAM read sector" in result.stdout
    line = next(l for l in result.stdout.splitlines() if "order flips" in l)
    flips = float(re.search(r"resamplings: ([0-9.]+) %", line).group(1))
    assert flips < 10


# ---- The held-out evaluations (Kimi review P4, 2026-11-16; heldout4 2026-11-17) -------------------------------------------
def test_heldout3_evaluation_of_the_earlier_model_is_on_record():
    """Five PolyBench kernels measured once (HeldOut3Frozen1) with the model of 2026-11-16 (SHA-256 a2192a090d43689d,
    built from Probes, Tensor1 / Profile, ComputeProfile1): scatter 5.8 %, max regret 0.9 % against 9.4 % by time, biases
    of -21 to +32 %. Since 2026-11-17 the set is reused (its kernels guided the rows probe and the timing fix); the
    numbers stay reproducible from the earlier runs."""
    result = tool("energy_model.py", "--power", "Probes,Tensor1", "--counters", "Profile,ComputeProfile1",
                  "--validation", os.path.join("heldout3", "HeldOut3Frozen1"))
    assert "120 of 120 configurations" in result.stdout
    assert "scatter 5.8 %, bias +4.6 %, total MAPE 4.7 %, max regret 0.9 % (time only 9.4 %)" in result.stdout
    biases = [float(m) for m in re.findall(r"HeldOut3Frozen1\s+\w+\s+24\s+[0-9.]+%\s+([+-][0-9.]+)%", result.stdout)]
    assert len(biases) == 5 and min(biases) == -21.3 and max(biases) == 31.6


def test_frozen_heldout4_evaluation_is_on_record():
    """Five more PolyBench kernels (kernels/HeldOut4.cu, written before any model change of 2026-11-17), measured once in
    AllMany2 with the fixed kernel time and scored once with the shipped model of 2026-11-17 (SHA-256 09b36a64a9fa3194,
    chosen before the scores were seen; docs/PROBES.md, "Which data was used for what"). If the shipped model's prices
    change, this evaluation is no longer of the shipped model: label it reused. It changed the same day (model 2d90caf0,
    with the inactive-lane event): the frozen evaluation is reproduced here from the runs of model 09b36a64."""
    result = tool("energy_model.py", "--power", "Probes,Tensor1,LatSweep1,Rows1", "--counters",
                  "Profile,ComputeProfile1,LatSweepProfile1,RowsProfile1",
                  "--validation", os.path.join("heldout4", "HeldOut4Frozen1"))
    assert "217 of 217 configurations" in result.stdout
    assert "scatter 2.5 %, bias +3.3 %, total MAPE 3.5 %, max regret 1.2 % (time only 4.8 %)" in result.stdout
    biases = [float(m) for m in re.findall(r"HeldOut4Frozen1\s+\w+\s+\d+\s+[0-9.]+%\s+([+-][0-9.]+)%", result.stdout)]
    assert len(biases) == 5 and min(biases) == -14.9 and max(biases) == 8.8


def test_frozen_heldout5_evaluation_is_on_record():
    """The CUDA kernels of KTT examples not used before (ATF's Gaussian convolution, Sort's radix reduce, Sort2's
    radixSortBlocks and findRadixOffsets; ValidationKernels.cpp, set heldout5), written after the model of 2026-11-17
    was fixed, measured once in KttMany1 and scored once with it. All four are over-predicted by 12-16 % (dynamic;
    sortreduce draws only 1-67 W above the baseline, so its relative dynamic error is large), the ranking holds. Since
    model 2d90caf0 (the same day) this is the evaluation of the earlier model 09b36a64, reproduced from its runs."""
    result = tool("energy_model.py", "--power", "Probes,Tensor1,LatSweep1,Rows1", "--counters",
                  "Profile,ComputeProfile1,LatSweepProfile1,RowsProfile1",
                  "--validation", os.path.join("heldout5", "KttMany1"))
    assert "140 of 140 configurations" in result.stdout
    assert "scatter 20.0 %, bias +20.7 %, total MAPE 5.2 %, max regret 0.9 % (time only 5.6 %)" in result.stdout
    biases = [float(m) for m in re.findall(r"KttMany1\s+\w+\s+\d+\s+[0-9.]+%\s+([+-][0-9.]+)%", result.stdout)]
    assert biases == [13.3, 41.7, 12.0, 15.7]


@needs_binary
def test_heldout5_and_data_modes_are_accepted():
    """The fifth set and the data modes parse (a dry run plans the probes; the kernels themselves need the GPU)."""
    for extra in ([], ["--validate-data", "zeros"], ["--validate-data", "random"]):
        result = program("--dry-run", "--mt4g", MT4G, "--validate", "heldout5", *extra)
        assert result.returncode == 0, result.stdout
    assert "heldout5 = atfconv" in program("--help").stdout


def test_data_values_record():
    """The same six kernels with their own data, zeros and random data (Data_example, Data_zeros, Data_random,
    2026-11-17; --validate-data). The predictions hardly move with the data, the measured power does: zeros draw 15-20 %
    less dynamic power, so every bias rises by 15-27 points. atfconv's own data (ATF's integers 1-100 as floats) accounts
    for its bias (+15.7 % own, +2.9 % random); the radix kernels' keys are random already and their bias stays. The
    ranking survives even zeros (max regret 3.5 %)."""
    def biases(mode):
        result = tool("energy_model.py", "--power", "Probes,Tensor1,LatSweep1,Rows1", "--counters",
                      "Profile,ComputeProfile1,LatSweepProfile1,RowsProfile1",
                      "--validation", os.path.join("datavalues", "Data_" + mode))
        found = re.findall(r"Data_" + mode + r"\s+(\w+)\s+\d+\s+[0-9.]+%\s+([+-][0-9.]+)%", result.stdout)
        return {k: float(b) for k, b in found}, result.stdout
    own, _ = biases("example")
    zeros, zeros_text = biases("zeros")
    rand, _ = biases("random")
    assert own == {"transpose": -2.5, "reduction": 6.7, "atfconv": 15.7, "radixblocks": 10.6, "radixoffsets": 15.4}
    assert rand == {"transpose": -2.1, "reduction": 6.2, "atfconv": 2.9, "radixblocks": 9.2, "radixoffsets": 14.6}
    assert all(zeros[k] - own[k] >= 5 for k in own) and min(zeros[k] - own[k] for k in own if k != "atfconv") >= 15
    assert "max regret 3.5 %" in zeros_text


# ---- Energy breakdown of measured kernels (2026-11-16) ------------------------------------------------------------------
def test_energy_breakdown_on_the_calibration_run(tmp_path):
    """The parts add up to the predicted total, the report covers every kernel at the model's clock, and the predicted
    saving from the median to the best configuration is close to the measured one (gemm: 45 against 43 %)."""
    out = tmp_path / "parts.csv"
    result = tool("energy_breakdown.py", "--model", MODEL, "--validation", "Validation", "--csv", str(out))
    assert "2 configurations more than 2 % off the model's clock left out" in result.stdout
    gemm = result.stdout.split("  gemm ")[2].splitlines()  # the breakdown table's gemm rows
    assert "45 % / 43 %" in gemm[1]
    rows = counter_table.read_rows(str(out))
    assert len(rows) == 204
    parts = ["baseline", "compute", "issue", "L1/shared", "constant", "L2", "DRAM"]
    for row in rows:
        assert sum(float(row[f"{p}_j"]) for p in parts) == pytest.approx(float(row["predicted_j"]), rel=1e-9)


# ---- Kernel time and the power between launches (2026-11-17) -------------------------------------------------------------
def test_validation_energy_charges_the_gap_below_the_baseline(tmp_path):
    """KTT's time per launch includes its NVML hook; the program now writes the kernel's own GPU time (counter
    gpu__time_duration) as kernel_s and how far below the baseline the GPU draws between launches. The tools add that
    power over the gap (wall - kernel time); a file without the column (runs before 2026-11-17) is read as before."""
    import energy_model
    with open(VALIDATION) as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = list(reader)
    table = {"baseline": {"slope_w_per_c": 0.3}}
    before = {(r["kernel"], r["configuration"]): r for r in energy_model.load_validation(os.path.join(DATA, "Validation"),
                                                                                       table)}
    for row in rows:
        row["gap_below_baseline_w"] = "5"
        if row["kernel_s"]:
            row["wall_s"] = str(float(row["kernel_s"]) + 0.0006)
    prefix = tmp_path / "Gap"
    with open(f"{prefix}_validation.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields + ["gap_below_baseline_w"])
        writer.writeheader()
        writer.writerows(rows)
    after = energy_model.load_validation(str(prefix), table)
    assert after
    for row in after:
        old = before[(row["kernel"], row["configuration"])]
        wall = float(row["wall_s"])
        expected = (float(row["power_w"]) - row["baseline_w"]) * wall + 5 * 0.0006
        assert row["dynamic_energy_j"] == pytest.approx(expected, rel=1e-9)
        assert row["baseline_w"] == pytest.approx(old["baseline_w"])


def test_counter_table_has_the_kernel_time_counter():
    """--validate and --tune take the kernel time from gpu__time_duration.sum (CounterKernelS in EnergyTuner.cpp); it is
    collected with every run, and only as a time (it feeds no event)."""
    entry = counter_table.ENTRIES["gpu__time_duration.sum"]
    assert "use" in entry and "feeds" not in entry


@needs_binary
def test_latency_group_sweeps_steady_rates():
    """The latency group (dependent loads, duty 100) has 1, 4 and 16 chains times 1, 2, 4 and 8 working warps per block
    at L2 and DRAM since 2026-11-17: steady DRAM traffic at many rates for the busy event's shape."""
    output = subprocess.run([BINARY, "--dry-run", "--only", "latency", "--mt4g", MT4G], capture_output=True, text=True,
                            check=True).stdout
    assert re.search(r"latency\s+24 at duty 100", output)


# ---- Energy-time trade-off and counter-pass pruning (2026-11-17) ------------------------------------------------------

def test_pareto_front_and_steps(model, validation):
    """The front holds exactly the configurations no other one beats in both time and energy; its steps start at the
    fastest point of the front and each saves more than 2 sd against the step before."""
    for kernel, configurations in validation.items():
        predictions = [model.predict(row, seconds, temperature, clock, label, group=kernel)
                       for label, row, seconds, temperature, clock in configurations]
        front = energy_predictor.pareto_front(predictions)
        dominated = {i for i, p in enumerate(predictions) for q in predictions
                     if q.kernel_s <= p.kernel_s and q.energy_j < p.energy_j
                     or q.kernel_s < p.kernel_s and q.energy_j <= p.energy_j}
        assert set(front) == set(range(len(predictions))) - dominated
        assert [predictions[i].kernel_s for i in front] == sorted(predictions[i].kernel_s for i in front)
        steps = energy_predictor.pareto_steps(model, predictions, front)
        assert steps[0] == front[0] and set(steps) <= set(front)
        for a, b in zip(steps, steps[1:]):
            assert predictions[a].energy_j - predictions[b].energy_j > 2 * model.pair_sd(predictions[b], predictions[a])
        choice = energy_predictor.recommend(model, predictions)
        assert choice.index in front


@needs_binary
def test_cpp_pareto_column_equals_python(tmp_path, model):
    for data, objective in ((VALIDATION, "energy"), (os.path.join(DATA, "AllMany2Dev_validation.csv"), "edp")):
        prefix = str(tmp_path / "cpp")
        result = program("--predict", data, "--model-file", MODEL, "--objective", objective, "--output", prefix)
        assert result.returncode == 0, result.stdout
        assert "energy-time trade-off" in result.stdout
        cpp = {(r["kernel"], r["configuration"]): r["pareto"] for r in counter_table.read_rows(prefix + "_predicted.csv")}
        for kernel, configurations in energy_predictor.from_validation_csv(data).items():
            predictions = [model.predict(row, seconds, temperature, clock, label, group=kernel)
                           for label, row, seconds, temperature, clock in configurations]
            front = energy_predictor.pareto_front(predictions)
            steps = energy_predictor.pareto_steps(model, predictions, front)
            for i, p in enumerate(predictions):
                assert cpp[(kernel, p.label)] == ("step" if i in steps else "front" if i in front else "")


def test_contender_margin(model):
    """The margin m solves m = z (r + b) sqrt((1 + m)^2 + 1); no baseline in the accounting, no margin."""
    m = energy_predictor.contender_margin(model)
    baseline_w = model.predict({}, 1.0, strict=False).energy_j
    sd = math.hypot(model.uncertainty["baseline_sd_w"], model.uncertainty["baseline_unknown_temperature_sd_w"])
    c = 2 * (model.uncertainty["dynamic_relative_sd"] + sd / baseline_w)
    assert m == pytest.approx(c * math.sqrt((1 + m) ** 2 + 1), rel=1e-9)
    assert 0.2 < m < 0.3
    dynamic = copy.copy(model)
    dynamic.accounting = "dynamic"
    assert energy_predictor.contender_margin(dynamic) is None


@pytest.mark.parametrize("data, least", [("AllMany2Dev_validation.csv", 0.2),
                                         (os.path.join("heldout5", "KttMany1_validation.csv"), 0.2),
                                         (os.path.join("heldout4", "HeldOut4Frozen1_validation.csv"), 0.0)])
def test_pruning_keeps_the_choice_on_recorded_runs(model, data, least):
    """On runs that profiled every configuration: the kernel's time is never below the bound taken from KTT's time, and
    the counter-pass pruning would leave out neither the choice nor a contender, for energy and edp, also with a
    slowdown limit. It leaves out about a quarter of the configurations, none of kernels shorter than a few ms
    (HeldOut4: 0.1-4 ms, where KTT's launch overhead hides the time)."""
    path = os.path.join(DATA, data)
    ktt = {(r["kernel"], r["configuration"]): float(r["ktt_kernel_s"]) for r in counter_table.read_rows(path)
           if r["valid"] == "1" and r.get("repeat", "0") in ("", "0") and r.get("ktt_kernel_s")}
    total = skipped = 0
    for kernel, configurations in energy_predictor.from_validation_csv(path).items():
        predictions = [model.predict(row, seconds, temperature, clock, label, strict=False, group=kernel)
                       for label, row, seconds, temperature, clock in configurations]
        ktt_s = [ktt[(kernel, p.label)] for p in predictions]
        for p, k in zip(predictions, ktt_s):
            assert p.kernel_s >= energy_predictor.PRUNE_TIME_FACTOR * k - energy_predictor.PRUNE_LAUNCH_OVERHEAD_S
        temperatures = [c[3] for c in configurations]
        for objective, slowdown in (("energy", None), ("edp", None), ("energy", 0.05)):
            pruned = energy_predictor.pruned_indices(
                model, ktt_s, temperatures,
                lambda i: (predictions[i].kernel_s, energy_predictor.objective_value(predictions[i], objective)),
                objective, max_slowdown=slowdown)
            choice = energy_predictor.recommend(model, predictions, objective, max_slowdown=slowdown)
            assert not pruned & ({choice.index} | set(choice.contenders)), (kernel, objective, slowdown)
            if slowdown is None and objective == "energy":
                total += len(predictions)
                skipped += len(pruned)
    assert least <= skipped / total < 0.4


def test_pruning_on_a_tune_run(model):
    """GPU runs of --tune (gemm, coulomb, atfconv, 80 random configurations each, 2026-11-17): on the run without
    pruning the rule would leave out 81 of 228 predicted configurations and never the choice or a contender; the run
    with pruning left out 78 (KTT's random searcher picks other configurations per run) and notes each."""
    rows = collections.defaultdict(list)
    for r in counter_table.read_rows(os.path.join(DATA, "tune", "NoPrune1_tune.csv")):
        if r["predicted_j"]:
            rows[r["kernel"]].append(r)
    skipped = 0
    for kernel, kernel_rows in rows.items():
        ktt_s = [float(r["ktt_kernel_s"]) for r in kernel_rows]
        seconds = [float(r["kernel_s"]) for r in kernel_rows]
        energy = [float(r["predicted_j"]) for r in kernel_rows]
        temperatures = [float(r["temperature_c"]) for r in kernel_rows]
        pruned = energy_predictor.pruned_indices(model, ktt_s, temperatures, lambda i: (seconds[i], energy[i]))
        assert not pruned & {i for i, r in enumerate(kernel_rows) if r["choice"] in ("best", "contender")}, kernel
        skipped += len(pruned)
    assert skipped == 81
    pruned_run = counter_table.read_rows(os.path.join(DATA, "tune", "Prune1_tune.csv"))
    assert sum("(pruned)" in r["note"] for r in pruned_run) == 78
    assert all(not r["predicted_j"] for r in pruned_run if "(pruned)" in r["note"])


# ---- Transfer to another GPU type, power limit in --predict and the breakdown (2026-11-17) ----------------------------

def quick_rows_of(source, target):
    """The rows a --quick run measures (the global, compute and shared probes and the idle and sleep rows), taken from a
    calibration run."""
    with open(source) as file, open(target, "w", newline="") as out:
        reader = csv.DictReader(file)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(r for r in reader if r["probe"] in ("global", "compute", "shared")
                         or r["probe"].startswith(("idle", "sleep")))


def test_check_model_transfers_to_another_gpu_type(tmp_path, model):
    """The RTX 3090 model and the quick probes of the RTX 2080 Ti calibration: TRANSFER with factors per part; on the 2080 Ti's
    validation run the transferred model's total error is close to the 2080 Ti's own calibration (4.7 against 4.5 %) and
    below the 3090 model with only the 2080 Ti's baseline (7.2 %), and it ranks as well (max regret 0.9 %)."""
    quick = tmp_path / "Quick.csv"
    quick_rows_of(os.path.join(DATA, "rtx2080ti", "Probes.csv"), quick)
    written = tmp_path / "transferred.model.json"
    name = "NVIDIA GeForce RTX 2080 Ti"
    result = tool("check_model.py", "--model", MODEL, "--run", str(tmp_path / "Quick"), "--other-gpu", name,
                  "--write", str(written))
    assert "TRANSFER" in result.stdout and "factors per part" in result.stdout
    transferred = energy_predictor.Model.load(str(written))
    assert transferred.device["name"] == name
    assert transferred.data["adjusted"]["factors"]["L2"] == pytest.approx(0.824, abs=0.01)
    assert transferred.baseline["mean_w"] == pytest.approx(74.0, abs=0.5)
    validation = os.path.join(DATA, "rtx2080ti", "Validation_validation.csv")
    measured = {(r["kernel"], r["configuration"]): float(r["energy_j"]) for r in counter_table.read_rows(validation)
                if r["valid"] == "1" and r.get("repeat", "0") in ("", "0")}
    errors, regrets = [], []
    for kernel, configurations in energy_predictor.from_validation_csv(validation).items():
        predictions = [transferred.predict(row, s, t, c, label, strict=False, group=kernel)
                       for label, row, s, t, c in configurations]
        energies = [measured[(kernel, p.label)] for p in predictions]
        errors += [abs(p.energy_j / e - 1) for p, e in zip(predictions, energies)]
        best = min(range(len(predictions)), key=lambda i: predictions[i].energy_j)
        regrets.append(energies[best] / min(energies) - 1)
    assert sum(errors) / len(errors) == pytest.approx(0.047, abs=0.003)
    assert max(regrets) < 0.01


def test_check_model_without_other_gpu_fails_on_another_type(tmp_path):
    quick = tmp_path / "Quick.csv"
    quick_rows_of(os.path.join(DATA, "rtx2080ti", "Probes.csv"), quick)
    result = tool("check_model.py", "--model", MODEL, "--run", str(tmp_path / "Quick"), check=False)
    assert result.returncode == 1 and "FAIL" in result.stdout


@needs_binary
def test_predict_applies_the_model_files_power_limit(tmp_path):
    prefix = str(tmp_path / "cpp")
    default = program("--predict", VALIDATION, "--model-file", MODEL, "--output", prefix)
    assert default.returncode == 0 and "power limit 350 W" in default.stdout
    none = program("--predict", VALIDATION, "--model-file", MODEL, "--output", prefix, "--power-limit", "0")
    assert none.returncode == 0 and "power limit none" in none.stdout
    low = program("--predict", VALIDATION, "--model-file", MODEL, "--output", prefix, "--power-limit", "200")
    assert low.returncode == 0 and "above the GPU's limit of 200 W" in low.stdout
    python = tool("energy_predictor.py", "--model", MODEL, "--csv", VALIDATION)
    assert "power limit 350 W" in python.stdout


def test_breakdown_parts_add_up_at_the_power_limit(tmp_path):
    """At a power limit the prediction is the limit times the time; the breakdown scales every part to match (Antepara
    et al.'s rescaling), so the parts still add up to the prediction."""
    out = tmp_path / "parts.csv"
    result = tool("energy_breakdown.py", "--model", MODEL, "--validation", "AllMany2Dev", "--power-limit", "250",
                  "--csv", str(out))
    assert "predicted above the power limit of 250 W" in result.stdout
    rows = counter_table.read_rows(str(out))
    parts = ("baseline_j", "compute_j", "issue_j", "L1/shared_j", "constant_j", "L2_j", "DRAM_j")
    capped = [r for r in rows if float(r["predicted_power_w"]) == pytest.approx(250.0, rel=1e-6)]
    assert len(capped) > 100
    for row in rows:
        assert sum(float(row[name]) for name in parts) == pytest.approx(float(row["predicted_j"]), rel=1e-9)


def test_lanes_probes_price_inactive_lanes(tmp_path, model):
    """Partly idle warps (run Lanes1, counters LanesProfile2, 2026-11-17): one FFMA lane costs about 100 pJ and eight about
    146 pJ per warp instruction, against 282 pJ for a full warp. With the thread counter the lanes probes price "inst
    inactive lanes" (about 69 pJ per warp instruction of inactive lanes) and fit within 7 %, the indexing probe's guarded
    moves (predicated off, not inactive) stay within 4 %. With the older counter run (no thread counter) they are checks."""
    rows = {r["operation"]: r for r in counter_table.read_rows(os.path.join(DATA, "Lanes1.csv"))
            if r["level"] == "compute" and r["duty_pct"] == "100"}
    assert float(rows["ffma-lanes1"]["pj_per_access"]) == pytest.approx(100.5, abs=1)
    assert 8 * float(rows["ffma-lanes8"]["pj_per_access"]) == pytest.approx(145.6, abs=2)
    assert model.pj["inst inactive lanes"] == pytest.approx(68.9, abs=0.5)
    fitted = {p["label"]: p["model_pj_per_access"] / p["pj_per_access"] - 1 for p in model.data["probes"]}
    for op in ("ffma", "imad"):
        for lanes in (1, 8):
            assert abs(fitted[f"compute {op}-lanes{lanes}"]) < 0.08
    assert abs(fitted["compute ffma-indexed"]) < 0.04
    old = tool("energy_model.py", "--power", "Probes,Tensor1,LatSweep1,Rows1,Lanes1", "--counters",
               "Profile,ComputeProfile1,LatSweepProfile1,RowsProfile1,LanesProfile1").stdout
    line = next(l for l in old.splitlines() if "compute ffma-lanes1" in l)
    assert float(line.split()[-2]) == pytest.approx(-59.0, abs=1)


def test_recounted_runs_score_as_before(model):
    """The 19 kernels of AllMany2 and the heldout5 kernels, counted again with the thread counter (--recount, merged with
    tools/merge_counters.py; power as measured) and scored with model 2d90caf0: as with the earlier model, which these
    configurations' bias and ranking hardly depend on (few configurations have inactive lanes: up to 50 % in five of
    bicg). Both held-out sets were scored before, so this is a reused evaluation."""
    for run, expected in (("AllMany2b", "scatter 5.0 %, bias +0.6 %, total MAPE 3.0 %, max regret 2.7 % (time only 19.7 %)"),
                          ("KttMany1b", "scatter 19.9 %, bias +20.5 %, total MAPE 5.1 %, max regret 0.9 % (time only 5.6 %)")):
        result = tool("energy_model.py", *REFERENCE_RUNS, "--validation", os.path.join("recount", run))
        assert expected in result.stdout, run


@needs_binary
def test_cpp_predicts_inactive_lanes_like_python(tmp_path, model):
    """The recounted 19 kernels (thread counter present): the C++ predictor counts and prices "inst inactive lanes" as
    the Python one does."""
    data = os.path.join(DATA, "recount", "AllMany2b_validation.csv")
    prefix = str(tmp_path / "cpp")
    result = program("--predict", data, "--model-file", MODEL, "--output", prefix)
    assert result.returncode == 0, result.stdout
    cpp = {(r["kernel"], r["configuration"]): float(r["predicted_dynamic_j"])
           for r in counter_table.read_rows(prefix + "_predicted.csv") if r["predicted_dynamic_j"]}
    compared = inactive = 0
    for kernel, configurations in energy_predictor.from_validation_csv(data).items():
        for label, row, seconds, temperature, clock in configurations:
            p = model.predict(row, seconds, temperature, clock, label, group=kernel)
            assert cpp[(kernel, label)] == pytest.approx(p.dynamic_j, rel=1e-9)
            compared += 1
            inactive += p.events["inst inactive lanes"] > 0
    assert compared == 995 and inactive > 30


# --- whole applications (tools/app_energy.py) and opcode probes (2026-11-17) ------------------------------------------

def test_application_energy_does_not_depend_on_the_averaging_window():
    """The integral over the run and a tail, minus the idle level, is the energy of the run whatever NVML's averaging
    window: a 2 s burst of 150 W above 40 W idle reads 300 J with a 0.1 s and a 1 s boxcar."""
    import app_energy
    def samples(window):
        step, out = 0.01, []
        for i in range(int(20 / step)):
            t = i * step
            # boxcar average of the true power (40 W, 190 W from 5 to 7 s) over [t - window, t]
            overlap = max(0.0, min(t, 7.0) - max(t - window, 5.0))
            out.append((t, 40.0 + 150.0 * overlap / window, 60.0, 1200.0, math.nan))
        return out
    for window in (0.1, 1.0):
        result = app_energy.application_energy(samples(window), 5.0, 7.0)
        # the run itself (2 s at 190 W = 380 J), since the idle level after the run is subtracted from the tail only
        assert result["energy_j"] == pytest.approx(380.0, rel=0.01)


def test_gap_energy_and_busy_union():
    import app_energy
    model = {"awake_idle_w": 100.0, "sleep_kernels_w": 105.0, "deep_idle_w": 40.0, "temperature_c": 60.0,
             "gap": {"fall_s": 1e-3, "idle_timer_s": 10.0, "exit_delay_s": 3.0}}
    assert app_energy.gap_energy(1.0, model, 60.0, 0.0) == pytest.approx(100.0 + 5.0 * 1e-3, rel=1e-3)
    # past the idle timer the deep idle power
    assert app_energy.gap_energy(15.0, model, 60.0, 0.0) == pytest.approx(1000.0 + 5e-3 + 200.0, rel=1e-3)
    # the temperature moves the awake powers by the slope
    assert app_energy.gap_energy(1.0, model, 70.0, 0.3) == pytest.approx(103.0, rel=1e-3)
    trace = [(0.0, 1.0, "a", "kernel", 0.0), (0.5, 1.0, "b", "kernel", 0.0), (2.0, 0.5, "c", "h2d pageable", 1.0)]
    busy, gaps = app_energy.busy_union(trace)
    assert busy == pytest.approx(2.0) and gaps == [pytest.approx(0.5)]


def test_app_model_fit_recovers_the_fall_time():
    import app_energy
    sleep, awake, fall = 106.0, 103.0, 4e-4
    rows = [{"probe": "gap", "gap_us": "0", "launches": "1000", "kernel_s": "2.0", "wall_s": "2.0", "power_w": sleep,
             "deep_idle_w": 42.0, "temperature_c": 60.0}]
    for gap in (2e-5, 1e-4, 5e-4, 2e-3, 1e-2, 0.2):
        launches, kernel = 1000, 2e-3
        energy = awake * gap + (sleep - awake) * fall * -math.expm1(-gap / fall)
        wall = launches * (kernel + gap)
        rows.append({"probe": "gap", "gap_us": str(gap * 1e6), "launches": str(launches), "kernel_s": str(launches * kernel),
                     "wall_s": str(wall), "power_w": (sleep * launches * kernel + energy * launches) / wall,
                     "deep_idle_w": 42.0, "temperature_c": 60.0})
    rows.append({"probe": "context", "deep_idle_w": 42.0, "awake_idle_w": 103.0, "idle_timer_s": 17.5, "exit_delay_s": 3.0,
                 "temperature_c": 60.0})
    model = app_energy.fit_app_model(rows, "test")
    assert model["awake_idle_w"] == pytest.approx(awake, abs=0.1)  # the 10 ms gap still holds 0.1 W of the fall
    assert model["gap"]["fall_s"] == pytest.approx(fall, rel=0.05)
    assert model["gap"]["idle_timer_s"] == 17.5 and model["deep_idle_w"] == 42.0


def test_ncu_instances_and_opcode_correction():
    import app_energy
    text = "15204352 (FFMA: 13107200; MUFU.SIN: 131072; FADD: 131072; IADD3: 2)"
    assert app_energy.parse_instances(text) == {"FFMA": 13107200.0, "MUFU.SIN": 131072.0, "FADD": 131072.0, "IADD3": 2.0}
    assert app_energy.opcode_family("MUFU.EX2") == "MUFU.EX2" and app_energy.opcode_family("FADD.FTZ") == "FADD"
    table = {"FADD": {"pj": 300.0, "pipe_pj": 280.0}, "FFMA": {"pj": 330.0, "pipe_pj": 280.0},
             "MUFU.SIN": {"pj": 400.0, "pipe_pj": 325.0}}
    # FFMA is a calibration opcode (no correction); FADD.FTZ matches FADD; MUFU.EX2 is not in the table
    correction = app_energy.opcode_correction_j({"FFMA": 1e9, "FADD.FTZ": 1e9, "MUFU.SIN": 1e6, "MUFU.EX2": 1e9}, table)
    assert correction == pytest.approx(1e9 * 20e-12 + 1e6 * 75e-12)


def test_sampled_launches_interpolate_by_launch_number():
    import app_energy
    trace = [(float(i), 0.5, "k(int)", "kernel", 0.0, "(1, 1, 1)") for i in range(30)]
    # windows of launches 1-2 and 21-22 profiled (application launch numbers 0, 1, 20, 21), x = the launch's number
    launches = [{"name": "k(int)", "grid": "(1, 1, 1)", "counters": {"x": float(i + 1)}, "opcodes": {},
                 "opcode_modifiers": {}, "index": i} for i in (0, 1, 20, 21)]
    priced = app_energy.kernel_counters(trace, launches)
    assert [p[0]["x"] for p in priced[:22]] == [pytest.approx(float(n)) for n in range(1, 23)]
    assert priced[29][0]["x"] == 22.0  # after the last profiled launch: the last
    assert app_energy.launch_windows(8190, 12, 8)[0] == (0, 8) and app_energy.launch_windows(8190, 12, 8)[-1] == (8182, 8)
    assert app_energy.launch_windows(76, None, 8) == [(0, 76)]


def test_antepara_energies_match_the_documented_values(model):
    import app_energy
    energies = app_energy.antepara_energies(model, 41.5)
    # related_work.md: 1.71 / 7.1 / 20.9 pJ/bit above the idle power (old Full run); this model's probes
    assert energies["L1"] * 1e12 / 8 == pytest.approx(1.7, abs=0.1)
    assert energies["L2"] * 1e12 / 8 == pytest.approx(7.3, abs=0.4)
    assert energies["DRAM"] * 1e12 / 8 == pytest.approx(21.4, abs=1.0)


def test_opcode_probes_are_checks_not_fit():
    import energy_model
    assert energy_model.OPCODE_PREFIX == "compute opcode-"


APPS = os.path.join(DATA, "apps")
APP_MODEL = os.path.join(ROOT, "models", "rtx3090_1200mhz.app.json")
OPCODE_TABLE = os.path.join(ROOT, "models", "rtx3090_1200mhz.opcodes.json")


def test_recorded_applications_score_as_documented(model):
    """Two of the eight Rodinia applications (lavaMD, needle; run Rodinia1nc, ncu without cache flush, counters trimmed
    to the model's): process energy, Antepara et al.'s form and the opcode table as docs/APPLICATIONS.md reports them."""
    import app_energy
    with open(APP_MODEL) as file:
        app_model = json.load(file)
    with open(OPCODE_TABLE) as file:
        table = json.load(file)["opcodes"]
    run = os.path.join(APPS, "Rodinia1nc")
    expected = {"lavaMD": (549.3, 0.037, -0.254, 0.022, 0.002), "needle": (239.3, 0.027, -0.529, 0.027, -0.045)}
    for name, (measured, error, antepara, opcode, nvml) in expected.items():
        result = app_energy.score_app(name, run, model, app_model, table)
        assert result["measured_j"] == pytest.approx(measured, abs=0.1)
        assert result["error"] == pytest.approx(error, abs=0.001)
        assert result["nvml_error"] == pytest.approx(nvml, abs=0.001)
        assert result["antepara_error"] == pytest.approx(antepara, abs=0.001)
        assert result["opcode_error"] == pytest.approx(opcode, abs=0.001)
    # lavaMD is one launch, needle twelve windows of eight launches out of 2047 (the rest interpolated)
    assert app_energy.score_app("needle", run, model, app_model)["launches"] == 2047


def test_app_model_file_holds_the_idle_states():
    with open(APP_MODEL) as file:
        app_model = json.load(file)
    gap = app_model["gap"]
    assert 40 < app_model["deep_idle_w"] < 45 and 100 < app_model["awake_idle_w"] < 106
    assert 15 < gap["idle_timer_s"] < 20 and 2 < gap["exit_delay_s"] < 4.5 and 0.3 < gap["wake_s"] < 0.8
    assert gap["fall_s"] < 2e-3
    assert 108 < app_model["copies"]["h2d pageable"]["power_w"] < 115


def test_app_model_refits_from_its_probe_rows():
    import app_energy
    with open(APP_MODEL) as file:
        app_model = json.load(file)
    refit = app_energy.fit_app_model(app_model["probes"], app_model["device"])
    for key in ("awake_idle_w", "sleep_kernels_w", "deep_idle_w"):
        assert refit[key] == pytest.approx(app_model[key])
    for key in ("fall_s", "idle_timer_s", "exit_delay_s", "wake_s"):
        assert refit["gap"][key] == pytest.approx(app_model["gap"][key])


@pytest.mark.skipif(not os.path.exists(os.path.join(os.environ.get("CUDA_PATH", "/usr/local/cuda"), "bin", "ptxas")),
                    reason="needs a CUDA toolkit for the probes' SASS")
def test_opcode_table_reproduces_from_the_recorded_runs(tmp_path):
    out = tmp_path / "opcodes.json"
    subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "opcode_energy.py"), "--power",
                    os.path.join(DATA, "Opcodes1") + "," + os.path.join(DATA, "Opcodes2"), "--counters",
                    os.path.join(DATA, "OpcodesProfile2") + "," + os.path.join(DATA, "OpcodesProfile1"), "--model", MODEL,
                    "--out", str(out)], check=True, capture_output=True, text=True)
    with open(out) as file:
        table = json.load(file)["opcodes"]
    with open(OPCODE_TABLE) as file:
        shipped = json.load(file)["opcodes"]
    assert set(table) == set(shipped)
    for opcode, entry in shipped.items():
        assert table[opcode]["pj"] == pytest.approx(entry["pj"], rel=1e-6)
    # what docs/PROBES.md reports
    assert table["DADD"]["pj"] / table["DADD"]["pipe_pj"] == pytest.approx(0.51, abs=0.01)
    assert table["FFMA"]["constant_operands_pj"] / table["FFMA"]["pj"] == pytest.approx(0.43, abs=0.01)
    assert table["MUFU.EX2"]["pj"] == pytest.approx(419, abs=2)


def test_lud_input_matrices(tmp_path):
    """apps/make_inputs.py --lud: the smooth matrix is lud's own (10 exp(-0.001 |i - j|) in float) and the random one
    is diagonally dominant; both read back exactly from the text."""
    numpy = pytest.importorskip("numpy")
    subprocess.run([sys.executable, os.path.join(ROOT, "apps", "make_inputs.py"), str(tmp_path), "--lud", "48"],
                   check=True)

    def read(name):
        with open(tmp_path / name) as file:
            size = int(file.readline())
            values = numpy.array(file.read().split(), dtype=numpy.float32)
        return values.reshape(size, size)
    smooth, rand = read("lud_smooth_48"), read("lud_random_48")
    i, j = numpy.indices(smooth.shape)
    expected = (10.0 * numpy.exp((numpy.float32(-0.001) * numpy.abs(i - j).astype(numpy.float32)).astype(numpy.float64))
                ).astype(numpy.float32)
    assert numpy.array_equal(smooth, expected)
    off = numpy.abs(rand).sum(axis=1) - numpy.abs(numpy.diag(rand))
    assert (numpy.abs(numpy.diag(rand)) > off).all()
    assert len(numpy.unique(rand)) == rand.size


def test_broadcast_probes_are_checks_and_price_as_documented(model):
    """The optional broadcast probes (run Broadcast1 with counters BroadcastProfile1, 2026-11-18): never in the fit; the
    model's errors per shared load as docs/PROBES.md reports them; bank reads follow the distinct words."""
    import contextlib
    import io
    import energy_model
    power, counters = os.path.join(DATA, "Broadcast1"), os.path.join(DATA, "BroadcastProfile1")
    with contextlib.redirect_stdout(io.StringIO()):
        fitted, _ = energy_model.probe_table([power], [counters], busy_scale=95.0)
        checks, _ = energy_model.probe_table([power], [counters], busy_scale=95.0, checks=True)
    assert not any("-bcast" in p.label for p in fitted)
    error = {p.label: sum(p.counts[e] * model.pj.get(e, 0.0) for e in p.counts) / p.energy - 1
             for p in checks if "-bcast" in p.label}
    expected = {"shared ld.shared-1way-bcast2 w4": 0.010, "shared ld.shared-1way-bcast16 w4": 0.175,
                "shared ld.shared-1way-bcast32 w4": 0.172, "shared ld.shared-1way-bcast2 w16": -0.148,
                "shared ld.shared-1way-bcast16 w16": 0.006, "shared ld.shared-1way-bcast32 w16": 0.188}
    assert set(error) == set(expected)
    for label, value in expected.items():
        assert error[label] == pytest.approx(value, abs=0.002), label
    with open(counters + ".csv") as file:
        rows = [r for r in csv.DictReader(file) if r["operation"].startswith("ld.shared-1way") and r["width_bytes"] == "4"]
    reads = {r["operation"]: float(r["l1tex__data_bank_reads.sum"]) / float(r["smsp__inst_executed_op_shared_ld.sum"])
             for r in rows}
    assert reads == pytest.approx({"ld.shared-1way": 32, "ld.shared-1way-bcast2": 16, "ld.shared-1way-bcast16": 2,
                                   "ld.shared-1way-bcast32": 1}, abs=0.05)


def test_lud_internal_diagnosis_scores_as_documented(tmp_path):
    """lud_internal and its parts measured steadily (--validate diagnose, run LudParts1, 2026-11-18): at lud's first step
    the model is +4.7 % in energy and +9 % in dynamic energy, about the same for every part (docs/APPLICATIONS.md)."""
    out = tmp_path / "parts.csv"
    subprocess.run([sys.executable, os.path.join(TOOLS, "energy_breakdown.py"), "--model", MODEL, "--validation",
                    os.path.join(DATA, "diagnose", "LudParts1"), "--csv", str(out)], check=True, capture_output=True)
    with open(out) as file:
        rows = {r["configuration"]: r for r in csv.DictReader(file)}
    expected = {0: (0.047, 0.093), 1: (0.047, 0.144), 2: (0.048, 0.088)}
    for part, (energy, dynamic) in expected.items():
        r = rows[f"LUD_OFFSET=0 LUD_PART={part}"]
        measured, predicted, baseline = (float(r[k]) for k in ("measured_j", "predicted_j", "baseline_j"))
        assert predicted / measured - 1 == pytest.approx(energy, abs=0.002)
        assert (predicted - baseline) / (measured - baseline) - 1 == pytest.approx(dynamic, abs=0.002)


def test_nvml_view_keeps_energy_and_smears_it_across_the_exit():
    """nvml_view: a segment well inside the process keeps its energy; one that ends at the exit loses part of it to the
    time after the exit, as NVML's 1 s mean, held 0.5 s, shows it later than the 0.5 s the split allows for; segments
    shorter than the grid step keep their energy."""
    import app_energy
    deep = 40.0
    inside = app_energy.nvml_view([(1.0, 1.0, 200.0)], 5.0, deep, 10.0)
    assert inside == pytest.approx(160.0 + deep * 5.0, rel=1e-3)
    tiny = app_energy.nvml_view([(1.0 + i * 1e-3, 2e-4, 1040.0) for i in range(1000)], 5.0, deep, 10.0)
    assert tiny == pytest.approx(1000 * 2e-4 * 1000.0 + deep * 5.0, rel=1e-3)
    at_exit = app_energy.nvml_view([(0.0, 2.0, 200.0)], 2.0, deep, 10.0)
    # the window's mean lags 0.5 s and the hold another 0.25 s on average, against the 0.5 s the split allows for: about
    # 0.29 s of the 160 W above deep (46 J) falls after the split
    assert at_exit - deep * 2.0 == pytest.approx(320.0 - 46.5, abs=1.0)


def test_clock_mismatch_warns_only_beyond_one_clock_step(tmp_path):
    """ClockMismatch (Common.cpp, the warm-up clock check of --expect-clock and --lock-clock, 2026-11-18): silent within one
    step of NVML's supported clocks (RTX 3090 SM: 15 MHz), a warning naming both clocks beyond it; the step comes from the
    clock nearest to the expected one; without a list the fallback step; unknown clocks are no warning."""
    ktt = os.path.abspath(os.path.join(ROOT, "..", ".."))
    source = tmp_path / "clock.cpp"
    source.write_text(r'''
#include <cmath>
#include <cstdio>
#include "Common.h"
int main()
{
    std::vector<double> sm;
    for (double f = 210; f <= 2100; f += 15) sm.push_back(f);
    const std::vector<double> memory = {405, 810, 5001, 9251, 9501, 9751};
    const double nan = std::nan("");
    const std::string cases[] = {
        ClockMismatch("SM", 1200, 1200, sm, 15), ClockMismatch("SM", 1200, 1215, sm, 15),
        ClockMismatch("SM", 1200, 1230, sm, 15), ClockMismatch("SM", 1200, 1185, {}, 15),
        ClockMismatch("SM", 1200, 1170, {}, 15), ClockMismatch("memory", 5001, 9751, memory, 50),
        ClockMismatch("memory", 9751, 9501, memory, 50), ClockMismatch("memory", 9751, 9250, memory, 50),
        ClockMismatch("SM", nan, 1200, sm, 15), ClockMismatch("SM", 1200, nan, sm, 15)};
    for (const auto& c : cases) std::printf("%d|%s\n", c.empty() ? 0 : 1, c.c_str());
}
''')
    output = tmp_path / "clock"
    command = ["g++", "-std=c++17", "-O1", "-I" + ROOT, "-I" + os.path.join(ktt, "Libraries", "Json-3.9.1"), str(source),
               os.path.join(ROOT, "Common.cpp"), "-o", str(output)]
    built = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert built.returncode == 0, built.stdout
    lines = subprocess.run([str(output)], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.splitlines()
    warned = [line.split("|", 1)[0] == "1" for line in lines]
    # 1200 exact / one step up / two steps / one step without a list / two steps without a list / memory 5001 -> 9751
    # (not applied) / one memory step (250) / beyond it / unknown expected / unknown observed
    assert warned == [False, False, True, False, True, True, False, True, False, False]
    assert "ran at 1230 MHz under load, not at the expected 1200 MHz" in lines[2]
    assert "memory clock ran at 9751 MHz" in lines[5] and "step of 4750 MHz" not in lines[5]


def test_instruction_coverage_shares():
    """instruction_coverage (tools/app_energy.py, 2026-11-18, Wattchmen's coverage figure): shares of executed warp
    instructions priced by a probe of their own (or the opcode table), only through their pipe, or in no probed pipe."""
    import app_energy
    opcodes = {"FFMA": 40, "LDG.E.128": 20, "MUFU.EX2": 10, "IADD3": 10, "BRA": 10, "HGMMA.64x128x16.F32": 5,
               "UTCHMMA": 3, "TEX.SCR.LL": 2}
    plain = app_energy.instruction_coverage(opcodes)
    assert plain["own"] == pytest.approx(0.6) and plain["pipe"] == pytest.approx(0.3)
    assert plain["none"] == pytest.approx(0.1)
    assert plain["largest_pipe"][0] == ("MUFU.EX2", pytest.approx(0.1))
    table = {"MUFU.EX2": {"pj": 1.0, "pipe_pj": 1.0}, "IADD3": {"pj": 1.0, "pipe_pj": 1.0}}
    with_table = app_energy.instruction_coverage(opcodes, table)
    assert with_table["own"] == pytest.approx(0.8) and with_table["pipe"] == pytest.approx(0.1)
    assert app_energy.instruction_coverage({}) == {}


def test_recorded_applications_instruction_coverage(model):
    """lavaMD and needle (run Rodinia1nc): the coverage docs/APPLICATIONS.md reports, without and with the opcode table."""
    import app_energy
    with open(APP_MODEL) as file:
        app_model = json.load(file)
    with open(OPCODE_TABLE) as file:
        table = json.load(file)["opcodes"]
    run = os.path.join(APPS, "Rodinia1nc")
    for name, (own, own_table) in {"lavaMD": (0.591, 0.869), "needle": (0.513, 0.648)}.items():
        assert app_energy.score_app(name, run, model, app_model)["coverage"]["own"] == pytest.approx(own, abs=0.001)
        covered = app_energy.score_app(name, run, model, app_model, table)["coverage"]
        assert covered["own"] == pytest.approx(own_table, abs=0.001) and covered["none"] == 0.0


@pytest.mark.skipif(not os.path.exists(os.path.join(os.environ.get("CUDA_PATH", "/usr/local/cuda"), "bin", "ptxas")),
                    reason="needs a CUDA toolkit for the probes' SASS")
def test_operand_reuse_probes_price_as_documented(tmp_path):
    """The optional reuse probes (runs Reuse1-3, counters ReuseProfile1, 2026-11-18; docs/PROBES.md "The operand reuse
    cache"): per warp FFMA, the same operand values cost 334 pJ from a register per chain and 284 / 280 pJ with one or two
    operands from the reuse cache; the second reused operand saves no more than the noise; the opcode table is untouched."""
    reuse = os.path.join(DATA, "reuse")
    out = tmp_path / "reuse.json"
    result = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "opcode_energy.py"), "--power",
                             ",".join(os.path.join(reuse, f"Reuse{i}") for i in (1, 2, 3)), "--counters",
                             os.path.join(reuse, "ReuseProfile1"), "--model", MODEL, "--out", str(out)],
                            check=True, capture_output=True, text=True)
    with open(out) as file:
        data = json.load(file)
    summary = data["operand_reuse"]
    assert summary["ffma-same"]["pj_per_ffma"] == pytest.approx(334.0, abs=1.0)
    assert summary["ffma-reuse1"]["pj_per_ffma"] == pytest.approx(283.9, abs=1.5)
    assert summary["ffma-reuse2"]["pj_per_ffma"] == pytest.approx(279.9, abs=1.5)
    assert summary["ffma-rr"]["pj_per_ffma"] == pytest.approx(349.1, abs=1.5)
    assert summary["ffma-reuse1"]["reused_per_ffma"] == pytest.approx(0.75)
    assert summary["ffma-reuse2"]["reused_per_ffma"] == pytest.approx(1.5)
    assert summary["ffma-same"]["same_bank_pairs_per_ffma"] == pytest.approx(1.0)
    assert summary["ffma-reuse2"]["same_bank_pairs_per_ffma"] == pytest.approx(0.25)
    assert summary["ffma-reuse1"]["pj_per_ffma"] - summary["ffma-reuse2"]["pj_per_ffma"] < 6.0
    # The reuse probes stay out of the opcode table: its FFMA entry is opcode 3's (ffma-rr).
    assert list(data["opcodes"]) == ["FFMA"] and data["opcodes"]["FFMA"]["probe"] == "compute opcode-ffma-rr"
    per_run = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "opcode_energy.py"), "--power",
                              ",".join(os.path.join(reuse, f"Reuse{i}") for i in (1, 2, 3)), "--counters",
                              os.path.join(reuse, "ReuseProfile1"), "--model", MODEL, "--per-run"],
                             check=True, capture_output=True, text=True).stdout
    assert "ffma-reuse1: saving per reused operand" in per_run and "mean 66.8 pJ" in per_run


def test_mixed_work_probes_price_as_documented():
    """The optional mix and saturate probes (run Mix1, counters MixProfile1, 2026-11-18; docs/PROBES.md "Mixed work"):
    an FFMA added to loads that hit L1 or L2 costs 315-345 pJ per warp instruction, near the same run's three-register
    FFMA probe (330 pJ) and above the model's fma-pipe price (280 pJ); the rows the 350 W cap clocked down are withheld."""
    import contextlib, io
    import energy_model
    mix = os.path.join(DATA, "mix")
    with contextlib.redirect_stdout(io.StringIO()):
        probes, _ = energy_model.probe_table([os.path.join(mix, "Mix1")], [os.path.join(mix, "MixProfile1")], 95.0)
    energy = {p.label: p.energy for p in probes}
    assert energy["compute ffma3"] * 32 == pytest.approx(330.1, abs=1.0)
    model = energy_predictor.Model.load(MODEL)
    assert model.pj["inst fma (fp32, imad)"] == pytest.approx(280.0, abs=0.5)
    for level in ("L1", "L2"):
        for width in (4, 16):
            added = (energy[f"{level} ld+ffma16 w{width}"] - energy[f"{level} ld w{width}"]) / 16 * 32
            assert 315.0 < added < 345.0, (level, width, added)
    # Clocked down by the power cap at duty 100: no energy (the fit needs both duties).
    for label in ("DRAM ld+ffma16 w4", "DRAM ld+shx2 w4", "DRAM ld+l1x8 w16"):
        assert label not in energy
    # The extra L1 and shared loads count as issued (the compiler did not merge them).
    rows = {(r["level"], r["operation"], r["width_bytes"]): r for r in csv.DictReader(open(os.path.join(mix, "MixProfile1.csv")))}
    plain, extra = rows[("L2", "ld", "16")], rows[("L2", "ld+shx8", "16")]
    per_access = lambda r, name: float(r[name]) / float(r["accesses_per_launch"])
    assert per_access(extra, "smsp__inst_executed_op_shared_ld.sum") * 32 == pytest.approx(8.0, rel=0.02)
    assert per_access(plain, "smsp__inst_executed_op_global_ld.sum") == pytest.approx(
        per_access(extra, "smsp__inst_executed_op_global_ld.sum"), rel=0.02)
