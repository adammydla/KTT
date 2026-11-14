"""Regression tests of the EnergyProbes tools and of the program's GPU-free modes.

    python3 -m pytest Examples/EnergyProbes/tests -q            # from the KTT root; about half a minute
    EP_BINARY=/path/to/EnergyProbesCuda python3 -m pytest ...   # another build (default: Build/x86_64_Release)

No GPU is needed. The data in tests/data are real runs of an RTX 3090 at 1200 MHz: the probe power run Preset3 with its
energy table, the counter runs ProfileModel1 and ProfileNew2, the misaligned and write-path probes MisalignedWrite2 (at
one duty, --full-rate), the validation run Validation13, and the MT4G file. The tests of the program
(--predict, --dry-run, option checks) are skipped when it is not built.
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
MT4G = os.path.join(DATA, "NVIDIA_GeForce_RTX_3090.json")
VALIDATION = os.path.join(DATA, "Validation13_validation.csv")

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
    for name in ("Preset3.csv", "ProfileModel1.csv", "Validation13_validation.csv"):
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
    busy = energy_events.add_busy(dict(counts), 1e9, busy=True, scale_gbs=95.0)["DRAM busy ns"]  # 32 GB/s for 1 s
    assert busy == pytest.approx(1e9 * (1 - math.exp(-32 / 95.0)))
    assert energy_events.add_busy(dict(counts), 1e9, busy=False)["DRAM busy ns"] == 0.0
    assert energy_events.add_busy(dict(counts), 0.0, busy=True)["DRAM busy ns"] == 0.0


def test_every_counter_of_the_events_is_in_the_counter_table():
    energy_events.events({})  # value_of raises for a name that is not in the table


def test_cpp_predictor_knows_the_same_events():
    with open(os.path.join(ROOT, "EnergyModel.cpp")) as file:
        source = file.read()
    for name in energy_events.EVENTS:
        assert source.count('"' + name + '"') >= 2, name  # the constructor's list and Events()


# ---- Fit and model file -------------------------------------------------------------------------------------------------

def test_the_model_file_is_what_the_reference_runs_give(tmp_path, model):
    """build_model.py on the reference runs reproduces the energies of the shipped model: a change in the fit, the events
    or the CSV reading shows here."""
    output = tmp_path / "rebuilt.model.json"
    result = tool("build_model.py", "--power", "Preset3", "--counters", "ProfileModel1,ProfileNew2", "--full-rate",
                  "MisalignedWrite2", "--calibration", "Validation13", "--output", str(output))
    rebuilt = energy_predictor.Model.load(str(output))
    for name, energy in model.pj.items():
        assert rebuilt.pj[name] == pytest.approx(energy, rel=1e-6, abs=1e-6), name
    assert rebuilt.baseline == model.baseline
    assert rebuilt.required == model.required
    # One validation run instead of five: the relative sd is of the same size.
    assert 0.04 < rebuilt.uncertainty["dynamic_relative_sd"] < 0.08
    assert "ordered right" in result.stdout


def test_reference_scores(model):
    """The scores of the model's fit (Preset3 + ProfileModel1, ProfileNew2 + MisalignedWrite2) on Validation13."""
    result = tool("score_ranking.py", "--power", "Preset3", "--counters", "ProfileModel1,ProfileNew2", "--full-rate",
                  "MisalignedWrite2", "--validation", "Validation13")
    line = next(l for l in result.stdout.splitlines() if "mean over the example kernels" in l)
    scatter, bias, mape, regret = (float(part.split()[-2 if "regret" not in part else 2].rstrip("%"))
                                   for part in line.split(":", 1)[1].split(",")[:4])
    assert scatter == pytest.approx(6.3, abs=0.3)
    assert abs(bias) < 5 and mape < 4.0 and regret <= 2.5


def test_predictor_equals_energy_model(model, validation):
    import contextlib
    import io
    import energy_model as m
    m.configure()
    with contextlib.redirect_stdout(io.StringIO()):
        probes, table = m.probe_table([os.path.join(DATA, "Preset3")],
                                      [os.path.join(DATA, "ProfileModel1"), os.path.join(DATA, "ProfileNew2")],
                                      [os.path.join(DATA, "MisalignedWrite2")])
        rows = m.load_validation(os.path.join(DATA, "Validation13"), table)
    probes = [p for p in probes if p.label not in m.TWO_REGISTER_FFMA and p.label not in m.CHECK_ONLY]
    energies, _, _ = m.fit(probes, m.EVENTS)
    simple, _, _ = m.fit(probes, m.SIMPLE)
    reference = {(r["kernel"], r["configuration"]): r["predicted_dynamic_j"] for r in m.predict(rows, energies, simple)}
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
    """On every kernel of Validation13 the model's first choice uses at most 2.5 % more energy than the measured best,
    and the measured best is the choice or one of its contenders."""
    rows = counter_table.read_rows(VALIDATION)
    measured = {(r["kernel"], r["configuration"]): counter_table.number(r["energy_j"]) for r in rows}
    for kernel, configurations in validation.items():
        if kernel == "mixed":
            continue
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
    kernels, 5.4 %, widens the sd to about 8.3 % per prediction)."""
    within = [prediction(1.0, 0.5, 1.0)._replace(group="a"), prediction(1.1, 0.55, 1.0)._replace(group="a")]
    assert energy_predictor.recommend(model, within).confident
    between = [within[0], within[1]._replace(group="b")]
    choice = energy_predictor.recommend(model, between)
    assert choice.index == 0 and choice.contenders == [1]
    assert model.between_kernels_sd() == pytest.approx(model.uncertainty["dynamic_bias_sd_between_kernels"])
    assert 0.03 < model.between_kernels_sd() < 0.08


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
        assert key in model.device  # null for runs from before 2026-11-13
    assert model.data["uncertainty"]["calibration"]["left_out_off_clock"] == 3
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
    rows = counter_table.read_rows(os.path.join(DATA, "Preset3.csv"))
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
    result = tool("check_model.py", "--model", MODEL, "--run", "Preset3")
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
    run = edited_rows(tmp_path, "Preset3", "Warm", higher_sleep)
    written = tmp_path / "baseline.model.json"
    result = tool("check_model.py", "--model", MODEL, "--run", run, "--write", str(written))
    assert "BASELINE" in result.stdout and "PASS" not in result.stdout
    adjusted = energy_predictor.Model.load(str(written))
    assert adjusted.baseline["mean_w"] == pytest.approx(1.05 * energy_predictor.Model.load(MODEL).baseline["mean_w"],
                                                        rel=0.01)


def test_check_model_compares_counts(tmp_path):
    result = tool("check_model.py", "--model", MODEL, "--run", "Preset3", "--counters", "ProfileModel1")
    assert "PASS" in result.stdout and "0 move" in result.stdout

    def doubled(row):
        if row.get("dram__sectors_read.sum"):
            row["dram__sectors_read.sum"] = repr(2 * float(row["dram__sectors_read.sum"]))
    run = edited_rows(tmp_path, "ProfileModel1", "Doubled", doubled)
    result = tool("check_model.py", "--model", MODEL, "--run", "Preset3", "--counters", run, check=False)
    assert result.returncode == 1 and "event counts differ" in result.stdout and "DRAM read sector" in result.stdout


# ---- The other tools still run on the reference runs --------------------------------------------------------------------

def test_energy_model_report():
    result = tool("energy_model.py", "--power", "Preset3", "--counters", "ProfileModel1", "--validation", "Validation13",
                  "--cross-validate")
    assert "Fit over 134 probes" in result.stdout and "Leave-one-out" in result.stdout


def test_check_run():
    result = tool("check_run.py", "Preset3", "--counters", "ProfileModel1", check=False)
    assert "Traceback" not in result.stdout and "locality, write, mixed hit their level" in result.stdout


def test_check_counters():
    result = tool("check_counters.py", "--csv", "ProfileModel1.csv", check=False)
    assert "Traceback" not in result.stdout


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
    assert default.returncode == 0
    preset = program("--dry-run", "--mt4g", MT4G, "--model")
    assert preset.returncode == 0 and "458" in preset.stdout
    quick = program("--dry-run", "--mt4g", MT4G, "--quick")
    assert quick.returncode == 0 and "30 configurations" in quick.stdout
    # An explicit option wins over a preset wherever it stands.
    assert "each at duty 50 %" in program("--dry-run", "--mt4g", MT4G, "--duty", "50", "--quick").stdout
    everything = program("--dry-run", "--mt4g", MT4G, "--all")
    assert everything.returncode == 0 and "918" in everything.stdout


@needs_binary
@pytest.mark.parametrize("arguments, message", [
    (["--duty", "abc"], "--duty"),
    (["--widths", "16x"], "--widths"),
    (["--block-size", "100"], "--block-size"),
    (["--max-power-diff", "-1"], "--max-power-diff"),
    (["--writes", "--unroll", "6"], "--writes"),
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
    shutil.copy(os.path.join(DATA, "Preset3.csv"), tmp_path / "Preset3.csv")
    prefix = str(tmp_path / "again")
    result = program("--analyze", str(tmp_path / "Preset3"), "--mt4g", MT4G, "--output", prefix)
    assert result.returncode == 0, result.stdout[-2000:]
    with open(prefix + "_energy_table.json") as file:
        again = json.load(file)
    with open(os.path.join(DATA, "Preset3_energy_table.json")) as file:
        reference = json.load(file)
    key = lambda f: (f["level"], f["operation"], f["width_bytes"], f["payload"])  # noqa: E731
    fits = {key(f): f["pj_per_access"] for f in reference["fits"]}
    assert len(again["fits"]) == len(fits)
    for fit in again["fits"]:
        assert fit["pj_per_access"] == pytest.approx(fits[key(fit)], rel=1e-6)
    assert again["baseline"]["slope_w_per_c"] == pytest.approx(reference["baseline"]["slope_w_per_c"], rel=1e-6)


# ---- SASS (slow: about 3 min; EP_SLOW=1) --------------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("EP_SLOW") or not shutil.which("nvcc", path=os.environ.get("PATH", "")
                                                                         + os.pathsep + "/usr/local/cuda/bin"),
                    reason="set EP_SLOW=1 (needs a CUDA toolkit, no GPU)")
def test_sass_checks():
    result = tool("check_sass.py", "--arch", "75,86,90", check=False)
    assert result.returncode == 0 and "all checks passed" in result.stdout, result.stdout[-3000:]


# ---- A real KTT output file ---------------------------------------------------------------------------------------------

def test_real_ktt_output_file(model):
    """Tune_gemm_counters.json is what KTT saved for the counter pass of a --tune run (24 Gemm configurations, RTX 3090),
    Tune_gemm_tune.csv the program's own predictions in that run from unprofiled times. The profiled launches in the JSON
    take up to 10 % longer, so the energies are a few percent higher; the choice is the same. The run's predictions are
    from the model of 2026-11-12; the model with the misaligned and write-path probes (2026-11-14) differs from it by
    -5 to +3 % on these dynamic energies."""
    kernels = energy_predictor.from_ktt_json(os.path.join(DATA, "Tune_gemm_counters.json"))
    assert list(kernels) == ["Gemm"] and len(kernels["Gemm"]) == 24
    predictions = [model.predict(c, s, t, clock, label) for label, c, s, t, clock in kernels["Gemm"]]
    choice = energy_predictor.recommend(model, predictions)
    rows = {r["configuration"]: r for r in counter_table.read_rows(os.path.join(DATA, "Tune_gemm_tune.csv"))}
    assert rows[predictions[choice.index].label]["choice"] == "best"
    for p in predictions:
        row = rows[p.label]
        assert p.dynamic_j == pytest.approx(float(row["predicted_dynamic_j"]), rel=0.06)
        assert p.energy_j == pytest.approx(float(row["predicted_j"]), rel=0.12)
    # The measured best of the run is the model's choice.
    measured = {label: float(r["measured_j"]) for label, r in rows.items()}
    assert measured[predictions[choice.index].label] == min(measured.values())


# ---- Search over configurations that were not profiled (2026-11-14) ------------------------------------------------------

@pytest.mark.parametrize("kind", ["energy", "counters"])
def test_energy_search_proposes_unprofiled_configurations(model, validation, kind):
    """From 16 profiled Gemm configurations of Validation13, the surrogate ranks the other 8 (whose counters it does not
    see): proposals are only unprofiled ones, best first, and their predicted energies are near the model's on their
    real counters (same kernel, same work: within a factor of 2)."""
    pytest.importorskip("sklearn")
    import energy_search
    rows = validation["gemm"]
    space = [label for label, *_ in rows]
    profiled = rows[:16]
    proposals = energy_search.propose(model, space, [r[0] for r in profiled], [r[1] for r in profiled],
                                      [r[2] for r in profiled], 3, kind)
    assert len(proposals) == 3
    assert all(label in space[16:] for label, _ in proposals)
    assert [e for _, e in proposals] == sorted(e for _, e in proposals)
    truth = {label: model.predict(c, s, strict=False).energy_j for label, c, s, _, _ in rows}
    for label, energy in proposals:
        assert 0.5 < energy / truth[label] < 2.0


def test_energy_search_features():
    import energy_search
    features = energy_search.Features(["A=1 B=x C=4", "A=3 B=y C=4", "A=7 B=x C=4"])
    assert features.names == ["A", "B"]  # C has one value
    assert features(["A=1 B=y"]).tolist() == [[1.0, 1.0]]


def test_energy_search_cli(tmp_path):
    pytest.importorskip("sklearn")
    space = tmp_path / "space.txt"
    labels = [row["configuration"] for row in counter_table.read_rows(VALIDATION) if row["kernel"] == "gemm"]
    space.write_text("\n".join(labels) + "\n")
    result = tool("energy_search.py", "--model", MODEL, "--csv", VALIDATION, "--kernel", "gemm", "--space", str(space),
                  "--count", "2", check=False)
    # Every configuration of the space was profiled: nothing to propose, no error.
    assert result.returncode == 0, result.stdout
    assert result.stdout.strip() == ""


# ---- AccelWattch import (2026-11-14) ---------------------------------------------------------------------------------------

def test_accelwattch_import(tmp_path):
    """A benchmark in AccelWattch's layout becomes a kernel awKernel under AW_BENCH with iterations from the script, its
    dimensions from its macros, unused pointers padded, and --set variants; texture and two-kernel programs are skipped."""
    source = tmp_path / "ubench"
    common = ("#include <stdio.h>\n#define THREADS_PER_BLOCK 256\n#define NUM_OF_BLOCKS 640\n"
              "#define TWICE(x) \\\n    ((x) + (x))\n")
    main_one = ("int main(int argc, char** argv){ int iterations = atoi(argv[1]); unsigned blocks = atoi(argv[2]);\n"
                " int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS; size_t size = N * sizeof(float);\n"
                " cudaMalloc((void**)&d_A, size); dim3 dimGrid(blocks,1); dim3 dimBlock(THREADS_PER_BLOCK,1);\n"
                " K<<<dimGrid,dimBlock>>>(d_A, d_C, iterations); }\n")
    files = {
        "functional_benchmarks/ONE/ONE.cu": common + "__global__ void K(const float* A, float* C, int iterations)\n"
            "{ float v = A[threadIdx.x]; for (int k = 0; k < iterations; ++k) v = TWICE(v); C[threadIdx.x] = v; }\n"
            + main_one,
        "functional_benchmarks/TEX/TEX.cu": common + "texture<float,1> t;\n__global__ void K(float* C, int n) {}\n"
            "int main(){ K<<<1,1>>>(d_C, 1); }\n",
        "mix_benchmarks/TWO/TWO.cu": common + "__global__ void K1(float* C, int n) {}\n__global__ void K2(float* C, int n) {}\n"
            "int main(){ int iterations = 1; K1<<<1,1>>>(d_C, iterations); K2<<<1,1>>>(d_C, iterations); }\n",
    }
    for name, text in files.items():
        (source / os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
        (source / name).write_text(text)
    out = tmp_path / "out"
    result = tool("accelwattch_import.py", str(source), str(out), "--set", "blocks=82,41", "--iterations", "7")
    assert "2 benchmarks written" in result.stdout
    assert "TEX: texture references" in result.stdout and "TWO: launches 2 kernels" in result.stdout
    table = [line.split("\t") for line in (out / "accelwattch.tsv").read_text().splitlines() if not line.startswith("#")]
    assert [(t[1], t[2], t[3], t[4], t[5]) for t in table] == [
        ("functional/ONE blocks=82", "82", "256", "7", "2"), ("functional/ONE blocks=41", "41", "256", "7", "2")]
    assert int(table[0][6]) == 16 << 20
    kernel = (out / "AccelWattch.cu").read_text()
    assert "#if AW_BENCH == 1" in kernel and "#define AW_ITER 7" in kernel
    assert 'extern "C" __global__ void awKernel(const float* A, float* C, void* awUnused2, void* awUnused3)' in kernel
    assert "const int iterations = awIterations;" in kernel and "awIterations = AW_ITER;" in kernel
    assert "#define TWICE(x) \\\n    ((x) + (x))" in kernel
    if shutil.which("nvcc", path=os.environ.get("PATH", "") + os.pathsep + "/usr/local/cuda/bin"):
        nvcc = shutil.which("nvcc", path=os.environ.get("PATH", "") + os.pathsep + "/usr/local/cuda/bin")
        compiled = subprocess.run([nvcc, "-arch=sm_86", "-DAW_BENCH=0", "-cubin", "-o", str(tmp_path / "k.cubin"),
                                   str(out / "AccelWattch.cu")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  universal_newlines=True)
        assert compiled.returncode == 0, compiled.stdout


@needs_binary
def test_accelwattch_is_a_validation_name():
    assert "accelwattch" in program("--validate", "bogus").stdout
    assert "--accelwattch-dir" in program("--help").stdout


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


def test_fit_stability_report():
    """tools/fit_stability.py (NEXT_STEPS_PLAN P4) on the test runs: prices with their CV, per-kernel stability and the
    families left out."""
    result = tool("fit_stability.py", "--power", "Preset3", "--counters", "ProfileModel1,ProfileNew2", "--full-rate",
                  "MisalignedWrite2", "--validation", "Validation13", "--resamples", "20")
    assert "stability: kernel means" in result.stdout and "within-kernel" in result.stdout
    assert "DRAM read sector" in result.stdout and "one family left out" in result.stdout


def test_fit_thermal_load_recovers_a_slope(tmp_path):
    """tools/fit_thermal_load.py (NEXT_STEPS_PLAN M1) on a synthetic trace: the sleep baseline rises 0.3 W/K, a probe
    0.3 + 0.8 W/K; temperatures run up (cool start) and down (hot start), with a drift in time that the fit separates."""
    lines = ["configuration,time_s,power_w,instant_power_w,average_power_w,energy_j,sm_clock_mhz,memory_clock_mhz,"
             "temperature_c,fan_pct,reasons"]
    clock = 0.0
    for name, base, slope in (("Hold_sleep_cool", 115.0, 0.3), ("Hold_sleep_hot", 115.0, 0.3),
                              ("Hold_ffma3_d100_cool", 200.0, 1.1), ("Hold_ffma3_d100_hot", 200.0, 1.1)):
        energy = 0.0
        for k in range(900):  # 90 s at 100 ms
            t = k * 0.1
            temperature = 60 + 10 * (1 - math.exp(-t / 30)) if name.endswith("cool") else 75 - 8 * (1 - math.exp(-t / 30))
            power = base + slope * (temperature - 65) + 0.01 * t
            energy += power * 0.1
            lines.append(f"{name},{clock + t:.3f},{power:.3f},{power:.3f},{power:.3f},{energy:.4f},1200,9501,"
                         f"{round(temperature)},50,")
        clock += 200
    (tmp_path / "T_trace.csv").write_text("\n".join(lines) + "\n")
    result = tool("fit_thermal_load.py", str(tmp_path / "T"))
    sleep = re.search(r"Sleep baseline: b = ([+-][0-9.]+)", result.stdout)
    assert sleep and float(sleep.group(1)) == pytest.approx(0.3, abs=0.06), result.stdout
    excess = re.search(r"Excess slope of the probes \(b - b_sleep\): mean ([+-][0-9.]+)", result.stdout)
    assert excess and float(excess.group(1)) == pytest.approx(0.8, abs=0.1), result.stdout
