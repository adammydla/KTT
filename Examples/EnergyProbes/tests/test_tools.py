"""Regression tests of the EnergyProbes tools and of the program's GPU-free modes.

    python3 -m pytest Examples/EnergyProbes/tests -q            # from the KTT root; about half a minute
    EP_BINARY=/path/to/EnergyProbesCuda python3 -m pytest ...   # another build (default: Build/x86_64_Release)

No GPU is needed. The data in tests/data are real runs of an RTX 3090 at 1200 MHz: the probe power run Preset3 with its
energy table, the counter run ProfileModel1, the validation run Validation13, and the MT4G file. The tests of the program
(--predict, --dry-run, option checks) are skipped when it is not built.
"""

import copy
import csv
import json
import math
import os
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
    result = tool("build_model.py", "--power", "Preset3", "--counters", "ProfileModel1", "--calibration", "Validation13",
                  "--output", str(output))
    rebuilt = energy_predictor.Model.load(str(output))
    for name, energy in model.pj.items():
        assert rebuilt.pj[name] == pytest.approx(energy, rel=1e-6, abs=1e-6), name
    assert rebuilt.baseline == model.baseline
    assert rebuilt.required == model.required
    # One validation run instead of five: the relative sd is of the same size.
    assert 0.04 < rebuilt.uncertainty["dynamic_relative_sd"] < 0.08
    assert "ordered right" in result.stdout


def test_reference_scores(model):
    """The scores the README quotes for Preset3 + ProfileModel1 on Validation13."""
    result = tool("score_ranking.py", "--power", "Preset3", "--counters", "ProfileModel1", "--validation", "Validation13")
    line = next(l for l in result.stdout.splitlines() if "mean over the example kernels" in l)
    scatter, bias, mape, regret = (float(part.split()[-2 if "regret" not in part else 2].rstrip("%"))
                                   for part in line.split(":", 1)[1].split(",")[:4])
    assert scatter == pytest.approx(6.4, abs=0.3)
    assert abs(bias) < 5 and mape < 4.5 and regret <= 2.5


def test_predictor_equals_energy_model(model, validation):
    import contextlib
    import io
    import energy_model as m
    m.configure()
    with contextlib.redirect_stdout(io.StringIO()):
        probes, table = m.probe_table([os.path.join(DATA, "Preset3")], [os.path.join(DATA, "ProfileModel1")])
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
    take up to 10 % longer, so the energies are a few percent higher; the choice is the same."""
    kernels = energy_predictor.from_ktt_json(os.path.join(DATA, "Tune_gemm_counters.json"))
    assert list(kernels) == ["Gemm"] and len(kernels["Gemm"]) == 24
    predictions = [model.predict(c, s, t, clock, label) for label, c, s, t, clock in kernels["Gemm"]]
    choice = energy_predictor.recommend(model, predictions)
    rows = {r["configuration"]: r for r in counter_table.read_rows(os.path.join(DATA, "Tune_gemm_tune.csv"))}
    assert rows[predictions[choice.index].label]["choice"] == "best"
    for p in predictions:
        row = rows[p.label]
        assert p.dynamic_j == pytest.approx(float(row["predicted_dynamic_j"]), rel=0.02)
        assert p.energy_j == pytest.approx(float(row["predicted_j"]), rel=0.12)
    # The measured best of the run is the model's choice.
    measured = {label: float(r["measured_j"]) for label, r in rows.items()}
    assert measured[predictions[choice.index].label] == min(measured.values())
