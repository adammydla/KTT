#!/usr/bin/env python3
"""Does the data that a kernel reads change its energy? The probes and the event energies use random data; the
validation kernels read uniform random floats of various ranges (ValidationKernels.cpp), whose words have fewer ones and
fewer toggles. The two-factor fits of the 16-byte load probes from --payload-bits runs (E = E0 + a ones + b toggles,
with the toggles between words 8 apart, the beats of a 32-byte sector, which explain the loads: fit_payload.py) give per
level how much a load of such data costs against random data. That difference goes onto the level's read-sector event
(L1 sector, L2 read sector, DRAM read sector), level by level (each probe's change minus what the levels above it explain
already). The model is otherwise unchanged; the control kernel reads random data (factor 1). Nothing is fitted on the
validation kernels.

    python3 tools/data_factor.py --payload PayloadBits1,PayloadBits2,PayloadBits3 [--stats stats.json] [--skip zero] \
        --power Full,Ffma3,Writes,Instr --counters ProfileFull4 --validation Validation9,...,HeldOut2

--stats replaces the runs' bit statistics (runs before 2026-11-06 21:00 lack the lagged toggles; --skip zero: their
"zero" read random data). Stores are not covered: no probe stores controlled data.
"""
import argparse, contextlib, csv, io, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--payload", required=True, help="--payload-bits runs, comma-separated")
parser.add_argument("--stats", help="bit statistics replacing the runs' _payloads.json")
parser.add_argument("--skip", default="", help="payloads to leave out of the fit, comma-separated; RUN:PAYLOAD for one run")
parser.add_argument("--power", required=True)
parser.add_argument("--counters", required=True)
parser.add_argument("--validation", required=True)
arguments = parser.parse_args()
runs, power, counters = arguments.payload.split(","), arguments.power, arguments.counters
validations = arguments.validation.split(",")
stats_file = arguments.stats
skip = set(filter(None, arguments.skip.split(",")))
TOGGLES = "toggles_lag8"

# The example kernels' inputs (ValidationKernels.cpp: uniform floats over these ranges), and their word statistics over
# 1e6 values. Coulomb's atoms: x, y, z in [0, 20) (w in [0, 0.5) is a quarter of its data and is left out).
RANGES = {"transpose": (0, 10), "gemm": (-2, 2), "nbody": (0, 20), "coulomb": (0, 20), "reduction": (0, 1000),
          "convolution": (-2, 2), "pnpoly": (0, 1)}
rng = np.random.default_rng(1)


def word_statistics(low, high):
    words = rng.uniform(low, high, 1_000_000).astype(np.float32).view(np.uint32)
    ones = np.unpackbits(words.view(np.uint8)).mean()
    toggles = np.unpackbits((words[1:] ^ words[:-1]).view(np.uint8)).mean()
    return ones, toggles


data = {k: word_statistics(*r) for k, r in RANGES.items()}

# Two-factor fit of each load level, on energies relative to the same run's random payload.
statistics, fits = {}, []
for run in runs:
    statistics.update(json.load(open(run + "_payloads.json")))
    fits.append((run, list(csv.DictReader(open(run + "_fits.csv")))))
if stats_file:
    statistics.update(json.load(open(stats_file)))
TRAIN = lambda p: p in ("zero", "random") or p.startswith(("ones:", "same:", "flip:"))
model = {}
for level in ("L1", "L2", "DRAM"):
    x, y = [], []
    for run, run_fits in fits:
        points = {f["payload"]: float(f["pj_per_access"]) for f in run_fits if f["level"] == level and f["operation"] == "ld"
                  and f["width_bytes"] == "16" and f["payload"] in statistics and f["pj_per_access"]
                  and f["payload"] not in skip and f"{run}:{f['payload']}" not in skip}
        for p, e in points.items():
            if TRAIN(p) and "random" in points:
                x.append([1, statistics[p]["words"]["ones"], statistics[p]["words"][TOGGLES]])
                y.append(e / points["random"])
    coefficients, *_ = np.linalg.lstsq(np.array(x), np.array(y), rcond=None)
    model[level] = coefficients
    print(f"{level}: relative E = {coefficients[0]:.3f} + {coefficients[1]:.3f} ones + {coefficients[2]:.3f} toggles "
          f"(random: {coefficients @ [1, 0.5, 0.5]:.3f})")

relative = lambda level, ones, toggles: (model[level] @ [1, ones, toggles]) / (model[level] @ [1, 0.5, 0.5])

with contextlib.redirect_stdout(io.StringIO()):
    probes, table = m.probe_table(power.split(","), counters.split(","))
if any(p.label == "compute ffma3" for p in probes):
    probes = [p for p in probes if p.label not in m.TWO_REGISTER_FFMA]
probes = [p for p in probes if p.label not in m.CHECK_ONLY]
energies, _, _ = m.fit(probes, m.EVENTS, False)
simple, _, _ = m.fit(probes, m.SIMPLE, False)
by_label = {p.label: p for p in probes}
EVENT = {"L1": "L1 sector", "L2": "L2 read sector", "DRAM": "DRAM read sector"}


def factors(ones, toggles):
    """The energy factor of each level's read-sector event for data with these statistics (level by level)."""
    g = {}
    for level in ("L1", "L2", "DRAM"):
        probe = by_label[f"{level} ld w16"]
        energy = sum(probe.counts[e] * energies[e] for e in m.EVENTS)
        change = energy * (relative(level, ones, toggles) - 1)
        for upper in g:  # what the levels above explain already
            change -= (g[upper] - 1) * probe.counts[EVENT[upper]] * energies[EVENT[upper]]
        own = probe.counts[EVENT[level]] * energies[EVENT[level]]
        g[level] = 1 + change / own if own > 0 else 1.0
    return g


for kernel, (ones, toggles) in data.items():
    g = factors(ones, toggles)
    print(f"  {kernel:12} ones {ones:.3f} toggles {toggles:.3f}: " + ", ".join(f"{EVENT[l]} x {g[l]:.3f}" for l in g))

print(f"\n{'run':14} {'kernel':12} {'scatter':>16} {'bias':>18} {'total MAPE':>16}   (unchanged -> with the data factor)")
summary = {"base": [], "data": []}
for name in validations:
    with contextlib.redirect_stdout(io.StringIO()):
        rows = m.load_validation(name, table)
    results = m.predict(rows, energies, simple)
    for kernel in dict.fromkeys(r["kernel"] for r in results):
        group = [r for r in results if r["kernel"] == kernel]
        g = factors(*data[kernel]) if kernel in data else {l: 1.0 for l in EVENT}
        out = {}
        for variant in ("base", "data"):
            dyn = np.array([r["predicted_dynamic_j"] * (1 + (sum((g[l] - 1) * r["shares"][EVENT[l]] for l in EVENT)
                                                                if variant == "data" else 0)) for r in group])
            measured_dyn = np.array([r["measured_dynamic_j"] for r in group])
            base = np.array([r["predicted_j"] - r["predicted_dynamic_j"] for r in group])
            measured = np.array([r["measured_j"] for r in group])
            error = dyn / measured_dyn - 1
            out[variant] = (error.std(), error.mean(), np.abs((base + dyn) / measured - 1).mean())
            if kernel != m.CONTROL:
                summary[variant].append(out[variant])
        b, d = out["base"], out["data"]
        print(f"{name:14} {kernel:12} {100*b[0]:6.1f} -> {100*d[0]:4.1f} % {100*b[1]:+6.1f} -> {100*d[1]:+5.1f} % "
              f"{100*b[2]:6.1f} -> {100*d[2]:4.1f} %" + ("   (control)" if kernel == m.CONTROL else ""))
for variant in ("base", "data"):
    s = np.mean(summary[variant], axis=0)
    print(f"mean over the example kernels, {variant}: scatter {100*s[0]:.2f} %, bias {100*s[1]:+.2f} %, total MAPE {100*s[2]:.2f} %")
