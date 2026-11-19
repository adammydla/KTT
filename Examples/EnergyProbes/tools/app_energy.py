#!/usr/bin/env python3
"""Energy of whole applications: measured with NVML, predicted from the kernel model plus the application probes.

The probes and the validation runs price single kernels launched back to back. A whole application also copies data
between the host and the GPU, leaves the GPU idle between kernels and while the host works, and runs many different
kernels once or a few times. Antepara et al. (SC'25) validate their benchmark-driven model on whole applications; this
tool does the same for the kernel model:

    energy = sum over kernel launches of the model's prediction (the awake baseline over the launch's time plus its
             event energies; counters from Nsight Compute, times from Nsight Systems)
           + sum over copies of the copy power times their time (application probes)
           + the GPU without work (application probes): creating a context wakes it; within a fraction of a millisecond
             after a kernel it falls to its awake idle power and stays there until an idle timer runs out (about 17.5 s
             after the last work on an RTX 3090) or until about 3 s after the application exits, whichever is first;
             then it falls to its deep idle power; before its context wakes it (CUDA's start-up, about 0.6 s after
             the start) it is in its deep idle

Steps (the GPU is CUDA_DEVICE_ORDER=PCI_BUS_ID device N, as EnergyProbes --device N; one GPU job at a time):

    apps/build_rodinia.sh RODINIA/cuda BUILD                     # applications and app_probes, once
    python3 apps/make_inputs.py BUILD/inputs                     # inputs of hotspot and hotspot3D, once
    python3 tools/app_energy.py probes  --device 1 --build BUILD --out models/rtx3090_1200mhz.app.json
    python3 tools/app_energy.py measure --device 1 --build BUILD --apps apps/rodinia.json --out RUN
    python3 tools/app_energy.py profile --device 1 --build BUILD --apps apps/rodinia.json --out RUN
    python3 tools/app_energy.py score   --model models/rtx3090_1200mhz.model.json \
                                        --app-model models/rtx3090_1200mhz.app.json --run RUN

measure: runs every application a few times on the GPU, samples NVML's power (the reading the probes are calibrated on)
and integrates it over the application's run. This process holds a CUDA context on the GPU, as a node that keeps one;
on an RTX 3090 that does not keep the GPU awake: before every run it is in its deep idle, and about 3 s after the
application exits it falls back to it. NVML averages the power over about 1 s; the integral runs on for TAIL_S seconds after
the application exits and the idle level at the end of that tail is subtracted, which gives the application's energy
(including the seconds the GPU stays awake after it) whatever the averaging window.

profile: one run under Nsight Systems (every kernel and copy with its GPU time) and runs under Nsight Compute (the
model's counters per launch, and the instructions per SASS opcode). Applications with thousands of launches profile
windows of consecutive launches spread over the run (the app list's "windows" and "window"; one ncu run each, about
0.2 s per profiled launch, more where ncu saves and restores large buffers for its passes); the launches in between take
the counters of the profiled launches of the same kernel around them, interpolated by the kernel's launch number.

score: predicted against measured energy per application, whole and above the idle power with a context, with the parts.

Needs: pynvml (nvidia-ml-py), libcuda, nsys and ncu (on PATH or in CUDA_PATH/bin). No SciPy.
"""

import argparse
import bisect
import csv
import ctypes
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402
import energy_predictor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT = "energyprobes-app-model"
SAMPLE_S = 0.01
# How long the integral runs on after the application exits, and how much of its end gives the idle level.
# On an RTX 3090 the GPU stays awake for about 3 s after an application exits, then falls to its deep idle power (about
# 42 W, with or without a context), which NVML shows about 1.5 s later: the tail must outlast both. The wait before a run
# must outlast the idle timer of the previous run's context (about 17.5 s) only if that run left one; it does not.
TAIL_S = 9.0
TAIL_MAX_S = 60.0
IDLE_TOLERANCE_W = 1.0
POST_S = 2.0
PRE_S = 3.0
# The mean delay of NVML's averaging (about a 1 s window): weights the idle levels before and after the application.
AVERAGING_DELAY_S = 0.5
# NVML's power reading on the RTX 3090: the mean over the last NVML_WINDOW_S, a new value every NVML_UPDATE_S (held in
# between; the application probes and the readings of a run, 2026-11-18). nvml_view() passes a predicted power timeline
# through the same filter.
NVML_WINDOW_S = 1.0
NVML_UPDATE_S = 0.5
# The application probes: gaps after sleep kernels of GAP_KERNEL_US, and copies of COPY_MB.
GAP_KERNEL_US = 2000
GAPS_US = [0, 20, 100, 500, 2000, 10000, 50000, 200000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000,
           4500000, 6000000]
COPIES = [("h2d", "pageable"), ("h2d", "pinned"), ("d2h", "pageable"), ("d2h", "pinned"), ("d2d", "pinned")]
COPY_MB = 256
PROBE_S = 8.0
# The context probe: from the deep idle (CONTEXT_SETTLE_S after the last work, longer than the idle timer), a context, then
# one kernel after CONTEXT_WAIT_S (longer than the idle timer), then the exit after CONTEXT_AFTER_S.
CONTEXT_SETTLE_S = 25.0
CONTEXT_WAIT_S = 30.0
CONTEXT_AFTER_S = 6.0
CONTEXT_TAIL_S = 12.0
# The wait after this process creates its context, longer than the idle timer.
INITIAL_WAIT_S = 22.0
# Per-opcode instruction counts that Nsight Compute adds to the model's counters (sass__* are SASS-patched metrics).
OPCODE_METRICS = ["sass__inst_executed_per_opcode", "sass__inst_executed_per_opcode_with_modifier_all",
                  "sass__thread_inst_executed_true_per_opcode"]
EXTRA_METRICS = ["gpu__time_duration.sum", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                 "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
                 "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
                 "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
                 # distinct bytes of shared loads (a broadcast load reads fewer than 32 words; the broadcast probes)
                 "sm__sass_data_bytes_mem_shared_op_ld.sum", "l1tex__data_bank_reads.sum"]


def tool(name):
    """nsys or ncu: on PATH, or in CUDA_PATH/bin."""
    found = shutil.which(name)
    if found:
        return found
    candidate = os.path.join(os.environ.get("CUDA_PATH", "/usr/local/cuda"), "bin", name)
    if os.path.exists(candidate):
        return candidate
    raise SystemExit(f"{name} not found (PATH, CUDA_PATH/bin)")


def app_environment(device, scratch):
    env = dict(os.environ)
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "APP_CUDA_DEVICE": str(device), "LC_ALL": "C", "TMPDIR": scratch})
    return env


# ----------------------------------------------------------------------------------------------------------------------
# The GPU: a CUDA context held by this process, and NVML samples.

class HeldContext:
    """Retains the primary CUDA context of device `ordinal` (CUDA_DEVICE_ORDER=PCI_BUS_ID) for this process's lifetime and
    returns the device's PCI bus id, to open the same GPU in NVML."""

    def __init__(self, ordinal):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self._check(self.cuda.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(self.cuda.cuDeviceGet(ctypes.byref(self.device), ordinal), "cuDeviceGet")
        self.context = ctypes.c_void_p()
        self._check(self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), self.device), "cuDevicePrimaryCtxRetain")
        self._check(self.cuda.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")
        bus = ctypes.create_string_buffer(64)
        self._check(self.cuda.cuDeviceGetPCIBusId(bus, 64, self.device), "cuDeviceGetPCIBusId")
        self.pci_bus_id = bus.value.decode()

    @staticmethod
    def _check(status, what):
        if status != 0:
            raise SystemExit(f"{what} failed with CUDA error {status}")


class Sampler:
    """NVML samples of one GPU every SAMPLE_S in a thread: (time, power W, temperature C, SM clock MHz, energy J)."""

    def __init__(self, pci_bus_id):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id.encode())
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else name
        # nvmlDeviceGetTotalEnergyConsumption is newer than some pynvml packages: call it through ctypes.
        self._energy = None
        try:
            function = pynvml._nvmlGetFunctionPointer("nvmlDeviceGetTotalEnergyConsumption")
            value = ctypes.c_ulonglong()
            if function(self.handle, ctypes.byref(value)) == 0:
                self._energy = function
        except Exception:
            pass
        self.samples = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample(self):
        nvml = self.nvml
        t = time.monotonic()
        power = nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        temperature = float(nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU))
        clock = float(nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM))
        energy = math.nan
        if self._energy is not None:
            value = ctypes.c_ulonglong()
            if self._energy(self.handle, ctypes.byref(value)) == 0:
                energy = value.value / 1000.0
        return (t, power, temperature, clock, energy)

    def _run(self):
        while not self._stop.is_set():
            sample = self._sample()
            with self._lock:
                self.samples.append(sample)
            self._stop.wait(SAMPLE_S)

    def between(self, a, b):
        with self._lock:
            return [s for s in self.samples if a <= s[0] <= b]

    def stop(self):
        self._stop.set()
        self._thread.join()


def integrate(samples, a, b):
    """The integral of the power over [a, b] (trapezoids, held flat to the ends), J."""
    points = [(t, p) for t, p, *_ in samples if a <= t <= b]
    if len(points) < 2:
        return math.nan
    total = (points[0][0] - a) * points[0][1] + (b - points[-1][0]) * points[-1][1]
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        total += (t1 - t0) * (p0 + p1) / 2
    return total


def mean_power(samples, a, b):
    values = [p for t, p, *_ in samples if a <= t <= b]
    return statistics.fmean(values) if values else math.nan


def application_energy(samples, a, b, tail_s=TAIL_S):
    """Energy of an application that ran from a to b: the power's integral from a to b + tail_s, minus the idle levels
    before (over the averaging delay) and after (over the rest of the tail). See the module's text. Split at the exit
    shifted by the averaging delay: the process (process_j, its own run) and what the GPU draws above the idle level
    after the exit (after_exit_j: it stays awake for a few seconds, longer after some applications)."""
    pre = mean_power(samples, a - PRE_S, a)
    post = mean_power(samples, b + tail_s - POST_S, b + tail_s)
    window = integrate(samples, a, b + tail_s)
    energy = window - pre * AVERAGING_DELAY_S - post * (tail_s - AVERAGING_DELAY_S)
    process = integrate(samples, a, b + AVERAGING_DELAY_S) - pre * AVERAGING_DELAY_S
    inside = [s for s in samples if a <= s[0] <= b]
    counter = [s[4] for s in samples if a - 0.05 <= s[0] <= b + tail_s and not math.isnan(s[4])]
    return {"wall_s": b - a, "energy_j": energy, "process_j": process, "after_exit_j": energy - process, "tail_s": tail_s,
            "pre_idle_w": pre, "post_idle_w": post,
            "mean_power_w": energy / (b - a) if b > a else math.nan,
            "temperature_c": statistics.fmean(s[2] for s in inside) if inside else math.nan,
            "min_clock_mhz": min((s[3] for s in inside), default=math.nan),
            "max_clock_mhz": max((s[3] for s in inside), default=math.nan),
            # The NVML energy counter over the same window, idle subtracted the same way (for comparison only: on an RTX
            # 3090 it covers more rails than the power reading the model is calibrated on).
            "counter_energy_j": (counter[-1] - counter[0]) - pre * AVERAGING_DELAY_S - post * (tail_s - AVERAGING_DELAY_S)
            if len(counter) >= 2 else math.nan,
            "samples": len(inside)}


def run_command(command, cwd, env, log):
    with open(log, "w") as out:
        a = time.monotonic()
        status = subprocess.run(command, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        b = time.monotonic()
    return a, b, status


def wait_for_idle(sampler, a, b):
    """Waits after an application's exit at b until NVML shows the idle level before its start at a again (within
    IDLE_TOLERANCE_W over POST_S), at least TAIL_S and at most TAIL_MAX_S; returns the tail's length."""
    pre = mean_power(sampler.samples, a - PRE_S, a)
    while True:
        now = time.monotonic()
        if now - b >= TAIL_MAX_S or (now - b >= TAIL_S and mean_power(sampler.samples, now - POST_S, now)
                                     <= pre + IDLE_TOLERANCE_W):
            return now - b
        time.sleep(0.25)


# ----------------------------------------------------------------------------------------------------------------------
# Application lists.

def load_apps(path, build, only=None):
    """The app list: [{"name", "command": [program, args...], "windows": n, "window": launches}]. {build} and {inputs}
    in the command are replaced; the program is relative to the build folder."""
    with open(path) as file:
        apps = json.load(file)["apps"]
    result = []
    for app in apps:
        if only and app["name"] not in only:
            continue
        command = [part.replace("{build}", build).replace("{inputs}", os.path.join(build, "inputs")) for part in app["command"]]
        command[0] = os.path.join(build, command[0])
        result.append(dict(app, command=command))
    if only:
        missing = set(only) - {app["name"] for app in result}
        if missing:
            raise SystemExit(f"not in {path}: {', '.join(sorted(missing))}")
    return result


def output_command(command, work):
    """Output files of the applications go to the run's work folder ({work} in the command)."""
    return [part.replace("{work}", work) for part in command]


# ----------------------------------------------------------------------------------------------------------------------
# measure

def measure(args):
    apps = load_apps(args.apps, os.path.abspath(args.build), args.only)
    os.makedirs(args.out, exist_ok=True)
    work = os.path.abspath(os.path.join(args.out, "work"))
    os.makedirs(work, exist_ok=True)
    context = HeldContext(args.device)
    sampler = Sampler(context.pci_bus_id)
    env = app_environment(args.device, work)
    results = {}
    path = os.path.join(args.out, "measure.json")
    if os.path.exists(path):
        with open(path) as file:
            results = json.load(file)["apps"]
    try:
        # Creating this process's context wakes the GPU for the idle timer: the first run starts from the deep idle.
        time.sleep(max(args.idle, INITIAL_WAIT_S))
        for app in apps:
            runs = []
            for repeat in range(args.repeats):
                time.sleep(args.idle)
                log = os.path.join(args.out, f"{app['name']}.{repeat}.log")
                a, b, status = run_command(output_command(app["command"], work), work, env, log)
                tail = wait_for_idle(sampler, a, b)
                if status != 0:
                    raise SystemExit(f"{app['name']} exited with {status} (see {log})")
                result = application_energy(sampler.samples, a, b, tail)
                runs.append(result)
                print(f"{app['name']} run {repeat + 1}: {result['wall_s']:.2f} s, {result['energy_j']:.1f} J "
                      f"(process {result['process_j']:.1f}, after exit {result['after_exit_j']:.1f} over {tail:.1f} s) "
                      f"(idle {result['pre_idle_w']:.1f} / {result['post_idle_w']:.1f} W, "
                      f"{result['temperature_c']:.0f} C, clock {result['min_clock_mhz']:.0f}-{result['max_clock_mhz']:.0f})",
                      flush=True)
                with open(os.path.join(args.out, f"{app['name']}.{repeat}.samples.csv"), "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["t_s", "power_w", "temperature_c", "sm_clock_mhz", "energy_counter_j"])
                    for s in sampler.between(a - PRE_S, b + tail):
                        writer.writerow([f"{s[0] - a:.4f}", s[1], s[2], s[3], s[4]])
            results[app["name"]] = {"command": app["command"], "runs": runs}
            with open(path, "w") as file:
                json.dump({"device": sampler.name, "pci_bus_id": context.pci_bus_id, "apps": results}, file, indent=1)
    finally:
        sampler.stop()
    print(f"wrote {path}")


# ----------------------------------------------------------------------------------------------------------------------
# probes

def probes(args):
    build = os.path.abspath(args.build)
    program = os.path.join(build, "app_probes")
    scratch = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.out)), "app_probes_work"))
    os.makedirs(scratch, exist_ok=True)
    context = HeldContext(args.device)
    sampler = Sampler(context.pci_bus_id)
    env = app_environment(args.device, scratch)
    rows = []
    if args.reuse:
        with open(args.reuse) as file:
            rows = [r for r in json.load(file)["probes"] if r["probe"] in ("gap", "copy")]

    def steady(command):
        # After args.idle seconds the GPU is in its deep idle (if args.idle outlasts the idle timer and NVML's averaging).
        time.sleep(args.idle)
        idle = mean_power(sampler.samples, time.monotonic() - 2.0, time.monotonic())
        log = os.path.join(scratch, "probe.log")
        a, b, status = run_command(command, scratch, env, log)
        if status != 0:
            raise SystemExit(f"{' '.join(command)} exited with {status}")
        with open(log) as file:
            line = next(l for l in file if l.startswith("result "))
        values = dict(part.split("=", 1) for part in line.split()[1:])
        # The steady part: the second half of the run (NVML averages over about 1 s).
        power = mean_power(sampler.samples, a + (b - a) / 2, b)
        temperature = statistics.fmean(s[2] for s in sampler.between(a + (b - a) / 2, b))
        return dict(values, power_w=power, deep_idle_w=idle, temperature_c=temperature)

    def context_probe():
        # From the deep idle: a process creates a context, waits CONTEXT_WAIT_S without work, runs one sleep kernel, waits
        # CONTEXT_AFTER_S and exits. The awake time without work after the context's creation (the idle timer) and after
        # the exit, each as the energy above the deep idle divided by the awake idle power above it.
        time.sleep(CONTEXT_SETTLE_S)
        deep = mean_power(sampler.samples, time.monotonic() - 3.0, time.monotonic())
        log = os.path.join(scratch, "probe.log")
        a, b, status = run_command([program, "context", str(CONTEXT_WAIT_S), str(CONTEXT_AFTER_S)], scratch, env, log)
        if status != 0:
            raise SystemExit("app_probes context failed")
        time.sleep(CONTEXT_TAIL_S)
        with open(log) as file:
            line = next(l for l in file if l.startswith("result "))
        values = {k: float(v) for k, v in (part.split("=", 1) for part in line.split()[2:])}
        awake = mean_power(sampler.samples, a + values["finished_s"] + 1.5, b)
        launched = a + values["launched_s"]
        # The wake-up: the process starts at a and its context wakes the GPU a little later (CUDA's start-up); NVML
        # shows it AVERAGING_DELAY_S late. The time at the deep idle that makes up the missing energy of the first 5 s.
        early = integrate(sampler.samples, a + AVERAGING_DELAY_S, a + AVERAGING_DELAY_S + 5.0)
        wake = 5.0 - (early - deep * 5.0) / (awake - deep)
        timer = (integrate(sampler.samples, a, launched - 0.5) - deep * (launched - 0.5 - a)) / (awake - deep)
        after = (integrate(sampler.samples, b, b + CONTEXT_TAIL_S) - deep * CONTEXT_TAIL_S) / (awake - deep)
        temperature = statistics.fmean(s[2] for s in sampler.between(a, b))
        return {"probe": "context", "deep_idle_w": deep, "awake_idle_w": awake, "idle_timer_s": timer, "wake_s": wake,
                "exit_delay_s": after, "temperature_c": temperature, **values, "wall_s": b - a}

    try:
        row = context_probe()
        print(f"context: awake {row['awake_idle_w']:.1f} W, deep {row['deep_idle_w']:.1f} W, wakes {row['wake_s']:.2f} s "
              f"after the start, idle timer {row['idle_timer_s']:.2f} s, awake after exit {row['exit_delay_s']:.2f} s",
              flush=True)
        rows.append(row)
        for gap in ([] if args.reuse else GAPS_US):
            # At least six gaps in the steady half of the run.
            seconds = max(PROBE_S, 12 * (gap + GAP_KERNEL_US) * 1e-6)
            row = steady([program, "gap", str(GAP_KERNEL_US), str(gap), str(seconds)])
            rows.append(row)
            print(f"gap {gap:>8} us: {row['power_w']:.2f} W, {int(row['launches'])} launches, kernel "
                  f"{float(row['kernel_s']):.3f} of {float(row['wall_s']):.3f} s, deep idle before {row['deep_idle_w']:.1f} W",
                  flush=True)
        for kind, memory in ([] if args.reuse else COPIES):
            row = steady([program, "copy", kind, memory, str(COPY_MB), str(PROBE_S)])
            rows.append(row)
            print(f"copy {kind} {memory}: {row['power_w']:.2f} W, "
                  f"{int(row['bytes']) * int(row['copies']) / float(row['copy_s']) / 1e9:.1f} GB/s", flush=True)
    finally:
        sampler.stop()
    result = fit_app_model(rows, sampler.name)
    with open(args.out, "w") as file:
        json.dump(result, file, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "probes"}, indent=1))
    print(f"wrote {args.out}")


def idle_power(model, temperature_c, slope_w_per_c):
    """Awake idle and sleep-kernel power at a temperature (moved by the kernel model's baseline slope), deep idle as
    measured."""
    shift = slope_w_per_c * (temperature_c - model["temperature_c"])
    return model["awake_idle_w"] + shift, model["sleep_kernels_w"] + shift, model["deep_idle_w"]


def gap_energy(gap_s, model, temperature_c, slope_w_per_c, awake_s=None):
    """Energy of the GPU over an idle gap of gap_s after work, J: from the awake power it falls to the awake idle power
    with time constant fall_s and stays there for awake_s (default: the idle timer), then the deep idle power."""
    awake_idle, sleep, deep = idle_power(model, temperature_c, slope_w_per_c)
    fall = model["gap"]["fall_s"]
    awake_part = min(gap_s, model["gap"]["idle_timer_s"] if awake_s is None else awake_s)
    energy = awake_idle * awake_part + (sleep - awake_idle) * fall * -math.expm1(-awake_part / fall)
    return energy + deep * max(gap_s - awake_part, 0.0)


def fit_app_model(rows, device):
    """The application terms from the probe rows. Gaps: the power over a gap is the cycle's energy minus the sleep
    kernels' (at the back-to-back power) over the gap's time; the awake idle power is the median over gaps of 10 ms to
    1 s, the fall time is fitted to the shorter gaps (scan, least squares); inside a process gaps of up to 6 s stay
    awake. The idle timer and the awake time after the exit come from the context probe. Copies: the mean power while
    copying back to back."""
    gaps = [r for r in rows if r["probe"] == "gap"]
    back_to_back = next(r for r in gaps if float(r["gap_us"]) == 0)
    sleep_w = back_to_back["power_w"]
    temperature = statistics.fmean(r["temperature_c"] for r in gaps)
    deep_w = statistics.median(r["deep_idle_w"] for r in rows)
    points = []
    for r in gaps:
        launches, kernel_s, wall_s = int(r["launches"]), float(r["kernel_s"]), float(r["wall_s"])
        gap_s = (wall_s - kernel_s) / launches
        if float(r["gap_us"]) > 0:
            points.append((gap_s, (r["power_w"] * wall_s - sleep_w * kernel_s) / launches))
    awake_w = statistics.median(e / g for g, e in points if 1e-2 <= g <= 1.0)

    def scan(low, high, error):
        best, x = None, low
        while x <= high:
            value = error(x)
            best = (value, x) if best is None or value < best[0] else best
            x *= 1.02
        return best[1]

    short = [(g, e) for g, e in points if g < 2e-3]
    fall = scan(1e-6, 2e-3, lambda f: sum(((awake_w * g + (sleep_w - awake_w) * f * -math.expm1(-g / f)) - e) ** 2
                                          for g, e in short))
    context = [r for r in rows if r["probe"] == "context"]
    timer = statistics.fmean(r["idle_timer_s"] for r in context) if context else math.inf
    exit_delay = statistics.fmean(r["exit_delay_s"] for r in context) if context else 0.0
    wake = statistics.fmean(r["wake_s"] for r in context if "wake_s" in r) if any("wake_s" in r for r in context) \
        else 0.0
    model = {"awake_idle_w": awake_w, "sleep_kernels_w": sleep_w, "deep_idle_w": deep_w, "temperature_c": temperature,
             "gap": {"fall_s": fall, "idle_timer_s": timer, "exit_delay_s": exit_delay, "wake_s": wake}}
    model["gap"]["points"] = [{"gap_s": g, "energy_j": e, "model_j": gap_energy(g, model, temperature, 0.0)}
                              for g, e in points]
    copies = {}
    for r in rows:
        if r["probe"] == "copy":
            copies[f"{r['kind']} {r['memory']}"] = {
                "power_w": r["power_w"], "gb_per_s": int(r["bytes"]) * int(r["copies"]) / float(r["copy_s"]) / 1e9,
                "temperature_c": r["temperature_c"]}
    return dict({"format": FORMAT, "version": 2, "device": device,
                 "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, **model, copies=copies, probes=rows)


# ----------------------------------------------------------------------------------------------------------------------
# profile

def profile(args):
    apps = load_apps(args.apps, os.path.abspath(args.build), args.only)
    model = energy_predictor.Model.load(args.model)
    metrics = available_metrics(model.counter_names() + EXTRA_METRICS)
    for app in apps:
        folder = os.path.abspath(os.path.join(args.out, app["name"]))
        work = os.path.join(folder, "work")
        os.makedirs(work, exist_ok=True)
        env = app_environment(args.device, work)
        command = output_command(app["command"], work)
        trace = os.path.join(folder, "trace")
        if not args.skip_trace:
            print(f"{app['name']}: nsys", flush=True)
            subprocess.run([tool("nsys"), "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                            "--force-overwrite", "true", "-o", trace] + command, cwd=work, env=env,
                           stdout=subprocess.DEVNULL, stderr=open(os.path.join(folder, "nsys.log"), "w"), check=True)
            subprocess.run([tool("nsys"), "stats", "--force-export=true", "--report", "cuda_gpu_trace", "--report",
                            "cuda_api_trace", "--format", "csv",
                            "--output", trace, trace + ".nsys-rep"], cwd=work, env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
        trace_ops = read_trace(trace + "_cuda_gpu_trace.csv")
        kernels = sum(1 for op in trace_ops if op[3] == "kernel")
        windows = launch_windows(kernels, app.get("windows"), app.get("window", 8))
        started = time.monotonic()
        for index, (skip, count) in enumerate(windows):
            print(f"{app['name']}: ncu, launches {skip + 1}-{skip + count} of {kernels}", flush=True)
            report = os.path.join(folder, f"counters_{index}")
            subprocess.run([tool("ncu"), "--clock-control", "none", "--cache-control", args.cache_control,
                            "--metrics", ",".join(metrics + OPCODE_METRICS),
                            "--launch-skip", str(skip), "--launch-count", str(count), "--export", report,
                            "--force-overwrite"] + command, cwd=work, env=env,
                           stdout=open(os.path.join(folder, f"ncu_{index}.log"), "w"), stderr=subprocess.STDOUT, check=True)
            with open(report + ".csv", "w") as out:
                subprocess.run([tool("ncu"), "--import", report + ".ncu-rep", "--csv", "--page", "raw", "--print-units",
                                "base", "--print-metric-instances", "details"], env=env, stdout=out, check=True)
        with open(os.path.join(folder, "windows.json"), "w") as file:
            json.dump({"launches": kernels, "windows": windows}, file)
        print(f"{app['name']}: done in {time.monotonic() - started:.0f} s", flush=True)


def launch_windows(launches, windows, window):
    """(skip, count) of the ncu runs: all launches in one run, or `windows` runs of `window` consecutive launches spread
    evenly over the application (the first and the last launches included)."""
    if not windows or windows * window >= launches:
        return [(0, launches)]
    starts = sorted({round(i * (launches - window) / (windows - 1)) for i in range(windows)})
    return [(start, window) for start in starts]


def available_metrics(names):
    """The metrics (base name before the roll-up) that this GPU's Nsight Compute knows; the others are dropped (aliases
    of other architectures)."""
    out = subprocess.run([tool("ncu"), "--query-metrics", "--chip", "ga102"], capture_output=True, text=True).stdout
    known = {line.split()[0] for line in out.splitlines() if line and not line.startswith(("-", "Device", "Metric"))}
    result = []
    for name in sorted(set(names)):
        base = name.rsplit(".", 1)[0] if name.endswith((".sum", ".avg", ".max", ".min")) else name
        if base in known:
            result.append(name)
    return result


# ----------------------------------------------------------------------------------------------------------------------
# score

def read_trace(path):
    """GPU operations from Nsight Systems' cuda_gpu_trace report, by start: (start s, duration s, name, kind, bytes)."""
    operations = []
    with open(path) as file:
        for row in csv.DictReader(file):
            name = row["Name"]
            start = float(row["Start (ns)"]) * 1e-9
            duration = float(row["Duration (ns)"]) * 1e-9
            if name.startswith("[CUDA memcpy"):
                kind = {"Host-to-Device": "h2d", "Device-to-Host": "d2h", "Device-to-Device": "d2d"}.get(
                    name[len("[CUDA memcpy "):-1], "d2d")
                host = row["SrcMemKd"] if kind == "h2d" else row["DstMemKd"]
                memory = "pinned" if host.lower().startswith("pinned") else "pageable"
                operations.append((start, duration, name, f"{kind} {memory}" if kind != "d2d" else "d2d pinned",
                                   float(row["Bytes (MB)"] or 0) * 1e6))
            elif name.startswith("[CUDA memset"):
                operations.append((start, duration, name, "memset", float(row["Bytes (MB)"] or 0) * 1e6))
            elif name.startswith("["):
                operations.append((start, duration, name, "other", 0.0))
            else:
                grid = "({}, {}, {})".format(*(row[k] for k in ("GrdX", "GrdY", "GrdZ")))
                operations.append((start, duration, name, "kernel", 0.0, grid))
    operations.sort()
    return operations


def parse_instances(text):
    """'total (A: 1; B: 2)' of --print-metric-instances details -> {A: 1.0, B: 2.0}."""
    text = text.strip()
    if "(" not in text:
        return {}
    inner = text[text.index("(") + 1:text.rindex(")")]
    result = {}
    for part in inner.split(";"):
        if ":" in part:
            key, value = part.rsplit(":", 1)
            result[key.strip()] = float(value)
    return result


def read_profiled(folder):
    """The profiled launches of an application (all windows), each with its launch number in the application (from 0,
    over all kernels)."""
    with open(os.path.join(folder, "windows.json")) as file:
        windows = json.load(file)["windows"]
    launches = []
    for index, (skip, _) in enumerate(windows):
        for j, launch in enumerate(read_counters(os.path.join(folder, f"counters_{index}.csv"))):
            launches.append(dict(launch, index=skip + j))
    return launches


def read_counters(path):
    """Nsight Compute rows by launch (in launch order): kernel name, counters (by metric name), opcode counts."""
    with open(path) as file:
        lines = [line for line in file if line.startswith('"')]
    rows = list(csv.reader(lines))
    header, data = rows[0], rows[2:]
    launches = []
    for row in data:
        record = dict(zip(header, row))
        counters = {}
        for key, value in record.items():
            if key.startswith("sass__"):
                continue
            number = counter_table.number(value)
            if not math.isnan(number):
                counters[key] = number
        launches.append({"name": record["Kernel Name"], "grid": record.get("Grid Size"), "block": record.get("Block Size"),
                         "counters": counters,
                         "opcodes": parse_instances(record.get("sass__inst_executed_per_opcode", "")),
                         "opcode_modifiers": parse_instances(record.get("sass__inst_executed_per_opcode_with_modifier_all",
                                                                        "")),
                         "opcode_threads": parse_instances(record.get("sass__thread_inst_executed_true_per_opcode", ""))})
    return launches


def short_name(name):
    return name.split("(")[0].strip()


def kernel_counters(trace, launches):
    """For every kernel operation of the trace, the counters to price it with and the profiled launch they come from:
    the profiled launch itself, or (launches between the profiled windows) the profiled launches of the same kernel
    before and after it, interpolated by the kernel's own launch number (the nearest one outside them). A profiled launch
    whose grid differs from the trace's launch of the same number is reported."""
    order = []  # (kernel, its own launch number) of every kernel operation, by the application's launch number
    seen = {}
    for op in trace:
        if op[3] == "kernel":
            name = short_name(op[2])
            seen[name] = seen.get(name, 0) + 1
            order.append((name, seen[name], op[5] if len(op) > 5 else None))
    by_name = {}
    mismatched = 0
    for launch in launches:
        if launch["index"] >= len(order):
            raise SystemExit(f"profiled launch {launch['index'] + 1}, but the trace has {len(order)} kernel launches")
        name, number, grid = order[launch["index"]]
        if name != short_name(launch["name"]):
            raise SystemExit(f"launch {launch['index'] + 1}: {launch['name']} profiled, {name} in the trace")
        if grid is not None and launch["grid"] and grid != launch["grid"]:
            mismatched += 1
        by_name.setdefault(name, []).append((number, launch))
    if mismatched:
        print(f"warning: {mismatched} profiled launches do not have the grid of the traced launch")
    result = []
    position = 0
    for op in trace:
        if op[3] != "kernel":
            result.append(None)
            continue
        name, number, _ = order[position]
        position += 1
        profiled = by_name.get(name)
        if not profiled:
            raise SystemExit(f"kernel {name} has no profiled launch")
        numbers = [n for n, _ in profiled]
        k = bisect.bisect_left(numbers, number)
        if k < len(numbers) and numbers[k] == number:
            result.append((profiled[k][1]["counters"], profiled[k][1]))
        elif k == 0 or k == len(numbers):
            nearest = profiled[0 if k == 0 else -1][1]
            result.append((nearest["counters"], nearest))
        else:
            (n0, a), (n1, b) = profiled[k - 1], profiled[k]
            weight = (number - n0) / (n1 - n0)
            counters = {key: (1 - weight) * value + weight * b["counters"].get(key, value)
                        for key, value in a["counters"].items()}
            result.append((counters, a if weight < 0.5 else b))
    return result


def first_api_call(path):
    """Start of the first CUDA API call in Nsight Systems' cuda_api_trace report, s (None without the report)."""
    if not os.path.exists(path):
        return None
    with open(path) as file:
        starts = [float(row["Start (ns)"]) * 1e-9 for row in csv.DictReader(file)]
    return min(starts) if starts else None


def busy_union(trace):
    """Time with at least one GPU operation running, s, and the idle gaps between operations."""
    busy = 0.0
    gaps = []
    end = None
    for start, duration, *_ in trace:
        if end is None or start >= end:
            if end is not None:
                gaps.append(start - end)
            busy += duration
            end = start + duration
        elif start + duration > end:
            busy += start + duration - end
            end = start + duration
    return busy, gaps


# Opcodes whose pipe price the probes calibrated on them (the fma pipe on FFMA and IMAD, fp64 on DFMA, xu on MUFU.RSQ):
# the opcode table's energies of these are checks of operands (three registers, constants), not corrections.
CALIBRATION_OPCODES = {"FFMA", "IMAD", "DFMA", "MUFU.RSQ", "HFMA2", "HMMA", "IMMA"}


def opcode_family(opcode):
    """The key an opcode table entry and a kernel's opcode are matched on: the opcode without modifiers, except MUFU
    (its function: MUFU.EX2)."""
    parts = opcode.split(".")
    return ".".join(parts[:2]) if parts[0] == "MUFU" else parts[0]


def opcode_correction_j(opcodes, table):
    """Energy the opcode table adds to a launch's pipe-priced energy: for every opcode it has (except the calibration
    opcodes), its warp instructions times its energy minus its pipe's price. `opcodes`: warp instructions per opcode
    with modifiers (Nsight Compute's sass__inst_executed_per_opcode_with_modifier_all)."""
    by_family = {opcode_family(op): entry for op, entry in table.items() if opcode_family(op) not in CALIBRATION_OPCODES}
    total = 0.0
    for opcode, count in opcodes.items():
        entry = by_family.get(opcode_family(opcode))
        if entry:
            total += count * (entry["pj"] - entry["pipe_pj"]) * 1e-12
    return total


# Instruction coverage (as Wattchmen reports it: the share of executed instructions its own benchmarks price). Opcodes a
# default probe runs as its measured instruction (the SASS of the probes' loops on sm_86, tools/check_sass.py), by family
# (opcode_family); with an opcode table, its opcodes count too.
PROBED_OPCODES = {"FFMA", "IMAD", "DFMA", "HFMA2", "HMMA", "IMMA", "MUFU.RSQ", "SHFL", "BAR", "LDG", "STG", "LDS", "STS",
                  "LDL", "STL", "LDC", "ATOMG", "ATOMS", "RED"}
# Opcodes of pipes no probe exercises: texture and surface instructions (KTT has no texture arguments) and the
# asynchronous tensor and copy instructions of Hopper and Blackwell (wgmma, tcgen05, TMA).
UNPROBED_OPCODES = {"TEX", "TLD", "TLD4", "TXD", "TXQ", "TMML", "SULD", "SUST", "SUATOM", "SURED", "HGMMA", "IGMMA",
                    "QGMMA", "BGMMA", "UBLKCP", "UBLKRED", "UBLKPF"}
UNPROBED_PREFIXES = ("UTC", "UTMA")


def instruction_coverage(opcodes, table=None):
    """Shares of executed warp instructions (`opcodes`: count per opcode, with or without modifiers): `own` priced by a
    probe of their own (PROBED_OPCODES, or an entry of the opcode table), `pipe` priced only through their pipe's event,
    `none` in no probed pipe (UNPROBED_OPCODES). Also the largest `pipe` and `none` opcodes. Empty without counts."""
    own_families = PROBED_OPCODES | {opcode_family(op) for op in (table or {})}
    shares = {"own": 0.0, "pipe": 0.0, "none": 0.0}
    by_family = {}
    for opcode, count in opcodes.items():
        family = opcode_family(opcode)
        by_family[family] = by_family.get(family, 0.0) + count
    total = sum(by_family.values())
    if total <= 0:
        return {}
    largest = {"pipe": [], "none": []}
    for family, count in by_family.items():
        kind = "own" if family in own_families else \
            "none" if family in UNPROBED_OPCODES or family.startswith(UNPROBED_PREFIXES) else "pipe"
        shares[kind] += count / total
        if kind in largest:
            largest[kind].append((count / total, family))
    return {**shares, **{f"largest_{kind}": [(f, share) for share, f in sorted(entries, reverse=True)[:3]]
                         for kind, entries in largest.items()}}


def nvml_view(segments, wall_s, deep_w, end_s, step_s=1e-3, phases=10):
    """The process energy of a predicted power timeline as the measurement takes it from NVML's reading
    (application_energy): segments (start s, duration s, power W) from the process start, deep_w where none is; the
    reading is the mean over NVML_WINDOW_S, updated every NVML_UPDATE_S and held, at `phases` evenly spread update
    phases; integrated from 0 to wall_s + AVERAGING_DELAY_S, minus the idle level over the delay. J (mean over phases)."""
    import numpy
    start = -2.0 * NVML_WINDOW_S
    times = numpy.arange(start, end_s, step_s)
    # The energy above deep_w up to every grid edge, exactly (segments shorter than a step keep their energy): the sum over
    # segments of (P - deep_w) x (overlap of [begin, begin + duration] with [start, edge]), from ramps at the segments'
    # ends: sum of w x max(edge - x, 0) over points x with weights w = +(P - deep_w) at begin, -(P - deep_w) at the end.
    positions = numpy.array([b for b, _, _ in segments] + [b + d for b, d, _ in segments], dtype=float)
    weights = numpy.array([w - deep_w for _, _, w in segments] + [deep_w - w for _, _, w in segments], dtype=float)
    order = numpy.argsort(positions)
    positions, weights = positions[order], weights[order]
    cumulative_w = numpy.concatenate([[0.0], numpy.cumsum(weights)])
    cumulative_wx = numpy.concatenate([[0.0], numpy.cumsum(weights * positions)])
    edges = numpy.concatenate([times, [times[-1] + step_s]])
    index = numpy.searchsorted(positions, edges, side="right")
    above = edges * cumulative_w[index] - cumulative_wx[index]
    energy = (edges - start) * deep_w + above
    width = int(round(NVML_WINDOW_S / step_s))
    mean = (energy[width:] - energy[:-width]) / NVML_WINDOW_S  # mean over the window ending at times[width - 1:]
    mean = numpy.concatenate([numpy.full(width - 1, float(deep_w)), mean[:times.size - width + 1]])
    hold = int(round(NVML_UPDATE_S / step_s))
    first = int(round(-start / step_s))
    last = int(round((wall_s + AVERAGING_DELAY_S - start) / step_s))
    results = []
    for k in range(phases):
        offset = k * hold // phases
        index = numpy.arange(times.size)
        shown = mean[numpy.maximum((index - offset) // hold * hold + offset, 0)]
        results.append(shown[first:last].sum() * step_s - deep_w * AVERAGING_DELAY_S)
    return float(numpy.mean(results))


def score_app(name, run, model, app_model, opcode_table=None, model_baseline=False):
    folder = os.path.join(run, name)
    trace = read_trace(os.path.join(folder, "trace_cuda_gpu_trace.csv"))
    launches = read_profiled(folder)
    with open(os.path.join(run, "measure.json")) as file:
        measured = json.load(file)["apps"][name]["runs"]
    wall = statistics.fmean(r["wall_s"] for r in measured)
    # The process's own energy (through the exit); the energy after the exit is scored apart.
    energy = statistics.fmean(r["process_j"] for r in measured)
    after_measured = statistics.fmean(r["after_exit_j"] for r in measured)
    temperature = statistics.fmean(r["temperature_c"] for r in measured)
    deep_measured = statistics.fmean(r["pre_idle_w"] for r in measured)
    slope = float(model.baseline["slope_w_per_c"])
    awake_idle_w, sleep_w, deep_w = idle_power(app_model, temperature, slope)
    # The application's first CUDA call (apps/device_select.cpp selects the device at the start of main) creates its
    # context and wakes the GPU.
    context_s = first_api_call(os.path.join(folder, "trace_cuda_api_trace.csv"))

    counters = kernel_counters(trace, launches)
    antepara = antepara_energies(model, deep_w)
    antepara_j = deep_w * wall
    opcode_j = 0.0
    parts = {"kernel baseline": 0.0, "kernel dynamic": 0.0, "copies": 0.0, "memset": 0.0}
    per_kernel = {}
    opcodes = {}
    executed = {}  # warp instructions per opcode (with modifiers) over all launches, sampled ones scaled
    op_j = []  # energy of every operation of the trace, for the timeline (nvml_view)
    for op, entry in zip(trace, counters):
        start, duration, op_name, kind, size = op[:5]
        before = sum(parts.values())
        if kind == "kernel":
            prediction = model.predict(entry[0], max(duration, 1e-9), temperature_c=temperature, strict=False)
            # The awake baseline as the application probes measured it in the state the applications run in (cooler and
            # with slower fans than the calibration run after its warm-up), unless model_baseline.
            baseline_j = prediction.baseline_j if model_baseline else sleep_w * duration
            parts["kernel baseline"] += baseline_j
            parts["kernel dynamic"] += prediction.dynamic_j
            antepara_j += antepara_kernel_j(entry[0], antepara)
            # A sampled launch takes the opcodes of the profiled launch it was matched with, scaled by its interpolated
            # instruction count.
            profiled = entry[1]
            scale = entry[0].get("smsp__inst_executed.sum", 0.0) / max(profiled["counters"].get("smsp__inst_executed.sum",
                                                                                                0.0), 1.0)
            for opcode, count in profiled["opcode_modifiers"].items():
                executed[opcode] = executed.get(opcode, 0.0) + scale * count
            if opcode_table:
                opcode_j += scale * opcode_correction_j(profiled["opcode_modifiers"], opcode_table)
            k = per_kernel.setdefault(short_name(op_name), {"launches": 0, "time_s": 0.0, "energy_j": 0.0,
                                                            "dynamic_j": 0.0})
            k["launches"] += 1
            k["time_s"] += duration
            k["energy_j"] += baseline_j + prediction.dynamic_j
            k["dynamic_j"] += prediction.dynamic_j
        elif kind == "memset" or kind == "other":
            parts["memset"] += model.baseline_w(temperature) * duration
        else:
            copy = app_model["copies"].get(kind)
            parts["copies"] += (copy["power_w"] if copy else model.baseline_w(temperature)) * duration
        op_j.append(sum(parts.values()) - before)
    for launch in launches:
        for opcode, count in launch["opcodes"].items():
            opcodes[opcode] = opcodes.get(opcode, 0.0) + count

    busy, gaps = busy_union(trace)
    first = trace[0][0]
    last = max(op[0] + op[1] for op in trace)
    # Times from the measured runs where they differ from the traced run: the host time after the last GPU operation is
    # the measured wall time minus the traced time from the context to the last operation.
    # The GPU wakes wake_s after the process starts (its context's creation); until then it is in its deep idle. The
    # traced run's time from the first CUDA call to the last operation is that of the measured run after the wake-up.
    wake = app_model["gap"].get("wake_s", 0.0)
    span = last - (context_s if context_s is not None else first)
    host_tail = max(wall - wake - span, 0.0)
    # Idle gaps: from the context's creation to the first operation, and between operations. Each stays awake for the
    # idle timer at most. After the last operation the GPU stays awake until the idle timer runs out or until
    # exit_delay_s after the exit, whichever is first; the part after the exit counts above the deep idle (the
    # measurement subtracts the deep idle after the exit).
    timer, exit_delay = app_model["gap"]["idle_timer_s"], app_model["gap"]["exit_delay_s"]
    idle_gaps = ([first - context_s] if context_s is not None and first > context_s else []) + gaps
    # Tracing slows the host: where the traced gaps do not fit into the measured wall time, they shrink in proportion.
    traced_idle = sum(idle_gaps)
    fits = max(wall - wake - busy, 0.0)
    if traced_idle > fits:
        idle_gaps = [g * fits / traced_idle for g in idle_gaps]
    awake_end = min(timer, host_tail + exit_delay)
    parts["idle (awake, deep)"] = sum(gap_energy(g, app_model, temperature, slope) for g in idle_gaps) \
        + gap_energy(host_tail, app_model, temperature, slope, awake_s=awake_end) + deep_w * min(wake, wall)
    deep_s = sum(max(g - timer, 0.0) for g in idle_gaps) + max(host_tail - awake_end, 0.0)
    after_predicted = (awake_idle_w - deep_w) * max(awake_end - host_tail, 0.0)
    predicted = sum(parts.values())
    # The same prediction as a power timeline from the process start, as NVML shows it (nvml_view): its process energy
    # carries the smearing of NVML's averaging across the exit, as the measured one does. Deep idle until the wake-up,
    # then the traced operations with the gaps between them (shrunk as above), the host time after the last one, and
    # after the exit the GPU awake until awake_end.
    shrink = min(fits / traced_idle, 1.0) if traced_idle > 0 else 1.0
    segments = []
    cursor = min(wake, wall)
    if context_s is not None and first > context_s:
        gap = (first - context_s) * shrink
        segments.append((cursor, gap, gap_energy(gap, app_model, temperature, slope) / gap if gap > 0 else 0.0))
        cursor += gap
    origin, end = cursor - first, None
    for op, energy_j in zip(trace, op_j):
        start, duration = op[0], op[1]
        if end is not None and start > end:
            gap = (start - end) * shrink
            if gap > 0:
                segments.append((end + origin, gap, gap_energy(gap, app_model, temperature, slope) / gap))
            origin -= (start - end) - gap
        segments.append((start + origin, duration, energy_j / duration if duration > 0 else 0.0))
        end = start + duration if end is None else max(end, start + duration)
    tail_start = end + origin
    if host_tail > 0:
        segments.append((tail_start, host_tail,
                         gap_energy(host_tail, app_model, temperature, slope, awake_s=awake_end) / host_tail))
    if awake_end > host_tail:
        segments.append((wall, awake_end - host_tail, awake_idle_w))
    nvml_process = nvml_view(segments, wall, deep_w, wall + max(awake_end - host_tail, 0.0) + 4.0)
    above_measured = energy - deep_measured * wall
    above_predicted = predicted - deep_w * wall
    return {"app": name, "wall_s": wall, "busy_s": busy, "kernel_s": sum(k["time_s"] for k in per_kernel.values()),
            "launches": sum(k["launches"] for k in per_kernel.values()), "profiled": len(launches),
            "measured_j": energy, "measured_sd_j": statistics.stdev(r["process_j"] for r in measured) if len(measured) > 1 else 0.0,
            "after_exit_measured_j": after_measured, "after_exit_predicted_j": after_predicted,
            "after_exit_sd_j": statistics.stdev(r["after_exit_j"] for r in measured) if len(measured) > 1 else 0.0,
            "total_error": (predicted + after_predicted) / (energy + after_measured) - 1,
            "predicted_j": predicted, "error": predicted / energy - 1,
            # As NVML shows it: the process energy of the timeline through NVML's averaging, and the rest after the exit.
            "nvml_predicted_j": nvml_process, "nvml_error": nvml_process / energy - 1,
            "nvml_after_exit_predicted_j": predicted + after_predicted - nvml_process,
            "above_idle_measured_j": above_measured, "above_idle_predicted_j": above_predicted,
            "above_idle_error": above_predicted / above_measured - 1 if above_measured > 0 else math.nan,
            "temperature_c": temperature, "deep_idle_w": deep_w, "deep_idle_measured_w": deep_measured,
            "traced_span_s": span, "host_tail_s": host_tail, "deep_s": deep_s, "parts": parts,
            "opcode_correction_j": opcode_j, "opcode_error": (predicted + opcode_j) / energy - 1,
            "opcode_above_idle_error": (above_predicted + opcode_j) / above_measured - 1 if above_measured > 0 else math.nan,
            # Antepara et al.'s form with this GPU's numbers: the idle power over the wall time plus bytes and FLOPs.
            "antepara_j": antepara_j, "antepara_error": antepara_j / energy - 1,
            "antepara_above_idle_error": (antepara_j - deep_w * wall) / above_measured - 1 if above_measured > 0 else math.nan,
            "kernels": per_kernel, "opcodes": opcodes, "coverage": instruction_coverage(executed, opcode_table)}


def antepara_energies(model, idle_w):
    """Energies per byte and per FLOP in the form of Antepara et al. (SC'25), from this model's own probes: above the idle
    power, at the full rate of one probe each (16-byte loads at L1, L2, DRAM; FFMA with three registers; DFMA), marginal
    per level (L2 over L1, DRAM over L2). J per byte or per FLOP."""
    probes = {p["label"]: p for p in model.data["probes"]}
    awake = model.baseline_w()

    def above_idle(label, per):
        p = probes[label]
        return ((awake - idle_w) * p["ns_per_access"] * 1e-9 + p["full_rate_pj_per_access"] * 1e-12) / per
    l1, l2, dram = (above_idle(f"{level} ld w16", 16.0) for level in ("L1", "L2", "DRAM"))
    return {"L1": l1, "L2": l2 - l1, "DRAM": dram - l2, "FP32": above_idle("compute ffma3", 2.0),
            "FP64": above_idle("compute dfma", 2.0)}


def antepara_kernel_j(counters, energies):
    """A kernel's energy above the idle power in Antepara et al.'s form: bytes at each level and FLOPs times their
    energies (counters of one launch)."""
    c = lambda name: max(counters.get(name, 0.0), 0.0)  # noqa: E731
    l1 = 32 * (c("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") + c("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
               + c("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum") + c("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"))
    l2 = 32 * (c("lts__t_sectors_srcunit_tex_op_read.sum") + c("lts__t_sectors_srcunit_tex_op_write.sum"))
    dram = 32 * (c("dram__sectors_read.sum") + c("dram__sectors_write.sum"))
    fp32 = 2 * c("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") \
        + c("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") + c("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
    fp64 = 2 * c("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum") \
        + c("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum") + c("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum")
    return energies["L1"] * l1 + energies["L2"] * l2 + energies["DRAM"] * dram + energies["FP32"] * fp32 \
        + energies["FP64"] * fp64


def score(args):
    model = energy_predictor.Model.load(args.model)
    with open(args.app_model) as file:
        app_model = json.load(file)
    with open(os.path.join(args.run, "measure.json")) as file:
        measured = json.load(file)["apps"]
    apps = {app["name"]: app for app in json.load(open(args.apps))["apps"]} if args.apps else {}
    opcode_table = None
    if args.opcodes:
        with open(args.opcodes) as file:
            opcode_table = json.load(file)["opcodes"]
    results = []
    for name in measured:
        if not os.path.exists(os.path.join(args.run, name, "windows.json")):
            print(f"{name}: not profiled, skipped")
            continue
        results.append(score_app(name, args.run, model, app_model, opcode_table, args.model_baseline))
    print(f"{'application':<16}{'wall s':>8}{'GPU busy':>10}{'process J':>12}{'sd':>6}{'predicted J':>13}{'error':>8}"
          f"{'above deep idle: meas.':>24}{'pred.':>9}{'error':>8}")
    for r in results:
        print(f"{r['app']:<16}{r['wall_s']:>8.2f}{100 * r['busy_s'] / r['wall_s']:>9.0f}%{r['measured_j']:>12.1f}"
              f"{r['measured_sd_j']:>6.1f}{r['predicted_j']:>13.1f}{100 * r['error']:>+7.1f}%"
              f"{r['above_idle_measured_j']:>24.1f}{r['above_idle_predicted_j']:>9.1f}{100 * r['above_idle_error']:>+7.1f}%")
    if results:
        print("after the exit (J; the GPU stays awake for a few seconds): measured (sd over runs) / predicted; whole run with it")
        for r in results:
            print(f"  {r['app']:<16}{r['after_exit_measured_j']:>8.1f} ({r['after_exit_sd_j']:.0f}) / "
                  f"{r['after_exit_predicted_j']:.1f}   whole run error {100 * r['total_error']:+.1f} %")
        print(f"mean |error|: process {100 * statistics.fmean(abs(r['error']) for r in results):.1f} %, above deep idle "
              f"{100 * statistics.fmean(abs(r['above_idle_error']) for r in results):.1f} %")
        print("as NVML shows it (the predicted timeline through NVML's averaging, which smears energy across the exit as in "
              "the measurement): process J, error; after the exit, measured / predicted")
        for r in results:
            print(f"  {r['app']:<16}{r['nvml_predicted_j']:>9.1f}{100 * r['nvml_error']:>+7.1f} %   "
                  f"{r['after_exit_measured_j']:>7.1f} / {r['nvml_after_exit_predicted_j']:.1f}")
        print(f"  mean |error|: process {100 * statistics.fmean(abs(r['nvml_error']) for r in results):.1f} %")
        if opcode_table:
            print("with the opcode table (opcodes priced apart from their pipe):")
            for r in results:
                print(f"  {r['app']:<16}{r['opcode_correction_j']:>+9.1f} J  error {100 * r['opcode_error']:+6.1f} %, above deep "
                      f"idle {100 * r['opcode_above_idle_error']:+6.1f} %")
            print(f"  mean |error|: whole {100 * statistics.fmean(abs(r['opcode_error']) for r in results):.1f} %, above "
                  f"deep idle {100 * statistics.fmean(abs(r['opcode_above_idle_error']) for r in results):.1f} %")
        if any(r["coverage"] for r in results):
            print("instruction coverage (share of executed warp instructions priced by a probe of their own"
                  + (" or the opcode table" if opcode_table else "") + ", only through their pipe, in no probed pipe):")
            for r in results:
                c = r["coverage"]
                if c:
                    print(f"  {r['app']:<16}own {100 * c['own']:5.1f} %  pipe {100 * c['pipe']:5.1f} %  none "
                          f"{100 * c['none']:4.1f} %   largest pipe-only: "
                          + ", ".join(f"{f} {100 * v:.1f} %" for f, v in c["largest_pipe"]))
        print("Antepara et al.'s form with this GPU's energies (idle power x wall time + bytes and FLOPs):")
        for r in results:
            print(f"  {r['app']:<16}{r['antepara_j']:>10.1f} J  error {100 * r['antepara_error']:+6.1f} %, above deep idle "
                  f"{100 * r['antepara_above_idle_error']:+6.1f} %")
        print(f"  mean |error|: whole {100 * statistics.fmean(abs(r['antepara_error']) for r in results):.1f} %, above "
              f"deep idle {100 * statistics.fmean(abs(r['antepara_above_idle_error']) for r in results):.1f} %")
        print("\nparts of the prediction (J):")
        names = list(results[0]["parts"])
        print(f"{'application':<16}" + "".join(f"{n:>20}" for n in names))
        for r in results:
            print(f"{r['app']:<16}" + "".join(f"{r['parts'][n]:>20.1f}" for n in names))
    if args.json:
        with open(args.json, "w") as file:
            json.dump(results, file, indent=1)
        print(f"wrote {args.json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probes", "measure", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--device", type=int, required=True, help="CUDA device (PCI bus order), as EnergyProbes --device")
        p.add_argument("--build", required=True, help="folder of apps/build_rodinia.sh")
        p.add_argument("--out", required=True)
        p.add_argument("--idle", type=float, default=7.0,
                       help="idle seconds before each run, longer than the awake time after an exit and NVML's averaging "
                            "(default 7)")
        if name == "probes":
            p.add_argument("--reuse", help="take the gap and copy probes from this earlier output; run the context probe only")
        if name != "probes":
            p.add_argument("--apps", required=True, help="application list (apps/rodinia.json)")
            p.add_argument("--only", type=lambda s: s.split(","), help="comma-separated application names")
        if name == "measure":
            p.add_argument("--repeats", type=int, default=3)
        if name == "profile":
            p.add_argument("--model", required=True, help="model file (its counters are collected)")
            p.add_argument("--skip-trace", action="store_true", help="keep an earlier Nsight Systems trace")
            # ncu flushes the caches before every profiled kernel by default (all): later launches of an iterative
            # application then read from DRAM what the real run finds in L2. KTT's counter passes, on which the model is
            # calibrated, do not flush.
            p.add_argument("--cache-control", choices=["none", "all"], default="none",
                           help="ncu's cache control (default none: caches as the previous launch left them)")
    p = sub.add_parser("score")
    p.add_argument("--model", required=True)
    p.add_argument("--app-model", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--apps", help="application list (for the sampling of each application)")
    p.add_argument("--opcodes", help="opcode table of tools/opcode_energy.py: also score with opcodes priced apart")
    p.add_argument("--model-baseline", action="store_true",
                   help="price the kernels' awake baseline with the model file's (default: the application probes' "
                        "sleep kernels, measured in the state the applications run in)")
    p.add_argument("--json", help="write the results here")
    args = parser.parse_args()
    {"probes": probes, "measure": measure, "profile": profile, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
