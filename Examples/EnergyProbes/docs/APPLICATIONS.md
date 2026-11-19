# Whole applications

The probes and the validation runs price single kernels, each launched back to back for seconds. Antepara et al.
(SC'25) validate their benchmark-driven model on whole applications (MILC, BerkeleyGW, LAMMPS, GPT-NeoX). A whole
application also copies data between the host and the GPU, leaves the GPU idle while the host works, and runs many
kernels once or a few times each. `tools/app_energy.py` (added 2026-11-17) measures such applications and predicts
their energy from the kernel model plus three terms of their own.

## What it measures

`app_energy.py measure` runs every application three times on the GPU and samples NVML's power, the reading the probes
are calibrated on. NVML averages it over about 1 s and updates it every 0.5 s on an RTX 3090; the instant power field
updates no faster. So the integral runs on after the application exits, until NVML shows the idle level from before
its start again. The idle level is then subtracted. This gives the energy whatever the averaging window is.

The integral is split at the exit, shifted by the averaging delay of 0.5 s:

- **process energy**: the application's own run, from its start to its exit;
- **after-exit energy**: what the GPU draws above its idle level after the exit (see below).

The harness holds a CUDA context on the GPU, as a node that keeps one. Before each run it waits until the GPU is in its
deep idle.

## The GPU without work (application probes)

`app_energy.py probes` runs `apps/AppProbes.cu` (`apps/build_rodinia.sh` builds it) and fits the application model
(`models/rtx3090_1200mhz.app.json`). Findings on the RTX 3090 at 1200 MHz:

- **Deep idle.** 41-42 W, with or without a CUDA context (the model file's `idle_w`).
- **Waking.** A process's first CUDA call creates its context and wakes the GPU, 0.53 s after the process starts
  (CUDA's start-up; the context probe).
- **Awake idle.** Awake without work, the GPU draws about 103 W (at 56 °C). Within 0.5 ms after a kernel, the power
  falls from the sleep-kernel level (106 W) to that awake idle level. It stays there until one of two things happens:
  - the **idle timer** runs out: 17.6-17.8 s without work (the context probe, two runs);
  - the process exits; the GPU then stays awake **about 3 s** more (3.0 and 3.4 s in two runs).

  Inside a running process, gaps of up to 6 s between kernels all stay awake. A context alone does not keep the GPU
  awake beyond the idle timer, so the harness's own context does not change what an application draws.
- **After the exit.** The awake time after the exit varies from run to run: 2.3 to 3.9 s in tests of the context probe,
  with no trend in the allocated memory (none, 1 GB, 8 GB). After lud (1 GB matrix) the power steps down through about 90 and
  76 W over 6 s, about 300 J instead of about 190 J. The after-exit energy is therefore scored apart from the process
  energy. The kernel model has no part in it.
- **Copies.** Host-to-device and device-to-host copies draw 110.8-111.9 W, pageable or pinned (10.6-13.2 GB/s), about
  5 W above the sleep-kernel level. Device-to-device copies draw 208 W at 418 GB/s.
- **The awake baseline is lower than the model's.** The applications run from a cool GPU with slow fans (56-58 °C).
  There the sleep kernels draw 106.0 W; the model's baseline gives 111.7 W at that temperature. The model was
  calibrated at 67 °C after a 120 s warm-up, and its temperature slope (0.29 W/K) does not carry the fans' state.
  `score` prices the kernels' baseline with the sleep kernels measured in the applications' state, as the validation
  runs use their own sleep baseline; `--model-baseline` uses the model's.

The model of an application's idle time follows from these:
- until its context wakes the GPU (0.53 s after the start), it is in its deep idle;
- from then to its first GPU operation, and between operations, the GPU is awake (for the idle timer at most);
- after the last operation it stays awake until the idle timer runs out or until 3 s after the exit, whichever comes
  first.

## The prediction

`app_energy.py profile` runs every application once under Nsight Systems, which gives every kernel and copy with its GPU
time and the first CUDA call. It then runs it once under Nsight Compute, which gives the model's counters per launch and
the warp instructions per SASS opcode. Applications with many launches are profiled in windows of
consecutive launches spread over the run, first and last included (the app list's `windows` and `window`; for
example 12 windows of 8 of gaussian's 8190 launches). The launches in between take the counters of the profiled ones
of the same kernel around them, interpolated by that kernel's launch number. The profiled launches' grid sizes are
checked against the trace's. Only the counters are sampled; every launch keeps its own traced time.

The sampling costs little. Leaving out one profiled window at a time and interpolating it from the others changes an
application's predicted dynamic energy by at most 0.6 % (gaussian, lud, hotspot, streamcluster; needle 3.7 % of its
1.3 J). The left-out launches themselves can be off by up to 30 % (gaussian) and 75 % (lud), always too high: their
work shrinks faster than linearly from column to column, but these late launches are small. Profiling 16 of
particlefilter's 76 launches instead of all of them changes its dynamic energy by -1.2 % (0.2 % of its process energy;
its launches differ by at most 1.2 % in instructions), and cuts the profile from 573 s to about 2 min.

`app_energy.py score` then predicts the process energy as the sum of three parts:

- every kernel launch priced by the model: the awake baseline over the launch's time plus its event energies, at the
  run's temperature;
- every copy at its probe's power;
- the idle time as above.

**As NVML shows it** (added 2026-11-18). The measured process energy is split from the after-exit energy at the exit
plus 0.5 s, half of NVML's averaging window. But NVML also holds each reading for 0.5 s, so on average it shows the power
0.75 s late, and the window smears a step in power over a second. When the power drops at the exit (lud: from about
226 W to about 105 W), part of the process's energy lands after the split. `score` therefore also lays the prediction
out as a power timeline from the process start (deep idle until the wake-up, the traced operations with their gaps,
the host time, the awake time after the exit), passes it through the same filter (`nvml_view`: the mean over 1 s,
updated every 0.5 s and held, averaged over the update phase) and integrates it over the window the measurement uses.
The total over process and after-exit is the same either way; only the split moves. The filter's two constants come
from NVML's behaviour (readings change every 0.5 s exactly; the context probe's 1 s ramp), not from the applications.

The measured wall time sets the host time after the last GPU operation. Where tracing made the host slower (the
traced gaps do not fit into the measured wall time: gaussian, with 8190 launches), the traced gaps shrink in
proportion.

The applications are Rodinia 3.1 CUDA programs (`apps/rodinia.json`, built unchanged from
[gpu-rodinia](https://github.com/yuhc/gpu-rodinia) with `apps/device_select.cpp` linked in). These are the ones that
make their own input or read generated files (`apps/make_inputs.py`), at sizes where the GPU part matters:

| Application | What it does | Launches | GPU busy |
|---|---|---|---|
| gaussian | Gaussian elimination, 4096 x 4096; two kernels per column | 8190 | 88 % |
| lud | LU decomposition, 16384 x 16384, shared-memory blocks | 3070 | 70 % |
| hotspot | thermal stencil, 2048 x 2048, 10000 steps | 5000 | 64 % |
| particlefilter | particle filter, 400000 particles, double precision | 76 | 97 % |
| lavaMD | molecular dynamics, 40^3 boxes, one long kernel | 1 | 79 % |
| streamcluster | online clustering, 262144 points; 10741 small copies | 1790 | 20 % |
| needle | Needleman-Wunsch, 16384; copy-heavy | 2047 | 30 % |
| pathfinder | dynamic programming, 400000 x 2000; host-heavy | 100 | 2 % |

None of these kernels was used to fit or calibrate the model, and nothing was tuned on them.

## Results (RTX 3090, 1200 MHz; run `Rodinia1nc`, model 2d90caf0)

Over three runs the process energy has a standard deviation of 0.3-3.5 % of its mean (5-11 J). Errors of the
prediction:

| Application | Process energy (J) | Model, as NVML shows it | Model, not filtered | With the opcode table (not filtered) | Antepara et al.'s form | After the exit, measured / predicted as NVML shows it (J) |
|---|---|---|---|---|---|---|
| gaussian | 494.7 | -3.0 % | +1.2 % | +1.4 % | -48.1 % | 206 / 226 |
| lud | 490.4 | +2.0 % | +10.8 % | +11.5 % | -24.1 % | 301 / 249 |
| hotspot | 1028.8 | +4.6 % | +6.3 % | +3.0 % | -44.5 % | 148 / 224 |
| particlefilter | 1481.6 | -0.6 % | +1.7 % | +2.2 % | -60.7 % | 187 / 241 |
| lavaMD | 549.3 | +0.2 % | +3.7 % | +2.2 % | -25.4 % | 196 / 226 |
| streamcluster | 1606.9 | +0.2 % | +1.2 % | +1.2 % | -52.5 % | 179 / 222 |
| needle | 239.3 | -4.5 % | +2.7 % | +2.7 % | -52.9 % | 143 / 223 |
| pathfinder | 1208.2 | -0.3 % | +1.1 % | +1.1 % | -57.2 % | 150 / 223 |
| mean of the absolute errors | | 1.9 % | 3.6 % | 3.2 % | 45.7 % | |

Above the deep idle (process energy minus the deep idle power over the wall time), the mean absolute error is 3.9 %
(3.1 % with the opcode table, 75 % for Antepara et al.'s form). With the after-exit energy, the whole run is predicted
within -5.3 to +18 % (needle's after-exit energy varies most between runs, sd 35 J).

What this shows:

- **The kernel model carries over to whole applications.** None of these kernels was used for the model. Compared as
  NVML shows it, every application is within -4.5 to +4.6 % (mean 1.9 %). Without the filter every one was
  over-predicted (+1.1 to +10.8 %): the end of a process is the part NVML shows late, and its energy was missing from
  the measured process energy.
- **lud (+2.0 %; +10.8 % not filtered)** was the largest miss, and the hunt for it found the filter. lud ends on its
  kernels at about 226 W, so more of its energy lands after the split than for any other application (see "lud at four
  sizes" below). Its data values are not the cause (tested 2026-11-18, run `LudData1`,
  `apps/lud_data.json`). Rodinia fills the matrix with 10·exp(-0.001·|i-j|), so neighbouring values differ only in low
  bits, and data with fewer toggling bits costs less (a DRAM load of zeros draws 22 % less). So lud read two matrices of
  the same size from files (`apps/make_inputs.py --lud 16384`): its own smooth matrix, bit for bit, and uniform random
  values with a large diagonal. The GPU phase (kernels, copies and the time after the exit) took 658 J (sd 5) with the
  smooth matrix and 665 J (sd 25; median 653) with the random one: the same within 1 %. The counters are the same too.
  With the smooth file the model is again 48 J high, as with `lud -s` (47 J). The file reading adds 54 s of host time
  with the GPU awake for its idle timer and then in its deep idle, as the idle model expects; the process energy then
  has errors of +1.5 and +3.4 %, diluted by that host time. Broadcast shared loads are not the cause either (tested the
  same day with the `broadcast` probes, see "Broadcast shared loads" in [PROBES.md](PROBES.md)): lud_internal reads
  each word for 2 lanes in 16 four-byte loads and 2 elements for 16 lanes each in 4 sixteen-byte loads per loop, and
  the model prices these two patterns within +1.0 and +0.6 %. See "lud's kernel measured on its own" below. Without the cache flush
  (Nsight Compute's default flushes before every profiled kernel), lud's counters hardly change: 9.9 % with the flush.
- **hotspot (+6.3 %)** mixes float data with double literals (`2.0 * t`). So it runs F2F conversions, DADD and a DFMA by a
  constant on the fp64 pipe, which the model prices at a random-operand DFMA. The opcode table prices them apart
  (+3.0 %); see "Opcodes within a pipe" in [PROBES.md](PROBES.md).
- **Host- and copy-heavy applications** (streamcluster, needle, pathfinder: 2-30 % GPU busy) are within 1.1-2.7 %. Here
  the idle states decide the energy. Without the wake-up delay they were 3.4-16 % high.
- **Antepara et al.'s form** (P = Pconst + sum of e x bandwidth + e x FLOP rate, with this GPU's energies per byte and
  per FLOP computed their way: above the idle power, at full rate, marginal per level; L1 1.7, L2 over L1 7.3, DRAM over
  L2 21.4 pJ/bit, FP32 9.4 and FP64 216 pJ/FLOP) misses by 24-61 %. Its energies per byte include the awake GPU's static
  power only at the probes' full rate. A kernel at a lower rate, a GPU awake between kernels and the host phases then
  draw only Pconst (the deep idle power). This is not their model on their GPUs (A100, GH200, at 1 s sampling); it is
  their form with our numbers. On the RTX 3090 the awake GPU's static power (about 60 W above deep idle) is too large
  to leave to per-byte energies.
- **The after-exit energy** (150-300 J per run, 10-40 % of the process energy) is predicted as the awake idle power for
  the measured 3.4 s; it varies from run to run and from application to application (lud's teardown keeps the GPU at
  76-90 W for seconds). No kernel counter describes it.

**Instruction coverage (2026-11-18, after Wattchmen).** `score` prints, per application, the share of executed warp
instructions (all launches, sampled ones scaled) whose opcode a probe of its own measured (`own`: FFMA, IMAD, DFMA,
HFMA2, HMMA, IMMA, MUFU.RSQ, SHFL, BAR and the loads, stores and atomics of the memory probes; with `--opcodes` also the
opcode table's entries), the share priced only through its pipe's event (`pipe`), and the share in a pipe no probe
exercises (`none`: texture and surface instructions, Hopper's and Blackwell's wgmma, tcgen05 and TMA). Wattchmen
reports the same figure (A100 70 -> 93 %, H100 66 -> 92 % with its bucketing).

| Application | own | own, with the opcode table | Largest pipe-only opcodes (with the table) |
|---|---|---|---|
| gaussian | 26.5 % | 34.2 % | S2R 15.4 %, ULDC 12.0 %, EXIT 9.4 % |
| lud | 71.9 % | 85.1 % | S2R 5.8 %, MOV 2.9 %, LEA 2.9 % |
| hotspot | 31.3 % | 56.7 % | ISETP 11.9 %, UIADD3 5.9 %, BRA 4.9 % |
| particlefilter | 22.5 % | 33.9 % | BRA 22.0 %, ISETP 11.0 %, MOV 11.0 % |
| lavaMD | 59.1 % | 86.9 % | BRA 3.2 %, LEA 2.3 %, BSYNC 2.1 % |
| streamcluster | 77.4 % | 95.3 % | BRA 1.4 %, ISETP 1.4 %, MOV 0.7 % |
| needle | 51.3 % | 64.8 % | IMNMX 9.4 %, BSYNC 9.2 %, BSSY 4.7 % |
| pathfinder | 43.0 % | 56.7 % | ISETP 16.2 %, PRMT 9.2 %, IMNMX 6.6 % |

No application runs an instruction of an unprobed pipe (`none` 0 % everywhere). The pipe-only rest is mostly loop,
index and control instructions (ISETP, BRA, S2R, MOV, LEA, uniform-datapath instructions). They are not unpriced: the
probes' loops run them too, so their pipes' prices were fitted with them, but no probe measures them alone. The share is
not an error estimate (gaussian, with the lowest coverage, is predicted within 3 %); on a new GPU a `none` share above
zero is the warning that matters. KTT's counter pass has no per-opcode counts, so `--predict` and `--tune` cannot print
it.

### lud's kernel measured on its own (2026-11-18)

lud spends 97 % of its kernel time in lud_internal. `kernels/LudParts.cu` has it unchanged and two parts of it, which
the program measures steadily like the validation kernels (`--validate diagnose`, run `LudParts1`; a diagnosis, nothing
fitted on it): part 1 without the read-modify-write of the matrix, part 2 without the inner loop (shared loads and
FFMA). At lud's first step (1023 x 1023 blocks, as the largest steps that take most of its time):

| Part | Time per launch | Measured power | Model, energy | Model, dynamic energy |
|---|---|---|---|---|
| lud_internal | 5.49 ms | 234 W | +4.7 % | +9.3 % |
| without the read-modify-write | 4.58 ms | 170 W | +4.7 % | +14.4 % |
| without the inner loop | 3.29 ms | 256 W | +4.8 % | +8.8 % |

- **No single part is mispriced.** The model is 6-14 % high on each part's dynamic energy: the inner loop (the
  difference of the full kernel and part 2) +11 %, the read-modify-write (full minus part 1) +6 %. That is within the
  model's usual scatter on held-out kernels.
- **The same holds at 65 k blocks** (offset 12288: +5.7 %, dynamic +11 %). The last steps (529 to 3969 blocks, 7-26 µs)
  are 20-40 % high, measured at 1-4 % kernel busy time; they are 0.02 % of lud's work.
- **Temperature is not it either.** lud_internal alternating with sleep kernels every 10 s from a cool GPU
  (`handoff/analysis_2026-11-18`): the power above the sleep kernels goes from 116 W at 61 °C to 118.5 W at 71 °C,
  0.2 W/K. lud runs 11 K cooler than the model was calibrated, which makes about 5 J.

The kernel's ordinary over-prediction (+9 % on a dynamic energy that is half of lud's process energy) makes about 22 J
of lud's 53 J. The rest is where NVML puts the energy, below.

### lud at four sizes (2026-11-18)

lud -s 4096, 8192, 12288 and 16384 (`apps/lud_sizes.json`, run `LudSizes1`): the work grows 64-fold, the error not
filtered was +1.8, +26, +40 and +45 J. An error of the kernel would grow with the work; this one appears once the run
ends on its kernels and then stays. Measured, the power after lud's exit falls from about 226 W over 1.5 s (NVML's
window plus its hold), and the measured after-exit energy was 115 J above the prediction for the two larger sizes,
while the whole run (process and after-exit) was -8 to -14 %. Through the NVML filter the process energy is predicted
at -13.5, -10.1, -3.2 and +0.4 %. The two short runs (0.43 and 0.79 s) end near the wake-up delay of 0.53 s, which the
application model takes from the context probe as one value; for them the wake-up decides the energy. lud's own
after-exit energy (about 320 J against 250 J predicted) stays outside the model: its teardown keeps the GPU at 76-107 W
for 7 s.

How these numbers came about. Three terms of the application model were found while looking at these applications, all
measured with probes and none fitted to the applications:
- the wake-up delay (context probe);
- the in-situ baseline (gap probe);
- the shrinking of traced gaps (from the measured wall time).

A fourth, the NVML filter of the prediction, was found while looking for lud's error; its constants are NVML's, not
fitted. The kernel model is unchanged (2d90caf0). Earlier states of the comparison, in order:
1. with the model's baseline and no wake-up delay: +3.4 to +19.7 %;
2. with the wake-up delay: +0.7 to +18.7 %;
3. with the in-situ baseline: +0.7 to +9.9 % (flushed counters).
4. no flush, not filtered: +1.1 to +10.8 % (mean 3.6 %); as NVML shows it: -4.5 to +4.6 % (mean 1.9 %).

Limits:
- one GPU at one locked clock;
- eight applications from one suite, in one size each (lud also at four sizes);
- the applications' input data is not random (for lud this was tested: no effect);
- the after-exit energy is outside the model.
- runs shorter than about 1 s depend on the wake-up delay, one value from the context probe (lud at 4096 and 8192:
  -14 and -10 %).

## Running it

```bash
apps/build_rodinia.sh ../../../third_party/rodinia_src/gpu-rodinia/cuda ../../../third_party/rodinia_build
python3 apps/make_inputs.py ../../../third_party/rodinia_build/inputs
B=../../../third_party/rodinia_build
python3 tools/app_energy.py probes  --device 1 --build $B --out my_gpu.app.json                     # 12 min
python3 tools/app_energy.py measure --device 1 --build $B --apps apps/rodinia.json --out Run        # 14 min
python3 tools/app_energy.py profile --device 1 --build $B --apps apps/rodinia.json --model models/rtx3090_1200mhz.model.json --out Run   # 15 min
python3 tools/app_energy.py score --model models/rtx3090_1200mhz.model.json --app-model my_gpu.app.json --run Run \
    --apps apps/rodinia.json --opcodes models/rtx3090_1200mhz.opcodes.json
```

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on machines with several GPUs. Nothing else may run on the GPU. `--only NAME,...`
picks applications. Nsight Systems needs a writable `TMPDIR` (the harness sets it to the run's work folder).
