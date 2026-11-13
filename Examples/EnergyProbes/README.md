# EnergyProbes (prototype)

Estimates the energy of one **access** (loads, stores, texture fetches and 32-bit atomics) served by L1, L2, DRAM,
shared memory and the constant caches, and of one **arithmetic operation** (FFMA, DFMA, IMAD, HFMA2, HMMA, MUFU) on
NVIDIA GPUs, using KTT's NVML power measurement. Cache sizes come from an [MT4G](https://github.com/caps-tum/mt4g)
result file so that the same program works on different GPUs. Optional studies follow published GPU energy models (see
[Studies](#studies-from-published-gpu-energy-models)): control vs datapath energy, energy per request vs per byte, the
power of awake and active SMs, and constant vs static power. The results are collected in an energy table (JSON).
`tools/energy_model.py` turns them into an energy model per CUPTI event and checks its predictions on kernels of other KTT
examples, which `--validate` measures (see [Validation on example kernels](#validation-on-example-kernels)).

This is a prototype. Treat the numbers as hypotheses until they pass the checks at the end of this file.

Version 2 changes after the first test run (an 80-SM Ampere card): the old baseline (a busy-polling kernel) drew more power
than the memory probes, so all energies came out negative, and the SM clock varied by 24 % between probes. The baseline is now
a sleeping kernel, the program can lock the clock itself, records why the GPU limited its clock, and withholds energies that
cannot be compared with the baseline.

## Build

KTT is built with [Premake](https://premake.github.io), which is not part of the repository. Download it once into the
`KTT` folder (KTT's `.gitignore` already ignores it). Use 5.0.0-beta1: newer release binaries need a newer C library than
many GPU servers have (5.0.0 needs glibc 2.34, beta5-beta8 need 2.38, beta2 needs 2.29; beta1 needs only 2.16; check yours
with `ldd --version`). Some archives ship the binary without the executable bit, hence the `chmod`:

```bash
cd KTT
curl -L -o premake.tar.gz https://github.com/premake/premake-core/releases/download/v5.0.0-beta1/premake-5.0.0-beta1-linux.tar.gz
tar -xzf premake.tar.gz && rm premake.tar.gz && chmod +x premake5
```

KTT finds CUDA through `CUDA_PATH` (without it, premake fails with "Compute API libraries were not found"). Typically
`/usr/local/cuda`; this command derives it from `nvcc`:

```bash
export CUDA_PATH=$(dirname $(dirname $(readlink -f $(which nvcc))))
```

The power path needs NVML and CUPTI. In this KTT version (2.3.1), `--power-usage` alone does not compile:
`CuptiKtt.h` only includes `cupti.h` when a profiling backend is enabled. Build with a profiling backend as well:

```bash
./premake5 gmake --power-usage --profiling=cupti --no-opencl
cd Build
make config=release_x86_64 -j$(nproc) EnergyProbesCuda
```

`make ... EnergyProbesCuda` builds the KTT library first, then the example; drop the target name to build all examples.
The example also links NVML (`nvidia-ml`) and the CUDA driver (`cuda`) directly, found through `CUDA_PATH` as well.
Use `--profiling=cupti-legacy` on Volta and older GPUs. Binaries end up in `Build/x86_64_Release`; the example finds
`libktt.so` next to it without `LD_LIBRARY_PATH`.

Tested with Premake 5.0.0-beta1 and 5.0.0 (the generated makefiles build KTT and this example). If `CUDA_PATH` was not
exported when you run `make`, the includes and library paths in the generated makefiles are empty, so keep the export in
the same shell. KTT needs GCC 9 or newer; on older systems load a newer compiler first (e.g. `module load gcc` on clusters).

## Run

Run MT4G once on the GPU (or use a file from `mt4g/sample_results`), then:

```bash
cd Build/x86_64_Release
./EnergyProbesCuda --mt4g path/to/<GPU_NAME>.json --dry-run                  # check the plan, no GPU needed
./EnergyProbesCuda --mt4g path/to/<GPU_NAME>.json --analyze Study --output Re  # analyse an earlier run again
sudo ./EnergyProbesCuda --mt4g path/to/<GPU_NAME>.json --lock-clock 1400     # measure with a locked SM clock
```

**Lock the clock.** Energy per operation depends on voltage, which follows the clock, so all probes must run at the same
clock. Pick a value below the lowest clock any probe reaches on its own (an unlocked run prints a suggestion). `--lock-clock`
needs root and resets the clock at exit; if the program is killed, reset it with `sudo nvidia-smi -rgc`. Alternatively lock
it yourself with `sudo nvidia-smi -lgc 1400,1400` and run without sudo. Without a locked clock, most energies are withheld
(see below); `--force-energy` computes them anyway and marks them in the notes.

The default run takes about 12-61 minutes: 3 x 20 s of idle power, a 120 s warm-up, then 59 configurations of 10-60 s each
(11 global probes and the two shared probes at 4 duties, plus 6 baselines and the spin reference), and 20 s of idle power at
the end. `--all` runs every study, 918 configurations (156-921 min). Options:

**How long a run really takes.** The printed range assumes 10-60 s per configuration. In practice KTT stops a
configuration once every power sample of its last 10 s is within 2 % of their mean. On an RTX 3090 that happened after
10.7 s on average, plus 0.3 s between configurations. The extra 0.7 s is the window sliding past the start-up lag of
NVML's 1 s average, which keeps that lag out of the result. So:
- `--all` takes about 2.8 h (918 x 11.0 s and 3 min of warm-up and idle);
- `--model` takes about 1.45 h (458 configurations, 86 min measured). It measures what `tools/energy_model.py` uses, and
  nothing else.

`--model` leaves out three things, which only the studies use:
- the zero payload (the model fits random data only);
- duty 10 % (the fits use duties of 25 % and up);
- the SM sweep's L1, L2, no-load, shared and FFMA probes. The sweep's baselines and dependent loads stay, on all SM
  counts, because `tools/fit_busy.py` measures the DRAM busy shape from them.

So the model's inputs are the same as with `--all`. A shorter window would not keep precision: a 5 s window moves the
dynamic power of rows 3-20 W above the baseline by 3 % (sd), as much as two whole runs differ. The time between
configurations is too small to be worth cutting.

**Checked on the RTX 3090.** One `--model` run (`Preset1`, 84 min) was compared with the four runs of the reference
model (`Full,Ffma3,Writes,Instr`):
- probe energies: 137 of 138 probes in both, median difference 2.3 %, mean −0.2 %. Earlier full runs differ from each
  other by a median of 1.9-4.7 %;
- scores on the six validation runs (reference in brackets): scatter 6.1 % (5.9 %), bias +3.5 % (+2.7 %), total MAPE
  3.2 % (3.0 %), max regret 2.4 % (2.4 %), pairs 2-5 % apart ordered correctly 83.6 % (83.8 %);
- DRAM busy shape (`tools/fit_busy.py`): 171 pJ/B + 15.9 W, scale 113 GB/s (sd 32), against 172 pJ/B + 13.4 W, 95 GB/s
  (sd 35).

These shifts are what one run gives against four. For example, its arithmetic probes came out 1.4 % higher, which
raises the bias and the compute-heavy mixed rows by 1-2 points. The largest probe differences (up to 27 %) are
bank-conflict probes only 1-12 W above the baseline; two earlier runs differ as much on them (68 against 96 pJ).
Three rows were withheld for a clock drop: two mixed rows at the 350 W power cap, as in earlier runs, and one
dependent-load row at a thermal limit. `check_run.py` reports this as a FAIL of the clock check, as it should.

**A sleep baseline after every probe group (since 2026-11-06).** Before, the baselines ran only at the start, after each
duty group and at the end, so two thirds of a `--model` run (54 min) had none in between. Its probes ran at 64-77 C,
its 5 sleeps at 64-67.4 C, and 223 of 408 rows were more than 2 C outside them. Now a sleep follows every group (9 more,
2 minutes). The GPU cools by only 1-2 C during a 10 s sleep, so each sleep runs at the temperature its group left
behind. On the RTX 3090 (`Preset2`):

| | `Preset1` (old placement) | `Preset2` (sleep after every group) |
|---|---|---|
| Sleeps | 5, 64.0-67.4 C | 14, 64.0-71.5 C |
| Slope of the baseline | 0.33 ± 0.10 W/C | 0.29 ± 0.02 W/C |
| Uncertainty of the baseline at the hottest row | ± 0.94 W | ± 0.13 W |
| Scatter / bias / total MAPE / max regret | 6.1 / +3.5 / 3.2 / 2.4 % | 5.9 / +2.8 / 3.0 / 2.4 % |

- The sleeps lie on a line (rms 0.13 W). A quadratic does not fit better (slope 0.29 at 65 C, 0.31 at 72 C), and the 5
  sleeps of the old placement predict the 9 new ones within ±0.2 W. So the old baseline was not wrong in this run; it
  was only uncertain.
- That uncertainty matters for the bias, not the ranking. Refitting `Preset1` with slopes from 0 to 0.78 W/C moves the
  bias from +6.4 to −0.6 % and the total MAPE from 3.9 to 2.9 %, but the scatter only from 6.3 to 6.0 %. ±0.10 W/C is
  about ±0.9 points of bias, ±0.02 W/C about ±0.2.
- The fan speed (now in the traces and the CSVs, `fan_pct`) follows the temperature (correlation 0.90 over the sleeps)
  and explains nothing beyond it (rms 0.12 against 0.13 W).
- `check_run.py` checks this as "sleeps cover the probes' temperatures": the slope's error times the distance from the
  sleeps to the farthest row must stay within 0.5 W.
- The slope differs between runs by more than its error: the validation runs' own sleeps give 0.15-0.36 W/C
  (`Validation9` 0.27 ± 0.02, `Validation11` 0.16 ± 0.03). It does not matter there: scoring `Preset2` on the seven
  validation runs with their own slopes, with 0.29 or with 0.20 W/C for all of them gives the same scatter (6.0 %), bias
  (+2.6 to +2.9 %) and regret (2.4 %). The validation runs already have a sleep after every kernel.
- One `--model` run with these baselines and the short counter run below score like the four-run reference model
  (5.9 %, +2.8 against +2.7 %, 3.0 %, 2.4 %). Two `--model` runs differ by a median of 2.0 % per probe (mean +0.9 %); the main memory
  families agree within about 1 %, and the noisiest are the bank-conflict and instruction probes (sd 9-10 %), whose 25 %
  rows are only 0.3-2.4 W above the baseline.

**The dependent loads before the DRAM-heavy groups (since 2026-11-06).** In `Preset1` and `Preset2` the 16-chain,
32-warp dependent-load row at the DRAM footprint (about 280 W) ran with the SM clock 28 % low, limited by `sw-thermal`,
and its energy was withheld (one of the model's probes). The GPU read only 72 C, and the GPU's own slowdown is at 95 C,
so the limit most likely came from the GDDR6X memory, whose temperature NVML does not report on GeForce cards ("N/A").
The latency group then ran after the locality, sub-sector, mixed and write groups, which load DRAM hardest. A retry on
a cool GPU (`Latency2`, `--latency --duty 100`, 10 min) ran all 31 rows at the locked clock: 762 pJ per access, against
760 and 779 in `Instr` and `LowRate`. With it the model has all 134 probes again (`--power Preset2,Latency2`) and
scores the same: scatter 6.0 %, bias +2.7 %, total MAPE 3.1 %, max regret 2.4 % over the seven validation runs. The
latency group now runs right after the shared probes, its place in `Instr` and `LowRate`. The two mixed rows at the
power cap are not retried: they draw more than the 350 W limit, at any temperature, and are not in the fit.

Checked with a third `--model` run in the new order (`Preset3`, 86 min): the 16-chain DRAM row ran at the locked clock
(763 pJ per access), and only the two mixed rows were withheld. Its baseline: 14 sleeps over 64.3-70.8 C, 0.275 ±
0.022 W/C. Over the three `--model` runs a probe's energy varies by a median of 1.8 % (CV; 90 % of the probes within
4.6 %). One run scores 6.0-6.2 % scatter, +2.6 to +3.3 % bias and 2.4 % max regret on the seven validation runs, and
the mean of all three runs scores 6.1 %, +3.2 % and 2.4 %. So measuring again does not sharpen the ranking: what is
left is the model, not the measurement.

**The counter run in 12 minutes.** The counts are a property of the code, not of the measurement window, so the counter
run needs only the probes the model uses, at duty 100, with short windows:

```bash
./EnergyProbesCuda $GPU --model --profile --duty 100 --min-time-ms 500 --max-time-ms 1000 --warmup-s 10 --idle-s 0 \
    --output ProfileModel1                                                               # ~12 min
```

On the RTX 3090 this took 12 minutes, against 25 for `--all --profile --duty 100 --min-time-ms 2000 --max-time-ms 4000`
(`ProfileFull4`). Its counts agree with `ProfileFull4` as closely as two full counter runs agree with each other, and the
model scores the same with either, to the last digit (scatter, bias, MAPE, regret and pairs ordered correctly).

| Option | Effect |
|---|---|
| `--lock-clock MHZ` | Lock the SM clock via NVML for the run (root) |
| `--clock-locked` | The clock is already locked (`nvidia-smi -lgc`): rows at another clock than the baseline are withheld |
| `--warmup-s N` | Heat the GPU for N seconds before measuring (default 120, 0 = off) |
| `--idle-s N` | Idle power without a kernel for N seconds: before KTT starts (no CUDA context), before the warm-up and at the end (default 20, 0 = off) |
| `--duty LIST` | Intensities in % of time issuing accesses (default 100,50,25,10), for the fit across access rates |
| `--burst-accesses N` | Accesses per thread between two sleeps (default 64, multiple of `--unroll`) |
| `--fit-min-duty N` | Fit power vs access rate only over duties of at least N % (default 25, the linear range) |
| `--payload LIST` | `random`, `zero` or `random,zero`: data of all probes (control vs datapath energy) |
| `--widths LIST` | Bytes per global load and thread, from 4,8,16 (default 16): energy per request vs per byte |
| `--compute` | Also run the arithmetic probes: FFMA (two and three register operands), DFMA, IMAD, HFMA2, HMMA, MUFU, and the instruction classes `shfl` (warp shuffle), `bar` (block barrier), `ffma-loop` (4 FFMAs per loop iteration: the branch), `fmul-fadd` and `mufu-sin` (the last two only check the model) |
| `--atomics` | Also run 32-bit atomic adds (`red`, `atom`) at the L2 and DRAM footprints, the 4-byte loads, stores and no-load loop they are compared with, and `red.shared` |
| `--constant` | Also run constant-memory loads at the constant L1 and L1.5 footprints |
| `--texture` | Also run texture fetches (16-byte texels) at the L1, L2 and DRAM footprints |
| `--dram-locality` | Also run 16-byte loads and stores that stream through the DRAM footprint or place runs of 32 B (one sector) to 8 KiB apart, with an L2 control |
| `--sub-sector` | Also run 4- and 8-byte loads and stores in which every thread walks a sector of its own (a warp instruction accesses one word of 32 sectors), with the sectors of a warp adjacent or scattered |
| `--indexing` | Also run 32 FFMA accumulators with static and with runtime register-array indices (the compares and guarded moves of the Coulomb example) |
| `--bank-conflicts` | Also run 4-, 8- and 16-byte `ld.shared` and `st.shared` whose lanes fall on fewer banks, so that an instruction takes 1, 2, 4, ... times the wavefronts of the conflict-free access, up to 32 (needs `--shared-kib` 4 or more) |
| `--mixed` | Also run 16-byte loads with 0 to 1024 FFMAs per load at the L1, L2 and DRAM footprints, from memory- to compute-bound (as Mixbench), at duty 100 only; with `--compute`, the study compares them with the sum of the load and FFMA probes |
| `--sm-sweep LIST` | Also run L1, L2, no-load, shared (and FFMA) probes and a baseline on only N SMs, for each N; with `--latency` also the dependent loads with 1 and 16 chains at L2 and DRAM (steady traffic at low rates) |
| `--writes` | Also run loads and stores together, which no other probe does: `copy` (a read and a write stream), `update` (store back where loaded, so that L2 evicts dirty lines) at the L2 and DRAM footprints, and `local` (a local-memory frame per thread, as spills) at the L1 footprint and with 1 KiB per thread; 4 and 16 bytes, random data |
| `--latency` | Also run chains of dependent 4-byte loads (each load's address depends on the data of the previous one) at the L2 and DRAM footprints, with 1, 4 or 16 chains per warp and 4 to 32 working warps per SM, at duty 100: what a load costs when latency, not bandwidth, limits |
| `--all` | `--compute --atomics --constant --texture --dram-locality --sub-sector --indexing --bank-conflicts --mixed --writes --latency --payload random,zero --widths 4,8,16` and an SM sweep over all SMs, halves, ..., 1 |
| `--model` | The probe families of `--all` on random data at duties 100/50/25, and the SM sweep with only its baselines and dependent loads: what the energy model uses (458 configurations, about 1.4 h on an RTX 3090) |
| `--sweep` | Energy vs footprint curve over all power-of-two footprints |
| `--fixed-order` | Run the global probes in a fixed order (default: random, decorrelates heating from probe order) |
| `--profile` | CUPTI counters per probe: hit rates (see validation) and the event counts of `tools/energy_model.py` |
| `--payload-bits LIST` | Also 16-byte loads at L1, L2 and DRAM, the no-load control, and FFMA, DFMA and HMMA (`probeOperands`, sm_80+ for HMMA) on payloads with controlled bits besides random and zero: `ones:Q`, `same:Q`, `flip:T`, `unique:K`, `lsb0:N`, `sorted` (`Payloads.h`); their bit statistics go to `PREFIX_payloads.json` (`tools/fit_payload.py`) |
| `--thermal-curve` | Instead of the probes, the sleep baseline's power over temperature (about 15 min): cooling after compute and after DRAM heating, and a staircase of short heat/sleep steps, traced (`tools/fit_thermal.py`) |
| `--counter-table FILE` | The counters to collect with `--profile` and `--validate` (default `tools/counters.json` of this example, which the Python tools read too) |
| `--validate LIST` | Instead of the probes, measure random configurations of KTT example kernels (`transpose`, `gemm`, `nbody`, `coulomb` or `all`; the held-out kernels `reduction`, `convolution`, `pnpoly` or `heldout`): power and time, then CUPTI counters. `mixed` adds the control kernel: the mixed-intensity probe, measured the same way (see [The control kernel](#validation-on-example-kernels)) |
| `--tune LIST --model-file FILE` | Instead of the probes, tune example kernels with the energy model: time and counters of random configurations (`--validate-configs`), no power measurement. `--objective energy\|edp\|ed2p\|time`, `--max-slowdown PCT`, `--contender-sd X`, `--lenient`. See [Using the model in a tuner](#using-the-model-in-a-tuner) |
| `--tune-measure contenders\|all` | With `--tune`: after the model's choice, measure the power of the choice, its contenders and the fastest configuration (or of all) with NVML, `--validate-ms` each, and take the measured best |
| `--quick` | The 5-minute run for `tools/check_model.py`: the default probes and `--compute` at duty 100, 5-10 s per configuration |
| `--predict FILE --model-file FILE` | The same choice from a `--validate` run's CSV, without a GPU |
| `--validate-configs N`, `--validate-ms N` | Configurations per example kernel (default 24) and measurement time per configuration (default 4000) |
| `--force-energy` | Compute energies even when the clock differs from the baseline |
| `--min-time-ms`, `--max-time-ms`, `--max-power-diff` | Measurement length and power stability per configuration |
| `--trace-ms N`, `--no-monitor` | NVML trace interval, or no direct NVML use at all |
| `--analyze PREFIX` | Analyse `PREFIX.csv` of an earlier run again with the current code (needs `--mt4g`, no GPU): baseline, energies, fits, summary and studies, written under `--output` |
| `--kernel-dir DIR` | Directory of the probe kernels (default `kernels/` of this example) |

Before measuring:

- Nothing else should run on the GPU (no display, no other jobs).
- Persistence mode helps: `sudo nvidia-smi -pm 1`.
- On multi-GPU machines set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. KTT reads power from NVML device number = CUDA device number,
  which is a different GPU when the two orders differ; the program warns when it detects this. `sudo` drops environment
  variables, so pass it explicitly: `sudo env CUDA_DEVICE_ORDER=PCI_BUS_ID ./EnergyProbesCuda ...`.
- Bare-metal Linux. NVML power readings are not reliable in WSL2 or many VMs.

## Code map

The program measures, then analyses. The analysis also runs on its own, on the CSV of an earlier run (`--analyze`), so a
change of the analysis can be checked on old measurements.

A probe run (`RunProbes` in `ProbeRun.cpp`) goes through these steps:

1. `GpuSession`: the idle reading before KTT creates its CUDA context, the tuner, the GPU's sizes (KTT and the MT4G file)
   and the footprints; the plan is printed (`DescribeSetup`), then the NVML monitor is opened (GPU check, clock lock).
2. `ProbeKernels`: the random payload, the arguments of all probes, one kernel definition per probe file, the texture
   objects, and what each probe family sets for a configuration (`ProbeSettings`).
3. The KTT kernels of the run, one per group of configurations (`CreateDutyKernels`, `CreateGroupKernels`,
   `CreateSweepKernels`): a kernel's tuning parameters are the variants of its probe. With `--profile`, the kernels are
   profiled with the CUPTI counters of `ModelCounters` (`CounterTable.cpp`, from `tools/counters.json`).
4. The measurement: idle reading, warm-up, then `MeasureProbes`, with sleep baselines between the groups. The launcher of
   every kernel records the kernel's own counts and the NVML trace; every configuration becomes a `Row`.
5. `ReportProbeRun`: the baseline model over temperature, the energies above it, the fits across duties, the tables, the
   summary and the studies, and the output files.

A validation run (`RunValidation` in `Validation.cpp`) uses the same `GpuSession` and baseline kernels, then measures the
tuning spaces of the example kernels in `ValidationKernels.cpp`.

| File | What it holds |
|---|---|
| `EnergyProbes.cpp` | `main`: the options, then a probe run, a validation run, `--dry-run` or `--analyze` |
| `Options.h/.cpp` | `Options` and the table of command-line options, from which parsing and the usage text come |
| `Common.h/.cpp` | Constants that must match the kernels, operation codes and names, formatting helpers |
| `Hardware.h/.cpp` | The GPU's sizes from KTT and MT4G, the footprint of each level, the level a footprint should hit |
| `NvmlMonitor.h/.cpp` | Direct NVML access: the GPU by PCI bus id, clock lock, trace of power, clocks and temperature |
| `GpuSession.h/.cpp` | What every run on the GPU needs: tuner, hardware, monitor, kernel launcher, idle readings, sleep baselines |
| `ProbeKernels.h/.cpp` | The probes as KTT kernels: sources, arguments, definitions, settings per configuration, the launcher, the row of a result, texture objects |
| `Payloads.h/.cpp` | The payloads of `--payload-bits`: patterns with controlled one bits and toggles, and their bit statistics (also of the operands `probeOperands` builds) |
| `CounterTable.h/.cpp` | The CUPTI counters to collect (`ModelCounters`) and the canonical name of an alias, from `tools/counters.json` (`--counter-table`) |
| `ProbePlan.h/.cpp` | What a run measures (global-probe plan, operations, SM sweep) and the setup it prints |
| `ProbeRun.h/.cpp` | The kernels of a probe run and the order they run in |
| `Rows.h/.cpp` | `Row` (one measured configuration) and its CSV format, written and read |
| `Analysis.h/.cpp` | Baseline model, energies, fits across duties, and the selection of the rows and fits a result comes from |
| `Report.h/.cpp` | The row and counter tables, the summary, and `ReportProbeRun` |
| `Studies.h/.cpp` | One function per study, and the energy table (JSON) |
| `Validation.h/.cpp`, `ValidationKernels.h/.cpp` | `--validate`: the example kernels' tuning spaces, the control kernel, the power pass and the counter pass |
| `kernels/Common.cuh` | What all probes share: the counting protocol, the duty mechanisms, SM slots, the access macros |
| `kernels/*.cu` | One probe family per file (`Baseline`, `Global`, `Shared`, `Compute`, `Constant`, `Texture`, `Scatter`, `Partial`, `Mixed`, `Write`, `Latency`, `Operands`), each compiled as a program of its own with `Common.cuh` in front (`KernelSource`) |
| `EnergyModel.h/.cpp` | The predictor for KTT programs: loads a model file, prices counters and a time, recommends a configuration (no KTT types in its interface) |
| `Tune.h/.cpp` | `--tune` (example kernels tuned with the model only) and `--predict` (the same from a validation CSV, without a GPU) |
| `models/` | Model files (`build_model.py`); `tests/`: the regression tests and their data |
| `tools/energy_events.py` | The events and how they are counted from CUPTI counters (standard library only), shared by the fit and the predictor |
| `tools/build_model.py`, `tools/energy_predictor.py`, `tools/check_model.py` | The model file from calibration runs; energy and recommendation from counters and time (also from a KTT output JSON); the 5-minute check of a model on a GPU |
| `tools/energy_model.py` | The energy model per CUPTI event, fitted over the probes and checked on validation runs and on the mixed-intensity probes |
| `tools/compare_runs.py` | Repeatability: every value of several runs with its coefficient of variation |
| `tools/check_sass.py` | SASS checks of every probe variant, compiled as the program compiles them |
| `tools/check_run.py` | The validation checklist as tests of a run's outputs (and its profiled run's counters): PASS, WARN or FAIL per check |
| `tools/counters.json`, `tools/counter_table.py` | The counter table, the one place for counter names: what each counter feeds or checks, its aliases on other chips, its alternative (fp16 on its own pipe or in the fma pipe), its optional group and the compute capability it needs. The program collects every counter in it, and the Python tools read it through `counter_table.py`; a counter that is not in it cannot be read |
| `tools/check_counters.py` | Which counters of the table a GPU lacks (`--chip` from Nsight Compute's metric list, or `--csv` of a profiled run): covered by an alias or alternative, not applicable on this compute capability (`--cc`, else from the chip or the run), optional, without an event, or missing, with the model events that would read 0 |
| `tools/fit_thermal.py` | The sleep baseline over temperature and fan speed from a `--thermal-curve` run: line, leakage form, temperature + fan, and the two cooling curves compared |
| `tools/fit_payload.py` | Energy over the data from `--payload-bits` runs: E = E0 + a x ones + b x toggles per probe, single-factor fits, held-out patterns, the no-load control |
| `tools/transfer_gpu.py` | A probe table transferred from a reference GPU with a subset of probes measured on the new one (as Wattchmen), the leave-out test by subset size, and the models' scores |
| `tools/compare_gpus.py` | A port's comparison with a reference GPU: plain probe energies, combined costs of the fit and the energy of every event |
| `tools/score_ranking.py` | How well the model ranks the configurations of each kernel over validation runs: scatter and bias of the error, regret, and pairs ordered correctly next to a ranking by time alone |
| `tools/fit_busy.py` | The fixed power of steady DRAM traffic from a run's dependent-load rows (`--latency`, with `--sm-sweep`): the shape that the model's DRAM busy event uses (`energy_model.py --busy-scale`) |

**Checking a change.** For the kernels, `python3 tools/check_sass.py --against <copy of kernels/ before the change>` must
report identical SASS for the variants that were measured, or the numbers no longer describe the code (the probes keep
their loops spelled out for this reason: helper structs changed the register allocation). For the analysis, `--analyze`
of earlier runs before and after the change shows every number that moved.

## What the probes do

All kernels run on every SM (`blocks per SM x block size`, default 4 x 256 threads) and are **time-bounded**: each thread
loops until `--launch-ms` (default 200 ms) has passed and counts its accesses. KTT repeats the launch (precise measurement)
until the NVML power reading is stable, then reports mean power and mean duration. The kernels add their counts to a buffer
that the launcher clears once per configuration, and count the launches, so accesses per launch are averaged over the same
launches as power and duration.

| Probe | Kernel | What it keeps busy |
|---|---|---|
| sleep (before/after) | `probeSleep` | Warps resident but sleeping (`nanosleep`): GPU awake at the same clock, almost nothing issued. **Baseline.** |
| spin | `probeSpin` | Every thread polls the timer in a tight loop. Heavy instruction issue, no memory traffic. Reference only. |
| `ld` + footprint | `probeGlobal`, `LOAD_OP=0` | Plain `ld.global.v4` (SASS `LDG.E.128`), the load normal code uses. Footprint selects L1 / L2 / DRAM. `WIDTH` 4 / 8 / 16 bytes gives `LDG.E` / `LDG.E.64` / `LDG.E.128`. |
| `ld.cg` | `LOAD_OP=1` | Bypasses L1 (`LDG.E.128.STRONG.GPU`): L2 hits at the L2 footprint, or DRAM. (A tiny footprint concentrates all traffic on a few L2 slices; the first run measured 10x lower bandwidth that way.) |
| `ld.ca` | `LOAD_OP=3` | Explicit L1 caching as in MT4G. Compiles to a *strong* SM-scoped load (`STRONG.SM`, `STRONG.CTA` on Volta), not the plain load. |
| `ld.nc` | `LOAD_OP=4` | Read-only data path (`LDG.E.128.CONSTANT`, MT4G's read-only cache), at the L1 and L2 footprints. |
| `st` | `LOAD_OP=5` | Plain `st.global.v4` (`STG.E.128`). L1 is write-through for global stores, so the L2 footprint measures L2 writes and the DRAM footprint writes that L2 evicts to DRAM. The stored words follow a 4-word XOR recurrence (one instruction per store, like the XOR that consumes a load) started from the payload. |
| `red` | `LOAD_OP=6`, `WIDTH=4` | `red.global.add.u32` (`RED.E.ADD.STRONG.GPU`), what `atomicAdd` with an unused result compiles to. Global atomics are performed in L2, so like stores they run at the L2 and DRAM footprints. Every thread adds to its own word (no contention); the operands follow the store's recurrence. Same SASS as the 4-byte `st` apart from the memory instruction. |
| `atom` | `LOAD_OP=7`, `WIDTH=4` | `atom.global.add.u32` (`ATOMG.E.ADD.STRONG.GPU`), which returns the old value (XORed like a loaded word). |
| `no-load` | `LOAD_OP=2` | Same loop and address arithmetic without the load. Instruction-overhead reference, one per width. |
| shared | `probeShared`, `SHARED_OP` | Conflict-free `ld.shared.v4` (`LDS.128`), `st.shared.v4` (`STS.128`) or, with `--atomics`, `red.shared.add.u32` (`ATOMS.ADD`, one word per thread, a warp covers each bank once). |
| bank conflicts | `probeSharedBanks`, `SHARED_OP` 0 / 1, `SHARED_WIDTH` 4 / 8 / 16, `BANK_WAYS` | `ld.shared` (`LDS`, `LDS.64`, `LDS.128`) or `st.shared` (`STS`, ...) of 4, 8 or 16 bytes per thread. A conflict-free access takes 1, 2 or 4 wavefronts (a whole, half or quarter warp each); lane l accesses element l x ways, so the lanes of a wavefront fall on 1 / ways of its banks and every wavefront is replayed `ways` times (up to 32 wavefronts). The degree is a kernel argument, so all degrees of a width run the same SASS; every iteration moves the threads by one element more than the block's window, so the data changes and the banks rotate. Rows are named `ld.shared-4way` etc., with the width as their W. |
| texture | `probeTexture`, `FOOTPRINT` | `tex.1d.v4.u32` (`TLD.SCR.LZ`) from a texture object over linear memory: the texture path of L1TEX instead of the load path. 4 x 32-bit integer texels (16 bytes, like the default load), point sampling. The footprints, stripes and slots are those of `probeGlobal`. |
| locality | `probeScatter`, `RUN_BYTES` | 16-byte `ld`/`st` where every 2 threads access one 32-byte sector, numbered by one counter for the whole GPU. `RUN_BYTES` = 0 streams through the footprint (coalesced like `probeGlobal`); otherwise runs of that many consecutive bytes are placed apart by an odd multiplier, from 32 B (every sector its own L2 request and DRAM row) to 8 KiB. The run length, multiplier and footprint are kernel arguments, so the SASS is identical for all of them. |
| sub-sector | `probePartial`, `SPREAD` | 4- or 8-byte `ld`/`st` (`WIDTH`) in which every thread walks a 32-byte sector of its own, one word per access, as a transpose writes or reads a column: a warp instruction accesses one word of 32 sectors, and L2 merges the partial writes of a sector before writing it to DRAM. `SPREAD` 0 puts the sectors of a warp next to each other (4 per L2 request), 1 scatters them (1 per request). Placement and footprint are kernel arguments. |
| constant | `probeConstant`, `CONST_BYTES` | `ld.const` (`LDC`): all threads of a warp read the same word, so one warp instruction is one constant-cache request. The footprint selects the constant L1 (half its MT4G size, 1 KiB on an RTX 3090) or the L1.5 (32 KiB, which misses the 2 KiB L1 on every access). Random data only. |
| indexing | `probeCompute`, `COMPUTE_OP` 7 / 8 | 32 FFMA accumulators updated in one loop, fully unrolled (static indices, FFMAs only) or unrolled 4 times as the Coulomb example's inner loop over `Z_ITERATIONS` (a runtime index: ptxas gathers and scatters every group of 4 through 64 guarded `IMAD.MOV`s, most with the guard false in every lane). One operation is one update of one thread. |
| mixed intensity | `probeMixed`, `MIX_FMAS` | The 16-byte `ld` loop of `probeGlobal` (same stripes, slots and timer checks; L1, L2 or DRAM footprint), and after every 16 loads a loop of groups of 64 FFMAs in 4 independent chains (x -> x*x + c), 0 to 1024 FFMAs per load: from memory-bound to compute-bound at every level. The FFMAs do not depend on the loads; the warps of an SM in their load phase and in their FFMA phase keep both units busy. The count is an argument (`MIX_FMAS` only names the configuration), so all intensities run the same SASS; the loop adds 3 instructions (`IADD3`, `ISETP`, `BRA`) per 64 FFMAs. One access is one load. Duty 100, random data. |
| writes | `probeWrite`, `WRITE_OP` | One access is one load and one store of WIDTH (4 or 16) bytes. `copy` loads from the first half of the `probeGlobal` stripe and stores to the second; `update` stores back to the element it loaded. Both run at the L2 and DRAM footprints, with the stripes and slots of `probeGlobal`. `local` loads and stores an element of a 1 KiB local-memory frame per thread (`ld.local`, `st.local`), walking FOOTPRINT bytes per SM: the L1 footprint, or the whole frame (1 MiB per SM, 82 MiB in all). A group's 4 loads go first, then its 4 stores to the same offsets, so a store does not wait for its own load; the stored words are the loaded ones XOR the stored-word recurrence. The walked part of the local frame comes through memory (as in `ffma3`), so every footprint runs the same SASS but one immediate. On the RTX 3090, an `update` store finds its line in L2 (0.05 write misses per warp access of 4 bytes), and a `local` store at the L1 footprint goes on to L2 (L1 is write-through for local memory too). Random data, at every duty. |
| arithmetic | `probeCompute`, `COMPUTE_OP` | 4 independent chains per thread of FFMA, DFMA, IMAD, HFMA2, HMMA (`mma.sync.m16n8k16`, f16 in, f32 accumulate; sm_80+) or MUFU (`MUFU.RSQ` plus the FFMA that feeds the chain), or the same loop without operations (`loop`, overhead reference). `ffma3` (`COMPUTE_OP` 9) is an FFMA with three different register operands, as the FMAs of real kernels (`acc = a*b + acc`): one chain is held in four registers that stay bit-identical (a = a*b + c, b = c*d + c, c = a*c + c, d = b*d + c, the constants in registers too), so the values follow `x*x + c` while no FFMA reads a register twice or a constant. The copies and constants pass through memory once before the loop (a volatile store and load), so that ptxas cannot see that they are equal; it still takes 39 of the 192 operands per loop from the operand reuse cache. |

Arithmetic operands stay bounded but keep changing, so the datapath toggles as with real data (as in Antepara et al.):
floating point follows `x -> x*x + c` with `c = -1.9` from a start in [1, 1.5), which is chaotic and stays in [-1.9, 1.71]
(with `c = -2` rounding collapses orbits onto the fixed point 2); integers follow the LCG `x -> x*a + c`; HMMA accumulates
`D += A*B` with two alternating sets of A and B; MUFU follows `x -> 1.5 - rsqrt(|x|)` (the absolute value is an operand
modifier, so one step is `MUFU.RSQ` and one `FFMA`; float32 orbits visit about 6000 values). One counted operation is one
step of one thread; a warp instruction
is 32 of them. The `loop` probe counts the same number of operations per iteration, so its energy per operation is the loop
overhead to subtract.

The arithmetic probes use another duty mechanism. A few warps saturate an arithmetic pipeline, so with each warp sleeping at
its own time (as in the memory probes) the other warps filled the pipeline: at duty 50, FFMA still ran at 85 % of its full
rate, DFMA and HMMA at 99 %. Instead, all warps of the GPU work in the same windows of `%globaltimer`: in every 524 µs
period they work for the duty's share and sleep until the next period.

Duty, payload and the number of working SMs are kernel arguments (`idleRatio`, `payloadMask` or the data buffer,
`activeSms`), not compile-time parameters, so the SASS does not change between them.

Footprint rules (MT4G sizes; conservative defaults otherwise): L1 = L1/4 but at least two rows of one load per thread per
stripe (32 KiB at the defaults), L2 = half of the L2 *segment* (one L2 partition on A100/H100, so hits stay in the near
partition), DRAM = at least 8 x L2. The payload is random so bits toggle like real data.

**No two blocks on one SM read the same lines.** When every block walked the whole footprint, a block trailing another one
hit L1 on the lines the leader had brought in, caught up and stayed behind it. These convoys turned about 3 of 4 accesses of
the L2 probe into L1 hits (7.6 TB/s and 14 pJ/B instead of 2 TB/s and 34 pJ/B on an RTX 3090). Now thread 0 of every block
claims a slot on its SM (`%smid` plus an atomic counter per SM), and the footprint is split into stripes:

- L1 and L2 footprints: the block in slot *s* of every SM walks stripe *s*. The blocks of one SM together still cover the
  whole footprint (per-SM working set = footprint), and all SMs share it in L2.
- DRAM footprints (from the size in the plan's `Stripes:` line): every block walks its own stripe, so no two blocks on the
  GPU share data and blocks on different SMs cannot convoy in L2 either. Only as many stripes as there are blocks are read
  (e.g. 164 MiB of 256 MiB); the plan prints the amount.

Each block reports its SM and slot. The program checks that every SM got exactly `--blocks-per-sm` blocks on distinct
stripes; otherwise the row gets a `placement:` note, and the summary has a `blocks per SM` line. With a `--blocks-per-sm`
that is not a power of two, one stripe per SM stays unused.

**Intensity (`--duty`).** Every thread issues `--burst-accesses` accesses (default 64), then sleeps once (`nanosleep`)
so that the burst took the duty's share of the time; at 100 % the sleep is 0. Fitting power above baseline against the
access rate over the duties of at least `--fit-min-duty` (25 %) separates the cost per access (slope) from power that does
not scale with the rate (`fixed [W]`). The sleep code costs the same instructions per access at every level and every
duty:

- A burst is a fixed number of accesses, not a minimum time. With a 4 µs minimum burst, DRAM ran 9.4 and L1 7.3 warp
  instructions per access at duty 50.
- The duty is a kernel argument, and the burst code also runs at 100 %. As a compile-time constant, `ptxas` removed the
  code at 100 % (6.50 instead of 7.08 instructions per access, so the 100 % row did not lie on the line of the others) and
  compiled a different division for every duty.

The SASS is identical for all duties, and CUPTI (`smsp__inst_executed`) counts the same instructions per access at every
level and duty. `nanosleep` sleeps between 0 and twice the requested time, so the achieved duty is approximate; accesses
are counted, so that does not matter. On an RTX 3090 the access rate at duty 10 is 8-15 % of the full rate.

**Constant memory.** The kernel reads a 64 KiB `__constant__` array (the most a module can have), which KTT fills from the
payload when it loads the module (`AddArgumentSymbol`). As in `probeGlobal`, warps of one SM must not read lines another
warp just brought in: warp *w* of the block in slot *s* walks stripe *s x (block size / 32) + w* of the footprint, in steps of 68 bytes, so
every access is in another 64-byte line and the stripe's words are visited in turn. The footprint is a kernel argument: as a
compile-time value, `ptxas` folded it into the masks and shifts of every access. The SASS (`UIADD3`, `ULOP3`, `IADD3`, `LDC`
per access) is identical for all footprints.

For all footprints the `probeGlobal` SASS is identical except for one setup instruction that holds the footprint size
(checked with NVRTC + `ptxas` for sm_70 to sm_90, for every `LOAD_OP` and `WIDTH`: `tools/check_sass.py`). Differences between footprints therefore
come from the memory system, not from different instructions. Whether a footprint is split over the whole grid is a kernel
argument, not a parameter: as a compile-time constant, `ptxas` folded it and the SASS of split footprints differed. The
16-byte load loop is the same as in the earlier, read-only version of this program (same instructions, other registers).

## Studies from published GPU energy models

Each study is an option (`--all` runs all of them). Its results are printed after the summary and written to
`<prefix>_energy_table.json`.

| Study | Option | Idea from | What it gives |
|---|---|---|---|
| Constant vs static power | `--idle-s` (on by default) | Wattchmen (ICS '26), Antepara et al. (SC '25) | Idle power before KTT creates its CUDA context (*constant*: the lowest power state), with the context open but no kernel (the driver keeps the memory at its full clock), and the sleep baseline (an awake GPU at the probe clocks). *Static* = sleep baseline - constant, split into the context's and the sleeping SMs' share. Energies per access are above all of it. |
| Control vs datapath | `--payload random,zero` | Antepara et al. (SC '25) | The same probes on zeros switch only control (fetch, decode, addresses, tags); the difference to random data is datapath energy. The no-load loop moves no data, so its datapath share must be about 0. |
| Per request vs per byte | `--widths 4,8,16` | Wattchmen (ICS '26) | Energy per access net of the no-load loop of the same width, fitted as `per request + width x per byte` for L1, L2 and DRAM. |
| Arithmetic | `--compute` | Antepara et al., Wattchmen, AccelWattch | pJ per operation, net of the loop, per warp instruction and per FLOP. |
| Atomics | `--atomics` | (not in Wattchmen or Antepara et al.) | pJ per 32-bit atomic add in L2 (with and without returned value) and in shared memory, and what it costs above the 4-byte store (`red`) or load (`atom`) at the same level. |
| Texture | `--texture` | GPUWattch (texture cache as a component) | pJ/B of texture fetches at L1, L2 and DRAM, and what the texture path costs above the plain load of the level. |
| Access granularity and DRAM locality | `--dram-locality` | (the Transpose validation error) | pJ/B of loads and stores at DRAM when every 1, 2, 4, 16, 64 or 256 sectors are placed apart, against a stream, and the same at the L2 footprint: energy per request vs per sector, and per row. |
| Bank conflicts | `--bank-conflicts` | (the wavefront event of the energy model) | pJ per 4-, 8- and 16-byte shared access with conflicts of up to 32 wavefronts, the energy of a replayed wavefront, compared with an extra full wavefront (the wider conflict-free accesses over the 4-byte one): whether a conflict costs energy or only time. |
| Register-array indexing | `--indexing` | (the Coulomb validation error) | pJ per FFMA update with static indices and with a runtime index (compares and guarded moves): what guarded moves cost, and that a lane whose guard is false costs almost nothing. |
| Mixed intensity | `--mixed` (with `--compute`) | Antepara et al. (SC '25), who check their model over Mixbench's arithmetic intensities | Loads of each level with 0 to 1024 FFMAs per load: bandwidth, FLOP rate and dynamic power, against the sum of the `ld` probe of the level at the load rate and the FFMA probe (net of its loop) at the FFMA rate. Whether the energies of memory and arithmetic add up where both units are busy. `tools/energy_model.py --mixed` predicts the same rows from their counters. |
| Constant memory | `--constant` | GPUWattch (constant cache as a component) | pJ per warp request to the constant L1 and to the L1.5 behind it; their difference is the energy of a constant L1 miss. |
| Working SMs | `--sm-sweep LIST` | AccelWattch (MICRO '21) | The sleep baseline and the L1, L2, no-load, shared (and FFMA) probes with only N SMs working (blocks on the other SMs return at once). Power vs N gives the power per awake SM and per working SM, and a chip-wide part that comes with the first SM. Probes are compared with the sleep baseline on the same SMs. |
| NVML readings | always | Wattchmen found NVML's integrated power within 1 % of its energy counter on V100/A100/H100 | The trace records `nvmlDeviceGetPowerUsage` (KTT's source, a 1 s average on Ampere and newer), the instant and 1 s average power fields and the energy counter; the summary compares them. |

Not done here: an external power meter (GPUWattch measured with current sensors at 2 MS/s) and a clock sweep (needs the
clock lock at several values; AccelWattch fits constant power over the clock). The joint fit of all probes (AccelWattch uses
constrained quadratic programming, Wattchmen a non-negative linear system) is in `tools/energy_model.py`; the energy table itself
keeps the direct differences, which need one probe per component but cannot fail to converge.

## What the numbers mean

For each probe:

```
E_dyn         = (P_probe - P_baseline(T)) * t       P_baseline = sleep runs, modelled over temperature
pJ per access = E_dyn / accesses                    one access = one load or store of WIDTH bytes by one thread
pJ per byte   = pJ per access / WIDTH               (16 by default)
pJ per op     = E_dyn / operations                  arithmetic: one instruction of one thread; per FLOP: / 2 for an FMA
```

- **Path energy, not component energy.** "L1" includes issuing the load (LSU, address path) plus the L1 hit.
  "L2" includes the L1 miss, the crossbar and the L2 hit. "DRAM" adds the memory controller, PHY and the DRAM device.
- **Incremental energies.** `L2 - L1` and `DRAM - L2` (printed in the summary) remove the shared issue cost because the
  instructions are identical. These are the most trustworthy numbers this prototype produces.
- `L1 - no-load` subtracts loop and address arithmetic. It is approximate: per access, the reference lacks the load and the
  XOR that consumes it (2 SASS instructions; `ptxas` folds the XOR because the stand-in values are copies of each other).
- **Withheld energies.** A probe's energy is only computed when its SM clock is within 2 % of the baseline's and its power
  is above the baseline. Otherwise the `note` column says why. Clock limiters reported by NVML (`sw-power-cap`,
  `hw-slowdown`, thermal) are also noted: they mean the GPU lowered the clock below what you asked for.
- The baseline contains static power. If a probe runs hotter than the sleep kernel, leakage rises and is attributed to the
  memory access. Watch the temperature column and the baseline drift warning.
- The spin row shows how much plain instruction issue costs. It is not subtracted from anything.
- Board power (NVML) includes memory, VRM losses and fans.

Outputs:

- `EnergyProbes.csv`: one row per configuration (width, payload, working SMs, power, the other NVML readings, clocks,
  temperature, clock limiters, accesses per launch, launches, energies, notes, counters).
- `EnergyProbes_fits.csv`: the fits across intensities, per probe, width and payload.
- `EnergyProbes_energy_table.json`: constant and static power, the fits, arithmetic energies, atomics, constant memory,
  control vs datapath, per request vs per byte and the SM sweep, for a model of application energy.
- `EnergyProbes_trace.csv`: NVML samples every 50 ms during each configuration (power, instant and 1 s average power,
  energy counter, SM/memory clock, temperature, fan speed, clock reasons). Plot power over time to see the reading settle, throttling,
  or heating.
- `EnergyProbes_<probe>.json`: raw KTT results.
- With `--validate`: `EnergyProbes_validation.csv` (one row per example configuration: time per launch on the GPU and on the
  host, busy fraction, power, baseline, dynamic and total energy per launch, counters per launch); `EnergyProbes.csv` then
  holds only the idle and sleep rows.

The `ctr [W]` column is the mean power from NVML's cumulative energy counter over the second half of each configuration,
measured independently of KTT. It should agree with KTT's `P [W]` within a few percent; the summary reports the largest
difference.

## Validation on example kernels

The probes give energies per access; whether they add up to the energy of real code is a separate question. `--validate`
measures kernels of other KTT examples, and `tools/energy_model.py` predicts them from the probes:

```bash
cd Build/x86_64_Release
GPU="--device 1 --clock-locked --mt4g ../../NVIDIA_GeForce_RTX_3090.json"
./EnergyProbesCuda $GPU --compute --widths 4,8,16 --duty 100,50,25 --output Model           # probe energies, ~20 min
./EnergyProbesCuda $GPU --atomics --constant --duty 100,50,25 --payload random,zero \
    --output Atomics                                                                     # atomics, constant, ~30 min
./EnergyProbesCuda $GPU --texture --duty 100,50,25 --payload random,zero --output Texture  # texture, ~22 min
./EnergyProbesCuda $GPU --dram-locality --duty 100,50,25 --output Locality                 # request size, ~21 min
./EnergyProbesCuda $GPU --sub-sector --duty 100,50,25 --output Partial                     # partial sectors, ~18 min
./EnergyProbesCuda $GPU --indexing --duty 100,50,25 --output Indexing                      # indexing, ~13 min
./EnergyProbesCuda $GPU --bank-conflicts --duty 100,50,25 --output Banks2                  # bank conflicts, ~37 min
./EnergyProbesCuda $GPU --compute --atomics --constant --texture --dram-locality --sub-sector --indexing \
    --widths 4,8,16 --duty 100 --profile --min-time-ms 2000 --max-time-ms 4000 --warmup-s 10 --idle-s 0 \
    --output Profile9                                                                    # CUPTI counts, ~6 min
./EnergyProbesCuda $GPU --validate all --output Validation3                                # example kernels, ~12 min
python3 ../../Examples/EnergyProbes/tools/energy_model.py --power Model,Atomics,Texture,Locality,Partial,Indexing \
    --counters Profile9 --validation Validation3 --csv Validation3_predictions.csv
```

With the bank-conflict probes (`Banks` was an earlier run with the 4-byte accesses only): the counts of the conflict
probes (and the bank-conflict counters) come from a profiled run of their own, and a validation run made after the
conflict counters were added to `ModelCounters()`:

```bash
./EnergyProbesCuda $GPU --bank-conflicts --atomics --profile --duty 100 --min-time-ms 1000 --max-time-ms 2000 \
    --warmup-s 0 --idle-s 0 --output ProfileBanks3                                       # conflict counts, ~5 min
./EnergyProbesCuda $GPU --validate all --output Validation4                                # example kernels, ~12 min
python3 ../../Examples/EnergyProbes/tools/energy_model.py \
    --power Study,Study2,Study3,Model,Atomics,Texture,Locality,Partial,Indexing,Banks,Banks2 \
    --counters Profile9,ProfileBanks3 --validation Validation4
```

`--power` takes several runs (a probe's energy is the mean over the runs that have it; the results below also average the
three `--all` runs `Study`, `Study2`, `Study3`), `--counters` several profiled runs (the first that has a probe is used).

| Kernel | Example | Size | Mostly |
|---|---|---|---|
| `transpose` | Transpose | 16384 x 16384 floats | DRAM reads and writes, shared-memory tiles in some configurations |
| `gemm` | ClTuneGemm (reduced parameter set) | 4096 x 4096 x 4096 floats | FFMA, shared memory, L1/L2 |
| `nbody` | Nbody (inner unroll 0-8) | 128 Ki bodies | FFMA and one `MUFU.RSQ` per interaction |
| `coulomb` | CoulombSum3d (no register-limit parameter) | 256^3 grid, 2048 atoms | FFMA and `MUFU.RSQ`, atom data from L1 |
| `reduction` (held out) | Reduction (single-pass variants with atomics) | 1 Gi floats (4 GiB) | DRAM streaming, shared-memory tree with a barrier per step, one atomic add per block |
| `convolution` (held out) | ClTuneConvolution | 16384 x 16384 floats, 7 x 7 filter | FFMA with tiles in shared memory (with or without halo threads) or through L1, vector loads |
| `pnpoly` (held out) | KernelTunerPnpoly | 20 M points, 600 vertices | Branches, compares and selects; the vertices in constant memory |

The held-out kernels came after the model was built: none of its design decisions (events, probes, fitting) looked at
them. `reduction` keeps only the example's single-pass variants (`USE_ATOMICS=1`), because without atomics the example's
launcher runs further passes. `pnpoly` fills the kernel's `__constant__` vertex array as a KTT symbol argument; the example
never fills it. `convolution` drops configurations that cannot launch (over 1024 threads or 48 KiB of shared memory).

The tuning spaces are the examples' (CUDA variants, in `ValidationKernels.cpp`, which reads the examples' kernel files);
outputs are not checked against references. Sizes are larger than in the examples where a launch would take only about a
millisecond.

How a configuration is measured:

- **Power pass.** One untimed launch (compilation, first touch of the data), then launches back to back for
  `--validate-ms` while the NVML trace runs. With `--power-usage`, KTT synchronises and queries NVML after every launch, so
  the GPU idles between launches (`busy` = GPU time / host time). The dynamic energy per launch is therefore
  `(P - P_baseline(T)) x host time per launch`, which assumes the gaps draw about baseline power, and the total is
  `P_baseline(T) x kernel time + dynamic`. P is `nvmlDeviceGetPowerUsage` over the second half, as for the probes.
- **Baseline.** The sleep kernel before the first example and after every one; its temperature model is the run's own.
- **Counter pass.** After all power measurements, the same configurations once more with CUPTI (`ModelCounters()` in
  `ProbeKernels.cpp`, the same counters as `--profile` collects for the probes). Profiling replays the kernel and raises
  power, so it never overlaps the power pass.
- **Control kernel** (`--validate all,mixed`). The mixed-intensity probe (16-byte loads at the L1, L2 and DRAM footprints
  with 0, 16, 128 or 1024 FFMAs per load, launches of 5 or 50 ms; 24 configurations) runs like an example kernel, through
  the same launcher, baseline and counter pass. A probe run with `--mixed` measured the same configurations the probes'
  way, and the probe's own access counts are read in the power pass (`accesses_per_launch` in the CSV). So the control
  separates the measurement from the model: `energy_model.py --mixed <probe run>` prints both energies per load of every
  configuration, and the control is predicted apart from the example kernels (not in their totals).

The model (`tools/energy_model.py`) has one energy per event: a warp instruction of each pipe (alu, fma incl. IMAD, fp16,
fp64, tensor, xu = MUFU, lsu, tex, other; alu, fma and fp16 counted in active lanes / 32, so that a lane that is inactive
or whose guard predicate is false costs nothing; lsu split into shared-memory instructions, global-path instructions
(global, generic and local loads, stores and atomics, LDGSTS) and the rest of the pipe, from the instruction counters
by memory space, which runs profiled before 2026-11-04 lack: there all lsu instructions are one event), a sector at L1 (global and local loads and stores; texture fetches
separately), L2 (reads, writes from L1, atomics with and without result) and DRAM (reads, writes), an L2 request (one
per line an instruction touches) and an L2 request that misses (one DRAM access), a shared-memory wavefront (loads,
stores, atomics) and a wavefront replayed for a bank conflict (loads, stores; the bank-conflict counters count exactly
these, N - 1 per warp instruction for an N-way conflict), and a constant-cache request and constant L1 miss. It fits them over the probes (energy per access
from the power runs, counts per access from the profiled run) by non-negative least squares on relative errors, then
predicts every validation configuration. DRAM writes are priced by their sectors, whether the store that dirtied the
line missed L2 or not; only write miss requests beyond one per four written-back sectors (partial writes) have an event
of their own (see "Write-backs without a write miss"). With `--weighted`, every probe's error is weighed 1 / sqrt(n), n the probes of its
family (a study: the plain loads and stores of a level, bank conflicts, locality, sub-sector, arithmetic, ...;
`family()`), so that the 30 bank-conflict probes do not outweigh the few plain loads of a level; that helps runs without
the lsu split (see "Shared and global memory instructions"). The probe runs and the kernels they predict must have the
same counters (the script stops otherwise). With `--fma-halves`, the fma pipe is two events, its heavy half (FP32 and
IMAD) and its lite half (FP32 only), from the counters collected since `ProfileFull3` (see "The fma pipe's two halves";
off by default). When the probes include `ffma3`, the two FFMA probes that compute `x*x + c`
(`ffma`, `ffma-static`) stay out of the fit: the FMAs of real kernels read three registers. A probe whose fit across duties is not a line (fixed power above half its
full-rate dynamic power: only the constant L1.5) enters with its full-rate energy, as in the program's summary.
Counters that a run did not collect count as 0, so the validation run must collect the same counters as the profiled
probe run. It reports, per kernel, the mean absolute percentage error of the total and of the dynamic energy, Pearson and
Spearman correlations, and the regret (how much more energy the configuration the model ranks best uses than the
measured best). Two references show what the counters add: a constant power over the kernel time, and the fit with only
instructions and DRAM sectors. With `--cross-validate` it also predicts every probe from a fit without it (leave-one-out);
probes with an event of their own (the only tensor, fp64 or MUFU probe) cannot be predicted that way and are listed apart.

**Results on an RTX 3090 at 1200 MHz** (2026-11-03; probe energies averaged over `Study`, `Study2`, `Study3`, `Model`,
`Atomics`, `Texture`, `Locality`, `Partial` and `Indexing`, counts from `Profile9`, 93 configurations of a third
validation run, `Validation3`, every one 100 % busy; the earlier ones did not count L2 miss requests or active lanes):

The fit reproduces the 69 probes other than the no-load loops with a mean error of 4.3 % (the 12 sub-sector probes within
-17 to +34 %, the others within -11 to +12 %; the three no-load loops within -12 to +19 %). Energies per event, in pJ per
warp instruction (of 32 active lanes for alu, fma and fp16), sector, wavefront or request:

| alu | fma (fp32, IMAD) | fp16 | fp64 | tensor | xu (MUFU) | lsu | other | L1 sector | shared ld / st wavefront |
|---|---|---|---|---|---|---|---|---|---|
| 210 | 258 | 331 | 2340 | 2341 | 333 | 563 | 54 | 125 | 367 / 291 |

| | L2 read | L2 write | DRAM read | DRAM write |
|---|---|---|---|---|
| per sector (a partial write counts as one) | 804 | 747 | 2331 | 1791 |
| per request (one line an instruction touches) | 544 | 773 | | |
| per request that misses L2 (one DRAM access) | | | 6628 | 8100 |

| texture fetch (instruction + 16 sectors) | L2 atom / red sector | shared atom wavefront | constant request | constant L1 miss (to L1.5) |
|---|---|---|---|---|
| 5167 | 2375 / 1493 | 456 | 350 | 1848 |

A whole line read from DRAM costs 6.6 + 4 x 2.3 = 16.0 nJ (4.0 nJ per sector, as when the model had sectors only); a
single sector costs 9.0 nJ, 2.2 times as much per byte. An atomic that misses L2 counts as one read and one write miss
request (it reads its line and later writes it back). In every texture probe a warp instruction fetches 16 sectors, so
the fit cannot split a fetch between the instruction and its sectors (depending on the counter set it put the whole cost
on one or the other); only their sum is determined, 5.2 nJ per 512-byte warp fetch, against 2.7 nJ for a load. Several
events trade energy among themselves as probes are added (lsu instructions 379-563 pJ, shared wavefronts 290-410 pJ),
so single values are less certain than the predictions they add up to.

| Kernel | Total energy: MAPE (bias) | Dynamic energy: MAPE (bias) | Spearman ρ (total) | Regret | Constant power: MAPE, regret |
|---|---|---|---|---|---|
| transpose (24) | 4.2 % (-3.8 %) | 8.4 % (-7.7 %) | 0.991 | 0 % | 7.6 %, 0 % |
| gemm (24) | 4.3 % (-2.1 %) | 7.9 % (-3.5 %) | 0.977 | 0 % | 12.3 %, 0 % |
| nbody (21) | 6.8 % (-6.3 %) | 13.6 % (-12 %) | 0.969 | 13.2 % | 9.7 %, 0 % |
| coulomb (24) | 3.7 % (-3.6 %) | 9.1 % (-8.9 %) | 0.997 | 1.4 % | 25.9 %, 1.4 % |
| all (93) | 4.7 % (-3.9 %) | 9.6 % (-7.9 %) | | | |

On the same 93 configurations, the model before the indexing probe (warp counts, probes up to `Partial`) had a total MAPE
of 7.1 %, all of the difference in Coulomb (12.8 %, dynamic 40 %); the kernels other than Coulomb have all lanes active
(Transpose 80-100 % of its alu lanes), so counting lanes changes nothing there. Further back, on the 88 configurations of
`Validation2`: 8.3 % without request events, 7.7 % with them. None of the four kernels uses atomics, constant memory or
textures.

- **Ranking.** In Transpose and GEMM the configuration the model ranks best is the measured best, in Coulomb within 1.4 %;
  in Nbody it picked one 13 % above the best, the one it underpredicts most (see below). Autotuning for energy needs the
  ranking more than the absolute values. At a locked clock energy follows time closely, so a constant power ranks almost as
  well (here even better for Nbody; on `Validation2` its regret was 8.0 % for GEMM and 4.3 % for Nbody, the model's 0); the
  counters matter for the energy itself where power differs between configurations (Coulomb 3.7 vs 26 %, GEMM 4.3 vs 12 %).
- **Request size, not row locality.** Transpose was underpredicted by up to 46 %, and the error followed the DRAM
  bandwidth (r = 0.88): its configurations that read or write one sector per L2 request (strided columns) run at 200-300
  GB/s and implied 270 pJ/B, twice the streaming value. The locality probes show the same: one sector per request costs
  2.1-2.3x per byte at DRAM and 200-260 GB/s, while runs of whole lines placed apart cost at most 18 % more than a stream
  (see Example results). Every earlier probe moved 4 sectors per request, so the model had put the cost of a request into
  its sectors. With request and miss-request events the Transpose configurations of `Validation2` are within -27 to +8 %
  (dynamic; 12 of 24 below -10 %, before up to -46 %).
- **Partial sectors.** Strided 4- and 8-byte writes, which L2 merges into partial sectors (3-6 L2 write sectors per DRAM
  write sector in Transpose), are what the sub-sector probes measure; the existing events describe them (a partial write
  is a sector and a request of its own), and adding the probes brought Transpose on `Validation2` to 5.5 % (dynamic 12 %,
  regret 0). Its worst configurations there (-23 to -29 % dynamic) read one sector per request with coalesced writes; the
  partial-write ones are at -13 to -21 %.
- **Guarded moves cost almost nothing when their guard is false.** The Coulomb configurations with `Z_ITERATIONS=32` and
  partial unrolling index their result array at runtime; `ptxas` turns every access into compares and guarded moves
  (`ISETP`, `@P MOV`, `@P IMAD.MOV`), most of which have the guard false in every lane: only 35-38 % of their fma lanes
  and about 50 % of their alu lanes are active. With instructions counted per warp, the model overpredicted them by up to
  2x. The indexing probe reproduces the pattern (21 warp instructions per update, 66 % of the lanes guarded off) and costs
  48.6 pJ per update against 9.9 pJ with static indices; counted in active lanes, the same pipe energies predict it within
  2.3 %, without a parameter of its own. Coulomb is now within 3.7 % (its indexed configurations within -16 to +2 %).
- **Bank conflicts: the measured price fits the conflicting configurations better, the total gets worse.** The example
  kernels do conflict: in `Validation4` (the first validation run with the conflict counters, 90 configurations) 81 % of
  Transpose's shared-load wavefronts are conflict replays (unpadded tiles), 22 % of GEMM's loads and 49 % of its stores,
  4 % of Nbody's loads. Before the conflict events, the model charged every replay as a full wavefront. The bank-conflict
  probes price a replayed wavefront at 62 pJ (loads) and 180 pJ (stores), against 500 / 417 pJ for a wavefront of a
  conflict-free access:

  | Model on `Validation4` | Total MAPE (bias) | Transpose | GEMM | Coulomb | GEMM: r(error, replay share) | lsu instruction |
  |---|---|---|---|---|---|---|
  | probes up to `Indexing`, replays priced as full wavefronts | 4.9 % (-3.7 %) | 5.8 % | 4.4 % | 4.1 % | +0.47 | 563 pJ |
  | the same probes, replays not priced | 5.0 % (-4.4 %) | 5.8 % | 4.9 % | 4.1 % | -0.04 | 563 pJ |
  | with the 4-byte conflict probes (`Banks`) | 5.4 % (-5.1 %) | 5.9 % | 5.5 % | 4.7 % | +0.17 | 425 pJ |
  | with all conflict probes (`Banks`, `Banks2`) | 5.7 % (-5.3 %) | 5.9 % | 5.5 % | 5.2 % | +0.30 | 319 pJ |

  Regret is 0 % in every kernel and model. With replays at full price, GEMM's dynamic error grows with the share of its
  wavefronts that are replays (r = +0.47, up to +24 % for the most conflicting configuration); at the probe price the
  correlation falls to +0.17-0.30 and the worst overprediction to +7 %, so per configuration the replays are as cheap as
  the probes say. (In Transpose the correlation stays at +0.70 even with free replays: its configurations with shared
  tiles, the conflicting ones, also differ in their global accesses.) The total gets worse for another reason: the
  conflict-free shared probes move energy from the lsu instruction (563 to 319 pJ) to the shared wavefront (367 to 500
  pJ), and lsu instructions are in every kernel, also in Coulomb, which uses no shared memory (4.1 to 5.2 %). The single
  lsu event cannot be both: shared and global memory instructions probably cost differently. The hardware counts them
  separately (`smsp__inst_executed_op_shared_ld` etc.), which now splits the event (see "Shared and global memory
  instructions" below). The reference model above stays the one without the conflict probes.
- **Fast Nbody configurations are underpredicted.** The two worst (-29 and -25 % dynamic) run 240-250 G FMA warp
  instructions/s at 240 W with 16-byte vector loads of a structure-of-arrays layout. One cause was suspected: the FFMA
  probe computes `x*x + c` and reads one register twice, while Nbody's FMAs read three different registers. Measured since
  ("FFMA with three register operands" below), it explains only a small part.

**Shared and global memory instructions** (2026-11-04). The lsu pipe's instructions are split by memory space with the
counters `smsp__inst_executed_op_{shared,global,generic,local}_*`, `_ldsm`, `_ldgsts`.
- The probes split exactly. A global load, store or atomic is 1.000 global instruction per warp access. A shared load,
  store or atomic is 1.000 shared instruction. `LDC` is not on the lsu pipe.
- The fit gives a global-path instruction 615 pJ and a shared one 177 pJ, where the single lsu event gave both 243. The
  shared-memory wavefront changes little: 556 pJ instead of 530.
- In Transpose, 19-27 % of the lsu instructions are neither (probably the barriers around its shared tiles). No probe has
  them, so they cost 0. Pricing them like global instructions changes Transpose by 0.2 %.

The probes' counters are from `ProfileFull2` (`--all --profile ...`, with the new counters), their energies from `Full`.
Old and new model on two new validation runs, `Validation6` and `Validation7` (86 and 90 configurations), mean over both:

| Model | Within-kernel sd of the dynamic error | Bias | Total MAPE | Pairs ordered correctly, 2-5 / 5-10 / 10-25 % apart | Max regret |
|---|---|---|---|---|---|
| One lsu event | 8.8 % | -10.9 % | 5.8 % | 76 / 92 / 97 % | 1.7 % |
| One lsu event, `--weighted` | 8.4 % | -9.1 % | 5.1 % | 77 / 90 / 98 % | 1.7 % |
| **lsu by memory space** | **7.9 %** | **-8.2 %** | **4.7 %** | **81 / 94 / 99 %** | 1.7 % |
| lsu by memory space, `--weighted` | 7.9 % | -8.2 % | 4.7 % | 78 / 92 / 98 % | 1.7 % |

- GEMM gains most: its scatter falls from 9.6 to 7.8 % and its total MAPE from 6.2 to 4.5 %. Its 11 configurations
  without shared memory (they read through L1) are underpredicted by -7.9 % instead of -15.8 %, those with shared memory
  by -5.5 % instead of -7.9 %. The single event priced their many global instructions as shared ones, and that gap
  decided which configuration the model ranked first.
- Coulomb uses no shared memory; its total MAPE falls from 4.8 to 3.3 %.
- Per run, the total MAPE is 4.2 % (`Validation6`) and 4.9 % (`Validation7`).
- Weighting the probe families (`--weighted`, each probe 1 / sqrt(probes of its study)) was the other candidate.
  - Without the split, it lowered the total MAPE on `Validation4`/`5` from 5.4 to 4.8 %: it kept the 30 bank-conflict
    probes from pricing all lsu instructions as shared ones.
  - With the split it changes nothing, so it is off by default.
- Still open at the time: Nbody, at -13 % dynamic; the 3-operand FFMA (next) explains little of it.

**FFMA with three register operands** (2026-11-04).
- Runs: `Ffma3` (`--compute --duty 100,50,25 --payload random,zero`, 126 of 126 rows clock-matched) and its counters
  `ProfileFfma3`.
- In the same run, `ffma3` costs 10.16 pJ per FFMA against 8.65 pJ for `x*x + c`. That is +1.52 pJ (+17.5 %), or +49 pJ
  per warp instruction.
- On zeros the difference is still +1.06 pJ (5.23 vs 4.17), so two thirds of it is the access to the register file
  itself, not the bits that toggle.
- The residuals of the example kernels had suggested about twice as much (+86-96 pJ per warp FFMA).

Using `ffma3` instead of the two `x*x + c` FFMA probes raises the fitted energy of an fma-pipe instruction from 260 to
273 pJ. The IMAD and MUFU probes share the event and keep it lower. Scored on both validation sets, with power from
`Full,Ffma3`:

| Validation runs | Total MAPE (before → after) | Bias of the dynamic energy | Nbody | Scatter, pair order, regret |
|---|---|---|---|---|
| `Validation6`/`7` (lsu split) | 4.7 → 4.4 % | -8.3 → -7.2 % | -13.8 → -12.0 % | unchanged |
| `Validation4`/`5` (one lsu event) | 5.5 → 5.1 % | -10.4 → -9.3 % | -13.1 → -11.5 % | unchanged |

- Per run, the total MAPE is now 3.9 % (`Validation6`) and 4.5 % (`Validation7`).
- The mixed-intensity probes, whose FFMAs compute `x*x + c`, move the other way, from 1.6/3.3/1.8 % to 2.2/3.8/1.7 %.
- So the register operands are real energy, but they are not where most of the missing 7-9 % of the dynamic energy is.
- The ranking does not change: the scatter between configurations comes from elsewhere.

**Is it the measurement? The control kernel** (2026-11-04). The validation runs measure differently from the probe runs:
many short launches through KTT instead of 200 ms launches in KTT's precise measurement, host time instead of GPU time,
the run's own baseline at each row's temperature. Three validation runs with the control kernel (`Validation8`,
`Validation9`, `Validation10`: `--validate all,mixed`), compared with the probe run `Mixed`:

| Run | Control, validation vs probe measurement (5 / 50 ms launches) | Model on the control: dynamic MAPE (bias) | Model on the example kernels in the same run: dynamic MAPE (bias) |
|---|---|---|---|
| `Validation8` | +0.5 / +1.0 % | 3.0 % (+2.2 %) | 8.5 % (-5.7 %) |
| `Validation9` | +2.0 / +1.7 % | 2.6 % (+1.2 %) | 9.7 % (-8.3 %) |
| `Validation10` | +2.2 / +2.1 % | 2.8 % (+0.9 %) | 7.6 % (-5.1 %) |

- Over all 66 rows with a probe-run energy, the validation runs measure the control +1.6 % (sd 1.7 %) above the probe run
  (DRAM with 128 FFMAs per load has none: that probe row hit the power cap). The temperatures differ by a few degrees in
  either direction, the busy fraction is 99.6-100 %, and the launch length does not matter.
- So the measurement accounts for at most 1-2 points of the bias. The model misses the example kernels, not the control.
- Before this, the KTT power (the probes' `power_w`) and the NVML trace over the second half (the validation's `power_w`)
  agreed within 0.1 % on 834 probe rows, and the errors of the example kernels did not depend on temperature, launch
  length or busy fraction (|r| ≤ 0.05).
- One condition is real: `--validate-ms` must stay well above 1 s. `nvmlDeviceGetPowerUsage` is a 1 s average that updates
  every 0.5 s and settles about 1.3 s after a change, so with 1 s per configuration the second half still contains the
  previous configuration (a test with `--validate-ms 1000` was off by up to -58 %). The default 4000 ms is clear of it.

**The fma pipe's two halves** (2026-11-04). With one fma event the fit prices an FFMA 8 % under `ffma3` (9.31 vs 10.16
pJ), because the IMAD and indexing probes share the event.
- `sm__sass_thread_inst_executed_op_{ffma,fadd,fmul}_pred_on` would count FP32 by opcode. KTT's CUPTI pass accepts them
  but returns nonsense (0.004 FFMA per FFMA): they need SASS patching. They are not collected.
- What works are the two halves of the pipe, `sm__thread_inst_executed_pipe_{fmaheavy,fmalite}_pred_on`:
  - IMAD runs on the heavy half only (1.00 / 0.00 per IMAD).
  - Throughput-bound FFMA splits exactly in half: the FFMA probes 0.50 / 0.50, the mixed probes K/2 on lite.
  - The FFMAs of a dependent chain go to lite: the MUFU probe 0.07 / 0.95.
  - An fp16 instruction counts in both halves.
- `ProfileFull3` (`--all --profile ...`) has these counters, and so do `Validation9` and `Validation10`.
- In the example kernels, lite carries 34-60 % of GEMM's fma pipe, 53-80 % of Nbody's and 43-93 % of Coulomb's, and 0 % of
  Transpose's (all IMAD).

With `--fma-halves`, the two halves are two events. The fit then reproduces `ffma3` within 1.2 %, with heavy 252 pJ and
lite 340 pJ. Lite is not dearer to run: an FFMA costs the mean of the two, and an IMAD costs heavy. MUFU falls from 329 to
267 pJ, because its probe's FFMAs run on lite. Pricing FP32 as 2 x lite (same cost on both halves) gives the same scores.

Mean over `Validation9` and `Validation10` (88 configurations each), power `Full,Ffma3`, counters `ProfileFull3`:

| Model | Within-kernel sd of the dynamic error | Bias | Total MAPE | Pairs ordered correctly, 0-2 / 2-5 / 5-10 / 10-25 % apart (time only) | Max regret | Control (dynamic bias) |
|---|---|---|---|---|---|---|
| **one fma event (default)** | **8.2 %** | **-6.9 %** | **4.3 %** | **62 / 72 / 90 / 99 %** (56 / 55 / 75 / 88 %) | **0.7 %** | **+1.0 %** |
| `--fma-halves` | 8.8 % | -4.9 % | 4.2 % | 61 / 72 / 90 / 99 % | 0.7 % | +4.8 % |

- The split lowers the bias, mostly in Nbody (-8.5 to -5.6 %) and GEMM (-4.8 to -3.1 %).
- But it widens the scatter in Nbody and Coulomb, and orders no pair better. Ranking is what the model is for, so the split
  is off by default.
- The control gets worse with the split, as expected: its FFMAs compute `x*x + c`, and now cost as much as `ffma3`.
- The default model per run: total MAPE 4.7 % on `Validation9` and 3.8 % on `Validation10`.

```bash
./EnergyProbesCuda $GPU --all --profile --duty 100 --min-time-ms 2000 --max-time-ms 4000 --warmup-s 10 --idle-s 0 \
    --output ProfileFull3                                                                # counters, ~21 min
./EnergyProbesCuda $GPU --validate all,mixed --output Validation9                          # with the control, ~14 min
python3 ../../Examples/EnergyProbes/tools/energy_model.py --power Full,Ffma3 --counters ProfileFull3 \
    --validation Validation9 --mixed Mixed                                               # add --fma-halves for the split
```

**Where the rest is: the memory write path** (2026-11-04).
- On `Validation9`/`10`, the 81 configurations whose predicted dynamic energy is more than half L2 and DRAM events are
  underpredicted by -11 % on average. The 51 where those events are below 20 % are at -2 %.
- Neither read/write mixing (both groups -11 %) nor a fixed power while DRAM is busy explains it. The intercepts of the
  L2 and DRAM duty fits are -11 to +3 W, and DRAM loads at 10 % duty lie on the line.
- Regressing the residual on the predicted energy of each event group puts it on the writes. L2 writes are under-priced by
  25 %, DRAM writes by 13-20 %. Reads and instructions are within a few percent (R² 0.17-0.26, so a direction, not a
  price).
- The probes agree. The write probes are the worst fitted:
  - DRAM `st` w4 -11 %, DRAM `atom` -11 %, DRAM `red` -15 % (read-modify-write);
  - the partial stores -17 to +38 %;
  - L2 `st-stream` -11 %.
  The loads are mostly within ±5 %.
- Nbody is where this matters most for ranking. Its configurations that spill to local memory (LDL/STL 11-30 % of their
  instructions, at 16-67 G instructions/s against about 300) write the spills through L2 to DRAM. Most of them are
  underpredicted by 9-20 %; the two whose spills stay in L1 are over by 5-12 %. Its fast configurations scatter around
  -5 %. Transpose, which writes as much as it reads, is at -7 to -12 %.
- The write probes (next) were built to cover what the kernels do: stores mixed with loads, dirty evictions from L2, and
  spills through local memory.

**Loads and stores together: the write probes** (2026-11-05). Runs: `Writes` (`--writes --widths 4,16 --duty
100,50,25`, 87 of 87 probe rows clock-matched) and its counters `ProfileWrites` (`--writes --widths 4,16 --profile ...`).
- The counters show each pattern as designed:
  - `copy` at DRAM: per warp access, 4 read and 4 write sectors (4-byte elements), each with an L2 miss request.
  - `update` at DRAM: the stores find their lines in L2 (0.05 write misses per 4-byte warp access), and every sector is
    written back to DRAM later.
  - `local` at the L1 footprint: the loads hit L1, but the stores go on to L2 (4 sectors per 4-byte warp access). L1 is
    write-through for local memory too. With the whole frame (82 MiB), both go to DRAM.
- Mixing costs nothing extra. Against the plain load plus store of the same level and width, `copy` and `update` are
  0.96-1.00x at 16 bytes (same run) and 0.98-1.06x at 4 bytes (stores from `Full`). `local` costs as much per byte as
  global memory at DRAM (165-180 pJ/B) and 25-28 pJ/B at L1.
- In the fit, they raise the DRAM write sector from 1.94 to 2.39 nJ and the read sector from 2.35 to 2.51 nJ, and lower the
  L2 write miss request from 7.9 to 6.8 nJ. The plain store probes had carried the write path alone, and their 16-byte
  streams said little about what a write costs when it comes with reads.

Mean over `Validation9` and `Validation10`, power `Full,Ffma3` with and without `Writes`, counters `ProfileFull3` (and
`ProfileWrites`):

| Model | Within-kernel sd of the dynamic error | Bias | Total MAPE | Memory-dominated configurations (bias) | Pairs ordered correctly, 0-2 / 2-5 / 5-10 / 10-25 % apart | Max regret | Control |
|---|---|---|---|---|---|---|---|
| without the write probes | 8.2 % | -6.9 % | 4.3 % | -11.0 % | 62 / 72 / 90 / 99 % | 0.7 % | +1.1 % |
| **with the write probes** | **7.5 %** | **-5.0 %** | **3.6 %** | **-8.2 %** | **61 / 73 / 92 / 99 %** | **0.7 %** | **+1.4 %** |

- Nbody's scatter falls from 7.6 to 6.6 %, Coulomb's from 8.7 to 6.8 %. Transpose and GEMM keep theirs, with a smaller
  bias.
- Per run, the total MAPE is 3.9 % on `Validation9` (was 4.7 %) and 3.2 % on `Validation10` (was 3.8 %).
- `--weighted` and `--fma-halves` change little on top (sd 7.4 / 7.9 %, bias -6.1 / -3.1 %), so the defaults stay.
- What the fit still missed were the DRAM patterns that load and store (`update` w4 -19 %, `copy` w4 -12 %, `local` w4
  -10 %, `local` w16 -16 %): see the next part.
- The first `Writes` run had a bug of this change: the plain global probes got no level. It was re-measured, and the
  first run's write probes gave the same scores.

**Write-backs without a write miss** (2026-11-05). At DRAM, a 4-byte access costs 15 % more per byte than a 16-byte one
in every probe (`ld` 186 vs 161 pJ/B, `st` 188 vs 161). Taking each probe's energy as a part per warp instruction plus a
part per byte, the per-instruction part is 0.63 nJ without memory, 1.28 nJ at L1, 1.65 nJ at L2 and 4.3-4.7 nJ at DRAM.
So it looked like a cost per instruction that grows with the level the instruction waits on.
- Counters for that: `sm__inst_pending_pipe_lsu_mem_lg_cmd_read` (loads in flight, summed over cycles) is listed by ncu
  but not collected through KTT. LSU-throttle stalls (`smsp__warps_issue_stalled_lg_throttle`) belong to 16-byte stores
  and do not follow the misses (r = +0.56 over 28 probes, carried by one). `dram__cycles_active` is 4 x the sectors on
  this GPU, a derived count with no timing in it.
- With the write probes in the fit, the plain 4-byte DRAM load is only 5 % low. What remained were the patterns that load
  and store the same lines: `update` and `local` move as many DRAM sectors as `copy` (32 per 16-byte warp access), but
  their stores hit L2 (2.1 and 1.8 write misses against copy's 4), and they were 6-19 % low while `copy` fit.
- The model had priced a DRAM write mostly in the L2 write miss request (6.8 nJ, against 2.4 nJ per written sector). A
  dirty line that L2 evicts after a store hit costs a DRAM write too, but has no write miss: update in place, spills,
  read-modify-write. No GA102 counter counts write-back requests.
- So writes are now priced by their DRAM sectors, and only write miss requests beyond one per four written-back sectors
  (partial writes, 1-3 sectors per request) keep a price of their own (`L2 partial write miss request`). A line written
  back now costs 15.7 nJ, about what one costs to read (16.4 nJ); before, a write-back after a hit cost 9.5 nJ.
- The probes fit better: MAPE 6.0 -> 5.6 %, the DRAM probes 7.2 -> 6.1 %. `update` w4 is at -7 % instead of -19 %, `local`
  w4 at +3 % instead of -10 %, `update` w16 at -1 % instead of -6 %. The 4-byte stores and atomics stay 10-14 % low.

| Validation runs | Model | Within-kernel sd | Bias | Total MAPE | Pairs ordered correctly, 0-2 / 2-5 / 5-10 / 10-25 % | Max regret |
|---|---|---|---|---|---|---|
| `Validation9`/`10` (chosen on) | write miss request as the DRAM write access | 7.5 % | -5.0 % | 3.6 % | 61 / 73 / 92 / 99 % | 0.7 % |
| | **partial write miss requests** | **6.9 %** | **-3.7 %** | **3.1 %** | 59 / 73 / 92 / 99 % | 0.7 % |
| `Validation6`/`7`/`8` (not used for the choice) | write miss request as the DRAM write access | 7.4 % | -5.3 % | 3.8 % | 59 / 79 / 96 / 99 % | 1.7 % |
| | **partial write miss requests** | 7.5 % | **-4.2 %** | **3.4 %** | 62 / 79 / 95 / 99 % | 1.7 % |

- The bias and the total error fall on both sets. The scatter falls only on the set the change was chosen on, so the
  ranking is about the same as before.
- Per run, the total MAPE is now 3.2 % (`Validation6`), 3.7 % (`7`), 3.2 % (`8`), 3.7 % (`9`) and 2.7 % (`10`).
- Configurations dominated by L2 and DRAM traffic are now -5 % on average (-11 % before the write probes).

```bash
./EnergyProbesCuda $GPU --writes --widths 4,16 --duty 100,50,25 --output Writes                # write probes, ~20 min
./EnergyProbesCuda $GPU --writes --widths 4,16 --profile --duty 100 --min-time-ms 2000 --max-time-ms 4000 \
    --warmup-s 10 --idle-s 0 --output ProfileWrites                                      # their counters, ~3 min
python3 ../../Examples/EnergyProbes/tools/energy_model.py --power Full,Ffma3,Writes --counters ProfileFull3,ProfileWrites \
    --validation Validation9 --mixed Mixed
```

**Held-out kernels** (2026-11-05). Reduction, ClTuneConvolution and KernelTunerPnpoly were added after the model was
built, and scored with it unchanged: power `Full,Ffma3,Writes`, counters `ProfileFull3,ProfileWrites`, default options.
Runs: `HeldOut1` and `HeldOut2` (`--validate heldout,mixed`, 70 configurations each). Per run, two configurations need
more registers than a block can have (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`) and are left out.

| Kernel (2 runs) | Total MAPE | Dynamic bias | Within-kernel scatter | Regret (time only) |
|---|---|---|---|---|
| reduction | 3.3-4.0 % | +4.0 / +5.3 % | 5.2-9.5 % | 0 % (7.7-8.6 %) |
| convolution | 2.5 % | +0.3 / +2.1 % | 5.4-5.6 % | 0-2.4 % (2.4-4.2 %) |
| pnpoly | 1.9-2.2 % | -2.3 / -2.7 % | 5.8-6.6 % | 0 % (0 %) |

| Scores (`score_ranking.py`) | Within-kernel scatter | Bias | Total MAPE | Pairs ordered correctly, 0-2 / 2-5 / 5-10 / 10-25 % (time only) | Max regret (time only) |
|---|---|---|---|---|---|
| held-out kernels (`HeldOut1`, `HeldOut2`) | 6.4 % | +1.1 % | 2.7 % | 67 / 87 / 100 / 100 % (59 / 68 / 86 / 100 %) | 2.4 % (8.6 %) |
| the four design kernels (`Validation9`, `10`) | 6.9 % | -3.7 % | 3.1 % | 59 / 73 / 92 / 99 % (56 / 55 / 75 / 88 %) | 0.7 % (13.0 %) |

- The model does as well on kernels it was not designed on as on those it was: no sign of tuning to the four kernels.
- The control kernel agrees between the two measurements again: +0.6 / +0.4 % (`HeldOut1`) and +2.4 / +2.4 % (`HeldOut2`)
  for 5 / 50 ms launches. The model predicts it at +2.7 and +0.9 %.
- Reduction shows the two remaining errors:
  - Its bandwidth-bound configurations (about 5 ms per launch) come out 5-14 % high, as the streaming DRAM load probe in
    the fit (+3 %).
  - Its slowest configuration (82 blocks, 21.7 ms, latency-bound) comes out 18 % low. A GPU that waits costs energy the
    events do not count; an event per busy SM cycle with a probe of its own (pointer chasing at L2 and DRAM latency) is the
    next step for that.
- Pnpoly's branches and compares are priced well (1.9-2.2 %), although no probe covers them: its 'other' instructions are
  2-8 % of the total, and its alu and fma pipes carry 85-90 % of the dynamic energy.

**Instruction classes without a probe** (2026-11-05). Runs:
- `Instr` (`--compute --latency --duty 100,50,25`) measures the power of the new probes.
- `ProfileFull4` (`--all --profile`) profiles every probe with the counters added in this batch.
- `Validation11` and `Validation12` (`--validate all,heldout,mixed`) cover the seven example kernels and the control
  kernel with the same counters.

Five arithmetic probes run instructions that no earlier probe ran, with the SASS of all earlier probes unchanged on
sm_70/80/86/90:

| Probe | What it runs | pJ per thread op (per warp instruction) | Event | Fit |
|---|---|---|---|---|
| `shfl` | `SHFL.BFLY` | 26.0 (806) | inst lsu other (no probe before, so priced 0) | +0.5 % |
| `bar` | `BAR.SYNC` | 5.9 (164) | inst other (`BAR` issues to the adu pipe) | -7.7 % |
| `ffma-loop` | 4 FFMAs per loop iteration, plus `IADD3`, `ISETP`, `BRA` | 13.9 (15.0 with three-register FFMAs) | inst alu, inst other | -6.9 % |
| `fmul-fadd` | `FMUL`, then `FADD` | 16.0 per pair | only checks the model | model +16 % |
| `mufu-sin` | `MUFU.SIN` with its `FMUL.RZ` | 28.9 | only checks the model | model -4 % |

- **Shuffles.** A warp shuffle costs about three FFMAs. Until now the lsu instructions outside shared and global memory had
  no probe, so kernels with shuffles came out too low by that much.
- **Branches.** On GA102 a loop's `BRA` issues to none of the pipes CUPTI names; in the loop, `smsp__inst_executed_pipe_cbu`
  reads about 0. `smsp__inst_executed_op_branch` does count it: exactly one per 4 FFMAs.
  - As an event of its own the branch cannot be identified. Every probe loop has one branch per iteration in a fixed
    proportion to its other instructions, so the fit priced it 0 and `ffma-loop` came out 12 % low.
  - Branches therefore stay in "inst other" (129 pJ). The counter is still collected, to check `ffma-loop`.
- **FMUL and FADD** cost about 8 pJ each, against 10.3 for a three-register FFMA. The model counts them as FFMAs, because
  telling them apart needs the SASS opcode counters, which read wrong values through KTT (see the limitations).
- **On the example kernels** the probes move the bias, not the ranking. With them, the scatter stays at 6.7 % and the
  bias goes from -2.8 to +0.4 % (`Validation11`, `12`; without the busy events of the next paragraph).
- **Counter guard.** `energy_model.py` and `score_ranking.py` refuse to combine runs in which a counter that an event needs
  is present in some and missing in others (it would count as 0 there).
  - The optional counter groups must be in all runs or in none: lsu instructions by memory space, and the fma halves with
    `--fma-halves`.
  - Counters that another GPU names differently are read under either name (`ALIASES`: GB202's DRAM sectors), and the
    separate fp16 pipe of Turing and GA100 stands in for the fma pipe's fp16 counters.
  - `tools/check_counters.py --chip <chip>` lists what a GPU lacks before anything is measured.

**Steady DRAM traffic costs a power of its own** (2026-11-05). This came from Reduction's latency-bound configuration
(18 % too low).

The latency probe (`--latency`, `kernels/Latency.cu`) walks chains of dependent loads:
- Each load is one coalesced 128-byte line. The next line's address depends on the loaded data, so a chain has exactly one
  load in flight.
- Parallelism runs from 4 to 512 loads in flight per SM: 1/4/16 chains per warp times 4/8/32 working warps per SM.
- With `--latency`, the SM sweep also runs it with 1 and 16 chains on 1 to 82 SMs. Its lines are random over the whole
  footprint, so it stays in DRAM on any number of SMs.

Runs: `Instr` (all SMs) and `LowRate` (`--latency --duty 100 --sm-sweep 82,41,20,10,5,2,1`). Together they give 32 rows
from 1.3 to 900 GB/s at the DRAM footprint.

| Loads in flight per SM (all SMs) | 4 | 8 | 16 | 32 | 128 | 512 |
|---|---|---|---|---|---|---|
| DRAM: GB/s, pJ/B | 113, 268 | 224, 233 | 387, 207 | 674-768, 183-189 | 831-898, 184-188 | 899, 190 |
| L2: GB/s, pJ/B | 208, 60 | 414, 56 | 742, 52 | 1390-1547, 50-52 | 1932-2142, 50-52 | 2132, 50 |

- **The shape.** The power above the baseline is a cost per byte plus a fixed part that grows with the DRAM rate and then
  saturates: 172 pJ/B + 13.4 W x (1 - exp(-rate / 95 GB/s)), rms 2.4 W over the 32 rows (r0 ± 35 GB/s).
  - The alternatives fit worse: a fixed power that does not depend on the rate (rms 3.8 W), or one per working SM (2.9 W).
  - This is what DRAM power-down looks like. With requests at random times, DRAM rests whenever a gap outlasts its idle
    timer, and such gaps get exponentially rarer as the rate rises. At 1-27 GB/s the fixed part is 0.5-1.6 W; at 113 GB/s,
    10.5 W.
  - L2 shows nothing of the kind (49-51 pJ/B, the fixed term within its error).
- **Why the earlier probes missed it.** Their lower duties are bursts at the full rate with idle time between them, and DRAM
  rests between the bursts. So their energies per access include the fixed power at the full rate, and their fits across
  duties show no fixed part.
- **Stall cycles do not price it.** An event per warp cycle stalled on memory (`smsp__warps_issue_stalled_long_scoreboard`)
  got 0.8 pJ per cycle in the fit. Waiting warps cost nothing; what costs is DRAM kept awake, per unit of time. The event
  was dropped.
- **In the model: "DRAM busy ns".** It counts the nanoseconds DRAM is kept awake, times 1 - exp(-rate / 95 GB/s) of the
  DRAM rate (sectors read and written per ns):
  - for a probe, its time per access at duty 100, from the power run;
  - for a kernel, its kernel time.

  The sweep gives only the shape; the fit prices the event at 12.8 W, against the sweep's 13.4 W. `--no-busy` turns it off.
  `tools/fit_busy.py LowRate,Instr` fits the shape from a run's dependent-load rows. On another GPU,
  `--busy-scale <its scale>` uses that GPU's own; the default, 95 GB/s, is the RTX 3090's.
  - Fit residuals: the DRAM chase probes go from -31 to 0 % to -9 to -1 %. The streaming 16-byte DRAM load goes from +6.7
    to +4.2 %. The L2 chase probes stay at -12 to +3 %; the worst is 4 loads in flight per SM.
- **On the example kernels** (`score_ranking.py`; power `Full,Ffma3,Writes,Instr`, counters `ProfileFull4`; the reference
  is power `Full,Ffma3,Writes` with the same counters):

| Kernels (runs) | Model | Within-kernel scatter | Bias | Total MAPE | Pairs ordered correctly, 0-2 / 2-5 / 5-10 % | Max regret |
|---|---|---|---|---|---|---|
| design: transpose, gemm, nbody, coulomb (`Validation9`-`12`) | reference | 6.6-6.9 % | -3.8 to -4.0 % | 3.1 % | 59-73 / 73-77 / 92-97 % | 0-0.7 % |
| | + instruction and latency probes | 6.7-6.9 % | -0.5 to -0.7 % | 2.8 % | 56-68 / 72-77 / 92-97 % | 0-0.7 % |
| | + DRAM busy event | 6.6-6.7 % | +1.6 to +1.7 % | 2.9 % | 59-71 / 70-78 / 94-98 % | 0-0.7 % |
| held out: reduction, convolution, pnpoly (`Validation11`, `12`, `HeldOut1`, `2`) | reference | 6.4-7.0 % | -1.1 to +1.1 % | 2.7 % | 66-67 / 87-88 / 99-100 % | 2.0-2.4 % |
| | + instruction and latency probes | 6.1-6.7 % | +1.6 to +3.5 % | 3.4-3.6 % | 66 / 86-88 / 100 % | 2.0-2.4 % |
| | + DRAM busy event | 4.8-5.2 % | +3.2 to +5.0 % | 3.0-3.4 % | 66-68 / 87-89 / 99-100 % | 2.0-2.4 % |
| all six runs | reference | 6.7 % | -2.2 % | 3.0 % | 66 / 84 / 97 % | 2.4 % |
| | all of it (the default now) | 5.9 % | +2.7 % | 3.0 % | 66 / 84 / 98 % | 2.4 % |

- **The default was chosen on the four design kernels**, where the event is neutral (scatter -0.1 to -0.2 points, MAPE +0.1).
  It is the default because it is measured, its price agrees with the sweep, and it helps wherever the regime occurs.
- **The held-out numbers are not a clean test of this change.** Reduction motivated it, and Convolution's and Pnpoly's
  biases were looked at while it was built (the scale r0 was then set from the sweep alone).
- **Reduction's scatter** goes from 5-11 to 4-6 %; its slowest configuration from -18 to +1 %. Transpose's goes from
  8.6-9.5 to 7.2-8.1 %.
- **Gemm and Convolution come out 3-7 % high.** They load tiles in phases and compute in between. The counters give only a
  kernel's average DRAM rate, so the event treats DRAM as awake for the whole kernel, while it can rest between the phases.
- **Reduction's bandwidth-bound configurations stay 4-13 % high**, from two causes:
  - The fit places the streaming DRAM load 4 % high.
  - Reduction's data, floats in [0, 1000), toggles fewer bits than the probes' random words. With the zero payload, the
    same 16-byte DRAM load draws 22 % less (110 vs 140 W) and the L2 one 47 % less.

  Both are the same for every configuration of a kernel, so they move its bias, not its ranking.

## Using the model in a tuner

The probes, `energy_model.py` and the validation runs are the calibration. A tuner needs less: one file with the model
and one call that prices a configuration from its counters and its time, without measuring power.

```bash
# once per GPU type and clock: the model file, from the calibration runs
python3 tools/build_model.py --power Preset3 --counters ProfileModel1 \
    --calibration Validation11,Validation12,Validation13,HeldOut1,HeldOut2 --output rtx3090_1200mhz.model.json
# a KTT output file (JSON) of any tuning run with CUPTI profiling: the configuration to use
python3 tools/energy_predictor.py --model rtx3090_1200mhz.model.json --ktt TuningResults.json --objective energy
# the same inside a KTT program (EnergyModel.h); as a demonstration on the example kernels:
./EnergyProbesCuda $GPU --tune gemm,coulomb,transpose --model-file ../../Examples/EnergyProbes/models/rtx3090_1200mhz.model.json
```

- **The model file** (`models/rtx3090_1200mhz.model.json`, format version 1) holds the GPU and the SM clock of the
  calibration, the energy per event, the baseline line, the uncertainty and how it was measured, the counters the
  events need, every probe's full-rate energy and time (for the self-check below) and the SHA-256 of the runs it was
  built from.
- **`tools/energy_predictor.py`** (standard library only) and **`EnergyModel.h/.cpp`** (standard library and
  nlohmann::json; no KTT types in the interface) give the same numbers: `tests/` compares them on a validation run to
  1e-9. Both refuse a model of a newer version, an unknown event, and counters the model needs but does not get (they
  would count as 0 and make a kernel look cheap); `strict=False` / `--lenient` turns the last into a warning.
- **Objectives**: energy, EDP, ED2P or time, optionally only among configurations at most x % slower than the fastest
  (`--max-slowdown`), or the fastest among those within x % of the lowest energy.
- **Uncertainty**. The relative error of the dynamic energy differs between the configurations of one kernel with a
  standard deviation of 6.3 % (27 kernel runs, 604 configurations; 6.1 to 6.6 % with any one kernel left out). A
  prediction carries that as `sd_j`. Two configurations closer than 2 sd of their difference are *contenders*.
  On the calibration runs, 79 % of all pairs are told apart at 2 sd and every one of those 5214 is ordered right; the
  other 21 % are still ordered right in 83 % (by time alone: 73 %), so the first choice stands and the contenders are
  what to measure with NVML if the last percent matters. The first choice's regret is 0.2 % on average and 2.4 % at
  most; with the contenders measured, 0. The bias (+3.7 % on average, -3 to +11 % per kernel) is not part of `sd_j`:
  it moves all configurations of a kernel together.
- **Cost**. `--tune` timed and profiled 72 configurations of three kernels in 87 s (1.2 s each, with compilation),
  against about 10 s per configuration for an NVML measurement.
- **Self-check** (`tools/check_model.py`): a probe run of 5 minutes at full rate on the target GPU (`--quick`), compared with the
  calibration's own full-rate rows. PASS, SCALED (a common factor; `--write` applies it and this GPU's baseline) or
  FAIL. On the calibration GPU a week later: time per access 0.0 %, energy median -2.8 % (5 s windows read low),
  all 22 probes within 10 %, PASS.

### The model at another operating point

A model is valid at the clock it was calibrated at. The second RTX 3090 of the test machine is not clock-locked, and
locking needs root, so it shows what happens otherwise (`ValFree1`, `--validate all,heldout,mixed` on that card):

- Its SM clock moves between 1170 and 1935 MHz from one configuration to the next (one Reduction configuration: 490),
  because 82 % of the configurations draw more than 320 W and the GPU holds its 350 W limit by lowering the clock. NVML
  names the power cap for 119 of the 154 configurations and thermal slowdown for 42 (the card was at 81-84 C). The
  sleep baseline is 179 W (at 1200 MHz: 115 W).
- There, energy is close to the power limit times the time. Time alone orders 96.9 % of the pairs that differ by more
  than 2 % (energy over the kernel's time at the measured power); the 1200 MHz model 94.4 %, with a total MAPE of 26 %.
  Capping the model's power at 350 W with this card's baseline gives 96.8-97.5 %: no better than time.
- Among the 115 configurations at 330 W and more, time alone orders 99.7 % of the pairs and the model 93.7 %; among the
  39 below, both about 92 % (142 pairs: too few to separate them). In the 23 configurations that kept the full clock
  (1880 MHz and more), power is 228 W + 1.08 x the model's dynamic power (R2 0.78): the model's events still explain
  the power there, on a higher baseline.
- With `--power-limit` (the program reads it from NVML) a configuration that the model predicts above the limit gets
  the limit times its time, with a warning; that covers the model's own clock, not a free one.
- `check_model.py` refuses the card (2 of 13 probes at a constant clock), and both predictors print a notice when most
  configurations ran at another clock than the model's.

So the rule for a tuner: at the power limit, tune for time, and energy follows. The model earns its place where the
GPU is below its limit and the clock is constant (a locked clock, a lower power limit, smaller kernels), which is
where configurations of equal time differ in energy: there time alone has a regret of up to 12.6 % and the model 2.4 %.
A clock sweep (the same calibration at three or four locked clocks) is the missing piece and needs root.

### A second card of the same type

Between two runs the second RTX 3090 was locked at 1200 MHz as well (by the machine's administrator), which made it a
test of the model on a card it was not calibrated on, with nothing refitted:

| | Result on the second card |
|---|---|
| `--quick` + `check_model.py` (`Quick0`, 3.5 min) | PASS: time per access 0.0 %, energy median -1.7 %, 21 of 22 probes within 10 %, baseline 117.2 W against 116.9 W expected |
| `--validate all,heldout,mixed` (`ValFree2`, 162 configurations) | scatter 7.2 %, bias +6.7 %, total MAPE 4.0 %, max regret 2.7 % (time alone 6.2 %); pairs 5-10 % apart ordered right 95.7 % (time alone 84.8 %) |
| `--tune all,heldout --tune-measure all` (`TuneAll0`, 155 configurations, every one then measured) | the model's first choice is the measured best in 6 of 7 kernels and 0.1 % above it in Coulomb, where the best is one of its 2 contenders; the fastest configuration is up to 5.0 % above the best (Pnpoly); pairs more than 2 % apart ordered right 97.4 % (time alone 94.0 %) |

The calibration card's own validation runs have a scatter of 6.0 % and a regret of at most 2.4 %, so the model loses
little on the second card. Its baseline rises faster with temperature there (0.73 against 0.28 W/C, at 73-77 C), which
is the +6.7 % bias; `check_model.py --write` puts the card's own baseline into a copy of the model.

`--tune` took 14 min for this, of which the 155 power measurements (4 s each) are about 10 min; with `--tune-measure contenders` it
measures 27 + 7 configurations (the contenders and each kernel's fastest). A KTT output file of a profiled run gives the
same choices through `energy_predictor.py --ktt`; its launch times are those of profiled launches (up to 10 % longer),
so the energies are a few percent higher than with unprofiled times, which `--tune` uses.

### Tests

```bash
python3 -m pytest Examples/EnergyProbes/tests -q        # from the KTT root, 39 tests, a few seconds, no GPU
EP_SLOW=1 python3 -m pytest Examples/EnergyProbes/tests -q   # also the SASS checks of every probe variant, 3 min
```

`tests/data` holds real runs (RTX 3090, 1200 MHz). The tests rebuild the model from them and compare it with the shipped
file, compare the predictor with `energy_model.py` and the C++ predictor with the Python one, check the recommendation
on the validation run (measured best = the choice or a contender), the model-file checks, `check_model.py`'s three
verdicts, the CSV repair, and the program's GPU-free modes (`--predict`, `--dry-run`, `--analyze`, refused options).

## Validation checklist

`tools/check_run.py RUN --counters PROFILED_RUN` (run from the output directory) tests items 4, 5, 7, 8 and 9 below on a
run's outputs, and more that follow from how the probes are built: every configuration ran and has an energy above the
baseline, the sleep baselines agree, the baseline follows temperature (it warns when the analysis fell back to a flat
baseline although the sleeps show a clear slope), every fit is a line (R²), NVML's energy counter and sampled power keep
the same ratio in every row, probes that take the same path agree (`ld.cg` and `ld` at L2, `ld.nc` and `ld` at L1, the
locality stream and the global load), a shared load costs no more than an L1 hit, zeros never cost more than random
data, pJ per byte falls with the width, and energy rises with the conflict degree, with scattered sub-sector accesses
and with runtime indexing. With the counters: every global
probe hits its level, the instructions per access are the same at every level, shared wavefronts and conflicts are as
designed, and the constant probe reaches the L1 and the L1.5. Its exit code is 1 if a check fails, so it can gate a run.
Items 1-3 need runs of their own (`compare_runs.py` over them); `energy_model.py --cross-validate` predicts every probe
from a fit over the others, which tests whether the energies per event are a consistent decomposition.

1. **Sensor behaviour of your GPU.** NVML's default reading is a 1 s average on Ampere (except GA100) and newer, and only part
   of the runtime is sampled on A100/H100 ([Yang et al.](https://arxiv.org/abs/2312.02741)). Keep `--min-time-ms` well above
   the averaging window. Try `--min-time-ms 30000` and check that the numbers don't move.
2. **Repeatability.** Run 3-5 times with the same options, ideally on different days. Coefficient of variation should be a
   few percent. `python3 ../../Examples/EnergyProbes/tools/compare_runs.py A B C [--csv out.csv]` (run from the output directory) lists
   every energy of runs `--output A`, `B` and `C` with its mean, coefficient of variation and range, and recomputes the SM
   sweep with the temperature-corrected baseline.
3. **Linearity.** Change `--launch-ms` (100 / 200 / 400). Energy per access must not change.
4. **Ordering.** The summary checks L1 < L2 < DRAM in energy and the reverse in bandwidth. `ld.cg` at the L2 footprint
   should match `ld` at the L2 footprint.
5. **Hit levels.** Run with `--profile` (KTT built with `--profiling=cupti`). The counter table gives sectors per requested
   sector at L1, L2 and DRAM (reads plus writes) and shared wavefronts, each 1.0 where that level serves the access: L1 rows
   with a high `l1tex__t_sector_hit_rate`, L2 rows (loads and stores) with L2 = 1 and DRAM = 0, DRAM rows with DRAM = 1.
   Atomics count in L2 (they bypass L1) and at the DRAM footprint have DRAM = 2 (read and write-back). Texture rows
   count texture sectors as L1 and must look like the plain loads of the same footprint. `inst/acc` (warp
   instructions per warp access or operation) must be the same for all levels of a width. Constant rows: `IDC` must be 1
   request per warp access, `to L1.5` about 0 at the L1 footprint and about 1 at the L1.5 footprint (0.92 on an RTX 3090).
   These counter names are for Turing and newer.
6. **Independent energy source.** The `ctr [W]` column and the summary compare KTT's power with NVML's energy counter and
   with the instant and 1 s average power fields. An external power meter remains the only way to check NVML itself.
7. **Control vs datapath check.** With `--payload random,zero`, the no-load loop (which moves no data) must show a datapath
   share of about 0.
8. **Clock.** The summary checks that all rows ran within 2 % of the same SM clock and lists clock limiters per probe.
9. **Placement.** The summary's `blocks per SM` line must say ok; otherwise blocks on one SM shared a stripe and can convoy.
   `ld` and `ld.cg` at the L2 footprint should give the same bandwidth and energy.

### The checklist on an RTX 3090

NVML measures the whole board, so no test can show that an energy is the true energy of one component. These runs test the
next best things:
- the measurement itself;
- that every probe exercises the component it is named after;
- that the energies relate as the probes are built;
- that they add up to the energy of kernels they were not fitted on.

The runs:
- one run of every probe (`--all --output Full`, 2.2 h) and its counters (`--all --profile ... --output ProfileFull`);
- a validation repeat (`--validate all --output Validation5`);
- `--compute --duty 100,50,25` with the default 200 ms launches (`LinRef`), with `--launch-ms 100` and `400`, and with
  `--min-time-ms 30000 --max-time-ms 90000` (`Sensor30`).

```bash
./EnergyProbesCuda $GPU --all --output Full                                                # every probe, ~2.2 h
./EnergyProbesCuda $GPU --all --profile --duty 100 --min-time-ms 2000 --max-time-ms 4000 --warmup-s 10 --idle-s 0 \
    --output ProfileFull                                                                 # their counts, ~16 min
./EnergyProbesCuda $GPU --validate all --output Validation5                                # example kernels, ~12 min
./EnergyProbesCuda $GPU --compute --duty 100,50,25 --output LinRef                         # ~15 min each
./EnergyProbesCuda $GPU --compute --duty 100,50,25 --launch-ms 100 --output Lin100
./EnergyProbesCuda $GPU --compute --duty 100,50,25 --launch-ms 400 --output Lin400
./EnergyProbesCuda $GPU --compute --duty 100,50,25 --min-time-ms 30000 --max-time-ms 90000 --output Sensor30   # ~37 min
python3 ../../Examples/EnergyProbes/tools/check_run.py Full --counters ProfileFull
python3 ../../Examples/EnergyProbes/tools/compare_runs.py LinRef Lin100 Lin400
python3 ../../Examples/EnergyProbes/tools/compare_runs.py LinRef Sensor30
python3 ../../Examples/EnergyProbes/tools/energy_model.py --power Full --counters ProfileFull --validation Validation5 \
    --cross-validate
```

| Check | Result |
|---|---|
| Runs | `Full` 723 of 723 rows within 2 % of the baseline clock (largest difference 0.00 %); `ProfileFull`, `LinRef`, `Lin100`, `Lin400` and `Sensor30` all of theirs. `Validation5`: 8 Nbody configurations did not compile or launch, as in earlier runs |
| `check_run.py Full --counters ProfileFull` | 55 passed, 2 warnings, 1 failed (the baseline, below) |
| Counters | See the counters breakdown below the table |
| 1. Sensor window | 10 s vs 30 s minimum: fits agree within 0.0-5.1 % (CV), duty-100 rows within 0.05-3.8 %. The largest, DRAM `ld`, is 6 % higher in the longer run, and NVML reports no memory temperature to check that |
| 2. Repeatability | `Full` against each earlier run with the same probes (`Banks2`, `Partial`, `Atomics`, `Texture`, `Locality`, `Indexing`): median CV 1.8-2.2 %. Against `Study`/`Study2`/`Study3`: 3.7 %, from the baseline slope (below). Of 3 configurations measured in both validation runs, all agree within 0.5 % |
| 3. Linearity | 100/200/400 ms launches: fits agree within 0.1-3.4 %, duty-100 rows within 0.2-1.9 %. Out of sample: the line through the 100/50/25 % rows predicts the unused 10 % row within 1.6 W median (2.3 % of the full-rate dynamic power, mean signed -0.05 W, 172 fits) |
| Additivity | `Full` + `ProfileFull` alone (one fresh run and its counters) predict the example kernels with a total MAPE of 5.4 % on `Validation5` and 5.3 % on `Validation4` (the reference run set: 5.9 and 5.7 %). Bias: -4.9 % total, -10 % dynamic. `--cross-validate` over 102 probes: median 5.5 %, 68 within 10 % |
| Plausibility | `ld` 16 B: L1 0.98, L2 over L1 3.6 and DRAM over L2 15.5 pJ/bit, a 20.1 pJ/bit path, against L1 1.59, L2 over L1 4.71 and HBM over L2 13.11 pJ/bit on an A100 (Berkeley SC'25), a 19.4 pJ/bit path. Berkeley's energies are above the idle power at the full rate; computed that way, ours are L1 1.71, L2 over L1 7.1 and DRAM over L2 20.9 pJ/bit. Micron gives [7.25 pJ/bit](https://www.tomshardware.com/news/micron-reveals-gddr6x-details-the-future-of-memory-or-a-proprietary-dram) for the GDDR6X device alone. FFMA 4.7 pJ/FLOP, HMMA 0.65 pJ/FLOP |

Counters:
- Every global and texture probe hits its level, at the same instructions per access at every level.
- Shared wavefronts and conflicts are as designed in all 33 shared rows.
- The constant probe makes 1.000 request per warp access, 0 % of them to L1.5 at the L1 footprint and 92 % at the L1.5
  footprint.
- Each arithmetic probe runs 1.00 warp instruction of its pipe per warp operation, plus 0.17 for the loop.
- The locality probes make 16, 8 and 4 L2 requests per warp instruction for 32-, 64- and 128-byte or longer runs, and
  15.4-16 DRAM sectors.

- **Baseline over temperature: a ±3 % uncertainty of every energy.**
  - The analysis moves the sleep baseline to each row's temperature only when the sleeps span 2 C. `Full`'s six sleeps
    spanned 1.4 C (63.8-65.2 C), although their slope is clear: 0.78 ± 0.18 W/C.
  - So `Full` used a flat baseline, and rows that ran warmer count their extra leakage as dynamic power. The failed check
    is one of them: the 32-way shared load had R² 0.88 because its 50 % row ran 2.6 C warmer than its 100 % row at the
    same rate. With the line, its R² is 1.000.
  - Which correction is right is less clear. Each run's own slope is a tangent: 0.6-0.8 W/C when the sleeps are cool
    (`Study`, `Full`), 0.16-0.34 W/C when they are warm (`Banks2`, `Partial`, `Atomics`, validation runs 0.26-0.29).
    Pooled over all runs' traces, sleep power rises about 0.7 W/C between 63 and 66 C, then about 0.25 W/C up to 70 C.
    The fan of GPU 1 runs at 46 % idle at 55 C and stays at 60-61 % from 68 to 72 C.
  - Refitting `Full` three ways:

    | Baseline | Fits with R² < 0.99 (of 172) | CV of 56-57 probes over 11 runs | Total MAPE `Validation4` / `5` | `--cross-validate` MAPE |
    |---|---|---|---|---|
    | Flat (what the analysis did) | 23 | 2.8 % | 5.3 / 5.4 % | 11.4 % |
    | The sleeps' line, 0.78 W/C | 14 | 2.4 % | 6.3 / 6.5 % | 10.7 % |
    | Pooled curve, at the run's level | 14 | 2.0 % | 5.5 / 5.7 % | 10.8 % |

  - Extrapolating a steep line from cool sleeps to probes at 70-74 C removes too much. A baseline measured across the
    probes' temperatures would remove that ambiguity.
  - **Done on 2026-11-06:** a sleep now follows every probe group (see "A sleep baseline after every probe group" under
    the run options). In `Preset2` 14 sleeps span 64.0-71.5 C and lie on one line, 0.29 ± 0.02 W/C (rms 0.13 W), with
    no curvature. So the steep 0.6-0.8 W/C of cool runs (sleeps spanning 1-2 C) and the bend of the pooled curve are
    more likely noise of short spans and differences between runs than a curve within one run.
- **Most fits rest on their 25 % row.** In 119 of 172 fits the 50 % duty row reaches at least 90 % of the full rate,
  because half the warps already saturate the component. The 25 % row reaches 30-65 %. The slope is then set by the 25 %
  row against two rows of nearly the same rate, and R² measures the noise between those two. The warning "zeros cost
  more than random data" (`st-sub-adjacent` DRAM, 8 B) is this: zeros cost less in every row (1926 vs 1938 pJ at 100 %,
  1729 vs 2114 pJ at 25 %), but the slopes of the two fits do not keep that order.
- **The model misses about 10 % of the dynamic energy in every kernel**, a median 10.6 W.
  - With the kernel held fixed, the miss does not grow with the predicted power (0.02 ± 0.05 W per W).
  - It is larger in warmer configurations. Their extra power also heats them, so this does not identify the cause.
  - It is not an interaction of memory and arithmetic. The mixed-intensity probes, which run loads and FFMAs together
    from memory-bound to compute-bound, are predicted within 1.7-4.3 % with a bias of +0.7 to +4.3 % (see Example
    results).

## Example results (RTX 3090, SM clock locked at 1200 MHz, `--all`)

Fits over duties 100/50/25 (R² 0.998-1.000), random data, pJ per byte for memory and per operation of one thread for
arithmetic:

| Probe | Total | Control (zeros) | Datapath share |
|---|---|---|---|
| shared ld / st | 5.76 / 6.19 | 3.60 / 3.98 | 38 / 36 % |
| L1 ld / ld.nc | 7.55 / 7.53 | 5.09 / 4.92 | 33 / 35 % |
| L2 ld / st | 34.5 / 36.5 | 19.6 / 23.6 | 43 / 35 % |
| DRAM ld / st | 161 / 160 | 122 / 121 | 24 / 24 % |
| no-load loop | 2.09 | 2.11 | -1 % (check) |
| FFMA / IMAD / HFMA2 | 8.67 / 8.83 / 11.0 | 4.13 / 4.74 / 4.74 | 52 / 46 / 57 % |
| DFMA / HMMA (per thread) | 72.0 / 72.7 | 11.3 / 13.0 | 84 / 82 % |

- Net of the loop (0.75 pJ per operation): FFMA 3.96 pJ/FLOP, HFMA2 2.56, DFMA 35.6, HMMA m16n8k16 0.56 (2.30 nJ per warp
  instruction). IMAD costs the same as FFMA, as Antepara et al. found on A100.
- Per request vs per byte: L2 32.7 pJ per request + 30.4 pJ/B, DRAM 60.3 + 154.6 pJ/B; for L1 the line does not fit well
  (R² 0.96).
- Constant power (no CUDA context, memory at 810 MHz) 43.0 W; sleep baseline 113.9 W, so static power is 71 W. SMs with
  sleeping warps cost nothing measurable: the sleep baseline is the same with 1 to 82 such SMs.
- Each working SM adds about 1.1 W for L1 and FFMA, 0.94 W for shared memory, 0.90 W for L2 (whose fit has a +2.8 W
  chip-wide intercept) and 0.50 W for the loop alone (sleep baseline on the same SMs, moved to the probe's temperature;
  means of three runs).
- Repeatability over three `--all` runs on two days (`tools/compare_runs.py Study Study2 Study3`): the fitted energies above have
  a coefficient of variation of 0.5-2.5 %, except `ld` at L2 (5.9 %: 34.5, 38.6, 35.6 pJ/B) and the no-load loop (3.8 %).
  The sleep baseline repeats within 0.02 % (113.9 W), the idle power without a context within 5.8 % (43.0, 38.7, 39.2 W),
  the power per working SM within 0.3-2.3 %. Derived numbers are less stable: datapath shares 1-9 %, and the per-request
  terms of the width model 15-63 % (only the per-byte terms, 3-9 %, are usable).

**Atomics and constant memory** (one run, `Atomics`: `--atomics --constant --duty 100,50,25 --payload random,zero`, 144 of
144 rows clock-matched; fits R² 0.9988-1.0000 except the constant L1.5). Per access of one thread (4 bytes for the
atomics); a warp instruction is 32 of them:

| Probe | pJ per access | nJ per warp instruction | Above the 4-byte store / load of the level | Datapath share |
|---|---|---|---|---|
| L2 `red` | 233 | 7.4 | +39 pJ over `st` (+20 %) | 39 % |
| L2 `atom` | 343 | 11.0 | +162 pJ over `ld` (+89 %) | 48 % |
| DRAM `red` | 1364 | 43.6 | +637 pJ over `st` (+88 %) | 12 % |
| DRAM `atom` | 1449 | 46.4 | +727 pJ over `ld` (+101 %) | 20 % |
| `red.shared` | 57 | 1.83 | (`ld.shared` 16 B: 93 pJ) | 35 % |
| constant L1 (1 KiB) | 30.2 | 0.96 | | |
| constant L1.5 (32 KiB) | 83.5 at full rate, 36.0 at duty 25 | 2.67, 1.15 | | |

- An atomic add in L2 costs a store plus 1.25 nJ per warp; returning the old value adds another 110 pJ per access (3.5 nJ
  per warp). Per L2 sector the model gives 2.39 nJ (`atom`) and 1.52 nJ (`red`), against 0.94 / 0.98 nJ for a read /
  write sector.
- At the DRAM footprint an atomic reads its sector from DRAM and writes it back (CUPTI: 2.0 DRAM sectors per requested
  sector; the 4-byte store writes whole sectors and reads nothing), so DRAM `red` is 1.9x DRAM `st`. DRAM `red` - L2 `red`
  (1131 pJ) matches a DRAM read (`ld` DRAM - L2: 541 pJ) plus a write-back (`st` DRAM - L2: 533 pJ) within 5 %.
- Shared atomics cost 634 pJ per wavefront (a warp's 32 words), loads 405 and stores 331.
- A constant-cache request costs 0.96 nJ per warp with the loop around it (5.44 warp instructions per access); the model
  puts 344 pJ on the request itself. The L1.5 serves only 4.4 G requests/s for the whole GPU (47 G/s from the L1), so at
  duty 50 the probe ran at the same rate as at duty 100 and the fit across duties is not a line; the summary uses the
  full-rate row. There, a request costs 2.67 nJ (1.85 nJ more than an L1 hit, per request the L1 sends to the L1.5); at
  duty 25 only 1.15 nJ. At full rate a miss that finds no free tag is replayed (1.35 lookups per request), so the energy of
  a constant miss depends on how many warps wait for the L1.5.

**Texture fetches** (one run, `Texture`: `--texture --duty 100,50,25 --payload random,zero`, 96 of 96 rows clock-matched,
fits R² 0.9991-0.99999), 16-byte texels against the plain 16-byte load of the same run:

| Level | `tex` pJ/B | `ld` pJ/B | Extra per warp fetch (512 B) | Datapath share |
|---|---|---|---|---|
| L1 | 12.58 | 8.09 | +2.30 nJ (+55 %) | 41 % |
| L2 | 41.83 | 37.87 | +2.03 nJ (+10 %) | 48 % |
| DRAM | 165.8 | 165.2 | +0.33 nJ (within run-to-run noise, about ±3 pJ/B) | 24 % |

- The texture path costs about 2 nJ more per warp fetch than the load path, the same at L1 and L2, so it is energy of the
  path (addressing, filtering and formatting the texels), not of the level. The probe runs 0.56 warp instructions more per
  warp access (the texel index), which accounts for about 0.1 nJ of it.
- CUPTI: one texture-pipe instruction and 16 texture sectors per warp access, 100 % L1 hits at the L1 footprint, one L2
  (and DRAM) sector per requested sector at the others. From L1 the texture path delivers half the load path's rate (393
  vs 765 G accesses/s on all SMs); from L2 and DRAM the rates are the same.

**Access granularity and DRAM locality** (one run, `Locality`: `--dram-locality --duty 100,50,25`, 93 of 93 rows
clock-matched, fits R² 0.89-0.9999), 16-byte accesses, pJ/B:

| Runs placed apart | `ld` at DRAM | `st` at DRAM | Bandwidth (ld / st) | L2 sectors per request |
|---|---|---|---|---|
| stream | 159 | 150 | 920 / 857 GB/s | 4 |
| 32 B | 330 (2.1x) | 348 (2.3x) | 258 / 207 GB/s | 1 |
| 64 B | 243 (1.5x) | 247 (1.6x) | 510 / 397 GB/s | 2 |
| 128 B | 174 (1.09x) | 176 (1.18x) | 923 / 725 GB/s | 4 |
| 512 B - 8 KiB | 157-160 | 151-162 | 919 / 856 GB/s | 4 |

At the L2 footprint, the stream costs 35.9 (ld) and 43.2 (st) pJ/B, 32-byte runs 54.8 and 52.5 pJ/B.

- What costs energy is the size of a request, not the row locality. Runs of whole lines placed apart cost at most 18 %
  more than the stream (2.2 and 2.8 nJ per run for ld and st, fitted over 1 / run bytes), and nothing from 512 bytes on:
  losing row locality at full-line granularity is cheap on GDDR6X, and the bandwidth hardly drops. When every sector is a
  request of its own, the same bytes cost twice as much at DRAM and 1.2-1.5x at L2, and the bandwidth falls to 200-260
  GB/s.
- That is the range of the Transpose configurations the model underpredicted: they read or write one sector per L2
  request (strided columns), while every earlier probe moved 4 sectors per request, so the model put the cost of a request
  into its sectors.
- `dram__cycles_active` is no measure of locality on an RTX 3090: it was exactly 4 per DRAM sector for every pattern here.

**Sub-sector accesses** (one run, `Partial`: `--sub-sector --duty 100,50,25`, 75 of 75 rows clock-matched, fits R²
0.958-0.9998; every thread walks a sector of its own, one 4- or 8-byte word per access), pJ/B against the coalesced 16-byte
access of the level:

| Level, sectors of a warp | `ld` 4 B | `ld` 8 B | `st` 4 B | `st` 8 B | `st` GB/s (4 / 8 B) |
|---|---|---|---|---|---|
| DRAM, adjacent | 161 (0.99x) | 169 (1.03x) | 536 (3.3x) | 214 (1.3x) | 144 / 289 |
| DRAM, scattered | 453 (2.8x) | 445 (2.7x) | 2029 (12.6x) | 983 (6.1x) | 35 / 70 |
| L2, scattered | 81 (2.1x) | 75 (1.9x) | 507 (13.2x) | 207 (5.4x) | 92 / 184 |

- Partial stores are the most expensive accesses measured: every 4- or 8-byte store to a sector of its own is an L2
  write of its own (8 or 4 per sector), about 2 nJ per 4-byte store at L2. Loads of the same pattern cost far less: the
  first word brings the sector into L1, and 62-87 % of the accesses hit there.
- Whether L2 merges the partial writes before DRAM depends on the lines they occupy. With the sectors of a warp adjacent
  (4 per line), 1.2-1.5 DRAM write sectors per sector of data reach DRAM; scattered, every sector holds a line of its own,
  the 84 K threads' lines (11 MB) exceed the 6 MB L2, partial sectors are evicted before they are complete, and 4-byte
  stores write each sector to DRAM 4.8 times (59 % of the write requests miss L2).


**Register-array indexing** (one run, `Indexing`: `--indexing --duty 100,50,25`, 45 of 45 rows clock-matched, fits R²
0.9998-0.9999), per update of one accumulator by one thread:

| Indices | pJ per update | Warp instructions per warp update | Active fma lanes | Rate at duty 100 |
|---|---|---|---|---|
| static (fully unrolled) | 9.85 | 1.34 | 100 % | 8.7 T updates/s |
| runtime (unrolled 4 times, as Coulomb) | 48.55 | 21.2 (17.1 of them fma: `@P IMAD.MOV`) | 18 % | 0.38 T updates/s |

- The compares and guarded moves add 38.7 pJ per update (1.24 nJ per warp update): 4 times the FFMA they serve, but only
  about 73 pJ per warp instruction, a third of an alu or fma instruction, because two thirds of the lanes are guarded off.
  With alu and fma counted in active lanes, the probe is predicted within 2.3 % from the same pipe energies as the
  arithmetic probes.
- The guarded moves keep the probe at 17 W above the baseline at full rate, close to the range where power is less than
  linear in activity (see the limitations); the fit over duties 100/50/25 still has R² 0.9998.

**Bank conflicts** (run `Banks2`: `--bank-conflicts --duty 100,50,25`, 129 of 129 probe rows clock-matched, fits R²
0.983-1.000; wavefronts and replays per warp instruction from `ProfileBanks3`, exactly as designed), in nJ per warp
instruction (32 thread accesses):

| Ways (wavefronts x) | ld 4 B | ld 8 B | ld 16 B | st 4 B | st 8 B | st 16 B |
|---|---|---|---|---|---|---|
| 1 (1, 2 or 4 wavefronts) | 1.43 | 2.05 | 3.27 | 1.69 | 2.32 | 3.69 |
| 2 | 1.45 | 2.13 | 3.68 | 1.71 | 2.86 | 4.13 |
| 4 | 1.75 | 2.68 | 4.42 | 2.06 | 3.38 | 5.05 |
| 8 | 1.91 | 3.27 | 5.97 | 2.59 | 5.25 | 9.12 |
| 16 | 2.05 (duty 100) | 4.15 | | 4.32 | 7.58 | |
| 32 | 2.98 | | | 7.06 | | |

- **A conflict costs time more than energy, at every width.** 32 wavefronts take 16 (4 and 8 bytes) or 8 times (16
  bytes) as long as the conflict-free access, but cost only 1.8-2.1 times its energy for loads and 2.5-4.2 times for
  stores. A replayed wavefront costs 0.02-0.11 nJ for loads (0.04-0.10 at 8 bytes, 0.10 at 16) and 0.02-0.27 nJ for
  stores (0.1-0.2 nJ at most degrees); an extra full wavefront (the conflict-free 8- and 16-byte accesses over the 4-byte one) costs
  0.61-0.63 nJ for loads and 0.64-0.67 nJ for stores. So the energy of a wavefront is mostly the data it moves, and a
  replay moves only the elements of the lanes that waited, wide or narrow. In the energy model a replay costs 62 pJ
  (loads) and 180 pJ (stores); the 30 conflict probes fit within -16 to +14 %.
- **Two ways are almost free for 4-byte accesses.** They run at 64 bytes per clock and SM with one and with two ways
  (half a warp instruction per clock: something other than the banks limits them) and cost the same. The 8- and 16-byte
  accesses run at 128 bytes per clock (one wavefront per clock) without conflicts, so for them every doubling of the ways
  halves the rate, from 2 ways on.
- Stores cost more per replay than loads. The instructions around the access are the same for every degree, so the
  difference is in the store path itself.
- Repeatability: the 4-byte rows of an earlier run (`Banks`) agree within 11 %, except the 16-way load (2.65 nJ there,
  2.05 here, where its fit was not a line and the duty-100 row is used): that degree is not stable.

**Mixed intensity** (run `Mixed`: `--mixed --compute --duty 100,50,25`, 89 of 90 probe rows clock-matched; counters from
`ProfileMixed`: `--mixed --compute --profile --duty 100 ...`). These are 16-byte loads with K FFMAs per load, at duty 100.
"Sum of probes" means the `ld` probe of the level at the load rate plus the FFMA probe of the same run (net of its loop)
at the FFMA rate. "Event model" means `tools/energy_model.py --power Full --counters ProfileFull,ProfileMixed --mixed
Mixed`, which fits the probes of another run (`Full`):

| Level | FFMA per load | GB/s | TFLOP/s | Dynamic power | Sum of probes | Event model |
|---|---|---|---|---|---|---|
| L1 | 0 | 12141 | 0 | 90.9 W | +4.4 % | +4.7 % |
| L1 | 16 | 7009 | 14.0 | 114.4 W | -2.4 % | +0.6 % |
| L1 | 1024 | 186 | 23.8 | 99.3 W | -1.4 % | +3.7 % |
| L2 | 0 | 2067 | 0 | 76.0 W | +2.4 % | +2.9 % |
| L2 | 64 | 1903 | 15.2 | 132.4 W | +0.7 % | +3.3 % |
| L2 | 128 | 1348 | 21.6 | 137.3 W | +0.6 % | +4.2 % |
| DRAM | 0 | 893 | 0 | 143.9 W | +0.4 % | +1.2 % |
| DRAM | 64 | 893 | 7.1 | 172.3 W | +0.6 % | +2.3 % |
| DRAM | 256 | 706 | 22.6 | 215.7 W (power cap) | -4.7 % | +1.7 % |
| DRAM | 1024 | 186 | 23.8 | 133.1 W | -4.9 % | -0.9 % |
| **All 29 clock-matched rows**, mean absolute error per level | | | | | L1 2.7, L2 2.2, DRAM 2.6 % | L1 2.2, L2 4.3, DRAM 1.7 % |

- **The energies of memory and arithmetic add up**, also where both units work at full rate. The sum of the two probes
  has a bias of -1.8 to +1.7 % per level; its largest error is -6.1 % (L1, 512 FFMAs per load). The event model, fitted
  on another run's probes, predicts every row within -1.9 to +8.4 %.
  - The +4.4 % at L1 without FFMAs is the `ld` fit's fixed term: the sum uses the slope.
  - So the 10 % of dynamic energy that the model misses in the example kernels has another cause.
- **Roofline.** At L2 and DRAM the bandwidth holds its maximum until near the balance (16 FFMAs per load at L2, 64 at
  DRAM) and then falls as 1/K. At L1 it falls from 4 FFMAs per load on, because the loads and the FFMAs compete for the
  same issue slots.
  - The FLOP rate reaches 23.8 TFLOP/s, 94 % of the 25.2 TFLOP/s peak at 1200 MHz (82 SMs x 128 lanes x 2 FLOPs).
  - The FFMA probe reaches only 20.7 TFLOP/s, because of its loop around every 64 FFMAs (timer check, duty window).
- **The power limit.** At DRAM with 128 and 256 FFMAs per load, both units run near full rate. The board drew 318 and 334 W
  (1 s average), the most of any probe, and the software power cap (350 W limit) lowered the SM clock below the lock:
  - to 839 MHz on average at 128 FFMAs per load, so that energy is withheld and the row is not in the error summary;
  - to 1179 MHz at 256, within the 2 % tolerance.

  This is the `min(TDP, ...)` term of Antepara et al., and the first probe that reached it. In the example kernels it
  limited one Nbody configuration of `Validation5` (1133 MHz) and three of `Validation2`, too few to explain the 10 %
  that the model misses.
- **Counters.** The row with 0 FFMAs has the counters of the `ld` probe (hit rates, DRAM sectors) and 0.125 more warp
  instructions per load: the check whether to enter the FFMA loop, once per 16 loads. With 1024 FFMAs per load, a warp load
  comes with 1079 warp instructions: 1024 FFMAs, 48 for the loop and 7 for the load.

## Porting to another GPU

An RTX 2080 Ti (Turing, sm_75, 68 SMs) ran the bundle of 2026-11-05 with one counter alias and one L1 override. The
unchanged model ranked its configurations as well as on the RTX 3090 (max regret 0.5 %, against 13.9 % for a ranking by
time). What that port needed by hand, or found only by reading the output closely, is now handled:

| On the 2080 Ti | Now |
|---|---|
| Turing counts fp16 on a pipe of its own, under other counter names | The counter table (`tools/counters.json`) lists both names as alternatives; the program collects both, `check_counters.py` reports "covered" |
| `check_counters.py` failed on LDGSTS, which Turing does not have | A counter with `min_cc` is "not applicable" below it. The compute capability comes from `--cc`, the chip, the run's energy table (runs now record it) or the GPU's name |
| MT4G measured 63.25 KiB of L1, so the 32 KiB footprint was above half of it and every L1 row was labelled "mixed" (fixed with `--l1-kib 64`) | A measured size at most 3 % below a power of two or a multiple of 8 KiB is taken as that size (the 3090's 105.75 KiB is not). The dry run warns when the L1 footprint would not be labelled L1, and when it is more than a third of the L1 |
| The 16-byte write probes need 80-82 registers on sm_75, so fewer than 4 blocks fit per SM, and the placement check failed without saying why | Every row records its registers (KTT's compilation data) and the blocks that fit per SM; the note and `check_run.py` name the register limit. `check_sass.py` prints each variant's occupancy instead of a hand-written exception list, fails on spills, and takes `--sms` for `GRID_STRIPES` |
| KTT's power reading did not settle in 266 of 816 configurations, which then ran to `--max-time-ms`: `Full` took 7.1 h, and a 5 h queue timeout would have killed it | Rows record `power_settled` and `power_spread_pct`; the run's summary and `check_run.py` count them. The dry run prints the timeout a queue must allow (the upper bound plus 10 min) |
| No HMMA probe below sm_80 | HMMA in the GPU's shape: `m16n8k8` on sm_75, `m8n8k4` on sm_70 (ptxas compiles both; not yet measured on such a GPU). From sm_80 the probe is unchanged |
| Check limits were set on the 3090's noise; several fired on plain noise | `check_run.py` prints the run's power noise (the sleeps' sd around their temperature line), does not count a low R² whose rows stray less than 3 times the noise, and scales the tolerance of the conflict order with the noise of the two energies ("too noisy to check" where the noise allows more than 25 %). Every verdict on the 3090 runs is the same as before |
| An event without a probe (on Turing "inst lsu other", with no SHFL probe in that bundle) costs 0 without a word | `energy_model.py` and `score_ranking.py` warn when validation kernels count an event that no probe exercises, with the counts per kernel |
| A placement note with a comma ("... without 4 blocks, 11 with blocks sharing a stripe") was written unquoted, so in the profiled run the CUPTI counters of two write probes (update, 16 bytes, L2 and DRAM) were read one column off: no instructions, "tensor" instructions, no DRAM read sectors. Found on 2026-11-11 | Notes have no commas and the CSV writer replaces any in a text cell; `--analyze` and the Python tools (`counter_table.read_rows`) put the extra cell of an old file back into the note. The 2080 Ti numbers below are recomputed: the two probes fit within -5 and +8 % instead of +51 and -21 % |
| Single event energies differ between GPUs because events that occur together trade off in the fit | `energy_model.py` prints combined costs (a line read from DRAM, a line written back, a conflict-free shared load, an FFMA); `tools/compare_gpus.py` compares two GPUs on the plain probes and these |

Steps for a new GPU:
1. Before building: `tools/check_counters.py --chip <chip>` (what the GPU lacks) and
   `tools/check_sass.py --arch <sm> --sms <SMs>` (what does not compile, spills or fits fewer blocks).
2. An MT4G result, then `--dry-run --all`: the expected levels, the warnings, and the timeout line.
3. The runs (HANDOFF.md of the bundle): a power run, the counter run (`--model --profile --duty 100 --min-time-ms 500
   --max-time-ms 1000`), validation runs.
4. `check_counters.py --csv`, `check_run.py`, `score_ranking.py`, and `fit_busy.py` on a run with `--latency` and
   `--sm-sweep` for the shape of the DRAM busy event (`--busy-scale`).
5. `compare_gpus.py` against the 3090's reference runs.

**A shorter port: the 3090's probe table with the new GPU's counters** (`tools/transfer_gpu.py`). Wattchmen transfers its
table between two V100 systems from 10 % of the microbenchmarks and an affine fit. Tested here between real GPUs, the
RTX 3090's table (`Full,Ffma3,Writes,Instr`) and the RTX 2080 Ti's (`Full,Full2`, 114 probes in both), scored on the 2080
Ti's validation runs with its own counters (`ProfileFull`), without the DRAM busy event (`--no-busy`: its probe times
come from the power runs, which would be the 3090's):

| Probe energies used | Scatter | Bias | Total MAPE | Max regret |
|---|---|---|---|---|
| All measured on the 2080 Ti | 9.3 % | -9.8 % | 5.0 % | 0.5 % |
| The 3090's, unchanged | 10.3 % | -5.6 % | 5.5 % | 0.5 % |
| Per-kind factors from a random 10 % of the probes (mean of 20 draws) | 13.3 % | -10.9 % | 8.2 % | 0.6 % |
| ... from 20 % / 30 % / 50 % | 10.5 / 10.4 / 10.1 % | -12.0 / -13.1 / -12.1 % | 6.4 / 6.3 / 5.8 % | 0.5 % |

- Per probe the transfer helps little: the 2080 Ti's energies are a median 0.91 of the 3090's (L2 0.77, DRAM 0.96), and
  an affine map from a subset predicts the others within a median 12.5-14 %, against 15 % with no map at all. The
  differences are not one scale factor (DFMA 0.58, shared 1.04) and include the noisy probes.
- For the model, the 3090's table used as it is ranks the 2080 Ti's configurations as well as its own measured table
  (regret 0.5 % in both), with a total MAPE of 5.5 against 5.0 %; a transfer from a subset is no better at any size.
  The counts carry the ranking.
- With the busy event, the 3090's probe times paired with the 2080 Ti's counts gave 14.0 % scatter and +10.4 % bias
  (regret still 0.5 %; measured before the CSV repair of 2026-11-11 and not repeated), so leave it out of this short
  protocol.
- On the 3090 against itself (`Preset3`), no transfer is off by the run-to-run noise (a median 2.4 %), and small subsets
  only add noise.
- So for a new GPU: first the counter run (`--model --profile ...`, about 12 min) and the validation runs, scored with the
  reference table unchanged (`score_ranking.py --no-busy --power <3090 runs> --counters <new counter run>`); then a full
  `--model` power run for absolute energies. Only one second architecture (Turing) was tested.

**The 2080 Ti data with the current tools.** The model as the 2080 Ti test ran it (no DRAM busy event): total MAPE 5.0 %,
bias -9.8 %, scatter 9.3 %, max regret 0.5 % (`Full` + `Full2`, validation runs 1 and 2). With the DRAM busy event in its
RTX 3090 shape (95 GB/s): total MAPE 3.6 %, bias -2.7 %, scatter 10.1 %, max regret 0.5 %. The event moves the bias the
right way and widens the scatter by under a point. Its price came out 14.3 W (`Full`; 14.8 W with `Full2`), against
12.8 W on the 3090, and the 2080 Ti runs had no dependent-load probes to measure its shape. (Before the CSV repair of
2026-11-11, with two write probes' counters read one column off, these read 4.4 %, +3.9 %, 11.7 % and 18.9 W.) So measure the shape on every GPU (`fit_busy.py`) before
trusting the scores with the busy event; compare with `--no-busy`.

## Energy over the data (`--payload-bits`)

The probes run on random data (and, with `--payload zero`, on zeros); real data lies in between. `--payload-bits` adds
16-byte loads at the L1, L2 and DRAM footprints, the no-load loop as a control, and FFMA, DFMA and HMMA on operands read
from the payload (`kernels/Operands.cu`), each on payloads with controlled bits (`Payloads.h`). `tools/fit_payload.py`
fits the energy against the payloads' bit statistics, `tools/data_factor.py` applies the result to the validation
kernels. On the RTX 3090, runs `PayloadBits1-4` (each about 29 min; 9 training payloads, 5 held out), figure in
`/home/xmydla/thesis/thesis_material/payload_bits.png`:

| Probe | What explains the energy | R2 (two factors) | rms | Held out | Zero / random |
|---|---|---|---|---|---|
| L2 ld | bits that toggle between words 8 apart (the beats of a 32-byte sector on a 256-bit path) | 0.994 | 1.5 % | -3.3 to +1.2 %, sorted -15.7 % | 0.53 |
| DRAM ld | the same | 0.991 | 1.1 % | -1.6 to +0.9 %, sorted -8.2 % | 0.75 |
| L1 ld | the same | 0.81 | 6.1 % | -2.7 to +4.2 % | 0.60 |
| FFMA | how much neighbouring lanes' operands differ | 0.88 | 5.8 % | -5.1 to +3.0 % | 0.54 |
| DFMA | the same | 0.96 | 5.4 % | -3.2 to -1.5 %, zeroed low bits -11 to -17 % | 0.41 |
| HMMA m16n8k16 | the same | 0.97 | 4.4 % | -3.2 to -0.2 %, zeroed low bits -14 to -18 % | 0.40 |

- The fraction of one bits adds almost nothing once the toggles are in (VAMPIRE found the opposite for DDR3L devices,
  [arXiv 1807.05102](https://arxiv.org/abs/1807.05102); here the whole path from the SM is measured, not a DRAM device).
- The toggles that count are not those between neighbouring words: a word alternating with its complement (`flip:1`,
  every bit toggles between neighbours) costs as little as a constant word, because on a 256-bit path the same bits come
  back every beat (0 toggles between words 8 apart). For the arithmetic, the same holds for lanes: operands that are the
  same in every lane are cheap however they toggle from one instruction to the next.
- Zero data (measured in `PayloadBits4`) agrees with the earlier zero runs (L2 0.50-0.65, DRAM 0.73-0.83 of random).
- The control: at full rate the no-load loop draws 40.6-42.3 W on every payload, while the data moves the L2 load over
  37.9-75.2 W and the DRAM load over 96-144 W. Its fitted energy varies by ±6 %, from the noisier low-duty rows of a
  probe only 33 pJ per access.
- Not covered: stores (the store probes write an XOR recurrence of the payload, which collapses for structured seeds),
  and the bits that sorted data and zeroed low bits toggle in particular positions (they are the worst held-out cases).
- `PayloadBits1-3` were measured before a fix: their "zero" read random data (left out with
  `--skip PayloadBits1:zero,...`), and their `_payloads.json` lack the lagged and lane toggles (computed offline, from the
  same generator, in `Build/x86_64_Release/PayloadBits1-3_stats.json`, for `--stats`).

**Applied to the validation kernels** (`tools/data_factor.py`). The kernels read uniform random floats
(`ValidationKernels.cpp`), whose words have 0.43-0.58 ones and 0.40-0.45 toggles, against 0.5 and 0.5 for random words.
The two-factor fits of the loads give each level's read-sector event a factor (0.87-0.97). The rest of the model is
unchanged, nothing is fitted on the kernels, and the control kernel reads random data (factor 1):

| Over the example kernels (7 validation runs) | Scatter | Bias | Total MAPE |
|---|---|---|---|
| Unchanged model | 5.97 % | +2.57 % | 3.06 % |
| With the data factor | 5.76 % | +0.75 % | 2.52 % |

The held-out Reduction moves from +10.3 to +6.3 % (bias) and 5.3 to 3.2 % (total MAPE) in `HeldOut1`, Convolution from
+4.5 to +1.8 %; PnPoly from -0.9 to -1.1 %. The factor needs each kernel's input statistics, which a tuner would have
to know, so it stays a separate tool, not the default model.

## Measurement limits

What the power readings can and cannot say, against the good practice of Yang et al. for NVIDIA's sensor
([arXiv 2312.02741](https://arxiv.org/abs/2312.02741), measured on RTX 3090s among others):

| Their practice | EnergyProbes | |
|---|---|---|
| At least 32 repetitions or 5 s per measurement | Every configuration runs for at least 10 s (`--min-time-ms`) in launches of 200 ms; validation configurations for 4 s, thousands of launches for kernels of milliseconds | complies |
| Four trials with random delays | One trial per configuration and run; the probes of a group run in random order, a sleep baseline follows every group, and runs are repeated (three `--model` runs: a probe's energy varies by a median 1.8 %; seven validation runs). No random delays between configurations (0.3 s) | partly |
| Drop the rise time (about 250 ms; the 1 s average lags further) | KTT keeps measuring until its samples of the last `--min-time-ms` stay within `--max-power-diff` (2 %) of their mean, which slides the window past the start. Rows record whether that happened (`power_settled`) and the spread over that window (`power_spread_pct`): on the RTX 3090 the settled rows spread by 0.04-2.05 %, while the first second still holds the previous configuration's power | complies |
| Shift the samples by 100 ms against the kernel | Means over at least 10 s of steady state, of which the start is dropped; a 100 ms shift does not change them | not needed |
| Know the sensor's window | `nvmlDeviceGetPowerUsage` is a 1 s average on Ampere; the 100 ms instant field, the 1 s average field and the trace agree with it within 1 % at every duty (`Full.csv`), so windowing does not explain the sublinear readings at low activity | checked |
| Calibrate against an external meter | Not available. Yang found a proportional error of up to ±5 % with a gain and an offset per card | see below |

**KTT's stop rule** (2026-11-06; runs `Stop2a`, `Stop2b`, `Stop05`, `Stop20s`). KTT ends a configuration once its power
samples of the last 10 s stay within 2 % of their mean. 2 % of the 114 W baseline is 2.3 W, against 4-17 W of dynamic
power at low duty, so it might hide ramps. The default probes at duties 100/50/25/10 were measured with 2 %, 0.5 % (at
most 30 s) and 2 % with a 20 s minimum, and 2 % again as the control:

| Against the first 2 % run | Fit energies, median difference | Dynamic power at duty 100 / 50 / 25 / 10 | Settled |
|---|---|---|---|
| 2 % again (control) | 1.9 % | 1.0 / 0.4 / 2.4 / 4.1 % | 52 of 52 |
| 0.5 % | 1.9 % | 1.6 / 0.8 / 3.5 / 3.9 % | 51 of 52 |
| 20 s minimum | 2.2 % | 2.4 / 0.8 / 3.4 / 2.9 % | 52 of 52 |

- Neither a tighter tolerance nor a longer window moves the results beyond what a repeat of the same run does: checked,
  no effect. The low-duty rows differ by 3-4 % between any two runs; that is their noise (a few watts above the
  baseline), not the stop rule.
- Under steady DRAM load the power keeps rising for more than a minute (the `Thermal1` trace, constant DRAM loads):
  250.8 W over seconds 1-11, 254.9 W over 10-20, 258.6 W over 20-40 and 261.9 W over 80-120, 4.3 % of the total and
  about 7.6 % of the dynamic power. The fan (51 to 97 %) explains about 3.7 W of the 11 W and the die temperature about
  1 W; a later heating step started 5.8 W higher, so it depends on what ran before. Constant compute stays within
  0.6 % over 120 s. The 10 s windows of the probes therefore read DRAM-heavy probes a few percent lower than a kernel
  that runs for minutes would draw, and runs with other window lengths are not comparable (3 s windows read a median
  1.9 % and up to 10 % below the 10 s reference).
- On the RTX 3090 the power settles to 0.5 % within 30 s in all but one configuration, and KTT's reading agrees with the
  mean of the 50 ms trace within 0.1 % in every row. All fits stay above R2 0.99. (On the RTX 2080 Ti, a third of the
  configurations did not settle within 2 % at all; `power_settled` shows it there.)

**What a sensor error does to the results.** An offset of the reading cancels in the power above the sleep baseline.
A gain error scales every energy, every event price and every prediction by one factor (up to ±5 % on an RTX 3090 per
Yang), so ratios between probes, the shape of the model, the scatter of its errors within a kernel, the order of
configurations and the regret do not change; absolute pJ values carry that ±5 %. A measurement with an external meter
on the 12 V and 3.3 V rails would remove it: PowerSensor3 (open hardware on a modified riser, 20 kHz,
[arXiv 2504.17883](https://arxiv.org/abs/2504.17883)), NVIDIA's PMD (Yang et al.; it misses the 3.3 V rail, up to 10 W),
or PCAT. The NVML energy counter is no substitute: it covers other rails (0.80-0.96 of the dynamic power on the RTX
3090, 0.38 on an RTX 2080 Ti).

**The baseline over temperature** (`--thermal-curve`, `tools/fit_thermal.py`, 2026-11-06).
- Within a probe run, the sleeps after every group lie on a line: 0.29 ± 0.02 W/K in `Preset2` (rms 0.13 W). Pooled over
  26 power runs with an offset each, the slope is 0.33 W/K (95 % 0.31-0.36). The 0.78 W/K of `Full` came from sleeps that
  spanned 1.4 C.
- The leakage form of Price et al. ([arXiv 1407.8116](https://arxiv.org/abs/1407.8116), a T^2 exp(-b/T) + c) fits no
  better than a line over these spans (rms 0.416 against 0.413 W in a cooling curve), so the line stays.
- The fans draw board power and follow more than the GPU's temperature. Under DRAM loads they ran up to 99 % while the
  GPU cooled from 69 to 65 C (the memory heats, and NVML does not report its temperature on GeForce cards). Over 9626
  sleep samples with the fan at 43-99 %: 0.58 W/K plus 4.8 W x (fan/100)^3, about 1 W at 55 % and 4.7 W at 99 %.
  In `Preset3` the sleeps ran at 53-69 % and DRAM rows up to 99 %, so up to about 4 W of fan power can sit in a DRAM row's
  dynamic power (about 2 % of it). Removing it (the run's baseline plus the fan term) moves the energies by a median
  0.4 % and leaves the scores as they were (scatter 6.2 %, bias +2.9 against +3.3 %, regret 2.4 %), so it is not the
  default. Taking the cooling curve's 0.58 W/K as well lowers the bias to +0.8 % and the scatter to 6.0 %, but that line
  misses the run's own sleeps (rms 0.81 against 0.15 W): the gain is the bias effect of a steeper slope, not a better
  baseline. The cooling curve reached only 61-67 C, because the fans keep the GPU cool under the heaters.

## Known limitations of this version

- Tensor cores: one HMMA shape per architecture (f16 inputs, f32 accumulators). TF32, BF16, INT8 and FP8 (the RTX 3090 has
  no FP8), sparse MMA, and the tensor paths of Hopper (wgmma) and Blackwell (tcgen05/TMEM) have no probe; a kernel that
  uses them gets the warning about events that no probe exercises, or is priced by HMMA's energy per instruction.

- Bank conflicts were measured for loads and stores only; replays of shared atomics (conflicts and same-address
  serialisation) were not.
- Power is linear in activity only above roughly a quarter of a probe's full activity. Below that it rises less than
  linearly: at duties of 15 % and less (see the duty item below), and with only 5-20 of 82 SMs working. FFMA costs 8.4 pJ
  per operation on 82 SMs, 8.2 on 41, 6.7 on 20, 4.3 on 10 and 3.7 on 5, but 6.8 again on 1 or 2 SMs (means of three runs;
  L1, shared and L2 show the same dip). Both ways of lowering the activity show it, and it repeats, so it is not noise; as
  it depends on the total power above the baseline (a few watts), the cause is chip- or board-wide (sensor, voltage
  regulator), not local to an SM. A linear energy model will overestimate kernels with little activity.
- With a CUDA context but no kernel, the driver lowers the memory clock after a while (6.5-7.3 GHz on average over 20 s), so
  the split of the static power into context and SM shares is only printed when the memory still runs at the baseline's
  clock.
- The NVML energy counter on an RTX 3090 measures other rails than the power readings: 14 W vs 43 W idle, 92 vs 114 W for
  the baseline, and 0.80-0.96 of the dynamic power of the probes. All energies use `nvmlDeviceGetPowerUsage` (KTT), which
  equals the instant and the 1 s average power fields within 0.5 %. How much the counter covers depends on the GPU: on
  an RTX 2080 Ti it read 0.38 of the sampled power (`check_run.py` warns, nothing uses it for energies).
- The model counts alu, fma and fp16 instructions in active lanes (`sm__thread_inst_executed_pipe_*_pred_on`): an
  instruction whose guard is false in every lane costs nothing, which the indexing probe supports for `IMAD.MOV` on the
  fma pipe. No probe has guarded-off instructions on the alu pipe (ptxas emitted `@P MOV` there in Coulomb, but
  `IMAD.MOV` in the probe); the other pipes are counted per warp instruction.
- The memory path is still the least accurate part of the model. Example configurations dominated by L2 and DRAM traffic
  are about 5 % too low (11 % before the write probes and the write-back pricing), against 2 % for the others. The fit
  misses the 4-byte DRAM stores and atomics (-10 to -14 %) and the 8-byte partial stores (+41 %). GA102 has no counter
  for write-back requests, so a partial write-back is priced as whole-line sectors (see "Write-backs without a write
  miss").
- FP32 instructions by opcode (`sm__sass_thread_inst_executed_op_*`) need SASS patching, which KTT's CUPTI pass does not
  do: they come back without an error but wrong. The fma pipe can only be split into its heavy and lite halves.
- The model counts a partial write as an L2 write sector and request of its own and has no event of its own for it; it
  fits the sub-sector probes within -17 to +34 % (worst: 8-byte adjacent stores at DRAM). Transpose configurations with
  strided reads or partial writes are still 13-29 % too low in dynamic energy.
- KTT has no texture arguments and does not expose its buffers' device pointers, so the texture probe allocates its own
  copies of the payload and of zeros with the driver API, in KTT's CUDA context (current on the main thread since the
  tuner created it with `cuCtxCreate`), and passes the texture object's handle as a 64-bit scalar. Only 1D linear textures
  with point sampling are measured, not filtering, 2D/3D layouts or CUDA arrays.
- The atomic probes measure atomics without contention (every thread adds to its own word). Atomics of many threads on one
  address are serialized in L2 and cost more time per operation; that case is not measured.
- Constant memory holds random words only: a zero payload would need a second 64 KiB `__constant__` array, more than a
  module may have. Constant loads with compile-time offsets are instruction operands (`c[0x3][...]`) and not measured.
- The constant L1.5 is a shared bottleneck: lowering the duty of each warp does not lower the request rate until few warps
  are left, and the energy per request depends on the contention (2.67 nJ at full rate, 1.15 nJ at duty 25). A duty that
  makes all warps sleep at the same time (as for the arithmetic probes) would measure the full-contention cost at every
  duty. `idc__requests_lookup_hit` is not usable on an RTX 3090 (it is requests minus lookup misses, and replayed misses
  made it negative); the program counts requests sent to the L1.5 (`idc__requests_lookup_miss_tag_miss`) instead.
- Stores at the DRAM footprint include the write-back that L2 performs later; in steady state every stored sector is
  written back once, which the counters check (`DRAM` = 1.002 on an RTX 3090, and 0 at the L2 footprint).
- Stores run 8.9 warp instructions per access against 6.9 for loads (the rotation of the stored words costs moves), at L2 and
  DRAM alike. `st` L2 vs DRAM differences are clean; comparisons of stores with loads or with the no-load loop are not.
- CUPTI on an RTX 3090 (`--profile`): every load and store probe hits its level (sectors per requested sector 1.000 ± 0.002),
  `ld.nc` hits L1 at the L1 footprint, and warp instructions per access are equal across levels for each width (5.44, 5.94,
  6.94 for 4, 8, 16 bytes). The arithmetic probes run 1.17 warp instructions per operation (0.17 of them the loop).
- The SM sweep leaves out the plain DRAM loads: with few SMs only their blocks' stripes are accessed, which would fit in L2.
  With `--latency` it runs the dependent loads at DRAM instead, whose lines are random over the whole footprint.
- The DRAM busy event ("Steady DRAM traffic costs a power of its own") takes a kernel at its average DRAM rate over its
  whole time. A kernel whose traffic comes in phases is counted as if DRAM were awake throughout (Gemm and Convolution
  come out 3-7 % high). Its scale (95 ± 35 GB/s) rests on low-rate rows that are only 1-17 W above the baseline, where
  power readings are sublinear (see the SM-sweep item above).
- The probes run on random 32-bit words (or zeros). Data with fewer toggling bits costs less to move: the same 16-byte DRAM
  load draws 22 % less on zeros, the L2 one 47 % less, and Reduction's floats in [0, 1000) cost about 4 % less per DRAM
  sector than the probes. No counter sees data, so this is a bias per kernel; it hardly changes the ranking of one
  kernel's configurations, which all move the same data.
- HMMA runs from sm_70 in the shape of the GPU (see "Porting to another GPU"); the shapes of sm_70 and sm_75 compile but
  were not measured. The operand probe of `--payload-bits` has HMMA only from sm_80.
- The locality and the sub-sector probe walk, per SM, the footprint divided by the largest power of two in the block
  count (half of it on the RTX 3090's 82 SMs; with 64 or 80 SMs their L2 rows fit in L1, and the program warns). An odd
  step (the sectors of the grid plus one) was tried on 2026-11-12 and taken out again: it moves every warp off the
  128-byte lines, and the 16-byte stream load at the L2 footprint then hit L1 for 71 % of its sectors (11 % before;
  294 instead of 559 pJ per warp access in `Preset4`, whose other rows are a repeat of `Preset3`). A step that keeps
  whole lines and is not a multiple of the grid's sectors is the open fix; it needs a GPU with 64 or 80 SMs to test.
- `--unroll` must be a multiple of 4 with `--writes` (the write probe issues groups of 4; with another value it counted
  accesses it did not issue), of 8 with `--sub-sector` and of 16 with `--mixed`.
- Validation (`--validate`): the example kernels' outputs are not checked; Nbody's space keeps configurations that do
  not compile (4 and 8 of 24 in two runs gave `CUDA_ERROR_INVALID_PTX`), and large Pnpoly and Convolution blocks need
  more registers than a block can have (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`). These failures belong to the
  configuration: of 197 configurations that came up in more than one validation run, none failed once and ran once, so
  a retry fails again. Every launch through KTT is followed by a
  synchronisation and NVML queries; at the sizes used the GPU was busy 100 % of the host time, but for kernels of about
  a millisecond the idle gaps would matter (the dynamic energy assumes they draw baseline power). The counter pass
  re-uploads the buffers for every profiling pass; for the kernels here that did not show in the DRAM counts (Nbody's
  and Coulomb's inputs stay in L2; only spilled registers reach DRAM).
- For the instruction-heavy probes (L1, shared, no-load), power is linear in the access rate only from about 25 % duty
  up. In a sweep over 100/75/50/35/25/15/10/5 % (RTX 3090, 1200 MHz), the fits over 100-25 % have R² 0.998-0.999 and rms
  residuals of 0.7-1.1 W; at 15 % and below, L1 falls up to 2.5 W and shared about 5 W below the line. L2 and DRAM are
  linear over the whole range. The summary's fit therefore only uses duties of at least `--fit-min-duty` (25 %); lower
  duties are still measured and listed.
- Every row contains the burst code, also at 100 % (one `nanosleep 0` per burst). Compared with a kernel without it, this
  raised the full-rate L1 row from 6.9 to 7.8 pJ/B (shared 5.9 to 6.2, no-load 1.9 to 2.1). The cost is the same at every
  level, so differences between levels and the no-load subtraction remove it, but absolute L1 and shared numbers include
  it.
- Energy uses KTT's mean NVML power times mean duration. The energy counter is only used as a cross-check (`ctr [W]`).
- Before Volta there is no `nanosleep`, so the sleep baseline polls like the spin kernel and is too high.
- Pre-Volta GPUs: plain loads skip L1 by default, so use the `ld.ca` row as the L1 number there.
- Only NVIDIA. MT4G also supports AMD, but KTT's power measurement is NVML-only.
