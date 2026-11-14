# EnergyProbes: calibrating a GPU

How to measure a GPU with the probes and build a model from the runs. The tuner's side is in [../README.md](../README.md); the RTX 3090 results are in [FINDINGS.md](FINDINGS.md).

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
| `--lanes` | Also FFMA and integer-add chains with 1, 2, 4, 8, 16, 24 or 32 of a warp's lanes active, the others off by a guard predicate or by a branch (partially active warps; NEXT_STEPS_PLAN B7). ptxas issues about three quarters of the integer adds as IMAD on the fma pipe |
| `--misaligned` | Also the 4- and 16-byte stream load (L1, L2, DRAM) and store (L2, DRAM) started 16, 32, 64 or 96 bytes off the 128-byte line (B8; multiples of 16 only, as the store probe seeds its values with a 16-byte read). The payload is in a buffer of its own and the probe gets a pointer into it, so its SASS is that of the aligned probe |
| `--write-path` | Also 4-byte loads and stores of 1, 2, 4 or 8 words of each sector, with 4, 2 or 1 sectors per L2 request, at the L2 and DRAM footprints (the sub-sector probe with `PARTIAL_WORDS`; B9) |
| `--instr-classes` | Also instruction classes inside one event (B5): LOP3 (xor; ptxas merges pairs of xors), SHF (rotate), IMNMX (with an xor), FADD and FMUL alone, and MUFU RCP, SQRT, EX2, LG2 and RSQ, each with an FFMA that feeds it and an FADD that sums it |
| `--thermal-curve` | Instead of the probes, the sleep baseline's power over temperature (about 15 min): cooling after compute and after DRAM heating, and a staircase of short heat/sleep steps, traced (`tools/fit_thermal.py`) |
| `--counter-table FILE` | The counters to collect with `--profile` and `--validate` (default `tools/counters.json` of this example, which the Python tools read too) |
| `--validate LIST` | Instead of the probes, measure random configurations of KTT example kernels (`transpose`, `gemm`, `nbody`, `coulomb` or `all`; the held-out kernels `reduction`, `convolution`, `pnpoly` or `heldout`): power and time, then CUPTI counters. `mixed` adds the control kernel: the mixed-intensity probe, measured the same way (see [The control kernel](FINDINGS.md#validation-on-example-kernels)) |
| `--tune LIST --model-file FILE` | Instead of the probes, tune example kernels with the energy model: time and counters of random configurations (`--validate-configs`), no power measurement. `--objective energy\|edp\|ed2p\|time`, `--max-slowdown PCT`, `--max-energy-increase PCT`, `--accounting awake\|above-idle\|dynamic` (`--idle-w W`), `--contender-sd X`, `--lenient`. See [Using the model in a tuner](../README.md#using-the-model-in-a-tuner) |
| `--tune-measure contenders\|all` | With `--tune`: after the model's choice, measure the power of the choice, its contenders and the fastest configuration (or of all) with NVML, `--validate-ms` each, and take the measured best |
| `--quick` | The 5-minute run for `tools/check_model.py`: the default probes and `--compute` at duty 100, 5-10 s per configuration |
| `--predict FILE --model-file FILE` | The same choice from a `--validate` run's CSV, without a GPU |
| `--validate-configs N`, `--validate-ms N` | Configurations per example kernel (default 24) and measurement time per configuration (default 4000) |
| `--validate-repeat N` | Measure the power of every configuration N times, in rounds over all kernels with a sleep baseline after each, so that the repeats spread over the run; the CSV gets a `repeat` column and the tools read the first repeat unless asked (B2) |
| `--validate-seed N` | Pick the configurations from this seed (`SeededSearcher`), so that runs measure the same configurations (KTT's random searcher seeds from the system) |
| `--validate-data random\|zero\|small` | The example kernels' inputs: uniform floats (default), zeros, or integers 0 to 15 stored as floats (B12) |
| `--validate-size-scale F` | Gemm's m = n = k (4096) and convolution's image side (16384) times F (B14) |
| `--validate heldout2` | The second held-out set: `bicg` (PolyBench BiCG, fused, atomics) and `ktconvolution` (Kernel Tuner's convolution on ClTuneConvolution's task) |
| `--validate heldout3` | The third held-out set (2026-11-14): PolyBench/GPU's `gesummv`, `syrk`, `syr2k`, `conv2d`, `conv3d` with small tuning spaces (`kernels/HeldOut3.cu`) |
| `--l1-sweep` | Loads and stores at the L1 footprint with 1-32 sectors in 1, 4 or 32 lines per warp instruction, 4- and 16-byte words, every duty (M2) |
| `--mixed-int` | 16-byte loads with 0, 4, 8, 16 integer instructions (add, shift, and) per load at L1 and L2, duty 100 (M2; into the fit with `--full-rate`) |
| `--instr-classes` (2026-11-14) | also integer and float compare + select, int-float conversions (around an FADD), float min/max, IADD3 (M4) |
| `--lanes` (2026-11-14) | also 0 active lanes and the FFMA with rotating lanes (M5) |
| `--study-counters` | With profiling: also the screening counters of `tools/counters.json` (`"study": true`: L1 requests and misses, L1-L2 traffic, L2 tags, issued instructions, SASS counts per opcode class) |
| `--thermal-load`, `--thermal-hold S` | Instead of the probes: power over temperature under load (M1): five probes at every `--duty` and the sleep baseline, each held S s (90) after 60 s of sleep and after 60 s of DRAM stores (`tools/fit_thermal_load.py`) |
| `--tune-scale-configs N` | `--tune-measure`: the kernel's scale from N configurations spread over its predicted dynamic energy (default 5 with more than one kernel) |
| `--validate accelwattch --accelwattch-dir DIR` | AccelWattch's microbenchmarks (Kandiah et al., MICRO 2021) as one kernel with a configuration per benchmark (`AW_BENCH`); DIR is what `tools/accelwattch_import.py` wrote from the published sources (not in this repository). Use `--validate-configs` at least the number of benchmarks |
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
| `tools/fit_stability.py` | How much the kernels' predictions depend on the probe set: prices' CV over resamples within families, per kernel the sd of its mean and of its within-kernel differences, each family left out (P4) |
| `tools/fit_thermal_load.py` | Power over temperature per probe and duty from a `--thermal-load` run, up and down, and the three forms of the excess over the baseline's slope (M1) |
| `tools/screen_counters.py` | Every screening counter in turn as an extra event: the fit on the hard probes, the design kernels' scatter (M3) |
| `tools/energy_search.py` | Which configurations to profile next: a surrogate on the tuning parameters, fitted on the profiled configurations, ranks the others by the model's energy (a counter-guided search with an energy objective); `--evaluate` compares it with random picks on validation runs |
| `tools/accelwattch_import.py` | AccelWattch's microbenchmarks as one KTT kernel file for `--validate accelwattch`: kernels extracted, scalar arguments as constants, `--set` for command-line values, iterations scaled from a timing run |
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
