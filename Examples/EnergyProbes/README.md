# EnergyProbes

An energy model for NVIDIA GPUs that lets a tuner pick the kernel configuration that uses the least energy **without
measuring power for every candidate**. The model prices a configuration from its CUPTI counters and its time:

    energy = baseline power x kernel time + sum over events (count x energy per event)

An *event* is a warp instruction of a pipe, a sector served by L1, L2 or DRAM, an L2 request, a shared-memory
wavefront, a constant-cache request, or a nanosecond of DRAM kept awake. The energies per event come from *probes*:
small kernels that keep one part of the GPU busy while KTT measures the board power through NVML.

It is a calibrated, counter-based energy ranking at a fixed clock, not a physical model of each operation: the
prices are statistical proxies fitted to whole-board NVML power. On an RTX 3090 locked at 1200 MHz:

- **Frozen evaluation** (`heldout4`: five PolyBench kernels that calibrate nothing, written before the model change of
  2026-11-17, measured and scored once): the model's choice uses at most 1.2 % more energy than the measured best
  (picking the fastest: up to 4.8 %); within-kernel scatter 2.5 %. Different kernels' biases spread from -15 to +9 %.
  The earlier frozen set (`heldout3`, scored with the previous model: regret 0.9 %, biases -21 to +32 %) is now reused.
- **Kernels from other KTT examples** (`heldout5`: ATF's convolution, Sort's and Sort2's radix-sort kernels, 140
  configurations, scored once with the same model): regret at most 0.9 % (time only: 5.6 %), power error 5.5 %; all
  four are over-predicted by 12-16 % of their dynamic energy.
- **Development and calibration kernels** (the nine example kernels the model was designed and calibrated on, 511
  configurations, scored at the model's clock): regret up to 0.5 %, scatter 6.2 %. These are not independent.
- **Model change of 2026-11-17 (later): inactive lanes.** The shipped model (SHA-256 `2d90caf0...`) adds the event
  "inst inactive lanes" (warp instructions minus thread instructions / 32, from the counter
  `smsp__thread_inst_executed.sum`), priced by the new `lanes` probes: a warp instruction with one active lane costs
  about a third of a full one. Its scores on the recorded runs (counted again with the new counter, power as measured)
  are those of the model before, within 0.3 points: the 19 kernels of `AllMany2` total MAPE 3.0 %, max regret 2.7 %;
  heldout5 max regret 0.9 %. The heldout4 and heldout5 numbers above are therefore of the model before (`09b36a64`);
  any score of the new model on them is a reused evaluation.
- **RTX 2080 Ti** (Turing): the program ran unchanged; its own model picked the measured best in 7 of 9 kernels, with
  a wider scatter (10.6 %).

Details, the data ledger and limits are below and in [docs/PROBES.md](docs/PROBES.md).

## Build

KTT is built with [Premake](https://premake.github.io) (5.0.0-beta1 runs on old C libraries; newer releases need
glibc 2.29 or later). The power path needs NVML and CUPTI, so build with a profiling backend:

```bash
export CUDA_PATH=$(dirname $(dirname $(readlink -f $(which nvcc))))
./premake5 gmake --power-usage --profiling=cupti --no-opencl
cd Build && make config=release_x86_64 -j$(nproc) EnergyProbesCuda
```

The binary is `Build/x86_64_Release/EnergyProbesCuda`; run it from that directory (it finds the kernels of this and
of the other examples relative to it). The Python tools need NumPy and SciPy, except `energy_predictor.py` (standard
library only).

## Calibrating a GPU (once per GPU type and clock, about 80 minutes)

Energy per operation depends on voltage, which follows the clock, so **lock the SM clock**
(`sudo nvidia-smi -i <gpu> -lgc F,F`, then `--clock-locked`; or run as root with `--lock-clock F`). Pick F below the
lowest clock the GPU reaches under load. `--expect-clock SM[:MEM]` (MHz; implied for the SM clock by `--lock-clock`)
makes the program compare the clocks NVML reports under load during the warm-up with the ones set, and warn if one is
off by more than a clock step: NVML lists clock pairs some GPUs do not apply (Fan et al., ICPP'19). Nothing else may
run on the GPU. On machines with several GPUs set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so that KTT reads the power of the GPU it runs on (the program checks this).
The cache sizes come from an [MT4G](https://github.com/caps-tum/mt4g) result file (`models/` has the RTX 3090's and the RTX 2080 Ti's).

```bash
G="--device 0 --clock-locked --mt4g ../../Examples/EnergyProbes/models/NVIDIA_GeForce_RTX_3090.json"
./EnergyProbesCuda $G --dry-run                                           # the plan, no GPU needed
./EnergyProbesCuda $G --output Probes                                     # power of every probe      47 min
./EnergyProbesCuda $G --profile --output Profile                          # CUPTI counters per probe   8 min
./EnergyProbesCuda $G --validate all,heldout,heldout2 --output Validation # 9 example kernels         24 min
./EnergyProbesCuda $G --validate heldout5 --validate-configs 100 --output KttMany1   # 4 more examples   25 min
./EnergyProbesCuda $G --validate transpose,atfconv --validate-data zeros --output Zeros # input data zeros (or random)
./EnergyProbesCuda $G --only lanes --output Lanes && ./EnergyProbesCuda $G --only lanes --profile --output LanesProfile  # partly idle warps
./EnergyProbesCuda $G --recount Validation_validation.csv --output ValRecount # counters again, no power
./EnergyProbesCuda $G --only opcodes --output Opcodes && ./EnergyProbesCuda $G --only opcodes --profile --output OpcodesProfile  # single SASS opcodes
./EnergyProbesCuda $G --only banks,broadcast --output Broadcast && ./EnergyProbesCuda $G --only banks,broadcast --profile --output BroadcastProfile  # broadcast shared loads
./EnergyProbesCuda $G --only reuse --output Reuse && ./EnergyProbesCuda $G --only reuse --profile --output ReuseProfile  # operand reuse cache
./EnergyProbesCuda $G --only global,compute,shared,mix,saturate --output Mix && ./EnergyProbesCuda $G --only global,compute,shared,mix,saturate --profile --output MixProfile  # mixed work
T=../../Examples/EnergyProbes/tools
python3 $T/check_run.py Probes --counters Profile                         # is the measurement sound?
python3 $T/build_model.py --power Probes --counters Profile --calibration Validation --output my_gpu.model.json
python3 $T/energy_model.py --power Probes --counters Profile --validation Validation   # the fit and its scores
python3 $T/merge_counters.py Validation_validation.csv ValRecount_recount.csv Validation2_validation.csv  # new counters, same power
python3 $T/fit_busy.py --power Probes --counters Profile                  # the DRAM busy scale this GPU allows
python3 $T/opcode_energy.py --power Opcodes --counters OpcodesProfile --model my_gpu.model.json --out my_gpu.opcodes.json
```

- The **probe run** measures 265 configurations (171 probes): every probe at duty 100 % and 25 % (the share of time it
  issues accesses), 10 s each once the power reading is stable, with a sleeping kernel as the baseline after every
  group. The energy per access is the slope of power over access rate across the duties.
- The **counter run** collects what each probe executes. Counts are a property of the code, so it uses short windows.
- The **validation run** measures power, time and counters of random configurations of nine kernels of other KTT
  examples. `build_model.py` takes the model's uncertainty from it.

A model built on one card can be checked on another card of the same type in 4 minutes:
`./EnergyProbesCuda $G --quick --output Quick` and `python3 $T/check_model.py my_gpu.model.json Quick` (PASS, SCALED
with a common factor that `--write` applies, or FAIL).

On a GPU of another type, the same 4-minute run gives a stand-in model until the GPU is calibrated (after Wattchmen,
which transfers its energy table from a fraction of its benchmarks): `check_model.py --model rtx3090_1200mhz.model.json
--run Quick --other-gpu "NVIDIA GeForce RTX 2080 Ti" --write rtx2080ti_transferred.model.json` scales the energies of
each part (compute, L1, shared memory, L2, DRAM) by the median ratio of that part's probes, and takes the run's baseline
and idle powers. On the 2080 Ti's validation run this transferred model had a total error of 4.7 % (its own calibration:
4.5 %; the 3090 model with only the 2080 Ti's baseline: 7.2 %; one factor for everything: 6.2 %) and the same max regret
(0.9 %). Its dynamic energies are rougher (median bias over the kernels +10.5 % against +3.2 %), so calibrate before
comparing different kernels.

## Using the model in a tuner

```bash
# the configuration to use, from a KTT output file (JSON) of any tuning run with CUPTI profiling
python3 tools/energy_predictor.py --model models/rtx3090_1200mhz.model.json --ktt TuningResults.json --objective energy
# the same inside a KTT program, demonstrated on the example kernels (time and counters only, no power measurement)
./EnergyProbesCuda --tune gemm,coulomb,transpose --model-file ../../Examples/EnergyProbes/models/rtx3090_1200mhz.model.json
```

`EnergyTuner.h` holds the passes for any simple KTT kernel: a timing pass (the searcher's configurations, without
profiling), a counter pass (each once more with the counters the model needs) and, optionally, NVML's power of chosen
configurations. `EnergyModel.h` is the predictor (standard library and nlohmann::json, no KTT types in its interface).
`example/MinimalTuner.cpp` is the smallest program that uses both:

```cpp
EnergyModel model(argv[1]);
EnergyTuner energy(tuner);                                  // tuner: a ktt::Tuner with the kernel and its space
energy.AddKernel("saxpy", kernel, definition);
tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
energy.TimeConfigurations("saxpy", 25);
std::vector<TunedConfiguration> rows = energy.CollectCounters(model.Counters());
// per row: model.Predict(row.counters, row.kernelS, row.temperatureC, row.smClockMHz, true, row.kernel)
const EnergyRecommendation choice = model.Recommend(predictions, EnergyObjective::Energy);
```

What a tuner should know:

- **Objectives**: energy, EDP, ED2P or time, optionally only among configurations at most x % slower than the
  fastest (`--max-slowdown`) or at most x % above the lowest energy (`--max-energy-increase`).
- **The energy-time trade-off.** Next to the choice, `--tune` and `--predict` print the Pareto front: the
  configurations that no other one beats in both time and predicted energy. Only the points the model tells apart
  are listed (each saves more than 2 sd of energy against the next faster one), with their time and energy against
  the fastest; the CSV column `pareto` marks them `step` and the other points of the front `front`. At the fixed clock
  the fronts are short: on AllMany2 1-6 points per kernel, and 3 of the 19 kernels have a step beyond the fastest
  configuration (syr2k: +0.5 % time for -18 % energy). All those steps were on the measured front.
- **Pruning the counter pass** (`--tune`, default on; `--no-prune` turns it off). The predicted energy is at least the
  baseline over the kernel's time, as no event has a negative price. The counter pass goes fastest first and leaves
  out a configuration whose objective is, from its KTT time alone, more than a margin above the best predicted so far.
  The margin (23.6 % for this model) is what the model's error could hide, so a pruned configuration can be neither
  the choice nor a contender (`ContenderMargin`, `CounterPruning` in `EnergyTuner.h`). KTT's time is taken as at most
  1.5 ms more and 15 % less than the kernel's own (on 1,135 recorded configurations of 23 kernels the kernel's time
  was never below 1.18 times this bound). On recorded runs the rule leaves out about a quarter of the configurations
  (none of kernels shorter than a few ms) and never the choice or a contender. A `--tune` of gemm, coulomb and atfconv
  (80 random configurations each, runs Prune1 / NoPrune1) left out 78 of 240 and took 197 s against 285 s without
  pruning; on the run without, the rule would have left out 81 of 228 and again never the choice or a contender. It is
  off with `--tune-measure all`, and with `--max-energy-increase` unless the objective is energy. A kernel's scale
  (`--tune-scale-configs`) then comes from the profiled configurations only.
- **Uncertainty and contenders.** The model's relative error of the dynamic energy differs between the configurations
  of one kernel with a standard deviation of 6.1 % (the model file stores it). Two configurations whose predictions
  are closer than 2 sd of their difference are *contenders*: the model's first choice stands, and the contenders are
  what to measure with NVML if the last percent matters (`--tune-measure contenders`). On the calibration run
  83 % of all pairs are told apart at 2 sd and 99.97 % of those are ordered right; the rest still in 82 %.
  A kernel can have many contenders (Bicg: 39 of 250 random configurations); `--measure-max N` measures the choice
  and only the N - 1 best predicted contenders. On the random sweep of 2026-11-18 (8 kernels, 250 KTT-random
  configurations each) the measured best was the model's first choice in 5 kernels, among its 5 best in 7 and among
  its 10 best in all 8 (handoff/sweep_2026-11-18).
- **Comparing different kernels** (two implementations of one task) is less certain, because the model's bias
  differs from kernel to kernel (sd 5.6 % over the nine kernels in the calibration run; -5 % for Transpose and Nbody
  to +13 % for Bicg). Give each prediction its kernel as the group;
  pairs from different groups are then compared with the wider sd. Better: `--tune-scale-configs N` measures N
  configurations per kernel, spread over its predicted energy, and scales the kernel's predictions by the mean of
  measured / predicted.
- **Which baseline belongs in the energy** (`--accounting`): `awake` (default; kernels run back to back),
  `above-idle` (the GPU idles with a CUDA context between runs of the work) or `dynamic` (no baseline).
- **Power limit.** A configuration predicted above the GPU's power limit gets the limit times its time (the GPU lowers
  its clock to hold the limit), as in Antepara et al.'s P = min(TDP, ...). `--tune` takes NVML's limit, `--predict`,
  `energy_predictor.py` and `energy_breakdown.py` the model file's (350 W on the RTX 3090; `--power-limit W`, 0 = none).
  The breakdown then scales every part by the same factor, so the parts still add up. At the locked 1200 MHz only the
  heaviest DRAM mixes reach the limit. On an unlocked GPU the clock moves below the limit too, and the 1200 MHz energies
  per event are too low there (ValFree1: -22 % at the limit); one factor for the dynamic energy fitted there (x 1.49)
  halves the error but ranks no better than time, and there is no second unlocked run to test it on, so it is not used.
- **Inputs are checked.** Both predictors refuse a model of a newer version, an unknown event, a busy scale that is
  not a positive number, and counters the model needs but does not get (they would count as 0 and make a kernel look
  cheap); `--lenient` turns the last into a warning. A kernel that executes events no probe prices (texture fetches)
  gets a warning.
- **One GPU type per model.** `--tune` (and `energy_predictor.py --ktt` when the KTT output names the GPU) refuses a
  model calibrated on another GPU type; `--allow-uncalibrated-device` overrides it. Another card of the same type is
  checked with `check_model.py`.
- **Cost.** Timing and profiling take about 1.2 s per configuration, against about 5-10 s for a power measurement.

## Whole applications and opcodes

- **Whole applications** ([docs/APPLICATIONS.md](docs/APPLICATIONS.md)): `tools/app_energy.py` measures applications
  with NVML and predicts their energy from the kernel model (counters from Nsight Compute, times from Nsight Systems)
  plus what the GPU does without work. On this RTX 3090:
  - the GPU sleeps at 42 W even with a context;
  - it wakes at a process's first CUDA call;
  - it stays awake at about 103 W for up to 17.6 s without work, or about 3 s after the process exits.

  On eight Rodinia applications, none of them used for the model, the process energy is predicted within -4.5 to
  +4.6 % (mean 1.9 %), with the prediction passed through NVML's averaging as the measurement is (without that filter
  +1.1 to +10.8 %, mean 3.6 %). The same energies in Antepara et al.'s form (idle power plus energy per byte and per FLOP) miss
  by 24-61 %, because they lack the awake GPU's static power.
- **Opcodes within a pipe** ([docs/PROBES.md](docs/PROBES.md), "Opcodes within a pipe"): the optional `opcodes` probes
  price 19 SASS opcodes against their pipe (`tools/opcode_energy.py`, `models/rtx3090_1200mhz.opcodes.json`).
  - Within a pipe opcodes differ by up to 2x: DADD and DMUL cost half the DFMA the fp64 price was fitted on, and F2F
    about a tenth.
  - Constant operands cut an FMA's energy to 0.43-0.46.
  - The table needs per-opcode counts, which only Nsight Compute has (not KTT's counter pass). It is an option of
    `app_energy.py score` (mean error 3.6 to 3.2 %; hotspot 6.3 to 3.0 %), not part of the model.
- **Broadcast shared loads** ([docs/PROBES.md](docs/PROBES.md), "Broadcast shared loads"): the optional `broadcast`
  probes let 2, 16 or 32 lanes read one shared word. Pairs cost as much as distinct words; 16 or 32 lanes per word
  cost about 17 % less than the model gives. No counter tells all cases apart, so they stay checks.
- **The operand reuse cache** ([docs/PROBES.md](docs/PROBES.md), "The operand reuse cache"): the optional `reuse`
  probes run FFMAs on the same values from a register per chain or from one shared register, which ptxas reads from
  the operand reuse cache. Caching one operand saves 15 % per FFMA (and issues 33 % faster); caching a second saves
  nothing more, so it is not a price per register read. Corrections for it made the GEMM scatter worse; no event.
- **Instruction coverage** ([docs/APPLICATIONS.md](docs/APPLICATIONS.md)): `app_energy.py score` prints the share of
  executed warp instructions a probe of their own measured (26-77 % on the Rodinia applications, 34-95 % with the opcode
  table), the share priced only through their pipe (mostly loop and index instructions), and the share in a pipe no
  probe exercises (texture, Hopper/Blackwell wgmma, tcgen05, TMA: 0 % here). Needs Nsight Compute's per-opcode counts.
- **lud's +11 %** ([docs/APPLICATIONS.md](docs/APPLICATIONS.md)) was mostly NVML, not the model. NVML holds each reading
  0.5 s on top of its 1 s window, so energy at the end of a process lands after the split at the exit. lud ends on its
  kernels at about 226 W, so it lost the most. Through the same filter lud is +2.0 %. Its input data, broadcast loads
  and temperature were ruled out on the way; its kernel on its own (`--validate diagnose`) is +9 % in dynamic energy.

## Limits

- **A model is valid at the clock it was calibrated at.** On a GPU that is not clock-locked and runs at its power
  limit, the clock moves from one configuration to the next and energy is close to the limit times the time: rank
  by time there (both predictors say so when most configurations ran at another clock). The model earns its place
  where the GPU is below its power limit and the clock is constant, which is where configurations of equal time
  differ in energy.
- **Kernels unlike the calibration kernels.** The bias between kernels comes from nine kernels. On the five frozen
  PolyBench kernels the ranking inside each kernel held (regret at most 0.9 %), but the biases spread wider (-21 to
  +32 %, sd 19.6 % against 3.4 %). Measure a few configurations for a kernel's scale before comparing it with other
  kernels.
- **Data.** The probes run on random 32-bit words. Data with fewer toggling bits costs less to move (a DRAM load of
  zeros draws 22 % less), and an FMA with constant operands costs less than half; no counter sees data. This is a bias per kernel; it hardly changes the ranking of one
  kernel's configurations, which all move the same data.
- **Low activity.** Power is linear in the access rate only from about a quarter of a probe's full rate up; a linear
  model overestimates kernels that keep the GPU mostly idle.
- **Not priced:** texture fetches (KTT has no texture arguments), contended atomics, tensor-core shapes other than
  the one HMMA shape per architecture, constant loads with compile-time offsets. IMMA has a price of its own (36 % above
  HMMA on the RTX 3090); the optional `--only tensor` probes (DMMA, sparse HMMA, FP8) check how far the HMMA price
  carries; Hopper's WGMMA, Blackwell's `tcgen05` and
  tensor memory, TMA and distributed shared memory have counters but no probe ([docs/PROBES.md](docs/PROBES.md),
  "Newer architectures").
- **One model per GPU type.** The 3090's energies on the 2080 Ti rank as well but are 19.5 % high. On the 2080 Ti the
  scatter within a kernel is about 10 % and the kernels' biases spread from -14 to +24 % (3090: -5 to +6 %).
- **Phases.** The DRAM busy event takes a kernel at its average DRAM rate over its whole time; a kernel whose traffic
  comes in phases is priced a few percent high.
- Only NVIDIA (KTT's power measurement is NVML), Volta or newer for the sleep baseline, bare-metal Linux (NVML power
  readings are not reliable in WSL2 and many VMs).

## Files

| File | What it holds |
|---|---|
| `EnergyProbes.cpp`, `Options.*` | `main` and the command line (`--help` lists every option) |
| `ProbePlan.*` | The table of probe groups: what a probe run measures |
| `ProbeKernels.*`, `kernels/*.cu` | The probes as KTT kernels: sources, arguments, launcher, the row of a result |
| `ProbeRun.*`, `GpuSession.*`, `NvmlMonitor.*` | The run: order, sleep baselines, idle readings, NVML trace and clock lock |
| `Hardware.*` | Cache sizes from KTT and MT4G, the footprint of each memory level |
| `Rows.*`, `Analysis.*`, `Report.*` | Rows and their CSV, the baseline's temperature model, energies, fits, the energy table |
| `Validation.*`, `ValidationKernels.*` | `--validate`: the example kernels' tuning spaces, measured through `EnergyTuner` |
| `EnergyModel.*`, `EnergyTuner.*`, `Tune.*` | The predictor, the tuner's passes, and `--tune` / `--predict` |
| `CounterTable.*`, `tools/counters.json` | The CUPTI counters, one table for the program and the tools |
| `tools/energy_events.py` | The events and how they are counted from counters (shared by the fit and the predictor) |
| `tools/energy_model.py`, `tools/build_model.py` | The fit and its scores (at the model's clock; `--bootstrap N`); the model file |
| `tools/fit_busy.py` | The DRAM busy event's scale on this GPU (a scan of the probe fit) |
| `tools/energy_breakdown.py` | Predicted against measured power and energy per configuration of `--validate` runs, and where the energy goes (baseline, compute, issue, L1/shared, L2, DRAM) |
| `tools/energy_predictor.py` | Energy and recommendation from counters and time (also from a KTT output JSON) |
| `tools/check_run.py`, `tools/check_model.py`, `tools/check_sass.py` | Checks of a probe run, of a model on a GPU, of the probes' machine code |
| `tools/check_chip_metrics.py` | The counter table against a chip's metric list (`ncu --query-metrics --chip`), before the first run on it |
| `tools/app_energy.py`, `apps/` | Whole applications: the application probes (`AppProbes.cu`), the Rodinia build (`build_rodinia.sh`, `device_select.cpp`, `make_inputs.py`, `rodinia.json`), measurement, profiling and prediction ([docs/APPLICATIONS.md](docs/APPLICATIONS.md)) |
| `tools/opcode_energy.py` | Energy per SASS opcode from the optional opcode probes, against the model's pipe prices |
| `models/` | The RTX 3090 and RTX 2080 Ti models (1200 MHz) and the MT4G files they were calibrated with; for the RTX 3090 also the application model (`.app.json`) and the opcode table (`.opcodes.json`) |
| `tests/` | `python3 -m pytest Examples/EnergyProbes/tests -q` from the KTT root: no GPU, a few seconds |

**Checking a change.** For the kernels, `python3 tools/check_sass.py --against <kernels/ before the change>` must
report identical SASS, or the measured numbers no longer describe the code. For the analysis, `--analyze` of an
earlier run before and after the change shows every number that moved.
