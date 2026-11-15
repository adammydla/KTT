# EnergyProbes

An energy model for NVIDIA GPUs that lets a tuner pick the kernel configuration that uses the least energy **without
measuring power for every candidate**. The model prices a configuration from its CUPTI counters and its time:

    energy = baseline power x kernel time + sum over events (count x energy per event)

An *event* is a warp instruction of a pipe, a sector served by L1, L2 or DRAM, an L2 request, a shared-memory
wavefront, a constant-cache request, or a nanosecond of DRAM kept awake. The energies per event come from *probes*:
small kernels that keep one part of the GPU busy while KTT measures the board power through NVML.

On an RTX 3090 at a locked SM clock, over 9 example kernels (8 validation runs, 906 configurations), the
configuration the model picks uses 0.2 % more energy than the measured best on average and 2.4 % at most; picking the
fastest configuration instead costs 2.5 % on average and up to 20.8 %. Details and limits are below and in
[docs/PROBES.md](docs/PROBES.md).

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
lowest clock the GPU reaches under load. Nothing else may run on the GPU. On machines with several GPUs set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so that KTT reads the power of the GPU it runs on (the program checks this).
The cache sizes come from an [MT4G](https://github.com/caps-tum/mt4g) result file (`models/` has the RTX 3090's).

```bash
G="--device 0 --clock-locked --mt4g ../../Examples/EnergyProbes/models/NVIDIA_GeForce_RTX_3090.json"
./EnergyProbesCuda $G --dry-run                                           # the plan, no GPU needed
./EnergyProbesCuda $G --output Probes                                     # power of every probe      47 min
./EnergyProbesCuda $G --profile --output Profile                          # CUPTI counters per probe   8 min
./EnergyProbesCuda $G --validate all,heldout,heldout2 --output Validation # 9 example kernels         24 min
T=../../Examples/EnergyProbes/tools
python3 $T/check_run.py Probes --counters Profile                         # is the measurement sound?
python3 $T/build_model.py --power Probes --counters Profile --calibration Validation --output my_gpu.model.json
python3 $T/energy_model.py --power Probes --counters Profile --validation Validation   # the fit and its scores
```

- The **probe run** measures 242 configurations (150 probes): every probe at duty 100 % and 25 % (the share of time it
  issues accesses), 10 s each once the power reading is stable, with a sleeping kernel as the baseline after every
  group. The energy per access is the slope of power over access rate across the duties.
- The **counter run** collects what each probe executes. Counts are a property of the code, so it uses short windows.
- The **validation run** measures power, time and counters of random configurations of nine kernels of other KTT
  examples. `build_model.py` takes the model's uncertainty from it.

A model built on one card can be checked on another card of the same type in 4 minutes:
`./EnergyProbesCuda $G --quick --output Quick` and `python3 $T/check_model.py my_gpu.model.json Quick` (PASS, SCALED
with a common factor that `--write` applies, or FAIL).

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
- **Uncertainty and contenders.** The model's relative error of the dynamic energy differs between the configurations
  of one kernel with a standard deviation of 5.8 % (the model file stores it; 6.1 % over all eight validation runs). Two configurations whose predictions
  are closer than 2 sd of their difference are *contenders*: the model's first choice stands, and the contenders are
  what to measure with NVML if the last percent matters (`--tune-measure contenders`). On the calibration run
  84 % of all pairs are told apart at 2 sd and every one of those is ordered right; the rest still in 78 %.
- **Comparing different kernels** (two implementations of one task) is less certain, because the model's bias
  differs from kernel to kernel (sd 3.4 % over the nine kernels in the calibration run, 4.7 % over all eight runs;
  -4 % for Transpose to +12 % for Bicg). Give each prediction its kernel as the group;
  pairs from different groups are then compared with the wider sd. Better: `--tune-scale-configs N` measures N
  configurations per kernel, spread over its predicted energy, and scales the kernel's predictions by the mean of
  measured / predicted.
- **Which baseline belongs in the energy** (`--accounting`): `awake` (default; kernels run back to back),
  `above-idle` (the GPU idles with a CUDA context between runs of the work) or `dynamic` (no baseline).
- **Inputs are checked.** Both predictors refuse a model of a newer version, an unknown event, and counters the
  model needs but does not get (they would count as 0 and make a kernel look cheap); `--lenient` turns the last into
  a warning. A kernel that executes events no probe prices (texture fetches) gets a warning.
- **Cost.** Timing and profiling take about 1.2 s per configuration, against about 5-10 s for a power measurement.

## Limits

- **A model is valid at the clock it was calibrated at.** On a GPU that is not clock-locked and runs at its power
  limit, the clock moves from one configuration to the next and energy is close to the limit times the time: rank
  by time there (both predictors say so when most configurations ran at another clock). The model earns its place
  where the GPU is below its power limit and the clock is constant, which is where configurations of equal time
  differ in energy.
- **Kernels unlike the calibration kernels.** The bias between kernels comes from nine kernels. On five PolyBench
  kernels outside the calibration the ranking inside each kernel held (regret 0), but the biases spread wider
  (-14 to +35 %). Measure a few configurations for a kernel's scale before comparing it with other kernels.
- **Data.** The probes run on random 32-bit words. Data with fewer toggling bits costs less to move (a DRAM load of
  zeros draws 22 % less), and no counter sees data. This is a bias per kernel; it hardly changes the ranking of one
  kernel's configurations, which all move the same data.
- **Low activity.** Power is linear in the access rate only from about a quarter of a probe's full rate up; a linear
  model overestimates kernels that keep the GPU mostly idle.
- **Not priced:** texture fetches (KTT has no texture arguments), contended atomics, tensor-core shapes other than
  the one HMMA shape per architecture, constant loads with compile-time offsets.
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
| `tools/energy_model.py`, `tools/build_model.py` | The fit and its scores; the model file |
| `tools/energy_predictor.py` | Energy and recommendation from counters and time (also from a KTT output JSON) |
| `tools/check_run.py`, `tools/check_model.py`, `tools/check_sass.py` | Checks of a probe run, of a model on a GPU, of the probes' machine code |
| `models/` | The RTX 3090 model (1200 MHz) and the MT4G file it was calibrated with |
| `tests/` | `python3 -m pytest Examples/EnergyProbes/tests -q` from the KTT root: no GPU, a few seconds |

**Checking a change.** For the kernels, `python3 tools/check_sass.py --against <kernels/ before the change>` must
report identical SASS, or the measured numbers no longer describe the code. For the analysis, `--analyze` of an
earlier run before and after the change shows every number that moved.
