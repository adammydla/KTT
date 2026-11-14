# EnergyProbes (prototype)

Estimates the energy of one **access** (loads, stores, texture fetches and 32-bit atomics) served by L1, L2, DRAM,
shared memory and the constant caches, and of one **arithmetic operation** (FFMA, DFMA, IMAD, HFMA2, HMMA, MUFU) on
NVIDIA GPUs, using KTT's NVML power measurement. Cache sizes come from an [MT4G](https://github.com/caps-tum/mt4g)
result file so that the same program works on different GPUs. Optional studies follow published GPU energy models (see
[Studies](docs/FINDINGS.md#studies-from-published-gpu-energy-models)): control vs datapath energy, energy per request vs per byte, the
power of awake and active SMs, and constant vs static power. The results are collected in an energy table (JSON).
`tools/energy_model.py` turns them into an energy model per CUPTI event and checks its predictions on kernels of other KTT
examples, which `--validate` measures (see [Validation on example kernels](docs/FINDINGS.md#validation-on-example-kernels)).

This is a prototype. Treat the numbers as hypotheses until they pass the checks of the validation checklist ([docs/CALIBRATION.md](docs/CALIBRATION.md#validation-checklist)).

Version 2 changes after the first test run (an 80-SM Ampere card): the old baseline (a busy-polling kernel) drew more power
than the memory probes, so all energies came out negative, and the SM clock varied by 24 % between probes. The baseline is now
a sleeping kernel, the program can lock the clock itself, records why the GPU limited its clock, and withholds energies that
cannot be compared with the baseline.

## Documents

- This file: what the example is, how to build it, and how a tuner uses the model (the model file, the predictors, `--tune`,
  `tools/check_model.py`) and its limits.
- [docs/CALIBRATION.md](docs/CALIBRATION.md): how to measure a GPU: running the probes (all options), the code map, what the
  probes do, the validation checklist, porting to another GPU and the limits of the measurement.
- [docs/FINDINGS.md](docs/FINDINGS.md): the results on the RTX 3090 (the thesis findings): the studies, what the numbers
  mean, the validation on example kernels, the example results and the energy over the data.

The three files were one README until 2026-11-13; the text was moved, not rewritten.

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

## Using the model in a tuner

The probes, `energy_model.py` and the validation runs are the calibration. A tuner needs less: one file with the model
and one call that prices a configuration from its counters and its time, without measuring power.

```bash
# once per GPU type and clock: the model file, from the calibration runs
python3 tools/build_model.py --power Preset3 --counters ProfileModel1,ProfileNew2 --full-rate MisalignedWrite2 \
    --calibration Validation11,Validation12,Validation13,HeldOut1,HeldOut2,HeldOut2a,HeldOut2b \
    --output rtx3090_1200mhz.model.json
# a KTT output file (JSON) of any tuning run with CUPTI profiling: the configuration to use
python3 tools/energy_predictor.py --model rtx3090_1200mhz.model.json --ktt TuningResults.json --objective energy
# the same inside a KTT program (EnergyModel.h); as a demonstration on the example kernels:
./EnergyProbesCuda $GPU --tune gemm,coulomb,transpose --model-file ../../Examples/EnergyProbes/models/rtx3090_1200mhz.model.json
```

**A kernel of your own.** `EnergyTuner.h/.cpp` holds the two passes `--tune` runs, for any simple KTT kernel: a timing
pass (the searcher's configurations, one untimed launch and three timed ones each, no profiling) and a counter pass
(every timed configuration once more with the counters `EnergyModel::Counters()` names), plus an optional power
measurement through a callback (`PowerMeter`; `--tune` passes this example's NVML monitor). It depends on KTT and
`EnergyModel.h` only. `example/MinimalTuner.cpp` is the smallest program that uses it (built by `tests/`, not by
premake; the build command is in its header):

```cpp
EnergyModel model(argv[1]);
EnergyTuner energy(tuner);                                  // tuner: a ktt::Tuner with the kernel and its space
energy.AddKernel("saxpy", kernel, definition);
tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
energy.TimeConfigurations("saxpy", 25);
std::vector<TunedConfiguration> rows = energy.CollectCounters(model.Counters());
// predictions: model.Predict(row.counters, row.kernelS, row.temperatureC, row.smClockMHz, true, row.kernel) per row
const EnergyRecommendation choice = model.Recommend(predictions, EnergyObjective::Energy);
```

- **The model file** (`models/rtx3090_1200mhz.model.json`, format version 1) holds the GPU and the SM clock of the
  calibration, the energy per event, the baseline line, the uncertainty and how it was measured, the counters the
  events need, every probe's full-rate energy and time (for the self-check below) and the SHA-256 of the runs it was
  built from. Since 2026-11-14 its fit includes the 98 misaligned and write-path probes of `MisalignedWrite2` (at one
  duty, `--full-rate`): against the earlier file (`Preset3` + `ProfileModel1` alone) the scatter on the calibration
  runs fell from 6.0 to 5.8 %, the total MAPE from 3.4 to 2.9 %, the bias spread between kernels from 5.4 to 4.0 %
  (over the same 7 kernels); the largest regret stayed 2.4 % (details in [docs/FINDINGS.md](docs/FINDINGS.md)).
- **`tools/energy_predictor.py`** (standard library only) and **`EnergyModel.h/.cpp`** (standard library and
  nlohmann::json; no KTT types in the interface) give the same numbers: `tests/` compares them on a validation run to
  1e-9. Both refuse a model of a newer version, an unknown event, and counters the model needs but does not get (they
  would count as 0 and make a kernel look cheap); `strict=False` / `--lenient` turns the last into a warning.
- **Objectives**: energy, EDP, ED2P or time, optionally only among configurations at most x % slower than the fastest
  (`--max-slowdown`), or the fastest among those within x % of the lowest energy (`--max-energy-increase`). Both limits
  are taken against all configurations; when no configuration meets both, both predictors say so (how many each limit
  admits) instead of returning one.
- **Inputs are checked.** A model file without a key the predictors need is refused with the key's name, a file that is
  not JSON with that message. Counter values that are not finite are an error (`--lenient`: 0 with a warning). Negative
  values count as 0 with a warning, also in the fit: CUPTI reports the two L2 reduction counters negative now and then
  in kernels without reductions (24 of about 1900 validation rows, 9 probe rows of `ProfileModel1`, down to -1.8 million
  sectors). Clamping them moved no event energy by more than 0.01 %.
- **Uncertainty**. The relative error of the dynamic energy differs between the configurations of one kernel with a
  standard deviation of 6.4 % (31 kernel runs of 9 kernels, 697 configurations at the model's clock; 3 gemm
  configurations that ran throttled at 1105-1170 MHz are left out; 6.1 to 6.8 % with any one kernel left out; the
  largest of one kernel, 10.5 % for bicg, is stored next to it, and the predictors use the pooled value). A prediction
  carries that as `sd_j`. Two configurations closer than 2 sd of their difference are *contenders*. On the calibration
  runs, 80 % of all pairs are told apart at 2 sd and 6045 of those are ordered right (100.00 %); the other 20 % are
  still ordered right in 83 %, so the first choice stands and the contenders are what to measure with NVML if the last
  percent matters. The first choice's regret is 0.23 % on average and 2.4 % at most; with the contenders measured, 0.
- **Comparing different kernels** (two implementations of one task) is less certain, because the model's bias differs
  from kernel to kernel: -1.9 % (transpose) to +18.0 % (bicg), standard deviation 6.1 % (4.0 to 6.5 % with any one
  kernel left out: without bicg it is 4.0 %; 9 kernels are few). The bias moves all configurations of a kernel
  together, so it is not part of `sd_j`. When predictions carry a group (the kernel: `predict(..., group=...)`,
  `EnergyPrediction::group`), pairs from different groups are compared with sqrt(6.4^2 + 6.1^2) = 8.9 % per
  prediction. Checked on the calibration runs, with each kernel's energies scaled to the run's common median (the
  kernels solve different problems): 84 % of the 34685 pairs of configurations of different kernels are told apart at
  2 sd, and 99.96 % of those are ordered right; the rest 84 %. The bias spread comes from the same 9 kernels, so this
  is an in-sample check. On five PolyBench kernels that are not in the calibration (`--validate heldout3`) the biases
  spread much wider, -14 to +35 % (sd 18 %), so for kernels unlike these nine measure a few configurations for the
  kernel's scale (below) before comparing kernels.
- **At the power limit** the energy is the limit times the time. Its standard deviation is then how closely the GPU holds
  its limit, 2.1 % (the 98 configurations of `ValFree1` limited by the power cap alone read 308-350 W around 341 W),
  not 6.4 % of a dynamic energy that no longer enters; the pair sd uses the same.
- **Which baseline belongs in the energy** (`--accounting`, `Model.accounting`, `EnergyModel::SetAccounting`) depends on
  how the GPU is used, not on the model:
  - `awake` (default): the GPU awake at the model's clock over the kernel's time. Right when kernels run back to back.
  - `above-idle`: the awake baseline minus the GPU's idle power with a CUDA context (94.2 W on the RTX 3090, from the
    calibration run; `--idle-w` sets another, e.g. 41.7 W without a context). Right when a fixed amount of work comes
    once per period and the GPU idles between.
  - `dynamic`: no baseline, the event energies only.

  On `TuneAll0` (every configuration measured) the measured best of Nbody changes with the accounting, and the model's
  choice follows it under all three. For Transpose the model's first choice is 7.4 % (above-idle) and 7.3 % (dynamic)
  above the measured best, for Coulomb 1.6 and 2.6 %, against 0 and 0.1 % with `awake`. In all of them the measured best
  is among the contenders (Transpose: 7 and 12 contenders instead of 4), so measuring them gives 0. The narrower the
  accounting, the larger the model's relative error, because the part the model predicts is the whole energy.
- **Provenance.** The model file records the driver and CUDA version, the GPU's UUID and serial, the power limit (from
  the power runs' energy tables; runs before 2026-11-13 have none, so they are null), the git commit of the tools and
  the SHA-256 of `tools/energy_events.py`. `predict(..., device_name=...)` warns when the GPU differs from the model's
  (`energy_predictor.py --ktt` passes the name from the KTT file; `--tune` compares it too). Both predictors take the
  counter aliases, group columns and optional counters from the model file.
- **Cost**. `--tune` timed and profiled 72 configurations of three kernels in 87 s (1.2 s each, with compilation),
  against about 10 s per configuration for an NVML measurement. Through `EnergyTuner` (2026-11-13, `TuneCost1`): 93 s,
  1.3 s each. `example/MinimalTuner` tuned its own SAXPY (25 configurations) in 16 s.
- **Checked on the GPU** (`TuneMeasure1`, `--tune gemm,coulomb --tune-measure contenders`): in Coulomb a contender
  measured 4.348 J against the choice's 4.349 J, within the 1 % margin, so the model's choice stands (without the
  margin the final choice would have flipped on 0.02 %); in gemm the model's choice measured lowest, 7.7 % below the
  fastest configuration. The kernels' scales from 3-4 measurements: 0.895 (gemm) and 0.950 (Coulomb).
- **Measuring the contenders** (`--tune-measure`). Measurements closer than `--measure-margin` (default 1 %) to the
  lowest are not told apart: within one run a 4 s measurement repeats to 0.38 % (median; p90 0.81 %), so two
  measurements differ by about 0.55 % (sd), and 1 % is about 2 sd of that (see the noise floor in
  [docs/FINDINGS.md](docs/FINDINGS.md)). Among them the final choice is the model's, if it is one of them, else the
  fastest. The CSV has the GPU temperature of each measurement, and the "predicted" column compares the measurement with
  the prediction at that temperature (the timing pass ran cooler).
- **A kernel's scale from a few measurements** (`--tune-scale-configs N`, default 5 when more than one kernel is tuned).
  With `--tune-measure`, N more configurations of each kernel are measured, at evenly spaced quantiles of its predicted
  dynamic energy, and the mean of measured / predicted dynamic energy over them is the kernel's scale
  (`EnergyModel::KernelScale`, `Model.kernel_scale`, from the predictions' unscaled dynamic energy, so a second call
  cannot compound). It multiplies the kernel's other predictions (column `scaled_predicted_j`); against other kernels
  its bias is then replaced by what is left of it, sqrt((6.4 % / sqrt(N))^2 + 2.4 %^2) (the scale's error and the
  repeat of a measurement between runs). One factor, not one per event (two factors were worse up to 8 measurements).
  It does nothing for the order inside the kernel.
  - Why quantiles: over the 27 kernel runs of the calibration, a scale from 5 such configurations is 2.0 % (rms) off
    the scale over all configurations, from 5 random ones 2.5 %. The earlier choice, the model's choice + contenders +
    the fastest, was 5.8 % off, because those were picked for a low prediction.
  - Offline on `TuneAll0` (200 random draws): with 3 / 5 measured configurations per kernel the rms error of the
    others including the bias falls from 9.2 to 7.5 / 7.1 %, and pairs of configurations of different kernels are
    ordered right in 93.7 / 94.1 % instead of 92.3 %.
  - **Data that is not random** (zeros, small integers: up to +95 % bias): five configurations take most of it away
    (gemm on zeros +72 -> +13 %, reduction +41 -> -2 %). For a kernel whose configurations mix arithmetic and memory
    differently (nbody), the spread inside the kernel grows on such data (10 -> 39 % on zeros), which no scale removes.
    So for your own data: measure five configurations, and rank by the model only inside kernels like gemm or reduction.
- **Measured energy with gaps between launches.** `--tune-measure` charges the baseline over the kernel time and the
  power above it over the wall time per launch (as the validation does); the CSV has `measured_wall_s` and
  `measured_busy`, and a note below 95 % busy (`EnergyTuner` with `PowerMeter.baselineW`).
- **Profiling only part of a large space** (`tools/energy_search.py`, needs scikit-learn). From the configurations
  profiled so far, a surrogate on the tuning parameters predicts the model's energy of the others and proposes the
  cheapest to profile next. Offline on the validation runs, with 20 of 46-110 configurations profiled this way, the
  model's choice is 5.3 % above the measured best on average, against 8.2 % for 20 picked at random (with 36: 2.6 % against 4.1 %;
  with all: 0.2 %). See "Configurations that were not profiled" in [docs/FINDINGS.md](docs/FINDINGS.md).
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

## Known limitations of this version

- No signal of when a prediction extrapolates. Two ideas were tested and rejected: a bootstrap over the probes
  (correlation 0.22 with the errors), and on 2026-11-13 the share of a kernel's energy in events whose price moves by more
  than 20 % when one probe family is left out (correlation with the size of the kernel's bias +0.52 over 7 kernels, +0.45
  to +0.68 with one kernel left out; within kernels +0.21 to +0.29). Neither is clearly better, so every prediction gets
  the same relative sd.

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
