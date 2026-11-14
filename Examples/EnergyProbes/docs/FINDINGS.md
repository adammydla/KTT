# EnergyProbes: findings on the RTX 3090

The results of the probes and the model on an RTX 3090 at 1200 MHz, with the studies behind them. How to measure is in [CALIBRATION.md](CALIBRATION.md), how to use the model in [../README.md](../README.md).

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

**A second held-out set** (2026-11-13, `--validate heldout2,mixed`, runs `HeldOut2a`, `HeldOut2b`, 24 configurations
per kernel, the model file unchanged: `Preset3` + `ProfileModel1`). Two KTT examples that nothing was designed on:
Bicg (PolyBench BiCG, the fused kernel with atomics; 2 GiB matrix) and Kernel Tuner's convolution (`ktconvolution`)
on ClTuneConvolution's task (16384 x 16384 image, the same 7 x 7 filter), a pair of kernels for one task. Of the
plan's other candidates, Covariance, Convolution3d, CoulombSum2d and CoulombSum3dIterative have OpenCL sources only,
FluidSimulation has no kernel source, and Sort / Sort2 run several kernels per launch, which the validation does not
measure (one kernel per configuration).

| Kernel | Scatter | Bias | Total MAPE | Max regret (time alone) |
|---|---|---|---|---|
| bicg | 10.9 / 12.8 % | +15.4 / +17.5 % | 7.0 / 7.5 % | 0.0 / 0.0 % (0.0 / 20.8 %) |
| ktconvolution | 6.4 / 2.6 % | +7.7 / +7.0 % | 3.6 / 3.2 % | 1.3 / 0.6 % (1.3 / 0.6 %) |
| control (mixed) | 5.4 / 5.0 % | +7.1 / +7.2 % | 4.0 / 4.0 % | 0 % |

- Bicg is the hardest kernel so far (scatter 11-13 %, bias +16 %), yet the model's first choice is the measured best in
  both runs, where time alone loses up to 20.8 %. Pairs 2-5 % apart are ordered right in 80 % (time alone 59 %).
- The bias between kernels over 9 kernels has a standard deviation of 6.4 % (7 kernels: 5.4 %): -3.2 % (transpose) to
  +16.4 % (bicg). The model file keeps 5.4 %; comparing different kernels is less certain than it says.
- **The pair for one task.** The best Kernel Tuner configuration uses 35.1 % more energy than the best ClTune one
  (measured, different runs). The model predicts its choices 36.8 % apart, more than twice the pair's standard deviation
  (8.3 % with the between-kernel term), so it tells the two kernels apart, in the right order.
- As the plan asks, the per-configuration errors of this set are not looked at until a model change is finished.

**The noise floor of a measured energy** (2026-11-13, `--validate all,heldout --validate-configs 8 --validate-repeat 4`
at `--validate-ms` 2000, 4000 and 10000, runs `Repeat2000`, `Repeat4000`, `Repeat10000`). Every configuration was
measured four times, in rounds over all kernels, so that its repeats lie minutes apart within the run:

| Window | Spread of a configuration's dynamic energy between its repeats (median / p90 / mean) | Last repeat against the first |
|---|---|---|
| 2 s | 1.52 / 3.31 / 1.80 % | +1.2 to +5.5 % per kernel (the GPU still warms up) |
| 4 s | 0.38 / 0.81 / 0.46 % | -0.2 to +1.5 % per kernel |
| 10 s | 0.42 / 1.39 / 0.63 % | +0.6 to +3.3 % per kernel (transpose 1.57 % median spread) |

- Within one run, a 4 s measurement repeats to 0.4 %. The 2.4 % of NEXT_STEPS_PLAN 8.1 (the model's error for the same
  configuration in different runs) is therefore not the window: it is between runs (another day, another temperature
  and baseline).
- 10 s windows are no better than 4 s (median 0.42 %, p90 1.39 %). With 10 s a round over all configurations takes
  about 10 minutes instead of 4, so the repeats lie further apart, and the slow rise of the energy over the run (+0.6 to
  +3.3 % from the first to the last repeat, at about 1 K warmer) outweighs what the longer window removes. The window
  stays 4 s.
- 2 s windows are not enough: they read up to 5 % differently as the GPU warms.
- The model's scatter against the mean of four repeats is the scatter against one repeat within 0.6 points per kernel
  (4 s), so the measurement noise inside a run is no part of the 6.3 % worth naming; the run-to-run part is.
- For `--tune-measure` this sets the margin within which measured candidates are not told apart: 1 % (about 2 sd of
  the difference of two 4 s measurements).

**The kernels' input data** (2026-11-13, `--validate gemm,nbody,reduction --validate-configs 8 --validate-seed 7
--validate-data random|zero|small`, runs `Data_random`, `Data_zero`, `Data_small`: the same configurations on uniform
random floats, on zeros, and on integers 0 to 15 stored as floats):

| Kernel | Input | Total energy / random | Dynamic energy / random | Model's bias | Model's regret (time alone) |
|---|---|---|---|---|---|
| gemm | random / zero / small | 0 / -17.0 / -11.2 % | 0 / -32.5 / -21.2 % | +14.6 / +70.7 / +45.7 % | 1.4 / 1.8 / 1.3 % (0 %) |
| nbody | random / zero / small | 0 / -19.2 / -5.5 % | 0 / -45.5 / -13.0 % | +2.4 / +92.6 / +18.1 % | 0 % (12.4-14.0 %) |
| reduction | random / zero / small | 0 / -11.8 / -8.6 % | 0 / -21.5 / -15.7 % | +9.8 / +40.0 / +30.3 % | 0 % (0 %) |

- Low-entropy data lowers the dynamic energy by 13-46 %, and the model, which sees no data, misses it by the same
  amount. This is the largest error of the model found so far, and it is a bias per kernel and input.
- Inside each kernel the ranking does not change: the measured energies of the 7-8 configurations on zeros and on small
  integers have rank correlation +1.00 with those on random data, the best configuration is the same, and the model's
  regret is the same. Tuning on random data gives the right configuration for other data.
- Comparing two different kernels on low-entropy data is where it matters: their dynamic energies move by different
  shares (gemm -32 %, reduction -22 % on zeros). The three kernels here differ by 2-5x, so their order did not change.
  `tools/data_factor.py` (energy over the data) covers the loads; the arithmetic would need the lane statistics too.

**Problem size** (2026-11-13, `--validate gemm,convolution --validate-configs 12 --validate-seed 7
--validate-size-scale 0.5|1|1.5`, runs `Size_05`, `Size_1`, `Size_15`: gemm 2048-6144, convolution 8192-24576):
bias and scatter move by about one point (gemm +13.2 / +12.1 / +11.2 %, convolution +6.1 / +5.6 / +5.8 %), and the
measured energies of the same configurations have rank correlation 0.99-1.00 across sizes. For gemm at half size the
measured best changes, and the model's choice changes with it (regret 1.3 %). A prediction for one size holds for
nearby sizes.

**Instruction classes inside one event** (2026-11-13, `--instr-classes --duty 100`, run `ClassesLanes1`, counts from
`ProfileNew2`): the energy per warp instruction of the class's pipe, after the probe's other instructions are priced
with the model, against the price of the model's event:

| Event (price) | Class: energy per warp instruction (ratio to the price) |
|---|---|
| inst xu (319 pJ) | RCP 308 (0.97), SQRT 300 (0.94), EX2 296 (0.93), LG2 335 (1.05), RSQ 305 (0.96); the old MUFU probe 294 (0.92) |
| inst fma (295 pJ) | FADD 169 (0.57), FMUL 191 (0.65), FFMA x * x + c 247 (0.84), IMAD 254 (0.86), FFMA with three registers 302 (1.02) |
| inst alu (182 pJ) | LOP3 (xor) 337 (1.86), SHF (rotate) 329 (1.81), IMNMX 279 (1.53) |

- MUFU's variants cost the same within 7 %: one event is right for them.
- The fma pipe's instructions spread from 0.57 to 1.02 of one price, and the alu pipe's logic and shift instructions
  cost 1.5-1.9 times its price (set mostly by the memory probes' address arithmetic). By NEXT_STEPS_PLAN's gate (±15 %)
  lumping them is a source of error, which the counters cannot fix: they count pipes, not opcodes. Counts per opcode
  need the CUPTI SASS Metrics API (works here, see below). This fits Coulomb's unrolled code (116 MOV against 10) and
  Pnpoly's methods.
- FADD's operands alternate between two values (x -> 1.9 - x), so its toggles differ from the chaotic FFMA chain; the
  ratio is for that data.

**Accesses off the line** (2026-11-13, `--misaligned --duty 100`, run `MisalignedWrite2`, counts from `ProfileNew2`):
the stream probes started 16 bytes (half a sector) or 32, 64, 96 bytes (1-3 sectors) off the 128-byte line, against the
same probe started on the line in the same buffer, measured and as the model's events predict from the counters:
- 16-byte accesses: DRAM loads +6 to +11 % measured, the events within 2 points; DRAM stores +9 to +17 %, within 4
  points; L2 loads +4 to +12 %, the events 2-6 points low; L1 within 4 %.
- 4-byte accesses: L2 loads +16 to +35 % measured, the events 8-14 points low; L2 stores +10 to +48 % (the
  half-sector start), the events 20 points low there and within 5 elsewhere; DRAM loads +11 to +30 %, the events 7-11
  points high.
- So the events see most of what a 16-byte stream off the line costs, and miss the 4-byte ones by about 10 points in
  both directions (Convolution's halos and Gemm's narrow loads are of that kind).

**The write path** (2026-11-13, `--write-path --duty 100`: 4-byte loads and stores of 1, 2, 4 or 8 words of each
sector, with 4, 2 or 1 sectors per L2 request, at the L2 and DRAM footprints; same runs):
- L2 stores are priced right in every combination (within 4.5 points).
- L2 loads that use 1 or 2 words of a sector cost +401 to +487 % and +172 to +206 % per access against all 8 words;
  the events predict +317 to +335 % and +134 to +168 %: about a quarter too little where few words of a sector are read.
- One sector per request (scattered) against four costs +63 to +94 % (L2 loads), +82 to +94 % (L2 stores) and +123 to
  +177 % (DRAM loads); the events predict 25-70 points less. The request events price a sector of a scattered access too
  low.
- DRAM stores of 2 sectors per request (pairs) cost +13 to +32 %; the events predict +43 to +50 %, too much.
- So one "partial write miss request" event does not fit the grid: the misses lie in the requests (scattered and paired
  sectors) and in sparse reads at L2. These rows are the data for a new request event, to be decided on the probes'
  own fit first (NEXT_STEPS_PLAN 8.5); the model file is unchanged.

**Partially active warps** (2026-11-13, `--lanes --duty 100`, run `ClassesLanes1`): FFMA and integer-add chains with k
of 32 lanes active, the others off by a guard predicate on every instruction or by a branch around the loop body
(energy per warp instruction; about 24 pJ of the fixed part is the loop's overhead):

| Form | 1 lane | 32 lanes | 1 / 32 lanes | Line over k |
|---|---|---|---|---|
| FFMA, guard predicate | 139 pJ | 277 pJ | 50 % | 135 pJ + 4.7 pJ per lane |
| FFMA, branch | 102 pJ | 280 pJ | 36 % | 101 pJ + 5.8 pJ per lane |
| integer add, guard predicate | 182 pJ | 291 pJ | 63 % | 185 pJ + 3.9 pJ per lane |
| integer add, branch | 114 pJ | 296 pJ | 38 % | 104 pJ + 6.4 pJ per lane |

A warp instruction with any lane active has a large fixed part: with one lane an FFMA costs half of a full one, where
the model (lanes / 32) charges 1/32. With no lane active it costs almost nothing (the indexing probe, finding 5), as
AccelWattch models it: the first lane powers the warp's shared parts. The aggregate counters cannot tell an instruction
with one active lane from one with none, so the model cannot use this without per-instruction lane counts (the CUPTI
SASS Metrics API, below); until then it is a limit with this size. Pnpoly's divergent method (-3.2 %) is in line.

**Two NVML questions and the CUPTI APIs** (2026-11-13, read-only, on GPU 1): `nvmlDeviceGetFieldValues` with
`NVML_POWER_SCOPE_MEMORY` is "Not Supported" on the RTX 3090, so the memory's power cannot be read apart. The buffered
samples of `nvmlDeviceGetSamples(NVML_TOTAL_POWER_SAMPLES)` came 139 ms apart on average (11 ms to 1 s), not denser than
the monitor's 100 ms polling. CUPTI's SASS Metrics API (`smsp__sass_inst_executed` per instruction) and its PM Sampling
API (counters every 10 us) both work without root (NVIDIA's samples, built outside the CUDA tree). Using them inside a
KTT launcher, and the check against KTT's own counters, are not done.

**Power level** (2026-11-13, offline): the probe fit's error does not follow the probes' total power (correlation +0.05
over 134 probes; mean error -3.6, +1.1, -1.1 % at 100-160, 160-220, 220-280 W), so nothing points at the board's voltage
converters; the on-GPU test of NEXT_STEPS_PLAN B15 was not run.

**Warps per SM** (2026-11-13, `--compute --duty 100` with the default memory probes, runs `Warps_BxN` for block size B
and N blocks per SM; the model's probes run 256 x 4 = 32 warps per SM). Energy per operation against the 256 x 4 run,
over the 20-24 probes with a price (the barrier and the loop overhead left out: the barrier's cost depends on the
block size by design, and the loop's price is under 1 pJ):

| Warps per SM | Runs | Median | Range | Probes off by more than 3 % |
|---|---|---|---|---|
| 8 | 64 x 4, 256 x 1 | -7.7 %, -5.2 % | -19.0 to +3.2 % | 20 of 20, 16 of 20 |
| 16 | 128 x 4, 256 x 2 | -2.0 %, -1.4 % | -10.9 to +4.2 % | 8, 8 of 20 |
| 24 | 128 x 6 | +0.1 % | -5.5 to +4.4 % | 6 of 24 |
| 32 | 1024 x 1 (same warps, other blocks) | 0.0 % | -5.5 to +4.2 % | 7 of 24 |
| 48 | 512 x 3 | +0.3 % | -3.5 to +4.5 % | 3 of 20 |

The 1024 x 1 run has the same warps per SM as the reference and differs from it by up to 5.5 % per probe: that is the
spread between two runs, not an effect of occupancy. Against it, 24 to 48 warps per SM are flat. Below 24 warps the
energy per operation falls, most at 8 warps: DFMA -19 / -12 %, HFMA2 -15 / -9 %, L2 loads -12 to -15 % (64 x 4), DRAM
loads -7 to -9 % at the same 893 GB/s (251 W against 261 W). The temperature differed by 1-3 K, which explains about
1 W of that. So with few resident warps a kernel draws less power per operation than the probes, and the model prices
it 5-8 % high per operation at 8 warps per SM (2 % at 16). The question of NEXT_STEPS_PLAN B11 is not closed by "flat
within 3 %": an event that uses the resident warps (the tuner knows them from the grid and the block size) could take
this, but only kernels at 16 warps or fewer would gain, and the effect is the size of the spread between runs at 16.

**Burst length at duty 50** (2026-11-13, `--duty 50 --burst-accesses N`, runs `Burst_64` to `Burst_16384`; the default
memory probes). The aim was to see from which idle time on DRAM rests (NEXT_STEPS_PLAN B10b). Energy per access:

| Probe | 64 | 256 | 1024 | 4096 | 16384 | Duty 100 (`Warps_256x4`) |
|---|---|---|---|---|---|---|
| DRAM `ld` | 2565 pJ | 2566 | 2576 | 2590 | 2584 | 2617 |
| DRAM `ld.cg` | 2522 | 2547 | 2505 | 2536 | 2605 | 2615 |
| DRAM `st` | 2570 | 2536 | 2520 | 2555 | 2536 | 2574 |
| L2 `ld` | 593 | 580 | 572 | 578 | 553 | 610 |
| DRAM mean rate (loads) | 656 GB/s | 656 | 657 | 891 | 892 | 893 |

The energy per access does not change with the burst length (DRAM within 1-3 %, 2-4 % under duty 100). But this design
cannot show DRAM resting: every thread sleeps after its own burst, at its own time, so while some warps sleep the
others keep DRAM busy. The mean DRAM rate never fell below 656 GB/s, where 1 - exp(-rate / 95 GB/s) is 1.000; at bursts
of 4096 and more it was the full rate. The question stays open. To answer it, the memory probes need the GPU-wide
windows of the arithmetic probes (`IdleForDutyWindow`: all warps work in the same window, then all sleep) with a
period from microseconds to milliseconds.

**The busy scale with 30 s windows** (2026-11-13, run `LowRate30s`: the `LowRate` sweep again, `--latency --duty 100
--sm-sweep 82,41,20,10,5,2,1`, every row measured 31 s instead of about 11 s; `tools/fit_busy.py`). The aim was the
scale r0 to +-10 GB/s (NEXT_STEPS_PLAN B10a). The DRAM fit, exponential shape:

| Runs | Rows | pJ/B | Fixed | r0 (sd) | rms |
|---|---|---|---|---|---|
| `LowRate` | 23 | 174 | 12.1 W | 91 (42) GB/s | 2.39 W |
| `LowRate30s` | 23 | 171 | 16.3 W | 134 (54) GB/s | 2.20 W |
| `LowRate,Instr` (the model's) | 32 | 172 | 13.4 W | 95 (35) GB/s | 2.43 W |
| `LowRate30s,LowRate,Instr` | 55 | 172 | 14.3 W | 107 (29) GB/s | 2.44 W |

Longer windows did not narrow the scale. The rows below 30 W above the baseline moved by -0.4 to +0.8 W between the two
runs at the same rates, with three times the window: that is a shift between runs, not reading noise, and on rows
1-17 W above the baseline it is what decides r0. The sweep used the same chain counts as `LowRate` (the plan also asked
for denser rates between 1 and 100 GB/s; the existing options give 11 rates there). The model's 95 GB/s lies inside the
pooled 107 +- 29, so it stays. Getting r0 to +-10 GB/s would need many runs of the low rates, or a meter with less
shift between runs than NVML's.

**Configurations that were not profiled** (2026-11-14, offline, `tools/energy_search.py --evaluate`; the open item of
the related work: Filipovič et al. steer KTT's search with counters, here with the model's energy as the objective).
The model prices only configurations whose counters were collected. The search profiles B configurations of a kernel:
half at random, then two at a time the ones a surrogate predicts cheapest, refitted after each step. The surrogate is
fitted on the profiled configurations' tuning parameters (extremely randomised trees, log2 of the values). It predicts
either the model's energy directly (`energy`) or every counter and the time, which the model then prices (`counters`).
The model chooses among the profiled configurations, and the regret is the measured energy of its choice against the
measured best of all. Data: the configurations of `Validation11`-`13` and `HeldOut1`-`2` pooled per kernel (46-110 per
kernel, the first measurement of each), 20 draws per budget, against B configurations at random:

| Configurations profiled | Random | Counters | Energy |
|---|---|---|---|
| 12 | 12.1 % | 9.5 % | 10.6 % |
| 20 | 8.2 % | 6.8 % | 5.3 % |
| 36 | 4.1 % | 2.9 % | 2.6 % |

Mean regret over the 7 kernels. Per kernel the gain is largest where configurations differ most (gemm at 20: 5.3 % random,
1.5 % energy; transpose at 36: 6.0 %, 2.2-4.1 %). There is none in reduction (all within 2 %) and in Coulomb at 12
(one draw with 84 % in each column).

- The guided search lowers the regret by 20-37 % at the same number of profiled configurations, or reaches random's
  regret with about half of them (energy at 20: 5.3 %; random needs between 20 and 36).
- Predicting the counters is no better than predicting the energy. From 8 to 32 profiled configurations the counters'
  route (with the same trees) gets the energy of the unprofiled ones to within a median 2-67 % per kernel (rank
  correlation 0.2-0.85), so it
  serves to order candidates, not to price them. The energy surrogate is the default; the counters' route is kept for
  the comparison.
- With every configuration profiled the model's regret is 0.2 % on average (2.4 % at most). The search therefore trades
  a few percent of regret for profiling a third to a half of the space. Profiling costs about 1.3 s per configuration
  (`TuneCost1`), so this matters for large spaces, not for the 46-110 configurations here.
- The regrets mix configurations measured in five runs, so they include the 2.4 % between runs. The comparison of the
  columns is fair; the absolute values are upper bounds.
- Use: `energy_search.py --ktt RUN_counters.json --space space.txt --count 4` proposes the next configurations from those
  profiled so far. A C++ searcher inside KTT is not written.

**The plan's section 11, steps without a GPU** (2026-11-14; NEXT_STEPS_PLAN P1-P7, M7).
- *P1, the kernel's scale.* The scale now comes from `--tune-scale-configs` configurations at evenly spaced quantiles of
  the predicted dynamic energy (`handoff/analysis_2026-11-14/kernel_scale_strategies.py`, 27 kernel runs). The table
  gives the rms of the scale from the selection against the scale over all configurations:

  | Selection | n = 3 | n = 5 | n = 8 |
  |---|---|---|---|
  | random | 3.3 % | 2.5 % | 1.7 % |
  | quantiles of the predicted dynamic energy | 4.1 % | 2.0 % | 1.9 % |
  | quantiles of the predicted power | 3.2 % | 2.4 % | 1.4 % |
  | quantiles of the time | 3.5 % | 2.1 % | 1.4 % |
  | choice + contenders + fastest (before) | 5.8 % | | |

  A scale from n measurements leaves hypot(6.3 % / sqrt(n), 2.4 %) of the kernel's bias against other kernels (3.8 % at
  n = 5), which the predictors now use instead of nothing. Predictions keep their unscaled dynamic energy, and scales
  are computed from it. Python and C++ agree to 1e-9 (test).
- *P2.* Measured energy over the wall time, as in the README.
- *P3.* `--full-rate RUN` (`energy_model.py`, `build_model.py`, `fit_stability.py`) lets probes of one-duty runs into the
  fit with their duty-100 energy: `ClassesLanes1` gives 38, `MisalignedWrite2` 98. A `--power` run that gives no probe
  is named in a warning (before, its probes were left out silently).
- *P4.* `tools/fit_stability.py`: the prices' CV over 300 resamples within families, per kernel the sd of its mean
  prediction and of its within-kernel differences, and each family left out. Reference (`Preset3` + `ProfileModel1`, 5
  calibration runs): kernel means 1.3-5.4 % (mean 3.7), within-kernel 1.9-4.6 % (mean 2.9), between kernels 3.2 %.
  The least stable large prices: DRAM busy ns CV 47 %, inst lsu global 23 %, inst alu 16 %.
- *P5, two fits judged with P4.* Scatter and bias sd on the 5 calibration runs, then the stability means:

  | Fit | Probes | Scatter | Bias sd | Kernel means | Within-kernel | Busy CV |
  |---|---|---|---|---|---|---|
  | reference | 134 | 6.30 % | 5.39 % | 3.7 % | 2.9 % | 47 % |
  | + SM-sweep rows of `LowRate`, `LowRate30s` (`--sm-sweep`) | 162 | 7.22 % | 6.17 % | 4.4 % | 3.3 % | 18 % |
  | `--weighted` | 134 | 6.31 % | 4.85 % | 4.0 % | 3.2 % | 74 % |
  | both | 162 | 6.50 % | 4.85 % | 3.8 % | 3.0 % | 19 % |
  | + misaligned and write-path probes (`--full-rate MisalignedWrite2`) | 232 | 6.13 % | 3.99 % | 3.7 % | 2.7 % | 27 % |

  - The SM-sweep rows steady the busy price but pull "inst lsu global" from 792 to 214 pJ: all 28 are dependent loads.
    Worse scatter: not adopted.
  - `--weighted` improves the bias spread and makes the predictions depend more on the probe set: not adopted (rule
    8.5 with P4).
  - The misaligned and write-path probes pass both and are the candidate for the next model change, together with the
    probes of step M2.
- *P6.* `build_model.py` stores the largest within-kernel sd next to the pooled one
  (`uncertainty.dynamic_relative_sd_largest_kernel`). The predictors keep the pooled value (the user's choice). The
  calibration with the second held-out set follows the next model change.
- *P7.* `pareto_front` says that it runs over all configurations; a `-inf` counter is named once; `EnergyTuner.h` says
  that the object must outlive the tuner's use of its kernels.
- *M7, the scale on data that is not random* (`handoff/analysis_2026-11-14/data_scale.py`, runs `Data_*`, 7-8
  configurations per kernel; scale from 5 at quantiles, error of the other 2-3):

  | Data | Kernel | Bias without scale | Scale | Others after scaling: bias, rms |
  |---|---|---|---|---|
  | random | gemm, nbody, reduction | +14.6, +2.4, +9.8 % | 0.91, 1.03, 0.90 | +8.8 / 9.0, +16.0 / 16.5, -2.4 / 2.8 % |
  | zero | gemm, nbody, reduction | +71.9, +94.9, +40.8 % | 0.61, 0.55, 0.71 | +12.9 / 13.8, +22.1 / 31.9, -2.3 / 2.9 % |
  | small | gemm, nbody, reduction | +46.9, +19.8, +31.0 % | 0.71, 0.88, 0.76 | +11.8 / 12.1, +16.0 / 20.0, -2.9 / 3.7 % |

  The scale takes away most of the bias from the data. For gemm and reduction, what is left is the size it is on random
  data. Nbody's spread inside the kernel grows on zeros (scatter 10 -> 39 %): its configurations move different
  shares of arithmetic and memory, and the arithmetic loses more on zeros (DFMA 0.17, FFMA 0.5 of random). A scale
  cannot fix that; `tools/data_factor.py` would need the arithmetic's operand statistics. With 2-3 configurations left
  per kernel these numbers are rough.

**Power over temperature under load** (2026-11-14, NEXT_STEPS_PLAN M1, run `ThermalLoad1`, `--thermal-load --duty
100,50`, `tools/fit_thermal_load.py`). The review of 2026-11-14 found the same configuration drawing 0.5-1.1 W per K more
above the stored baseline line, whose slope is 0.2-0.33 W/K. Here five probes and the sleep baseline were each held 90 s
twice: after 60 s of sleep (temperature rising) and after 60 s of DRAM stores (falling). P = a + b T + c t was fitted over
both holds in 2 s windows of NVML's power readings:

| Probe | Duty 100: P, b | Duty 50: P, b |
|---|---|---|
| sleep baseline | 113 W, +0.61 W/K | |
| loop | 160 W, +0.64 | 136 W, +0.53 |
| FFMA (three registers) | 200 W, +0.91 | 156 W, +0.78 |
| L2 loads | 191 W, +1.34 | 163 W, +0.78 |
| DRAM loads | 263 W, -0.29 | 222 W, +0.11 |
| DRAM stores | 254 W, -0.18 | 254 W, -0.17 |

- The compute and L2 probes rise 0.53-1.34 W/K (mean about 0.85). That is the size of the review's finding and of the
  Leiden thesis on NVML calibration (Fejsko 2025: 0.55-0.69 W per degree at a fixed clock on an RTX 4000 Ada).
- But the sleep baseline itself rose 0.61 W/K here, twice the slope in the energy tables (0.2-0.33 W/K). That slope comes
  from sleeps within a few kelvin of each other. The thermal curve of 2026-11-06 gave the same 0.58 W/K over all its
  sleep samples (with a fan term); the 0.33 W/K pooled over 26 runs had an offset per run.
- The excess of the working GPU over the baseline's own slope is small and not resolved: -0.17 W/K flat (rms 0.52 over
  the probes), below the plan's threshold of 0.3 W/K. Of the three forms, flat, proportional to the duty and
  proportional to the dynamic power, none fits clearly better.
- So most of what the review saw is the baseline's slope being too shallow, not a term that grows with the load.
- The DRAM probes show no slope. Their temperature moved over 1-6 K only, and their heat is in the memory, whose
  temperature NVML does not report on this card (`NVML_FI_DEV_MEMORY_TEMP`: not supported, checked).
- The holds swept only 1-6 K each. A wider range needs longer heating; the slopes per probe are rough (±0.5 W/K).

In the model (offline, `handoff/analysis_2026-11-14/thermal_m1_check.py`): a steeper slope at every duty, applied to the
probes and the validation alike, with the 0.3-0.4 W/K this run gives on top of the stored slope:

| Extra slope | Probe fit median | Scatter | Mean bias | Bias sd |
|---|---|---|---|---|
| 0 (today) | 4.36 % | 6.30 % | +3.4 % | 5.39 % |
| 0.3 W/K | 4.58 % | 6.15 % | +2.4 % | 5.23 % |
| 0.4 W/K | 4.64 % | 6.11 % | +2.0 % | 5.18 % |

The scatter and the bias improve a little and the probes' own fit gets worse, so under rule 8.5 it is not adopted. What
would settle it is a baseline measured over a wide temperature range, the thermal curve of 2026-11-06 with longer
heating, before any probe correction.

**The third held-out set** (2026-11-14, NEXT_STEPS_PLAN M6, run `HeldOut3a`, `--validate heldout3 --validate-configs
24 --validate-seed 12`; unchanged model; only per-kernel numbers were looked at, as the plan asks). Five PolyBench/GPU
kernels (`kernels/HeldOut3.cu`; sources in `thesis/third_party/polybench-gpu`) with small tuning spaces: block size,
unroll factor, accumulating in a register or in global memory, rows per thread:

| Kernel | Dynamic bias | Scatter | Total MAPE | Rank corr. | Regret | Regret by time | Energies span |
|---|---|---|---|---|---|---|---|
| gesummv | -19.6 % | 4.7 % | 4.0 % | 1.00 | 0 | 0 | 7.7x |
| syrk | +18.1 % | 8.6 % | 4.4 % | 1.00 | 0 | 2.3 % | 3.7x |
| syr2k | +2.9 % | 6.7 % | 2.4 % | 0.99 | 0 | 0 | 10.1x |
| conv2d | +15.6 % | 3.4 % | 7.6 % | 0.91 | 0 | 7.3 % | 1.6x |
| conv3d | +28.9 % | 3.1 % | 7.6 % | 0.92 | 0 | 2.1 % | 1.1x |

- Inside each kernel the model ranks as on the design kernels: regret 0 in all five, scatter 3-9 % (6.3 % in the
  calibration).
- The bias between kernels is much larger: sd 18.6 % over these five, against 6.4 % over the nine kernels before. The
  two stencils (+16, +29 %) read every input word several times from L1, as the convolution examples do (+4 to +7 %),
  and gesummv (-20 %) reads rows with one thread per row, i.e. strided.
- So the ranking of configurations holds on kernels nobody designed the model on. Comparing different kernels does
  not without a measured scale (`--tune-scale-configs`, P1). The uncertainty in the model file (5.4 % between kernels)
  is too small for kernels like these; P6's recalibration should include this set once the model changes are done.
- The set may be reported once more after a model change; its per-configuration errors stay unread until then.

**Counts per opcode class through KTT, and a split of the fma pipe** (2026-11-14, NEXT_STEPS_PLAN M3 and M4; runs
`ProfileStudy1`, `ValidationStudy1` with `--study-counters`, `handoff/analysis_2026-11-14/m4_split.py`).
- The GA102 metric list (`docs/metrics_ga102.txt`, 4663 base names) has counts per SASS class, FADD, FMUL and FFMA
  separately among them (`smsp__sass_thread_inst_executed_op_{fadd,fmul,ffma,integer,bit,conversion,...}_pred_on`).
  They are collected by patching the kernel's instructions. Through KTT's profiling they come back with the right mix:
  - the `fadd` probe reads 85 % FADD, `fmul` 85 % FMUL, the conversion probe 63 % conversions, the MUFU probes half
    FADD and half FFMA;
  - on the validation kernels (fixed work) their sum equals the hardware's thread instructions within 1 % for gemm,
    reduction and convolution (0.91-0.94 for nbody and coulomb, 0.64 for pnpoly, whose float compares are in the class
    `fp32`, now collected too).
- On the probes their totals are about 250 times too small. A probe runs for a fixed time, and in the instrumented pass
  it gets through that many fewer iterations. So on probes only the shares are usable (the tools rescale by the
  hardware count).
- Reduction's arithmetic is 100 % FADD, nbody's 31 % FADD and 31 % FMUL, coulomb's 28 % FADD.
- *The split* "inst fadd" and "inst fmul" as events of their own (the SASS shares times the fma pipe's count), with the
  class probes in the fit:

  | Fit | Probe fit median | Design scatter | Bias sd | Nbody | Pnpoly | Reduction |
  |---|---|---|---|---|---|---|
  | reference | 4.36 % | 6.69 % | 5.46 % | -2.6 % | -3.8 % | +9.6 % |
  | class probes, events unchanged | 3.89 % | 7.63 % | 4.96 % | -1.4 % | +0.8 % | +10.4 % |
  | class probes + FADD, FMUL events | 3.75 % | 9.85 % | 9.71 % | -12.1 % | -14.8 % | +10.2 % |

  (Design scatter and biases on the one run `ValidationStudy1`.) The fit prices FADD at 182 and FMUL at 189 pJ against
  293 pJ for the rest, the 0.6x of the isolated probes (B5). It then underpredicts the kernels with many FADDs and
  FMULs, and Reduction's bias does not move at all, so FADD is not its cause. Rejected (rule 8.5): in real kernels an
  FADD or FMUL costs about what an FFMA does. Most likely the operand traffic (register reads, bank conflicts), which the
  probes' short chains do not have.
- *The screening of the other candidates* (`tools/screen_counters.py`) needs the screening counters for every probe
  family. `ProfileStudy1` covered only the default probes and the new groups, so on that subset the reference itself
  fails (design scatter 27 %). The counter run of the whole preset with them is `ProfileStudy2`. Its result is
  in "The screening of candidate events" below.

**Probes that separate the global-memory events** (2026-11-14, NEXT_STEPS_PLAN M2; run `Decor1`, `--l1-sweep
--mixed-int --instr-classes --lanes --duty 100,50,25`, 296 probe rows, all within 2 % of the SM clock; counts
`ProfileStudy2`; `handoff/analysis_2026-11-14/m2_decor.py`, `m2_sweep.py`, `m_combo.py`, `m_gate_misw.py`).
- The L1 sweep: 28 probes, loads and stores at the L1 footprint with 1-32 sectors in 1-32 lines per warp instruction,
  4- and 16-byte words. The loads with integer work: 8 probes, `L1 ld` and `L2 ld` with 0-16 add/shift/and per load.
- The present events do not describe the L1 sweep: median fit error 13-14 %. The pattern is the word width at equal
  sectors. 16-byte loads come out 15-30 % too low, 4-byte loads mostly too high (up to +50 %). Even when all lanes read the same 16
  bytes, the 16-byte load costs 118 pJ per access against 75 pJ for the 4-byte one. So the width per lane costs energy
  of its own (most likely the register write-back), which the sectors do not measure.
- The loads with integer work fit within 1-11 %: the integer work adds up, as the FFMA work of `--mixed` did.
- Fits, scored on the five calibration runs (fit median on the reference's 134 probes; P4 = kernel means / within /
  between):

  | Probes | Fit median | Scatter | Bias sd | P4 | L2 read request price (CV) |
  |---|---|---|---|---|---|
  | reference | 4.36 % | 6.30 % | 5.39 % | 3.7 / 2.9 / 3.2 % | 452 pJ (75 %) |
  | + L1 sweep, integer work | 4.45 % | 6.03 % | 5.06 % | 4.4 / 3.3 / 3.5 % | 545 pJ (78 %) |
  | + misaligned, write path | 4.45 % | 6.13 % | 3.99 % | 3.6 / 2.6 / 3.3 % | 616 pJ (14 %) |
  | + all three | 4.40 % | 6.01 % | 4.12 % | 3.6 / 2.6 / 3.3 % | 609 pJ (11 %) |
  | + the class probes | 4.36 % | 7.34 % | 7.23 % | 3.4 / 2.6 / 2.8 % | |
  | + the lane probes | 4.30 % | 6.11 % | 5.77 % | 3.6 / 3.0 / 3.2 % | |

- The L1 sweep makes the stability worse (P4 up) and is not adopted. The misaligned and write-path probes pin the L2
  request price (CV 75 → 14 %). Gemm's error by `VWM` (+2.4 / -0.2 / -7.6 % against its mean) and Convolution's by
  `LOCAL` (+2.6 / +2.2 / -5.5 %) barely move with any of these probe sets.
- The class probes (M4) pull Nbody to -13 % again, through the cheap FADD and FMUL probes (as in the split below).
  The lane probes put Coulomb and PnPoly at +10 and +12 %. Neither is adopted.

**The screening of candidate events** (2026-11-14, NEXT_STEPS_PLAN M3 step 4; `tools/screen_counters.py --power
Preset3 --counters ProfileStudy2 --full-rate MisalignedWrite2,Decor1 --validation ValidationStudy1`; confirmation
`handoff/analysis_2026-11-14/m3_confirm.py`). There are 26 study counters, each added in turn as one more event. The
fit uses 323 probes, 146 of them hard. Reference: hard-probe rms 11.7 %, design scatter 8.57 % (one run), fit
median 6.59 %.
- One candidate stands out: `smsp__sass_inst_executed_op_memory_128b` (128-bit memory instructions, SASS count scaled
  to the hardware count). It brings the hard-probe rms to 9.2 % and the design scatter to 7.11 %, at 959 pJ per warp
  instruction. That is the width effect of the L1 sweep. Four more pass both tests by at most 0.3 points, which is noise
  with 26 candidates: L1 global load requests, L1 set accesses, L2 tag requests, the FFMA class count. The other 21
  get price 0 or make the scatter worse.
- Confirmed on the first held-out set? No. With the 128-bit count, Coulomb's sd falls from 11.8 to 7.9 % and Nbody's
  from 8.5 to 7.0 %, but the held-out scatter rises from 5.30 to 5.67 % (Reduction 3.3 → 4.5 %). The others change
  the held-out scatter by less than 0.1 point. So no candidate enters. The width per lane stays a known gap; it would
  need a hardware counter (the SASS count needs the instrumented pass) and a probe set that prices it next to the
  sectors.

**Partially active warps with changing lanes** (2026-11-14, NEXT_STEPS_PLAN M5; `Decor1`,
`handoff/analysis_2026-11-14/m45_decor.py`). The FFMA lane probe with k of 32 lanes on, against the same with the k
lanes changing from one FFMA to the next (pJ per access, fit over three duties):

| k | 0 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| fixed lanes | 1.68 | 4.47 | 4.63 | 4.78 | 5.60 | 6.96 | 8.07 | 8.68 |
| changing lanes | 2.17 | 5.78 | 6.07 | 6.31 | 7.05 | 8.37 | 9.67 | 8.81 |

- The changing lanes cost 1.3-1.6 pJ more at k = 1-24. Most of it is the extra predicate work: the compiler
  recomputes the eight predicates in every iteration (ALU count 0.0083 against 0.0029 per access), which the reference
  prices at +1.1 pJ. What is left for the change of lanes itself is 0.2-0.5 pJ, against 2.8 pJ for the step from 0 to
  1 lane.
- So the cost of a partial warp is a fixed part per instruction, not the switching of lanes. The rule the plan
  proposes (the fixed part on every instruction except select chains) needs per-instruction counts (M4 route 2, the
  SASS Metrics API), which were not built. The plan says to stop here, and the limit stays as documented (PnPoly-like
  kernels: about 1.5 points of scatter).

**The class probes of M4** (same run). Under the reference prices the new ALU probes come out too low and FADD/FMUL
too high: ISETP+SEL -30 %, FSETP+FSEL -16 %, FMNMX -16 %, IADD3 -47 %, LOP3 -44 %, SHF -44 %, IMNMX -35 %, FADD +68 %,
FMUL +48 %. Per ALU instruction the probes cost 7.4-12.5 pJ per access, and the split does not follow the SASS classes
(IADD3, an integer op, costs as much as LOP3 and SHF, bit ops; compare with select costs less). A per-class event
cannot express this, and the FADD/FMUL split was already rejected (above). Not adopted.

**A candidate model with the misaligned and write-path probes** (2026-11-14, rule 8.5 and 11.5). `build_model.py
--power Preset3 --counters ProfileModel1,ProfileNew2 --full-rate MisalignedWrite2 --calibration
Validation11,Validation12,Validation13,HeldOut1,HeldOut2`. Same events, 98 more probes at their full-rate energy.
The user made it the default (below).

| | Reference | Candidate |
|---|---|---|
| Probe fit median (the reference's probes) | 4.36 % | 4.45 % |
| Misfit of the sub-sector families (DRAM / L2) | 13.4 / 19.2 % | 9.8 / 10.4 % |
| Scatter (calibration, score_ranking) | 6.0 % | 5.8 % |
| Design kernels' scatter (rms of 4) | 7.19 % | 7.19 % |
| Bias | +3.7 % | +3.4 % |
| Total MAPE | 3.4 % | 2.9 % |
| Largest regret | 2.4 % | 2.4 % |
| Bias sd between kernels | 5.4 % | 4.0 % |
| Pairs 2-5 % apart ordered right | 84.4 % | 86.0 % |
| P4: kernel means / within / between | 3.7 / 2.9 / 3.2 % | 3.6 / 2.6 / 3.3 % |
| First held-out set: scatter | 5.53 % | 5.15 % |
| Second held-out set (bicg, ktconvolution): scatter | 10.9 / 12.8 / 6.4 / 2.6 % | 9.5 / 11.1 / 6.4 / 2.8 % |
| Second held-out set: bias | +15.4 / +17.5 / +7.7 / +7.0 % | +16.8 / +19.1 / +8.0 / +6.9 % |
| Third held-out set (`HeldOut3a`): scatter, bias sd | 5.2 %, 18.6 % | 5.5 %, 18.1 % |
| Control kernel (`mixed`, HeldOut2a/b): scatter, bias | 5.4 %, +7.1 % | 7.3 %, +2.0 % |

- Rule 8.5 passes on its two conditions. (a) It closes most of the sub-sector families' misfit. (b) The design
  kernels' scatter is unchanged (Nbody 7.5 → 8.3 % and Coulomb 5.9 → 6.2 % worse; Gemm 7.7 → 7.0 % and Transpose
  7.5 → 7.1 % better). Every calibration number improves or stays.
- Against it: the probes' own fit is 0.09 points worse; on the second and third held-out sets the scatter is mixed
  (bicg better, syrk and syr2k worse); the control kernel's scatter rises by 2 points. Regret is 0 on every held-out
  kernel either way.
- The second held-out set has now been reported once with a model change, as the plan asks. Its per-configuration
  errors were not read. `score_ranking.py` gained `--full-rate` and `--sm-sweep` for this comparison.
- Making it the default was the user's decision: adopted on 2026-11-14.

**The shipped model since 2026-11-14, and the uncertainty from 9 kernels** (NEXT_STEPS_PLAN P6). The model file
`models/rtx3090_1200mhz.model.json` is now built with the misaligned and write-path probes and with the second
held-out set in the calibration:

    python3 tools/build_model.py --power Preset3 --counters ProfileModel1,ProfileNew2 --full-rate MisalignedWrite2 \
        --calibration Validation11,Validation12,Validation13,HeldOut1,HeldOut2,HeldOut2a,HeldOut2b

| | 7 kernels (27 runs) | 9 kernels (31 runs, + bicg, ktconvolution) |
|---|---|---|
| Pooled sd within a kernel | 6.1 % | 6.4 % |
| Largest of one kernel | 8.3 % (nbody) | 10.5 % (bicg) |
| Bias (mean) | +3.4 % | +4.6 % |
| Bias sd between kernels | 4.0 % | 6.1 % (4.0-6.5 % with one left out) |
| Pairs told apart at 2 sd / ordered right | 80 % / 99.98 % | 80 % / 100.00 % |
| Largest regret | 2.4 % | 2.4 % |

- bicg alone moves the between-kernel sd from 4.0 to 6.1 %, with a bias of +18 %. With 9 kernels the number rests on
  one or two outliers. The third held-out set (five PolyBench kernels, biases -14 to +35 %) says that kernels unlike
  these spread wider still. It is not in the calibration, so it can still be reported once more as unseen.
- The predictors keep the pooled sd (the user's choice for P6); the largest is stored for a cautious caller.
- Tests: `tests/data` gained `ProfileNew2.csv` and `MisalignedWrite2.csv` (with its energy table), so that the
  rebuilt model equals the shipped one. The comparison with the 2026-11-12 `--tune` run (`Tune_gemm_tune.csv`) now
  allows 6 % on the dynamic energy (the models differ by -5 to +3 % on Gemm); the choice is the same.

**AccelWattch's microbenchmarks** (2026-11-14, `--validate accelwattch --validate-configs 54 --validate-ms 4000`, run
`AccelWattch1`; unchanged model file). AccelWattch (Kandiah et al., MICRO 2021) calibrated its power model with about 100
CUDA microbenchmarks, published in Accel-Sim's gpu-app-collection (branch `release-accelwattch`). Nobody here designed
them, so they are a third held-out set. `tools/accelwattch_import.py` turned the sources into one KTT kernel file:
```
python3 tools/accelwattch_import.py ~/thesis/third_party/accelwattch-ubench accelwattch_gen --set blocks=82 --set div=1,8,32 --exclude shrd_cnst \
    --exclude MOV --exclude FE_IL0_HIT --exclude FE_IL1_HIT --exclude FE_IL1_MISS --exclude mul_mem_1 --exclude l2w1 \
    --iterations-from <timing> --target-ms 20
```
- Taken: 54 benchmarks from 48 programs; the static-power programs run with one 1024-thread block per SM, ACT_CORE and
  LIGHT_SM with 1, 8 and 32 active threads per warp.
- Left out:
  - 28 programs use texture references (removed in CUDA 12);
  - 7 launch several kernels or none, 14 allocate more than 2 GiB, 3 take five pointers, 1 uses 2048-thread blocks;
  - `shrd_cnst` writes shared memory out of bounds: it indexes block-local arrays with the global thread id, and on
    the RTX 3090 that is an illegal address;
  - six programs do not depend on their iteration count when compiled for sm_86 (MOV, the three instruction-cache
    programs, `mul_mem_1`, `l2w1`): their loops are removed, so they are empty kernels of about 10 µs.
- Iterations were scaled to about 20 ms per launch in four timing rounds (`mul_mem_2` grows with the square of its count
  and was set by hand to 940).
- The count is read at run time from a `__device__` variable, as the programs read it from argv. As a compile-time
  constant the compiler folded whole loops: integer adds became one multiply.

| Group | Benchmarks | Dynamic energy: mean abs error (median) | Total energy: mean abs error (median) |
|---|---|---|---|
| functional | 19 | 107 % (+41 %) | 10.7 % (+9.0 %) |
| memories | 13 | 39 % (+22 %) | 9.7 % (+10.7 %) |
| mix | 7 | 53 % (+23 %) | 8.2 % (+8.4 %) |
| static power | 15 | 109 % (-4.5 %) | 10.4 % (-0.1 %) |
| all | 54 | 84 % (+25 %) | 10.0 % (+9.0 %) |

The total is within 10 % on average because many of these kernels draw little above the baseline (the FP64 ones 116-120 W
against 114 W). The dynamic energy, which is what ranks configurations, is off by much more than on the KTT kernels
(scatter 6.3 %). The misses fall into three groups, each a limit measured before:

- **Operands that stop changing.** The probes keep their operands changing; many of these kernels do not. Repeated
  multiplies of values in [1, 2) overflow to infinity within a few dozen iterations, and several kernels start from
  uninitialised registers. On constant operands only the control switches.
  - Over-predicted: FP64 add/mul/fma +180 to +560 %, FP32 multiply +126 %, REG_FILE +82 %, FP32 add +44 to +69 %, FP32
    fma +38 to +46 %.
  - The zero/random ratios measured on the probes (DFMA 0.17, FFMA 0.5, `Ffma3_fits.csv`, `Full_fits.csv`) predict
    +490 % and +100 % on constant operands: the right size.
  - The integer kernels, whose values keep changing, are within ±32 % (adds, divides, logic, multiplies, `l1d_stress`:
    mean abs error 14.5 %).
- **Partially active warps.** ACT_CORE with 1, 8 and 32 active threads per warp: -85 %, -36 to -48 %, -19 to +29 %. As
  in the lanes probe above, a warp instruction with one active lane costs 36-63 % of a full one, while the model charges
  lanes / 32. LIGHT_SM is too small to tell (its dynamic energy is 0.01-0.02 J).
- **Memory streams +14 to +24 %** (16 memory and mixed benchmarks: median +21.7 %). The buffers hold floats in [1, 2),
  whose 9 sign and exponent bits never change, so the data toggle less than the probes' random words. Loads follow the
  toggles (`--payload-bits`), so this is plausibly part of it; not measured separately.

So on kernels written to stress single units with degenerate data, the model's dynamic energy is not reliable. Its total
is within 10 %, and its misses are the ones the data factor (`tools/data_factor.py`) and per-instruction lane counts
(B6/B7) would address. It does not change the ranking results on real kernels, whose operands are real data.

**Where the model misses, configuration by configuration** (2026-11-13, no GPU time). The tuning parameter that
explains most of a kernel's within-kernel error (NEXT_STEPS_PLAN 8.1) names a parameter, not a cause. For each such
parameter, the badly and a well predicted configuration that differ in it were compared by their event counts per launch
(every configuration of a kernel does the same work) on `Validation11`-`13` and `HeldOut1`-`2`, with the model file:

| Kernel, parameter | Error against the kernel's mean, per value | What differs in the counts (worst / good) | Suspected cause | Covered by |
|---|---|---|---|---|
| gemm `VWM` | 1: +2.3 %, 2: 0 %, 4: -8.2 % (worst +20.6 %, VWM 1) | 3.6x the global load instructions, 3.3x the L1 sectors | narrow L1-heavy loads priced too high, wide ones too low | B8 (accesses per line), width |
| convolution `LOCAL`, tile sizes | `LOCAL` 2: -5.5 %; sizes not a power of two -2 to -15 % (worst +26.7 %, `LOCAL` 0) | L1 sectors 30 % of the energy, 2.8-4.2x | overlapping stencil reads that hit L1, halos off the line | B8 |
| coulomb `INNER_UNROLL_FACTOR` | 16: -18.1 % (2 configurations), 8: -4.4 % | 0.13x the load instructions, 0.73x the FMAs, 0.42x "other" | instruction mix inside "alu" and "fma": the unrolled code (664 SASS instructions against 96, 48 registers against 22) has 116 MOV against 10. Not instruction-cache misses: 10 KiB of code | B5, B6 |
| transpose `WORK_GROUP_SIZE_Y`, `VECTOR_TYPE` | WGY 4: -7.0 %, 64: +8.9 %; vector 2 and 4: -4.6 / -4.8 % (worst -16.9 %) | 0.16x the L2 read miss requests at the same DRAM sectors, 4.9x the address arithmetic | DRAM sectors per request on the write path, the busy power | B9, B10 |
| reduction `UNBOUNDED_WG` | 1: +4.9 % (worst -21.8 %: 32-thread groups, 72 ms against 5 ms) | 1000x the shared instructions, 25000x the L2 reductions | a latency-bound launch with few warps: power that no event sees | B10, B11 |
| pnpoly `USE_METHOD` | 0: -3.2 %, 1: +4.3 % (worst -11.5 %) | 2x the branch instructions, half the ALU and FMA instructions | partially active (divergent) warps | B7 (with B6 for the lanes) |

The order of B5-B11 stands on this evidence: the access patterns of B8 explain the two largest single misses (gemm,
convolution), and instruction classes (B5/B6) and partially active warps (B7) the compute-bound kernels.

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
