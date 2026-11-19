# EnergyProbes: the probes and the measurement

How a probe measures an energy, what each probe keeps busy, and why the run contains these probes and no others.
The tuner's side is in [../README.md](../README.md).

## How a probe measures

All probe kernels run on every SM (`--blocks-per-sm` x `--block-size`, default 4 x 256 threads) and are
**time-bounded**: each thread loops until `--launch-ms` (200 ms) has passed and counts its accesses. KTT repeats the
launch until every NVML power sample of the last `--min-time-ms` (10 s) is within 2 % of their mean, then reports mean
power and mean duration. The kernels add their counts to a buffer that the launcher clears once per configuration, so
accesses per launch are averaged over the same launches as power and duration.

**Baseline.** Energy is counted above a kernel whose warps sleep (`probeSleep`): the GPU is awake at the same clock but
issues almost nothing. A sleep baseline follows every probe group, so the baselines run at the temperatures the probes
leave behind, and the baseline is a line over the temperature (0.2-0.6 W/K on an RTX 3090). Every probe is compared with
the baseline at its own temperature.

**Duty.** Every thread issues `--burst-accesses` accesses (64), then sleeps so that the burst took the duty's share of
the time. Power above the baseline is fitted against the access rate over the duties: the slope is the energy per
access, the intercept is power that comes with running the probe but does not scale with its rate. The arithmetic
probes use windows of the global timer instead (all warps work in the same windows), because a few warps saturate an
arithmetic pipeline. Duty is a kernel argument and the burst code also runs at 100 %, so the machine code is the same
at every duty.

**Clock.** A row whose SM clock differs from the baseline's by more than 2 % is compared with a baseline at another
clock and voltage. With a locked clock that means the GPU throttled, and the row's energy is withheld.

**No two blocks of an SM read the same lines.** A block trailing another one would hit L1 on lines the leader brought
in. Thread 0 of every block claims a slot on its SM (`%smid` plus an atomic counter per SM), and the footprint is split
into stripes: at the L1 and L2 footprints the block in slot *s* walks stripe *s*; at the DRAM footprint every block of
the grid walks a stripe of its own. The program checks the placement of every row.

**Footprints** (from the MT4G cache sizes): L1 = a quarter of the L1, L2 = half of the L2 (of one L2 partition where
MT4G reports segments), DRAM = at least 8 x L2. For all footprints the machine code of a probe is identical except
for one immediate; `tools/check_sass.py` checks this for sm_70 to sm_90.

**Data.** Random 32-bit words, so bits toggle as in real data. Arithmetic operands stay bounded but keep changing:
floating point follows `x -> x*x + c` with `c = -1.9` (chaotic, inside [-1.9, 1.71]), integers an LCG.

## The probe groups

One KTT kernel per group; its tuning parameters are the probe's variants (`ProbePlan.cpp`, `--dry-run` lists them).

| Group | Kernel | What it keeps busy |
|---|---|---|
| `global` | `probeGlobal` | Plain loads (`LDG`, 4, 8, 16 bytes per thread) at the L1, L2 and DRAM footprints; the same loop without the load (instruction overhead); stores (`STG`, 4 and 16 bytes) and 32-bit atomics (`RED`, `ATOMG`) at L2 and DRAM. L1 is write-through for stores, and atomics are performed in L2, so both start at L2. |
| `compute` | `probeCompute` | 4 independent chains per thread of FFMA (`x*x + c`, and with three different register operands as in real kernels), DFMA, IMAD, HFMA2, HMMA (the GPU's `mma.sync` shape), IMMA (s8, sm_80+; since 2026-11-16), MUFU (`rsqrt` plus the FFMA that feeds it); the empty loop; warp shuffle; block barrier; the FFMA chain in a loop with branches; 32 FFMA accumulators with a runtime index (guarded moves, as in the Coulomb example). |
| `shared` | `probeShared` | Conflict-free `LDS.128`, `STS.128` and `ATOMS.ADD`. |
| `latency` | `probeLatency` | Chains of dependent 4-byte loads at L2 and DRAM (each address depends on the previous load's data), 1, 4 or 16 chains per warp, 1, 2, 4 or 8 working warps per block (the 4-chain and 2- and 4-warp points since 2026-11-17): steady traffic from about 100 to 900 GB/s at DRAM, which prices the DRAM busy event and pins its shape. Duty 100 only (the rate varies with the parallelism). |
| `rows` | `probeLatency` (ROWS 1) | Every thread walking a row of its own with 4-byte loads at DRAM (one thread per matrix row, as PolyBench's gesummv and atax do): a warp instruction touches 32 lines, the rows lie in different DRAM pages, 1, 2, 4 or 8 working warps per block (at 1 and 2 warps the L1 tag stage, one lookup per cycle per SM, sets the rate). Added 2026-11-17 for the row-per-thread kernels the model under-predicted; the fit still under-predicts it by 3-10 % (see below). Duty 100 only. |
| `banks` | `probeSharedBanks` | Shared loads and stores of 4 and 16 bytes whose lanes fall on fewer banks: 1 and 8 times the wavefronts of the conflict-free access. The degree is a kernel argument. |
| `constant` | `probeConstant` | `LDC` (all threads of a warp read one word) at the constant L1 and L1.5 footprints. |
| `locality` | `probeScatter` | 16-byte loads and stores in one stream, or in runs of 32 B (one sector per L2 request) and 128 B placed apart: what a request costs beyond its sectors. |
| `sub-sector` | `probePartial` | 4- and 8-byte loads and stores in which every thread walks a sector of its own (as a transpose writes a column): partial sectors, adjacent or scattered. |
| `write-path` | `probePartial` | The same with 1 or 8 words of each sector and 4, 2 or 1 sectors per L2 request. Duty 100 only. |
| `misaligned` | `probeGlobal` | The stream load and store started 16, 32, 64 or 96 bytes off the 128-byte line (a view into a buffer of its own: the machine code is that of the aligned probe). Duty 100 only. |
| `writes` | `probeWrite` | Loads and stores together: copy (a read and a write stream), update in place (dirty lines evicted), local memory (`LDL`, `STL`, as register spills). |
| `lanes` (optional) | `probeCompute` | FFMA and IMAD in 1 or 8 lanes of every warp, the other lanes branching around them (partly idle warps, as in a reduction tree or AccelWattch's ACT_CORE benchmarks); only the working lanes count accesses. Runs only with `--only lanes`. Checks, not in the fit (see "Partly idle warps" below). |
| `opcodes` (optional) | `probeCompute` (COMPUTE_OP 17) | One SASS opcode per chain and step, operands from 16 registers that change from instruction to instruction (cancelling pairs keep the chains bounded): FADD, FMUL, FFMA and DADD, DMUL, DFMA with three registers, IADD3, LOP3, SHF, MUFU EX2/LG2/SIN/RCP/SQRT/TANH (helpers: the FFMA chain feeding it and a LOP3 that folds the result), I2FP and F2I, POPC, F2F to double and back (helper: a DMUL by a constant); the first seven also with almost constant operands (`-const`). Runs only with `--only opcodes`. Checks, not in the fit: `tools/opcode_energy.py` prices each opcode against its pipe (see "Opcodes within a pipe" below). |
| `reuse` (optional) | `probeCompute` (COMPUTE_OP 17, COMPUTE_OPCODE 3, 20-22) | FFMA x = x * m + o with three registers: opcode 3 of the `opcodes` group, and three variants with the same m and o in all four chains, read from a copy per chain (`ffma-same`, nothing reused), from one shared register of m (`ffma-reuse1`, 0.75 operands marked `.reuse` per FFMA on sm_86) or of m and o (`ffma-reuse2`, 1.5). The operand values are the same in the three; only the registers differ. Runs only with `--only reuse`. Checks, not in the fit (see "The operand reuse cache" below). |
| `mix` (optional) | `probeGlobal` (LOAD_OP 0, MIX_FFMA 4, 16) | Loads of 4 and 16 bytes at L1, L2 and DRAM with 4 or 16 FFMAs per load (three registers, the recurrence of the compute group's ffma3). Runs only with `--only mix`. Checks, not in the fit (see "Mixed work" below). |
| `saturate` (optional) | `probeGlobal` (LOAD_OP 0, MIX_L1 2, 8 or MIX_SHARED 2, 8) | Loads of 4 and 16 bytes streaming from L2 or DRAM with 2 or 8 more loads per access from two rows of the block's stripe (L1 hits) or 16-byte shared loads. Step and mask of those loads pass through memory, so that ptxas cannot merge them. Runs only with `--only saturate`. Checks, not in the fit (see "Mixed work" below). |
| `broadcast` (optional) | `probeSharedBanks` (SHARED_GROUP) | Shared loads of 4 and 16 bytes where 2, 16 or 32 neighbouring lanes read the same address, so a warp load reads 16, 2 or 1 distinct elements. Same loop as the conflict-free `banks` probe; only the start index differs. Runs only with `--only broadcast`. Checks, not in the fit (see "Broadcast shared loads" below). |
| `tensor` (optional) | `probeCompute` | The other tensor-core paths, one warp instruction per step as HMMA: `DMMA` (f64), sparse `HMMA.SP` (2:4) from sm_80, and FP8 `QMMA` (e4m3) on sm_89 and sm_12x only, where it is native (sm_90 and sm_100 emulate it with conversions and HMMA). Runs only with `--only tensor`, not in the calibration run. `energy_model.py` reports these probes as checks, measured against what the fitted prices predict: whether one "inst tensor" price covers them. |

## Why these probes and no others

The run was cut down on 2026-11-14 from 458 + 120 configurations (110 minutes) to 242 (47 minutes). Every cut was tested on the RTX 3090 against the model of the full set, on the same nine example kernels
(7 validation runs, 697 configurations). "Scatter" is the standard deviation of the model's relative error between the
configurations of one kernel, which decides the ranking; two repeat runs of the same probes differ by 0.3 points.

| Probe set | Probes | Scatter | Bias | Total MAPE | Max regret | Pairs > 2 % apart ordered right |
|---|---|---|---|---|---|---|
| Full set, duties 100/50/25 | 232 | 6.4 % | +5.2 % | 3.2 % | 2.4 % | 98.5 % |
| The same from a repeat run | 231 | 6.2 % | +4.3 % | 3.0 % | 2.4 % | 98.6 % |
| Lean set, duties 100/50/25 | 150 | 6.3 % | +4.6 % | 3.1 % | 2.4 % | 98.6 % |
| **Lean set, duties 100/25** (this run) | 150 | 6.4 % | +5.3 % | 3.3 % | 2.4 % | 98.6 % |
| Full set, duty 100 only | 232 | 6.9 % | +4.4 % | 3.5 % | 2.4 % | 98.6 % |

What was removed, and the test behind it:

- **The middle duty (50 %).** The slope of a line through duties 100, 50 and 25 rests on its ends: without the middle
  point the probe energies move by a median of 0.5 % (two repeat runs: 2.3 %). Measuring at duty 100 alone is worse
  (scatter 6.9 %, the bias spread between kernels 7.4 instead of 6.1 %), so the fit stays.
- **Points inside a grid.** Of the bank-conflict degrees (1 to 32), the latency grid, the locality runs (32 B to
  8 KiB) and the write-path grid (1, 2, 4, 8 words) only the ends remain. The model fitted without the removed
  probes predicts them with a median error of 4.9 % (90th percentile 17 %), against 5.1 % (15 %) when they are in the
  fit: they carried no information the ends do not have.
- **Duplicates.** `ld.cg`, `ld.ca` and `ld.nc` take the path of the plain load at their level, and the misaligned
  probe at offset 0 is the aligned probe.
- **Probes that never entered the fit**: mixed loads with FFMAs (a check that energies add up: they do, within
  2-4 %), FMUL+FADD and MUFU.SIN, the static-index twin of the indexing probe, the spin kernel as a reference row,
  and the rows on fewer SMs.
- **Texture fetches**: KTT has no texture arguments, so no kernel tuned through KTT can execute the event.
- **Studies**: zero and patterned payloads, thermal curves, partially active warps, instruction classes inside one
  pipe, an L1 sector sweep, screening counters. Each answered a question about the model (see the thesis text); none
  changed what the fit needs.

**Checked end to end.** The model built from one run of the lean set, scored on the same seven earlier validation
runs as the model of the full set: scatter 5.7 % (full set 6.0 %), total MAPE 2.7 % (3.2 %), largest regret 2.4 %
(2.4 %), pairs 2-5 % apart ordered right 85.3 % (85.1 %). Its bias is lower (+2.4 against +4.6 %), for a reason
that has nothing to do with the probe set: the two duties of a probe now run interleaved in one group, at the
same temperature, where the earlier runs measured the global and arithmetic probes at duty 100 first and at the
lower duties later, 3-8 K cooler. A working GPU's power rises with the temperature faster than the line through a run's sleep baselines
(0.3 W/K fitted, 0.5-1.3 W/K measured under load), so the earlier slopes were a few percent high, most for the weak
probes (DFMA: 82 pJ then, 67 pJ now, at identical access rates).

A shorter power window is **not** free: with 5 s instead of 10 s per configuration (replayed from the recorded power
traces) the energies of the duty-fitted probes shift by +1.5 % on average, and the model's bias with them. With any
one probe family left out, the choice of configuration did not change (regret and ordered pairs as in the table); the
scatter moved between 6.0 and 6.9 %, except without the arithmetic probes (8.3 %).

## What the prices are

The energies per event are **statistical proxy prices at one fixed clock over this probe set**: NNLS on relative
errors of about 170 probes measured as whole-board power through NVML (no external power meter). They predict the
probes and real kernels, but they are not isolated physical energies. Events that always occur together trade off
(`energy_model.py` prints the most correlated pairs and the condition number of the fit), and a price of 0 means the
probes do not identify the event, not that it is free.

`energy_model.py --bootstrap N` refits on N resamplings of the probes (each keeping a probe of every priced event).
On the RTX 3090 (200 resamplings): the predicted dynamic energy of a validation configuration moves by 5.9 % (median
sd), but mostly together: 1.9 % of the pairs of one kernel flip order in more than 10 % of the resamplings, and the
95th percentile of the largest regret is 3.9 %. On the RTX 2080 Ti: 6.8 %, 2.2 % and 3.6 %. Single prices move far
more (L2 read miss request 2000-5900 pJ, DRAM busy 0.3-35 W between the 5th and 95th percentile) than the sums that
real kernels count.

Three kinds of numbers must not be mixed:
- **probe-derived prices** (the fit, above), the same for every kernel;
- **real-kernel uncertainty** (`build_model.py --calibration`): the spread and bias of the model's error on the nine
  calibration kernels, stored in the model file;
- **a kernel's scale** (`--tune-scale-configs N`, `KernelScale()`): measured / predicted energy of a few
  configurations of one kernel, measured with NVML at tuning time. It is an empirical correction for that kernel, not a
  price.

## Which data was used for what

| Data | Used for | Status |
|---|---|---|
| The probes (`Probes`, `Profile`; `Tensor1`, `ComputeProfile1`; `LatSweep1`, `LatSweepProfile1` and `Rows1`, `RowsProfile1` since 2026-11-17) | the prices | fit |
| `all` (transpose, gemm, nbody, coulomb) | designing the events and probes since 2026-10; uncertainty | development |
| `heldout`, `heldout2` (reduction, convolution, pnpoly; bicg, ktconvolution) | added 2026-11-05 and 11-13 as held-out kernels; their errors guided later model changes, and since 2026-11-14 they calibrate the uncertainty (since 2026-11-17 from `AllMany2`, measured with the fixed kernel time) | development, reused |
| `heldout3` (gesummv, syrk, syr2k, conv2d, conv3d; PolyBench) | frozen evaluation of the model of 2026-11-16 (below); on 2026-11-17 gesummv's and conv3d's errors led to the kernel-time fix and the rows probe | reused |
| `heldout4` (atax1, mvt2, jacobi2d, fdtd2d, corrstd; PolyBench) | written 2026-11-17 before any model change, measured once in `AllMany2`, scored once with the model of 2026-11-17 (below) | frozen evaluation |
| `heldout5` (atfconv, sortreduce, radixblocks, radixoffsets; the CUDA kernels of the AtfSamples, Sort and Sort2 examples) | written 2026-11-17 after the model of that day was fixed, measured once in `KttMany1`, scored once with that model (below) | frozen evaluation |
| `LowRate2`, `LowRateProfile2` (latency and rows groups with one block per SM: DRAM 28-400 GB/s) | a check of the DRAM busy shape at low rates; not in the fit (its probe names are those of the default run) | check |
| AccelWattch microbenchmarks | 2026-11-14, older model (archive) | historical |

**Since the inactive-lane event (2026-11-17, later)** the shipped model is SHA-256 `2d90caf0...`: the same power runs
plus `Lanes1`, counted with `Profile2` (the whole calibration plan) and `LanesProfile2`, calibration `AllMany2Dev2` (the
nine development kernels of `AllMany2b`). `AllMany2b` and `KttMany1b` (`tests/data/recount/`) are `AllMany2` and
`KttMany1` with the counters of a `--recount` run (power as measured; counts within 0.43 % of the first run's). The
frozen evaluations below are of the model before (`09b36a64`); the new model scores the same on them within 0.3 points,
but that is a reused look.

So the scores of the nine calibration kernels (README) describe the data the model was developed and calibrated on,
not independent generalisation. The independent number is the frozen `heldout4` evaluation:

- model `models/rtx3090_1200mhz.model.json`, SHA-256 `09b36a64a9fa3194...` (2026-11-17: the probes above, busy scale
  95 GB/s, calibration `AllMany2Dev`, the nine development kernels of `AllMany2`); `kernels/HeldOut4.cu` SHA-256
  `ac8f376eea6a5af1...`; run `AllMany2` (RTX 3090, GPU 1, 1200 MHz locked, 2026-11-17 05:04-06:34, 217 configurations,
  all at the clock; in `tests/data/heldout4/`). The model was chosen before the scores were seen;
- per kernel (scatter / bias / total MAPE / regret, time-only regret): atax1 1.6 / -14.9 / 4.8 / 0.0 (0.0), mvt2
  3.4 / +7.9 / 2.1 / 0.0 (0.0), jacobi2d 2.2 / +8.8 / 5.0 / 0.0 (4.8), fdtd2d 1.6 / +6.4 / 3.6 / 1.2 (1.8), corrstd
  3.8 / +8.3 / 2.1 / 0.0 (4.3) %;
- mean scatter 2.5 %, total MAPE 3.5 %, max regret 1.2 % (time only 4.8 %); pairs 2-5 % apart ordered right 97.2 %
  (time only 73.7 %), 5-10 % 100 % (84.3 %); power error 3.3 %, energy rank correlation 0.94;
- the model of 2026-11-16 on the same rows, for reference: scatter 2.7 %, regret 1.2 %, biases atax1 -19.9, mvt2 +1.0,
  jacobi2d +7.7, fdtd2d +4.8, corrstd +1.3 %. The rows probe moves the row-per-thread kernel (atax1) up by 5 points
  and the others up by 1-7 points; the ranking within each kernel is the same.

The `heldout3` evaluation of the model of 2026-11-16 (SHA-256 `a2192a090d43689d...`, run `HeldOut3Frozen1`, 120
configurations, in `tests/data/heldout3/`): scatter 5.8 %, total MAPE 4.7 %, max regret 0.9 % (time only 9.4 %),
biases gesummv -21.3, syrk +5.0, syr2k -4.4, conv2d +11.9, conv3d +31.6 %. Most of conv3d's and conv2d's bias was the
timing error below; gesummv's is not.

**Kernel time (fixed 2026-11-17).** KTT starts its timer before `cuLaunchKernel`, and with `--power-usage` its CUPTI
callback (the NVML hook) runs in between: KTT's time of a launch includes about 0.6 ms of host time. A 0.2 ms kernel
(conv3d) was recorded as 0.82 ms, which inflated its time, made its DRAM rate look lower and its dynamic power look
smaller. The program now takes the kernel time from the counter `gpu__time_duration` (KTT's is kept as
`ktt_kernel_s`), measures an empty kernel right after the first sleep baseline, and adds how far below the baseline
the GPU draws between launches (`gap_below_baseline_w`, 0.4-0.6 W on the RTX 3090) over the gaps. The probes are not
affected (200 ms launches). Validation runs before 2026-11-17 have the error for kernels under a few ms.

**What the PolyBench biases are not** (2026-11-17). gesummv (-21 %) and atax1 (-15 %) walk one matrix row per thread.
The rows probe does the same and is under-predicted by 10-15 % even when it is in the fit, so the events miss part of
that access. Three explanations were tested and ruled out:
- the DRAM busy shape: the kernels alone would prefer a scale of 25-40 GB/s, but the dependent-load probes at 28-55
  GB/s (`LowRate2`) are over-predicted by 45-94 % at such scales and fit best at 95-150 GB/s;
- L1 data-stage wavefronts (one per line a warp instruction touches, 32 in a rows access against 1 in a stream): with
  the counter in the fit (a fresh full profile, `ProfileW1`) its price comes out 0;
- the kernel-time error, which explained conv2d and conv3d but not gesummv.

**The `heldout5` evaluation** (2026-11-17, the same model `09b36a64...`): the CUDA kernels of KTT examples not used
before, with the examples' sources unchanged and their tuning spaces copied from the examples' programs
(`ValidationKernels.cpp` SHA-256 `0ce64b932f970a85...` as measured; the later `--validate-data` option leaves the default data unchanged; hashes of the sources in
`handoff/probes_2026-11-17/heldout5_sources.sha256`). Run `KttMany1` (RTX 3090, GPU 1, 1200 MHz locked, 140 valid
configurations, all at the clock; 15 random ATF configurations did not launch, too many registers for their block; in
`tests/data/heldout5/`):
- atfconv (ATF's Gaussian convolution, AtfSamples, 85 of 11 M configurations), sortreduce (Sort's radix histogram,
  all 20), radixblocks (Sort2's block sort in shared memory, all 18), radixoffsets (Sort2's digit offsets, 17 of 18);
- per kernel (scatter / bias / total MAPE / regret, time-only regret): atfconv 5.6 / +13.3 / 6.0 / 0.0 (5.6),
  sortreduce 67.2 / +41.7 / 1.5 / 0.0 (0.0), radixblocks 5.2 / +12.0 / 5.3 / 0.9 (0.0), radixoffsets 2.0 / +15.7 /
  7.9 / 0.0 (0.0) %; sortreduce draws only 1-67 W above the baseline, so its relative dynamic error is large while
  its total is within 1.5 %;
- power error 5.5 % (largest 12.3 %), power rank correlation 0.97-0.99, energy rank 0.99-1.00, max regret 0.9 % (time
  only 5.6 %). All four are over-predicted by 12-16 % of the dynamic energy, more than any `heldout4` kernel; their
  predicted energy is 23-34 % DRAM.

**Data values** (2026-11-17, after the `heldout5` score; runs `Data_example`, `Data_zeros`, `Data_random`, option
`--validate-data`). With zeros instead of their own data, transpose, reduction, radixblocks and radixoffsets draw
15-20 % less dynamic power in the same configurations while their counters, and so the predictions, stay the same.
The probes use random payloads, so the model over-predicts kernels on low-entropy data. atfconv's input (the integers
1-100 as floats, ATF's example) is such data: with random floats its bias falls from +16 to +3 %. The radix kernels'
keys are random already, and their +9 to +15 % stays unexplained.

Any model change after this makes the `heldout4` and `heldout5` evaluations reused ones.

## Checking a run

`tools/check_run.py RUN --counters PROFILED_RUN` prints PASS, WARN or FAIL per check: every configuration ran and
has an energy; every row ran at the baseline's clock, without a clock limiter; the baselines lie on a line over the
temperature and cover the probes' temperatures; blocks were placed one stripe each; KTT's power reading settled; the
energies relate as the probes' construction implies (L1 < L2 < DRAM, energy per byte falls with the width, conflicts
and scattered sectors cost more); and, from the counters, every probe hit its level and issued what it was designed
to issue. With three or more duties (`--duty 100,50,25`) it also checks that power is linear in the access rate.

These checks cannot show that an energy is the true energy of a component (NVML measures the whole board). Whether
the energies add up to the energy of real kernels is what the validation run and `tools/energy_model.py` show.

## Another GPU

- Run MT4G on it (or pass `--l1-kib`, `--l2-kib`); without cache sizes the footprints are guesses.
- The counters are in `tools/counters.json`; a counter under another name on a newer chip is an alias there. KTT
  skips counters a GPU lacks with a warning, and the tools refuse to mix runs with different counters.
- **Grid walk (2026-11-16).** The locality, sub-sector and write-path probes number the sectors of the whole grid.
  A thread stays in one residue class of the footprint's sectors, and the 4 blocks of an SM (placed round-robin) cover
  the classes they start in: half of the L2 footprint with 82 SMs (RTX 3090), a quarter with 68 (RTX 2080 Ti), but
  footprint / SMs with 64, 80 or 144, where it fits in L1 and the L2 rows would measure L1. There the planner leaves
  these groups' L2 rows out (28 configurations, a warning in the dry run); their DRAM rows stay. A fix was tried and
  measured: advancing by an odd number of blocks makes every SM walk the whole footprint, but then an SM's 4 blocks
  share one cycle and hit each other's lines in L1. On the RTX 3090, forced (run `OddStepProfile1`), the L2 stream
  read 4.5 instead of 14.3 L2 sectors per warp access: about 70 % L1 hits instead of 10 %. It was removed. A real fix
  needs each block's slot on its SM (as probeGlobal has) and a check on such a GPU. Tests simulate the addresses for
  64, 68, 80, 82, 132 and 144 SMs (coalescing, sectors per step, the data per SM against the planner's formula) and
  check the planner's choice.
- HMMA runs in the shape of the architecture (sm_70, sm_75, sm_80+). The sm_80+ and sm_75 shapes were measured
  (RTX 3090, RTX 2080 Ti); the sm_70 shape was not.
- **The DRAM busy event's scale** (95 GB/s) was fitted on the RTX 3090 (dependent-load runs of 2026-11-05). Since
  2026-11-16 `build_model.py` checks it on every GPU (`tools/fit_busy.py` prints the scan): it refits the probes for
  scales from 10 to 2560 GB/s and without the event, and keeps 95 only if it lies in this GPU's profile-likelihood
  95 % interval, else it uses the GPU's best scale. The model file records the scale, its source and the interval.
  On both GPUs the probes need the event but bound its scale only from above: best 50 GB/s, interval below 10 to 202
  GB/s on the 3090; best 64, below 10 to 320 GB/s on the 2080 Ti. So 95 stays on both, and its price follows (18.4
  and 18.7 W). The four DRAM dependent-load rows alone cannot fix three parameters (the tool says so).
- A model is valid for one GPU type. The 3090's energies with the 2080 Ti's counters rank that card's configurations
  as well as its own model does, but their bias is +19.5 % instead of +6.0 %.
- **A stand-in model from a quick run (2026-11-17).** `check_model.py --other-gpu NAME --write FILE` scales the model's
  energies per part (compute, L1, shared, L2, DRAM) by the median ratio of the part's quick probes and takes the run's
  baseline. The quick probes of the 2080 Ti calibration gave L2 x 0.824, DRAM x 0.940, compute x 0.989, L1 x 0.998,
  shared x 1.054. Over all 150 common probes one factor does not describe the two GPUs (log-ratio sd 0.26, DRAM ratios
  0.68-2.37), yet on the 2080 Ti's validation run the stand-in had a total error of 4.7 % (own calibration 4.5 %) and
  the same max regret (0.9 %); a random tenth of all probes with one factor gave 5.7 %, per part 6.5 %.

### RTX 2080 Ti (Turing, sm_75)

The lean program of 2026-11-14 ran on an RTX 2080 Ti locked at 1200 MHz (68 SMs, 2026-11-15/16) without a code change
or an override. The runs and its model are in `tests/data/rtx2080ti/` and `models/rtx2080ti_1200mhz.model.json`; the full
report is `thesis/handoffs_from_different_servers/rtx2080ti_v2/REPORT_v2.md`.

- **Time:** probe run 2 h 34 min (3090: 47 min), counters 12 min, validation 30 min. KTT's power reading did not settle
  in 107 of 242 configurations, so many ran to `--max-time-ms`.
- **Fit:** 150 probes, median error 7.1 % (3090: 4.6 %), 98 within 10 % (3090: 123).
- **Scores of its own model** (two validation runs, 9 kernels): scatter within a kernel 10.6 %, bias +5.5 %, total MAPE
  4.7 %, max regret 2.9 % (time only: 11.8 %). The worst kernels are Bicg (scatter 21.5 %, bias +21 %), Reduction (bias
  +24 %) and Pnpoly (bias -14 %). On the 4 kernels of the first test (2026-11-06) the total MAPE fell from 4.9 to
  3.2 % and the bias from -9.2 to +0.9 %, while the scatter stayed at about 10 % (3090: 6.1 %).
- **Tuner:** `check_model.py` passes its own model on a `--quick` run and fails the 3090's (time +27 %, baseline 113
  against 75 W). With `--tune ... --tune-measure all` the model's first choice was the measured best in 7 of 9 kernels;
  the other two were within 1.0 and 3.5 %.
- **HMMA m16n8k8:** 1315 pJ per warp instruction, 0.65 pJ per FLOP (3090 m16n8k16: 0.60). Turing has no IMMA probe
  (m16n8k32 needs sm_80), so its model prices IMMA as an HMMA instruction (`build_model.py` does this for every run
  without an IMMA probe).
- **DRAM over-fetch:** for a load that uses one sector of a line, Turing reads two DRAM sectors (up to 1.98 per L2
  read sector; streams about 1). The model counts DRAM sectors, so it prices this; it is why the scattered and
  `run32` loads cost relatively more there than on the 3090 (9.9-17.5x against 3.3-5.1x an adjacent load).
- **Fixes from this test (2026-11-16):** ptxas for sm_75 gave the 16-byte write probes 80-82 registers and the
  16-chain latency probe 72, so only 2-3 of 4 blocks fit per SM (the placement check failed on 6 rows). On sm_7x and
  sm_10x the write probes' group loop is now not unrolled for 16-byte elements (39-46 registers), and on sm_75 the
  16-chain latency probe has `__launch_bounds__` (64 registers). `check_sass.py --against` shows the SASS of sm_80,
  86, 89, 90 and 120 unchanged, so the 3090's numbers stand. The latency check now tests requests and sectors per
  warp access each: CUPTI counted them in different kernel replays (0.96 requests, 4.04 sectors), which made their
  ratio 4.2.
- **Open:** a second probe run on the card (probe repeatability), the indexed FFMA probe (fitted -20 % there, +3 % on
  the 3090), the price of a shared-load bank conflict (0 there: collinear with the wavefront), and why the scatter
  stays near 10 %.

### Newer architectures (Ada, Hopper, Blackwell)

What the review of newer GPUs (`thesis/handoff/NEWER_ARCHITECTURES.md`, 2026-11-15) asked for, and what of it is
in the code (2026-11-16). Nothing here was measured on a newer GPU; the checks below run without one.

Before the first run on a new GPU:

```bash
python3 tools/check_chip_metrics.py --chip gh100        # the counter table against the chip's metric list (ncu, no GPU)
python3 tools/check_sass.py --arch 90 --sms 132         # every probe compiles; the tensor probes to their instruction
./EnergyProbesCuda --dry-run --mt4g <GPU>.json          # footprints and the warnings (walk on power-of-two SM counts)
```

- `check_chip_metrics.py` says which counters a chip has under their name, an alias or an alternative, and which
  events would lose a counter there (counting as 0). Offline lists of Nsight Compute 2025.2.1: Turing, GA100, GA102,
  AD102, GH100, GB100 and GB202 have every counter the events need; GB10 (DGX Spark, unified LPDDR5x) has no DRAM
  sectors, so the DRAM events would cost nothing there.
- `check_sass.py` passes on sm_70, 75, 80, 86, 89, 90, 120 (all probes, all at 4 blocks per SM since 2026-11-16) and
  sm_100 except the known `probeWrite WRITE_OP=2 WIDTH=16` footprint difference. It fails a tensor probe that does not compile to its instruction (FP8 on sm_90:
  `F2FP` conversions and `HMMA`, so it is not planned there).

What the program records now:
- the board in the energy table (`board`): ECC, power smoothing (Blackwell; with it on the sleep baseline and the low
  duties are wrong, and the program warns), memory temperature, memory power where the board reports that scope, the
  NVLink power state, and the time NVML's clock policies were active per reason over the run;
- the CUDA context (`cuda_context`): whether the GPU supports compressible memory (the probes' allocations are not
  compressible), the L2 fetch granularity hint and the persisting-L2 set-aside;
- per configuration, a note when NVML's cumulative counters show a clock policy active during it (power cap, thermal,
  power brake, board limit, reliability), which catches limits shorter than the sampling interval. On the RTX 3090 at
  its locked 1200 MHz the power-cap counter ran through the IMMA probe (192 W, far below the 350 W limit) while the
  sampled clock stayed at the lock, so the note says "clock limited" only when the sampled clock moved as well;
- clock reasons through `nvmlDeviceGetCurrentClocksEventReasons` (the throttle-reason call is deprecated).

The counters and the events:
- HFMA2.MMA that ptxas places on the tensor pipe (GA100, GH100) is priced as fp16 work, DMMA on the tensor pipe
  (GA100, GH100; a tensor sub-pipe on GB100) as fp64, as on GA102 where DMMA is part of the fp64 pipe. Chips without
  those counters count as before.
- New check counters, asked for only from their lowest compute capability (`min_cc`): the Blackwell `tc` and `tmem`
  pipes, Hopper WGMMA (`tensor_op_gmma`), TMA, distributed shared memory (`op_stas`, `op_redas`), L2 traffic between
  partitions (`ltcfabric`), inline compression (`lrc__ilc_output_sectors`, must stay 0), the uniform pipe and the L2
  clock. No probe prices them: a kernel that counts them is mispriced.
- `check_run.py --counters` checks that DRAM read sectors per L2 read sector stay at or below 1 (no over-fetch; a
  warning, since the model counts DRAM sectors, and the RTX 2080 Ti reads two for an isolated one), that
  inline compression is unused where the counter exists, and that the tensor probes count one instruction per step in
  the event that prices them.

Measured on the RTX 3090 at 1200 MHz (2026-11-16, runs `Tensor1` with `--only tensor` and `TensorProfile1` with
`--profile`; 4.5 min and 30 s). Each probe executes one matrix instruction per warp step, DMMA in the fp64 event
(`check_run.py`: 14 passed, 0 failed). Energy per warp instruction, and the error of the model's prices (fit
`LeanProbes1` + `LeanProfile1`, which priced HMMA and DFMA) on it:

| Probe | nJ per warp instruction | Operations | pJ per operation | Model |
|---|---|---|---|---|
| HMMA f16 m16n8k16 (in the fit) | 2.44 | 4096 FLOP | 0.60 | fitted |
| Sparse HMMA f16 m16n8k32 | 2.61 | 8192 FLOP dense-equivalent | 0.32 | -6.5 % |
| IMMA s8 m16n8k32 | 3.32 | 8192 operations | 0.41 | -26.7 % |
| DMMA f64 m8n8k4 | 2.43 | 512 FLOP | 4.7 (DFMA: 33.7 per FLOP) | -11.1 % |

- One "inst tensor" price covers sparse HMMA within the scatter, but priced IMMA a quarter low, so a kernel on the INT8
  path (the FP64 emulation of cuBLAS on Blackwell, quantised inference) was mispriced. **Since 2026-11-16 IMMA is an
  event of its own**, "inst tensor imma", from the tensor pipe's IMMA count (`pipe_tensor_op_imma` from Turing to
  Hopper, `tensor_subpipe_imma` on GB100 and GB202), priced by the IMMA probe, which joined the compute group (244
  configurations in the calibration run instead of 242). On the RTX 3090: 3293 pJ per warp instruction, against 2414
  for HMMA. No other price, the baseline, or any score of the validation runs moved (no validation kernel uses IMMA:
  scatter 5.9 %, total MAPE 2.7 %, regret 2.4 % over the eight runs, before and after); the IMMA probe's error went
  from -26.7 % to 0. The shipped model adds the IMMA probe of `Tensor1` with the counts of `ComputeProfile1` (a
  `--only compute --profile` run with the IMMA counter). A model file without the event prices IMMA as "inst tensor",
  as before; a counter run without the IMMA count leaves the IMMA probe out of the fit (it would be fitted as HMMA).
- DMMA costs about what a DFMA does per warp instruction but does 7 times the FLOP; priced with the fp64 pipe it comes
  out 11 % low.
- NVML's cumulative power-cap counter ran for 10.8 of the 11.0 s of the IMMA row at duty 100 (192 W, limit 350 W)
  while the sampled clock stayed at 1200 MHz and the sampled reasons were clear: on a locked GeForce clock that
  counter is not a lowered clock (the note says so since).
- The board record on the 3090: ECC, power smoothing, memory temperature and memory power not reported (GeForce);
  compressible memory supported, L2 fetch granularity 64 B, persisting set-aside 1.125 MiB.

Not done, because it needs the hardware or a change of KTT:
- probes for WGMMA (`sm_90a`), `tcgen05` with tensor memory (`sm_100a`), TMA and distributed shared memory: they need
  arch-specific compile targets, which KTT sets for the whole tuner, and cannot be checked without such a GPU;
- the near/far L2 split on partitioned chips (A100, H100, B200), SM placement (lowest versus spread `%smid`; the
  probes now always run on all SMs), PM sampling of the DRAM rate, NVML GPM as a fallback counter source, and a
  square-wave characterisation of the power sensor;
- the CUPTI Profiling API that KTT uses is deprecated from CUDA 13.0: stay on CUDA 12.8/12.9 up to Blackwell.
- The DRAM busy event, the payload factors and the loop corrections were measured on GDDR6X at 1200 MHz: measure
  them again on HBM3/3e, GDDR7 and LPDDR5x (a calibration run does).

## Partly idle warps (2026-11-17)

The model counts the alu, fma and fp16 pipes in active lanes (predicate on), which the indexing probe needs: its
guarded moves run with the guard false in every lane and cost almost nothing. A warp instruction whose lanes are
*inactive* (divergence) is another matter. The `lanes` probes (run `Lanes1`, counters `LanesProfile1`) measured one FFMA
lane at 100.5 pJ and eight at 18.2 pJ each (146 pJ per warp instruction), against 8.8 pJ per lane and 282 pJ per warp
instruction of a full warp: about 94 pJ (a third) per warp instruction plus 6.4 pJ per active lane. The model, fitted
without them, puts them 59 % (one lane) and 30-37 % (eight) low, and AccelWattch's ACT_CORE benchmarks with 1 and 8
lanes of 32 miss by -86 % and -40 % the same way.

An event for the idle lanes of the alu and fma pipes (warp instructions minus active lanes / 32) fitted with these
probes took 52 pJ per idle lane slot of a warp: ACT_CORE then missed by -4 to -14 %, but the indexing probe rose to +45 %,
because the pred_on counters count a predicated-off lane as idle too (on the fma pipe alone the same: the guarded moves
are IMAD.MOV). The counter `smsp__thread_inst_executed.sum` counts predicated-off lanes as active, so 32 x
`smsp__inst_executed.sum` minus it is the inactive lanes alone. With it (counter runs `Profile2`, `LanesProfile2`) the
event **"inst inactive lanes"** (in warp instructions of 32 lanes, all pipes) takes 68.9 pJ, the lanes probes fit within
7 % (one FFMA lane +5 %), the indexing probe stays at +3 %, and the other prices move by less than 1 %. It is in the
shipped model since then (2d90caf0). Few configurations of the recorded kernels have inactive lanes (up to 50 % in five
of bicg, 15-32 % in some of convolution, reduction and ktconvolution); their dynamic error moved by 0.5-1.8 points and
the scores by at most 0.3. A kernel profiled without the thread counter counts the event as 0, as before.

**The FP64 misses on AccelWattch's benchmarks are mostly data values, partly the opcode.** Their FP64 kernels multiply
the same two inputs in every iteration and an uninitialised value (zero) into the rest, so the operands hardly toggle.
The model over-predicts them 2.3-5.4x (DP_MUL 5.4x); the zero-payload DFMA probe costs 0.17x the random one, which is
what DP_MUL measured (1 / 5.4 = 0.18). The opcode probes (below) split this: a DMUL costs 0.57x the DFMA the fp64 price
was fitted on, and with constant operands 0.41x of that again (0.23x together, 1 / 4.3).

## Opcodes within a pipe (2026-11-17)

The model prices instructions per pipe; Wattchmen (Tran et al., ICS'26) prices every SASS opcode. The optional `opcodes`
group runs one opcode per chain and step (19 opcodes, the first seven also with almost constant operands; runs
`Opcodes1` and `Opcodes2` with counters `OpcodesProfile1` and `OpcodesProfile2`). `tools/opcode_energy.py` takes the
model's prediction of each probe (pipe prices times the counted events, helpers and loop included) and puts the rest of
the measured energy on the target opcode, with the target count from the SASS of the probe's loop. A second pass prices
the helpers that have probes of their own (LOP3, IADD3, FMUL, a DMUL by a constant) at their measured energy. The table
(`models/rtx3090_1200mhz.opcodes.json`, both runs averaged) has these entries, per warp instruction and as a ratio to the
pipe's price:

| Pipe (model price) | Opcode: pJ (ratio) | With almost constant operands |
|---|---|---|
| fma (280 pJ) | FADD 237 (0.84), FMUL 268 (0.96), FFMA with three registers 350 (1.25) | 0.68, 0.73, 0.43 of random |
| fp64 (2127 pJ, fitted on DFMA x * x + c) | DADD 1091 (0.51), DMUL 1215 (0.57), DFMA with three registers 2262 (1.06), F2F (to double and back) about 270 (0.13) | 0.41, 0.41, 0.46 |
| alu (213 pJ) | IADD3 with three inputs 391 (1.84), LOP3 with three inputs 377 (1.77), SHF 301 (1.41), I2FP 247 (1.16) | IADD3 0.67 |
| xu (325 pJ) | MUFU EX2 419, LG2 431, RCP 409, SQRT 429, TANH 409 (1.26-1.33), SIN 345 (1.06, its FMUL.RZ helper at the FMUL price), F2I 210 (0.65), POPC 214 (0.66) | - |

The FP32, integer and MUFU entries repeat within 1-5 % between the two runs. The FP64 ones repeat only within 5-30 %:
at a 64th of the FP32 rate their probes draw 10-30 W above the baseline, so the baseline's noise weighs more. F2F rests on
the constant-operand DMUL helper and is the least certain (about 270 ± 100 pJ).

What this says:

- **Within a pipe, opcodes differ by up to 2x (fp64 up to 8x with F2F).** The pipe price is a mix weighted the way the
  calibration probes and kernels use the pipe: the fma pipe sits between FADD and the three-register FFMA; the alu
  price comes from loop and address instructions (two-input IADD3, ISETP, MOV), cheaper than three-input IADD3 and LOP3
  with random data; the xu price mixes MUFU with the conversions and POPC that also run there.
- **Constant operands halve the energy of an FMA** (0.43 for FFMA, 0.46 for DFMA) and cut additions and multiplications
  to 0.4-0.7. This is the same effect as the payload factors (zero data), at the level of one operand. A counter cannot
  see it; static SASS can see immediate and constant-bank operands, not loop-invariant registers.
- **On whole applications** the opcode table moves the predictions where the opcode mix is unusual. hotspot (float data
  with double literals: F2F conversions, DADD, DFMA on the fp64 pipe) goes from +5.8 to +2.5 %. lavaMD (DADD, DMUL
  besides DFMA) goes from +2.9 to +1.4 %. The others stay within 1 point (`docs/APPLICATIONS.md`).
- **On the KTT kernels it cannot be applied as it stands:** KTT's counter pass reads CUPTI's hardware counters, which
  have no per-opcode breakdown. Nsight Compute's `sass__inst_executed_per_opcode` is a SASS-patching metric (about 0.2 s
  per launch).

The table is a separate file and a separate option (`app_energy.py score --opcodes`). It does not change the model's
prices: the pipe prices were fitted on the probes' opcode mix, and replacing them by opcode prices would need every
probe and kernel profiled per opcode.

### The operand reuse cache (2026-11-18)

SASS marks a source operand `.reuse` when the next instruction reads the same register in the same operand slot; the
operand is then read from a small cache next to the register file instead of from its bank (Huerta et al. 2025,
arXiv 2503.20481: two banks per sub-core, a one-cycle bubble when two operands sit in one bank; Gebhart et al.
ISCA'11: such a cache saved 36 % of register-file energy). No counter counts reused operands, and tuned GEMM tiles
use the cache much more than the probes (0 to 1.4 reused operands per FFMA over the 197 GEMM configurations of
`GemmMany1`). The optional `reuse` group (runs `Reuse1`-`Reuse3`, counters `ReuseProfile1`; test data
`tests/data/reuse`) measures it with three FFMA probes whose operand values are the same, x = x * m + o with one m
and one o per step for all four chains; only the registers differ. `opcode_energy.py --per-run` prices them; the
reused operands and same-bank pairs come from the SASS (`check_sass.py` checks and prints them):

| Probe | pJ per warp FFMA (spread of 3 runs) | Reused operands per FFMA | Same-bank register pairs per FFMA | FFMA per second (share of the 12.6 T/s peak) | Power |
|---|---|---|---|---|---|
| `ffma-rr` (opcode 3: other m and o in each chain) | 349 (1.0 %) | 0 | 1.00 | 7.23 T (57 %) | 196 W |
| `ffma-same` (a copy of m and o per chain) | 334 (0.5 %) | 0 | 1.00 | 6.43 T (51 %) | 184 W |
| `ffma-reuse1` (one register of m) | 284 (1.8 %) | 0.75 | 0.25 | 8.53 T (68 %) | 194 W |
| `ffma-reuse2` (one register of m and of o) | 280 (0.6 %) | 1.50 | 0.25 | 8.73 T (69 %) | 195 W |

- **Reading m from the cache saves 50 pJ per FFMA (15 %), far above the noise.** It is not a saving per operand
  read: reading o from the cache as well (`ffma-reuse2`, 0.75 more reused operands per FFMA) saves 4 pJ more, inside
  the spread. Per reused operand that is 67 pJ in the first case and 36 pJ in the second, so a price per register
  read (the event the plan had in mind) does not describe it.
- **What changes with the saving is the bank pattern and the issue rate.** With three register operands and two banks
  every FFMA of `ffma-same` has two operands in one bank; caching m leaves one such pair in four FFMAs, the FFMAs issue
  33 % faster, and the energy per FFMA drops. Caching o as well changes neither, nor the energy. The control bits of the
  four loops are the same (stall 1, yield on every FFMA), so ptxas does not schedule them differently. The rate does
  not follow the bank pairs alone (`ffma-rr` issues faster than `ffma-same` at the same pairs), and the bank count is
  not settled (with four banks the order of the four does not match either).
- **Same values in consecutive FFMAs cost 4 % less** (`ffma-rr` 349 against `ffma-same` 334 pJ), the data effect of
  the constant-operand probes in a small dose.
- **The model's FFMA price sits in the middle.** It was fitted on the three-register probe (COMPUTE_OP 9), whose loop
  has 0.61 reused operands and 0.62 same-bank pairs per FFMA; its price, 280 pJ, equals the reuse probes'.
- **It does not show in real kernels at our accuracy.** Over the 197 GEMM configurations of `GemmMany1` (offline, the
  k loop's SASS of each configuration as KTT compiles it; `handoff/analysis_2026-11-18/gemm_reuse/rescore.py`), a
  correction per non-reused register read (66.8 or 36.0 pJ) or per same-bank pair (67.6 pJ) against the fit's FFMA
  probe makes the dynamic error's scatter worse, 7.4 % to 7.6-7.9 %, and the error does not trend with the reuse or
  the bank pairs. GEMM's FFMAs do not issue back to back as in the probes, so a bubble costs no issue slot there.
- **Decision:** no event for it (step T2 of the Wattchmen follow-ups was not built). KTT's counter pass could not
  count it anyway (reused operands are static SASS, executed counts would need Nsight Compute's source counters).
  The probes stay as checks.

## Mixed work (2026-11-18)

The random sweep of 2026-11-18 (handoff/sweep_2026-11-18) left two questions. On gemm and nbody, FFMA looked
under-priced. On bicg, memory-instruction work at full DRAM bandwidth looked free. The optional `mix` and `saturate`
groups add work to the global load probe (run `Mix1` with the global, compute and shared groups, counters
`MixProfile1`; `tests/data/mix`). The analysis is `handoff/analysis_2026-11-18/sorting/mix_check.py`; the excess
(measured minus model) is taken against the plain load of the same run. Run-to-run noise of a probe is about 1.8 %
(median). Five DRAM rows at 4 bytes or with 8 extra loads hit the 350 W cap; three are withheld, two are left out
of the refits below.

| Added work | Level | Real cost / model cost of the added work |
|---|---|---|
| 16 FFMAs per load | L1, L2 | 1.14-1.20 (318-341 pJ per warp FFMA; the run's ffma3 probe: 330) |
| 4 FFMAs per load | L1, L2 | 1.14-1.54 (smaller added work, so noisier) |
| 4 or 16 FFMAs per load | DRAM | 0.55-1.08, within the noise of a 2.6 nJ access |
| 2 or 8 L1 loads per access | L2 | 0.95 at 4 bytes, 1.25-1.32 at 16 bytes |
| 2 or 8 shared loads per access | L2 | 1.11-1.32 |
| 2 or 8 L1 or shared loads per access | DRAM | 0.74-1.31, mostly within the noise |

- **An FFMA next to loads costs what the ffma3 probe measures, about 330 pJ.** The model's fma price is 280 pJ, because
  the fit trades it against the load-instruction and L1 prices (it prices its own ffma3 probe 6-8 % low). That fits
  the under-prediction of gemm's and nbody's FFMA-heavy configurations.
- **Memory-instruction work is not free at DRAM saturation.** Within the noise, added L1 and shared loads cost about
  their price, so bicg's over-prediction is not this.
- **Refits do not improve the sorting** (`handoff/analysis_2026-11-18/sorting/FINDINGS.md`):

  | Variant | Probe fit | Calibration scatter, bias | Sweep MAPE | Regret |
  |---|---|---|---|---|
  | shipped | 4.8 % | 5.9 %, +1.0 % | 3.3 % | unchanged |
  | with the 23 mixed probes (fma 296 pJ) | 4.6 % | 5.9 %, +2.0 % | 3.3 % | transpose +3.9 % |
  | fma price held at 330 pJ, the rest refitted | 4.5 % | 6.1 %, +3.9 % | 3.7 % | transpose +3.9 % |

  The mixed-probe refit gets nbody's bias right (dynamic -6.0 → -3.7 % on the sweep), but moves gemm and ktconvolution
  up. None changes which configuration the model picks. The model stays as shipped.

## Broadcast shared loads (2026-11-18)

Many kernels give one shared memory word to several lanes: a tile row in a matrix product, a coefficient in a stencil.
The model prices a shared load by its instruction and its wavefronts, which were fitted on loads of 32 distinct words. The
optional `broadcast` probes (`--only broadcast`; run `Broadcast1` with the `banks` probes, counters `BroadcastProfile1`)
let 2, 16 or 32 lanes read each element. lud_internal (Rodinia) does both kinds: 16 four-byte loads where each word goes
to 2 lanes and 4 sixteen-byte loads where 2 elements go to 16 lanes each, per loop.

The counter `l1tex__data_bank_reads.sum` (now in `tools/counters.json`, a check) counts the distinct words: 32, 16, 2
and 1 per four-byte load for 1, 2, 16 and 32 lanes per word. Conflicts do not add bank reads. Broadcast 16-byte loads take
2 wavefronts instead of 4 (the distinct bytes fit in fewer), which the model already counts. Nsight Compute's
`sm__sass_data_bytes_mem_shared_op_ld` counts the same distinct bytes; `tools/app_energy.py profile` collects both.

| Probe (lanes per address) | Bytes | Wavefronts | Bank reads | Measured pJ per warp load | Model | Error |
|---|---|---|---|---|---|---|
| conflict-free (1) | 4 | 1 | 32 | 1432 | 1388 | -3.1 % |
| 2 | 4 | 1 | 16 | 1374 | 1388 | +1.0 % |
| 16 | 4 | 1 | 2 | 1181 | 1388 | +17.5 % |
| 32 | 4 | 1 | 1 | 1184 | 1388 | +17.2 % |
| conflict-free (1) | 16 | 4 | 128 | 3239 | 3162 | -2.4 % |
| 2 | 16 | 2 | 64 | 2572 | 2192 | -14.8 % |
| 16 | 16 | 2 | 8 | 2178 | 2192 | +0.6 % |
| 32 | 16 | 2 | 8 | 1845 | 2192 | +18.8 % |

(One run; the loop's other instructions are included in each energy.)

- **Pairs of lanes per word cost as much as distinct words.** Half the bank reads save only 4 % (58 pJ) at 4 bytes. At
  16 bytes, half the wavefronts save 670 pJ, not the 2 x 485 pJ the wavefront price gives: the model is 15 % low.
- **16 or 32 lanes per word save about 250 pJ** (4 bytes), 17 % below the model. At 16 bytes, one element for all 32 lanes
  costs 330 pJ less than two elements for 16 lanes each, with the same counters.
- So the energy does not follow the bank reads in proportion, and no counter tells the 16-byte cases apart. A model event
  for broadcasts would need more probes (group sizes 4 and 8) first.
- **lud is priced right by this.** Its two patterns are the 4-byte pair (+1.0 %) and the 16-byte half-warp broadcast
  (+0.6 %). Broadcast loads do not explain lud's +11 % (docs/APPLICATIONS.md).
