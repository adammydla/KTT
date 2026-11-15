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
| `compute` | `probeCompute` | 4 independent chains per thread of FFMA (`x*x + c`, and with three different register operands as in real kernels), DFMA, IMAD, HFMA2, HMMA (the GPU's `mma.sync` shape), MUFU (`rsqrt` plus the FFMA that feeds it); the empty loop; warp shuffle; block barrier; the FFMA chain in a loop with branches; 32 FFMA accumulators with a runtime index (guarded moves, as in the Coulomb example). |
| `shared` | `probeShared` | Conflict-free `LDS.128`, `STS.128` and `ATOMS.ADD`. |
| `latency` | `probeLatency` | Chains of dependent 4-byte loads at L2 and DRAM (each address depends on the previous load's data), 1 or 16 chains per warp, 1 or 8 working warps per block: steady traffic at low to high rates, which prices the DRAM busy event. Duty 100 only (the rate varies with the parallelism). |
| `banks` | `probeSharedBanks` | Shared loads and stores of 4 and 16 bytes whose lanes fall on fewer banks: 1 and 8 times the wavefronts of the conflict-free access. The degree is a kernel argument. |
| `constant` | `probeConstant` | `LDC` (all threads of a warp read one word) at the constant L1 and L1.5 footprints. |
| `locality` | `probeScatter` | 16-byte loads and stores in one stream, or in runs of 32 B (one sector per L2 request) and 128 B placed apart: what a request costs beyond its sectors. |
| `sub-sector` | `probePartial` | 4- and 8-byte loads and stores in which every thread walks a sector of its own (as a transpose writes a column): partial sectors, adjacent or scattered. |
| `write-path` | `probePartial` | The same with 1 or 8 words of each sector and 4, 2 or 1 sectors per L2 request. Duty 100 only. |
| `misaligned` | `probeGlobal` | The stream load and store started 16, 32, 64 or 96 bytes off the 128-byte line (a view into a buffer of its own: the machine code is that of the aligned probe). Duty 100 only. |
| `writes` | `probeWrite` | Loads and stores together: copy (a read and a write stream), update in place (dirty lines evicted), local memory (`LDL`, `STL`, as register spills). |

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
- With 64 or 80 SMs the locality and sub-sector probes' L2 rows fit in L1 (the program warns); the model prices the
  events that happened, but those rows then say nothing about L2.
- HMMA runs in the shape of the architecture (sm_70, sm_75, sm_80+); only the sm_80+ shape was measured.
- The scale of the DRAM busy event (95 GB/s, `--busy-scale`) was measured on the RTX 3090.
