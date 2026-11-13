// What a probe run measures: the global-probe plan, the operations of the other probes, the SM sweep, and the
// description of the setup printed before a run (and by --dry-run).
#pragma once

#include <cstdint>
#include <set>
#include <tuple>
#include <vector>
#include "Hardware.h"
#include "Options.h"

// Which (LOAD_OP, FOOTPRINT, WIDTH) combinations of the global probe are measured.
using Plan = std::set<std::tuple<uint64_t, uint64_t, uint64_t>>;

Plan BuildPlan(const Options& options, const Footprints& footprints);

std::vector<uint64_t> SharedOps(const Options& options);

// Arithmetic probes for this GPU: HMMA m16n8k16 needs sm_80 or newer.
std::vector<uint64_t> ComputeOps(const Hardware& hardware);

// Active SM counts of the SM sweep: the list given, or all SMs, half of them, a quarter, ..., 1 (--all).
std::vector<uint64_t> SweepCounts(const Hardware& hardware, const Options& options);

void DescribeSetup(const Hardware& hardware, const Options& options, const Footprints& footprints, const Plan& plan);
