// The CUPTI counters of EnergyProbes, from the counter table tools/counters.json (--counter-table), which the Python tools
// read too: one place for the counter names, the events they feed, their aliases on other chips and their alternatives.
#pragma once

#include <string>
#include <vector>

// The counters to collect with --profile and in the validation's counter pass, in the table's order: the activity counts
// of the energy model in energy_model.py (warp instructions per pipe, the lsu pipe's also by memory space, sectors per
// level and operation, shared-memory wavefronts, constant-cache requests and misses) and the counters that check which
// level served the probes. KTT skips the ones a GPU lacks with a warning; tools/check_counters.py says what that costs.
// Probes and validation kernels collect the same set.
std::vector<std::string> ModelCounters(const std::string& tableFile);

// The canonical name of a counter: for an alias (the same counter under another chip's name, alias_of in the table) the
// name the tools read, otherwise the name itself. Uses the table that ModelCounters() loaded.
std::string CanonicalCounter(const std::string& name);
