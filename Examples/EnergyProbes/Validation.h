// Validation on kernels of other KTT examples (--validate): a power pass over random configurations of each kernel with
// sleep baselines between them, a counter pass over the same configurations, and the CSV that energy_model.py predicts.
#pragma once

#include "Options.h"

// Measures and writes PREFIX_validation.csv and PREFIX.csv (the idle and sleep rows); returns the exit code.
int RunValidation(const Options& options);

// --recount FILE: the counter pass again over the configurations of an earlier --validate run (FILE is its
// PREFIX_validation.csv), for counters that run did not collect; no power measurement. Writes PREFIX_recount.csv (kernel,
// configuration, kernel time, counters); tools/merge_counters.py puts its counters into a copy of FILE.
int RunRecount(const Options& options);
