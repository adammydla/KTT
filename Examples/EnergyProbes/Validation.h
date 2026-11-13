// Validation on kernels of other KTT examples (--validate): a power pass over random configurations of each kernel with
// sleep baselines between them, a counter pass over the same configurations, and the CSV that energy_model.py predicts.
#pragma once

#include "Options.h"

// Measures and writes PREFIX_validation.csv and PREFIX.csv (the idle and sleep rows); returns the exit code.
int RunValidation(const Options& options);
