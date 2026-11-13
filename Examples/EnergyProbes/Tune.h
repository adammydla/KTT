// The energy model used the way a tuner uses it (--tune, --predict): no power measurement, only the time and the CUPTI
// counters of every configuration, priced with a model file (EnergyModel.h).
#pragma once

#include "Options.h"

// --tune LIST --model-file FILE: random configurations of KTT example kernels, timed and profiled, then the model's choice
// per kernel for --objective. Writes PREFIX_tune.csv. Returns the exit code.
int RunTune(const Options& options);

// --predict FILE.csv --model-file FILE: the same choice from the rows of a --validate run's CSV (kernel, configuration,
// kernel_s and counters), without a GPU. Writes PREFIX_predicted.csv.
int RunPredict(const Options& options);
