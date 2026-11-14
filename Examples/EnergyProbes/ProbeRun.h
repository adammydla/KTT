// A probe run: the probes of the options in their order, with idle readings, a warm-up and sleep baselines between the
// groups, then the analysis of the rows (Report.h).
#pragma once

#include "Options.h"

// Measures, analyses and writes the outputs; returns the exit code.
int RunProbes(const Options& options);

// --thermal-curve: the sleep baseline's power over temperature while the GPU cools and heats (no probes).
int RunThermalCurve(const Options& options);
int RunThermalLoad(const Options& options);
