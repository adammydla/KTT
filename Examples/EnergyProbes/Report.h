// The analysis of a probe run's rows, printed and written to the files the tools read.
#pragma once

#include <vector>
#include "Analysis.h"
#include "Hardware.h"
#include "NvmlMonitor.h"
#include "Options.h"
#include "Rows.h"

// Energies above the baseline (`rows` gets them) and the fits across duties; prints the rows, the baseline and the fits,
// and writes PREFIX.csv (every row) and PREFIX_energy_table.json (device, baseline, idle power, fits, provenance), from
// which tools/build_model.py builds the model. tools/check_run.py checks the run.
void ReportProbeRun(std::vector<Row>& rows, const BaselineModel& baseline, const Hardware& hardware, const Options& options,
    const NvmlMonitor* monitor);
