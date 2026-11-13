// The printed report of a probe run: every row, the CUPTI counters and the summary.
#pragma once

#include <vector>
#include "Analysis.h"
#include "Hardware.h"
#include "NvmlMonitor.h"
#include "Options.h"
#include "Rows.h"

void PrintRows(const std::vector<Row>& rows);

void PrintCounters(const std::vector<Row>& rows);

void PrintSummary(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    const BaselineModel& baseline, const Options& options, const NvmlMonitor* monitor);

// The analysis of a probe run's rows: energies above the baseline (`rows` gets them), the fits across duties, the row and
// counter tables, the summary and the studies; writes PREFIX.csv, PREFIX_fits.csv and PREFIX_energy_table.json.
void ReportProbeRun(std::vector<Row>& rows, const BaselineModel& baseline, const Footprints& footprints,
    const Hardware& hardware, const Options& options, const NvmlMonitor* monitor);
