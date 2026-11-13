// The studies that follow published GPU energy models, printed and collected into the energy table.
#pragma once

#include <vector>
#include <json.hpp>
#include "Analysis.h"
#include "Hardware.h"
#include "Options.h"
#include "Rows.h"

// The studies that follow published GPU energy models. They are printed and collected into the energy table (JSON) for a
// model of application energy: constant and static power, energies per access and operation, control vs datapath energy,
// energy per request vs per byte, and the power of awake and active SMs.
nlohmann::json PrintStudies(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    const BaselineModel& baseline, const Hardware& hardware, const Options& options);
