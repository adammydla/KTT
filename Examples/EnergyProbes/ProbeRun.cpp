#include <iostream>

#include "Analysis.h"
#include "CounterTable.h"
#include "GpuSession.h"
#include "ProbeKernels.h"
#include "ProbePlan.h"
#include "ProbeRun.h"
#include "Report.h"

// The order of a run: idle power, warm-up, a sleep baseline, then every group followed by a baseline. A baseline after
// every group runs at about the temperature its probes left behind (the GPU cools by only 1-2 C in a 10 s sleep), so the
// baselines span the probes' temperatures and the baseline's temperature model is interpolated, not extrapolated.
int RunProbes(const Options& options)
{
    GpuSession session(options);
    const std::vector<ProbeGroup> groups = ProbeGroups(options, session.GetHardware(), session.GetFootprints());
    DescribeSetup(session.GetHardware(), options, session.GetFootprints(), groups);
    session.OpenMonitor();

    ktt::Tuner& tuner = session.GetTuner();
    const BaselineKernels baseline = AddBaselineKernels(tuner, options, session.GetBlocks(),
        session.GetHardware().multiprocessors);
    const ProbeSet probes(tuner, options, session.GetFootprints(), baseline, session.GetHardware().multiprocessors);

    // KTT needs every kernel before the first one runs. A group's kernel has the group's variants as its tuning
    // parameters, and KTT runs all its configurations in turn.
    const ktt::KernelId warmup = session.CreateKernel("warmup", baseline.spin, Configure{}, false, baseline.counts);
    std::vector<ktt::KernelId> kernels;

    for (const ProbeGroup& group : groups)
    {
        const ktt::KernelId kernel = session.CreateKernel(group.name, probes.Definition(group.kernel),
            probes.Settings(group.kernel), group.placement, baseline.counts);
        std::vector<std::string> names;

        for (const auto& [name, values] : group.parameters)
        {
            tuner.AddParameter(kernel, name, values);
            names.push_back(name);
        }

        tuner.AddParameter(kernel, "DUTY", group.duties);

        if (group.allowed)
        {
            tuner.AddConstraint(kernel, names, group.allowed);
        }

        kernels.push_back(kernel);
    }

    SleepBaselines sleeps(session, baseline.sleep, baseline.counts, 1 + groups.size());

    if (options.profile)
    {
        try
        {
            tuner.SetProfiling(true);
            tuner.SetProfilingCounters(ModelCounters(options.counterTable));
        }
        catch (const ktt::KttException& error)
        {
            std::cerr << "Profiling could not be enabled (build KTT with --profiling=cupti): " << error.what() << "\n";
            return 1;
        }
    }

    session.MeasureIdle("idle-start");
    session.WarmUp(warmup);
    sleeps.RunNext();

    for (size_t i = 0; i < groups.size(); ++i)
    {
        session.Run(kernels[i], groups[i].name, groups[i].probe, "", "");
        sleeps.RunNext();
    }

    session.MeasureIdle("idle-end");
    session.WriteTrace();

    std::vector<Row>& rows = session.GetRows();
    const BaselineModel model = FitBaseline(rows);

    if (model.runs == 0)
    {
        WriteCsv(rows, options.outputPrefix + ".csv");
        std::cout << "\nNo power data was collected. Rebuild KTT with --power-usage and run on a GPU that reports power "
            << "through NVML.\n";
        return 1;
    }

    ReportProbeRun(rows, model, session.GetHardware(), options, session.GetMonitor());
    return 0;
}
