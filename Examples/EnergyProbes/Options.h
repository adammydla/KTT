// Command-line options.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <Ktt.h>
#include "Common.h"

// Where the program finds the kernels of the probes and of the other examples, relative to the directory it runs in.
#if defined(_MSC_VER)
inline const std::string kernelPrefix = "";
#else
inline const std::string kernelPrefix = "../";
#endif

inline const std::string defaultKernelDirectory = kernelPrefix + "../Examples/EnergyProbes/kernels/";

struct Options
{
    // The GPU and the files.
    ktt::PlatformIndex platform = 0;
    ktt::DeviceIndex device = 0;
    std::string kernelDirectory = defaultKernelDirectory;
    std::string counterTable = defaultKernelDirectory + "../tools/counters.json"; // the CUPTI counters to collect
    std::string mt4gFile;
    std::string outputPrefix = "EnergyProbes";
    std::optional<uint64_t> l1Override;
    std::optional<uint64_t> l2Override;

    // What runs: the probes (default), --validate, --tune, --predict, --analyze or --dry-run.
    std::vector<std::string> only;     // --only: probe groups to run (empty = all)
    bool profile = false;              // --profile: the counter run (CUPTI counters of every probe)
    std::vector<std::string> validate; // --validate: KTT example kernels to measure instead of the probes
    uint64_t validateConfigs = 24;     // random configurations per kernel
    uint64_t validateMs = 4000;        // measurement time per configuration
    std::vector<std::string> tune;     // --tune: example kernels to tune with the energy model
    std::string modelFile;             // --model-file: the model for --tune and --predict (tools/build_model.py)
    std::string predictCsv;            // --predict: a --validate run's CSV to predict without a GPU
    std::string analyzePrefix;         // --analyze: analyse the CSV of an earlier run instead of measuring
    bool dryRun = false;

    // The recommendation of --tune and --predict.
    std::string objective = "energy";           // energy, edp, ed2p or time
    std::optional<double> maxSlowdownPct;       // only configurations at most this much slower than the fastest
    std::optional<double> maxEnergyIncreasePct; // only those at most this much above the lowest energy
    std::string accounting = "awake";           // awake, above-idle or dynamic (which baseline the energy includes)
    std::optional<double> idleW;                // the idle power for above-idle (default: the model file's)
    double contenderSd = 2.0;                   // configurations within this many sd of the best are contenders
    std::string tuneMeasure = "none";           // none, contenders or all (NVML power of the candidates)
    std::optional<uint64_t> tuneScaleConfigs;   // default: 5 with more than one kernel, else 0
    double measureMarginPct = 1.0;              // measured candidates this close to the best are not told apart
    bool lenient = false;                       // counters the model needs and does not find become warnings

    // The measurement.
    std::vector<uint64_t> duties{100, 25}; // percent of time issuing accesses; the fit across them gives the energy
    uint64_t fitMinDuty = 25;              // lowest duty used in the fit
    uint64_t launchMs = 200;
    uint64_t minTimeMs = 10000;
    uint64_t maxTimeMs = 60000;
    double maxPowerDiff = 0.02;
    uint64_t warmupS = 120;
    uint64_t idleS = 20; // idle power without a kernel: before KTT starts, before the warm-up and at the end
    uint64_t traceIntervalMs = 50;
    bool randomOrder = true;
    bool forceEnergy = false;
    bool monitor = true;
    std::optional<uint32_t> lockClockMHz;
    bool clockLockedExternally = false; // locked with nvidia-smi -lgc by someone else

    // The shape of the probes.
    uint64_t blockSize = 256;
    uint64_t blocksPerSm = 4;
    uint64_t unroll = 16;
    uint64_t burstAccesses = 64; // accesses per thread between two sleeps
    uint64_t sharedBytes = 8 * KiB;
};

bool ClockIsLocked(const Options& options);

void PrintUsage();

Options ParseOptions(int argc, char** argv);
