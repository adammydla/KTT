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
    ktt::PlatformIndex platform = 0;
    ktt::DeviceIndex device = 0;
    std::string kernelDirectory = defaultKernelDirectory; // the probe files (kernels/)
    std::string counterTable = defaultKernelDirectory + "../tools/counters.json"; // the CUPTI counters (--profile)
    std::string mt4gFile;
    std::string outputPrefix = "EnergyProbes";
    uint64_t launchMs = 200;
    uint64_t minTimeMs = 10000;
    uint64_t maxTimeMs = 60000;
    double maxPowerDiff = 0.02;
    uint64_t blockSize = 256;
    uint64_t blocksPerSm = 4;
    uint64_t unroll = 16;
    uint64_t burstAccesses = 64; // accesses per thread between two sleeps at duty below 100
    uint64_t sharedBytes = 8 * KiB;
    uint64_t warmupS = 120;
    uint64_t traceIntervalMs = 50;
    std::optional<uint32_t> lockClockMHz;
    std::vector<uint64_t> duties; // percent of time issuing accesses
    uint64_t fitMinDuty = 25;     // lowest duty used in the fit across intensities
    std::vector<uint64_t> payloads{PayloadRandom};
    std::vector<std::string> payloadBits; // --payload-bits: patterns of Payloads.h
    std::vector<uint64_t> widths{DefaultWidth}; // bytes per access of the global loads
    std::vector<uint64_t> smSweep;              // active SM counts of the SM sweep (empty = off)
    bool smSweepAuto = false;                   // --all: SM sweep over all SMs, halves, ..., 1
    bool sweepStudies = true;                   // SM sweep with its L1, L2, no-load, shared (and FFMA) probes; --model:
                                                // only the baselines and the dependent loads, which the model uses
    bool compute = false;
    bool atomics = false;  // global and shared atomics (and the 4-byte loads and stores they are compared with)
    bool constant = false; // constant-memory loads at the constant L1 and L1.5 footprints
    bool texture = false;  // texture fetches at the L1, L2 and DRAM footprints
    bool dramLocality = false; // DRAM loads and stores whose runs of consecutive lines are scattered (row locality)
    bool subSector = false;    // 4- and 8-byte accesses, one word of 32 sectors per warp instruction (partial sectors)
    bool indexing = false;     // FFMA accumulators with static vs runtime register-array indices (guarded moves)
    bool bankConflicts = false; // 32-bit shared loads and stores with 1- to 32-way bank conflicts
    bool mixed = false;         // 16-byte loads with 0 to 1024 FFMAs each (arithmetic intensity, as Mixbench)
    bool writes = false;        // loads and stores together: copy, update in place, local memory
    bool latency = false;       // dependent loads at low memory-level parallelism
    uint64_t idleS = 20; // idle power measurement without a kernel, before the warm-up and at the end
    std::optional<uint64_t> l1Override;
    std::optional<uint64_t> l2Override;
    bool sweep = false;
    bool randomOrder = true;
    bool profile = false;
    std::vector<std::string> validate; // KTT example kernels to measure instead of the probes (--validate)
    uint64_t validateConfigs = 24;     // random configurations per kernel
    uint64_t validateMs = 4000;        // measurement time per configuration
    std::vector<std::string> tune;     // --tune: example kernels to tune with the energy model (no power measurement)
    std::string modelFile;             // --model-file: the model for --tune and --predict (tools/build_model.py)
    std::string predictCsv;            // --predict: a --validate run's CSV to predict without a GPU
    std::string objective = "energy";  // --objective: energy, edp, ed2p or time
    std::optional<double> maxSlowdownPct; // --max-slowdown: only configurations at most this much slower than the fastest
    double contenderSd = 2.0;          // --contender-sd: configurations within this many sd of the best are contenders
    std::string tuneMeasure = "none";  // --tune-measure: none, contenders or all (NVML power of the candidates)
    bool lenient = false;              // --lenient: counters the model needs and does not find become warnings
    bool dryRun = false;
    bool thermalCurve = false; // --thermal-curve: the baseline's power over temperature, not the probes
    std::string analyzePrefix; // --analyze: analyse the CSV of an earlier run instead of measuring
    bool forceEnergy = false;
    bool clockLockedExternally = false; // locked with nvidia-smi -lgc by someone else
    bool monitor = true;
};

bool ClockIsLocked(const Options& options);

void PrintUsage();

Options ParseOptions(int argc, char** argv);
