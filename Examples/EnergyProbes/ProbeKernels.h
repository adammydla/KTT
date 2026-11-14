// The probes as KTT kernels: the launcher that runs a configuration and collects the kernels' own counts and
// the NVML trace, the row made of a KTT result, the CUPTI counters, and the texture objects of the texture probe.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <map>
#include <vector>
#include <cuda.h>
#include <Ktt.h>
#include <json.hpp>
#include "Hardware.h"
#include "NvmlMonitor.h"
#include "Options.h"
#include "Rows.h"

// Sets the runtime arguments (duty, active SMs, payload, ...) and buffers of a definition for one configuration.
using Configure = std::function<void(ktt::ComputeInterface&, const ktt::KernelConfiguration&)>;

// A texture object over a device copy of the payload (or zeros), for probeTexture: 4 x 32-bit integer texels, point
// sampling, read as integers. KTT has no texture arguments and does not expose the device pointers of its buffers, so the
// copy is allocated here with the driver API, in KTT's context: the tuner creates it with cuCtxCreate, which leaves it
// current on this thread. Must be destroyed before the tuner.
class PayloadTexture
{
public:
    // Copies `bytes` of `data` (zeros for nullptr) to the device and creates the texture object over them.
    PayloadTexture(const uint32_t* data, const size_t bytes, const ktt::DeviceIndex device);
    ~PayloadTexture();

    PayloadTexture(const PayloadTexture&) = delete;
    PayloadTexture& operator=(const PayloadTexture&) = delete;

    // The texture object, passed to probeTexture as a 64-bit argument.
    uint64_t GetHandle() const
    {
        return static_cast<uint64_t>(m_Texture);
    }

private:
    CUdeviceptr m_Buffer = 0;
    CUtexObject m_Texture = 0;
};

double CounterValue(const ktt::KernelProfilingCounter& counter);

std::optional<uint64_t> ParameterValue(const ktt::KernelConfiguration& configuration, const std::string& name);

// Short, comma-free label of a configuration for the trace file, e.g. "Global LOAD_OP=0 FOOTPRINT=16384".
std::string TraceLabel(const std::string& kernelName, const ktt::KernelConfiguration& configuration);

// The CUPTI counters of the model and of the probes' checks: ModelCounters() in CounterTable.h (tools/counters.json).

// Sleep time per burst time for a duty, (100 - duty) / duty in 16.16 fixed point (argument idleRatio of the kernels).
uint32_t IdleRatio(const uint64_t duty);

// The activeSms argument for a configuration: the ACTIVE_SMS parameter of SM sweep kernels, all SMs otherwise.
uint32_t ActiveSms(const ktt::KernelConfiguration& configuration);

// The parameters every probe kernel is compiled with (BLOCK_SIZE, BLOCKS_PER_SM, UNROLL, BURST_ACCESSES) and the thread
// modifiers that make its grid BLOCKS_PER_SM blocks of BLOCK_SIZE threads per SM.
void AddProbeLaunchParameters(ktt::Tuner& tuner, const ktt::KernelId kernel, const ktt::KernelDefinitionId definition,
    const Options& options);

ktt::KernelId CreateProbeKernel(ktt::Tuner& tuner, const std::string& name, const ktt::KernelDefinitionId definition,
    const Options& options, const ktt::ArgumentId& countsId, const uint64_t blocks, const Configure& configure,
    const bool checkPlacement, StatsMap& stats, NvmlMonitor* monitor);

Row MakeRow(const ktt::KernelResult& result, const std::string& probe, const std::string& operation,
    const std::string& levelWithoutParameters, const StatsMap& stats, const Hardware& hardware, const Footprints& footprints);

std::vector<ktt::KernelResult> RunProbe(ktt::Tuner& tuner, const ktt::KernelId kernel, const std::string& label,
    const std::string& outputPrefix, const ktt::PreciseMeasurementParameters& precise);

// The program a probe kernel is compiled from: kernels/Common.cuh followed by the probe's file in `directory`, with #line
// directives so that compiler messages name the right file. Every probe file is a program of its own.
std::string KernelSource(const std::string& directory, const std::string& file);

// The buffers and scalars of the probe kernels (KTT argument ids). The scalars are set by the launchers for every
// configuration (see ProbeSettings); the kernel files describe what each one means.
// --misaligned: the random payload in a buffer of its own (driver API, in KTT's context), so that the global probe can
// read it from a start moved by some bytes off the 128-byte line. The probe's SASS stays the same: only the pointer it
// gets differs (a KTT argument per offset over this buffer).
class OffsetBuffer
{
public:
    OffsetBuffer(const uint32_t* data, size_t bytes, size_t slack, ktt::DeviceIndex device);
    ~OffsetBuffer();
    OffsetBuffer(const OffsetBuffer&) = delete;
    OffsetBuffer& operator=(const OffsetBuffer&) = delete;
    uint64_t Address() const { return static_cast<uint64_t>(m_Buffer); }

private:
    CUdeviceptr m_Buffer = 0;
};

struct ProbeArguments
{
    ktt::ArgumentId data;       // random payload, read by the loads and overwritten by the stores and atomics
    ktt::ArgumentId zeroData;   // payload of zeros (the same as data without --payload zero)
    ktt::ArgumentId sink;       // one word per thread, so that the compiler keeps the accesses
    ktt::ArgumentId counts;     // per-block accesses and active time, SM slots, launches
    ktt::ArgumentId smCounters; // per-SM slot counters of the probes with SM slots
    ktt::ArgumentId gridSplit;
    ktt::ArgumentId activeSms;
    ktt::ArgumentId idleRatio;
    ktt::ArgumentId payloadMask;
    ktt::ArgumentId operandA;
    ktt::ArgumentId operandC;
    ktt::ArgumentId duration;
    ktt::ArgumentId constantBytes;
    ktt::ArgumentId texture;
    ktt::ArgumentId sectorMask;
    ktt::ArgumentId runBits;
    ktt::ArgumentId stride;
    ktt::ArgumentId bankWays;
    ktt::ArgumentId fmaGroups;
    ktt::ArgumentId activeWarps; // probeLatency: working warps per block
    ktt::ArgumentId lineMask;    // probeLatency: lines of the footprint - 1
    ktt::ArgumentId zero;        // probeLatency: 0, the dependency of an address on the previous load's data
    std::optional<ktt::ArgumentId> constantData; // probeConstant's __constant__ array (a symbol), with --constant
    std::vector<ktt::ArgumentId> patternData;    // the payloads of --payload-bits (PAYLOAD 2, 3, ...)
    nlohmann::json payloadStatistics;            // bit statistics of every payload (PayloadStatistics), by name
    std::shared_ptr<OffsetBuffer> offsetBuffer;  // --misaligned: the payload's own buffer
    std::map<uint64_t, ktt::ArgumentId> offsetData; // --misaligned: the payload from OFFSET_BYTES on
};

// The data the probes read (`bytes` of it), random so that bits toggle as in real data.
std::vector<uint32_t> RandomPayload(const uint64_t bytes);

// Adds the arguments for `blocks` blocks of --block-size threads. `payload` stays referenced by KTT and must outlive the
// run.
ProbeArguments AddProbeArguments(ktt::Tuner& tuner, std::vector<uint32_t>& payload, const Options& options,
    const uint64_t blocks);

// One kernel definition per probe kernel, the optional ones only with their option.
struct ProbeDefinitions
{
    ktt::KernelDefinitionId sleep;
    ktt::KernelDefinitionId spin;
    ktt::KernelDefinitionId global;
    ktt::KernelDefinitionId shared;
    ktt::KernelDefinitionId compute;
    std::optional<ktt::KernelDefinitionId> constant;
    std::optional<ktt::KernelDefinitionId> texture;
    std::optional<ktt::KernelDefinitionId> scatter;
    std::optional<ktt::KernelDefinitionId> partial;
    std::optional<ktt::KernelDefinitionId> l1Sweep; // --l1-sweep
    std::optional<ktt::KernelDefinitionId> banks;
    std::optional<ktt::KernelDefinitionId> mixed;
    std::optional<ktt::KernelDefinitionId> write;
    std::optional<ktt::KernelDefinitionId> latency;
    std::optional<ktt::KernelDefinitionId> operands; // probeOperands, with --payload-bits
};

// Adds the definitions (grid = one block per SM, multiplied by the launchers' thread modifiers) and their arguments.
ProbeDefinitions AddProbeDefinitions(ktt::Tuner& tuner, const ProbeArguments& arguments, const Options& options,
    const uint32_t multiprocessors);

// The definition of probeMixed alone (also used by the control kernel of --validate, without the other probes).
ktt::KernelDefinitionId AddMixedDefinition(ktt::Tuner& tuner, const ProbeArguments& arguments, const Options& options,
    const uint32_t multiprocessors);

// The texture objects of probeTexture over the random and the zero payload (with --texture; zero only with --payload
// zero). Must be destroyed before the tuner.
struct ProbeTextures
{
    std::unique_ptr<PayloadTexture> random;
    std::unique_ptr<PayloadTexture> zero;
};

ProbeTextures CreateProbeTextures(const std::vector<uint32_t>& payload, const Options& options);

// What the launcher of each probe sets for a configuration: `common` the arguments of every probe (duty, active SMs,
// payload mask; all of probeSleep and probeShared), the others those of their probe on top.
struct ProbeSettings
{
    Configure common;
    Configure global;
    Configure texture;
    Configure scatter;
    Configure partial;
    Configure l1Sweep;
    Configure banks;
    Configure mixed;
    Configure write;
    Configure latency;
    Configure operands;
    Configure constant;
    Configure compute;
};

ProbeSettings MakeProbeSettings(const ProbeArguments& arguments, const ProbeDefinitions& definitions,
    const Footprints& footprints, const ProbeTextures& textures);
