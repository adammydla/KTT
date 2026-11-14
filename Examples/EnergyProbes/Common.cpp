#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <sstream>

#include "Common.h"
#include "Payloads.h"

std::string PayloadName(const uint64_t payload)
{
    if (payload >= PayloadPatternFirst && payload - PayloadPatternFirst < PayloadPatterns().size())
    {
        return PayloadPatterns()[payload - PayloadPatternFirst];
    }

    return payload == PayloadZero ? "zero" : "random";
}

std::string OperandName(const uint64_t operation)
{
    const std::vector<std::string> names = {"ffma-bits", "dfma-bits", "hmma-bits"};
    return operation < names.size() ? names[operation] : "?";
}

std::string ComputeName(const uint64_t operation)
{
    const std::vector<std::string> names = {"loop", "ffma", "dfma", "imad", "hfma2", "hmma", "mufu", "ffma-static",
        "ffma-indexed", "ffma3", "shfl", "bar", "ffma-loop", "fmul-fadd", "mufu-sin", "ffma-pred", "ffma-branch", "iadd-pred",
        "iadd-branch", "lop3-xor", "shf-rotate", "imnmx-xor", "fadd", "fmul", "ffma-rcp-fadd", "ffma-sqrt-fadd", "ffma-ex2-fadd",
        "ffma-lg2-fadd", "ffma-rsq-fadd", "isetp-sel", "fsetp-sel", "cvt-i2f-f2i", "fmnmx", "iadd3", "ffma-pred-rotating"};
    return operation < names.size() ? names[operation] : "?";
}

double FlopsPerOperation(const uint64_t operation, const uint32_t computeCapabilityMajor)
{
    switch (operation)
    {
    case ComputeFfma:
    case ComputeDfma:
    case ComputeImad:
    case ComputeStaticIndex:
    case ComputeDynamicIndex:
    case ComputeFfma3:
    case ComputeFfmaLoop:
    case ComputeFmulFadd:
        return 2.0;
    case ComputeFfmaLanes:
    case ComputeFfmaLanesBranch:
    case ComputeFfmaLanesRotating:
        return NaN; // counted per thread, active or not
    case ComputeHfma2:
        return 4.0; // two FMAs on half2
    case ComputeHmma:
        // m16n8k16 per warp from sm_80 (4096 FLOPs), m16n8k8 on sm_75 and 4 x m8n8k4 on sm_70 (2048), shared by 32 threads
        return computeCapabilityMajor >= 8 || computeCapabilityMajor == 0 ? 2.0 * 16 * 8 * 16 / 32 : 2.0 * 16 * 8 * 8 / 32;
    default: // MUFU (rsqrt or sin, plus an FFMA), SHFL and BAR count no FLOPs
        return NaN;
    }
}

std::string OperationName(const uint64_t operation)
{
    switch (operation)
    {
    case LoadDefault:
        return "ld";
    case LoadCg:
        return "ld.cg";
    case LoadNone:
        return "no-load";
    case LoadCa:
        return "ld.ca";
    case LoadNc:
        return "ld.nc";
    case StoreDefault:
        return "st";
    case AtomicRed:
        return "red";
    case AtomicAdd:
        return "atom";
    default:
        return "?";
    }
}

bool IsPowerOfTwo(const uint64_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

uint64_t FloorPowerOfTwo(const uint64_t value)
{
    uint64_t result = 1;

    while (result * 2 <= value)
    {
        result *= 2;
    }

    return result;
}

uint64_t CeilPowerOfTwo(const uint64_t value)
{
    uint64_t result = 1;

    while (result < value)
    {
        result *= 2;
    }

    return result;
}

std::string FormatBytes(const uint64_t bytes)
{
    std::ostringstream stream;

    if (bytes >= GiB && bytes % GiB == 0)
    {
        stream << bytes / GiB << " GiB";
    }
    else if (bytes >= MiB && bytes % MiB == 0)
    {
        stream << bytes / MiB << " MiB";
    }
    else if (bytes >= KiB && bytes % KiB == 0)
    {
        stream << bytes / KiB << " KiB";
    }
    else
    {
        stream << bytes << " B";
    }

    return stream.str();
}

std::string Fixed(const double value, const int precision)
{
    if (std::isnan(value))
    {
        return "-";
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string CsvValue(const double value, const int precision)
{
    if (std::isnan(value))
    {
        return "";
    }

    std::ostringstream stream;
    stream << std::setprecision(precision) << value;
    return stream.str();
}

std::string CsvText(std::string text)
{
    std::replace_if(text.begin(), text.end(), [](const char c) { return c == ',' || c == '\n' || c == '\r'; }, ' ');
    return text;
}

void AppendNote(std::string& note, const std::string& text)
{
    note += (note.empty() ? "" : "; ") + text;
}

std::string SignedPercent(const double fraction)
{
    return (fraction > 0.0 ? "+" : "") + Fixed(fraction * 100.0, 1) + "%";
}

std::string JoinList(const std::vector<uint64_t>& values, const std::string& separator)
{
    std::string result;

    for (const uint64_t value : values)
    {
        result += (result.empty() ? "" : separator) + std::to_string(value);
    }

    return result;
}

nlohmann::json ToJson(const double value)
{
    return std::isnan(value) ? nlohmann::json() : nlohmann::json(value);
}

uint32_t FloatBits(const float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::vector<uint64_t> BankWaysOfWidth(const uint64_t width)
{
    std::vector<uint64_t> ways;
    std::copy_if(BankWays.cbegin(), BankWays.cend(), std::back_inserter(ways),
        [width](const uint64_t degree) { return degree * width <= WavefrontBytes; });
    return ways;
}

std::string MixedName(const uint64_t fmasPerLoad)
{
    return "ld-ffma" + std::to_string(fmasPerLoad);
}

std::string WriteName(const uint64_t operation)
{
    switch (operation)
    {
    case WriteCopy:
        return "copy";
    case WriteUpdate:
        return "update";
    case WriteLocal:
        return "local";
    default:
        return "write" + std::to_string(operation);
    }
}

std::string LatencyName(const uint64_t chains, const uint64_t warpsPerSm)
{
    return "chase-c" + std::to_string(chains) + "-w" + std::to_string(warpsPerSm);
}

bool IsWriteOperation(const std::string& operation)
{
    return operation == "copy" || operation == "update" || operation == "local";
}

std::optional<uint64_t> MixedFmasOf(const std::string& operation)
{
    const std::string prefix = "ld-ffma";

    if (operation.size() <= prefix.size() || operation.compare(0, prefix.size(), prefix) != 0
        || operation.find_first_not_of("0123456789", prefix.size()) != std::string::npos)
    {
        return std::nullopt;
    }

    return std::stoull(operation.substr(prefix.size()));
}
