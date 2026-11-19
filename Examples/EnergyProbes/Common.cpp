#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "Common.h"

std::string ClockMismatch(const std::string& domain, const double expectedMHz, const double observedMHz,
    const std::vector<double>& supportedMHz, const double fallbackStepMHz)
{
    if (std::isnan(expectedMHz) || std::isnan(observedMHz) || expectedMHz <= 0.0)
    {
        return "";
    }

    double step = fallbackStepMHz;
    std::vector<double> clocks = supportedMHz;
    std::sort(clocks.begin(), clocks.end());
    clocks.erase(std::unique(clocks.begin(), clocks.end()), clocks.end());

    if (clocks.size() >= 2)
    {
        const auto nearest = std::min_element(clocks.cbegin(), clocks.cend(),
            [expectedMHz](const double a, const double b) { return std::abs(a - expectedMHz) < std::abs(b - expectedMHz); });
        step = std::numeric_limits<double>::infinity();

        if (nearest != clocks.cbegin())
        {
            step = std::min(step, *nearest - *(nearest - 1));
        }

        if (nearest + 1 != clocks.cend())
        {
            step = std::min(step, *(nearest + 1) - *nearest);
        }
    }

    // Half a megahertz more for NVML's rounding of the reported clock.
    if (std::abs(observedMHz - expectedMHz) <= step + 0.5)
    {
        return "";
    }

    std::ostringstream text;
    text << "WARNING: the " << domain << " clock ran at " << Fixed(observedMHz, 0) << " MHz under load, not at the expected "
         << Fixed(expectedMHz, 0) << " MHz (more than one clock step of " << Fixed(step, 0) << " MHz). The GPU may not "
         << "apply this clock setting; the rows are labelled with the clock they ran at.";
    return text.str();
}

std::string OperationName(const uint64_t loadOp)
{
    switch (loadOp)
    {
    case LoadDefault: return "ld";
    case LoadNone: return "no-load";
    case StoreDefault: return "st";
    case AtomicRed: return "red";
    case AtomicAdd: return "atom";
    default: return "?";
    }
}

std::string ComputeName(const uint64_t computeOp)
{
    const std::vector<std::string> names = {"loop", "ffma", "dfma", "imad", "hfma2", "hmma", "mufu", "ffma-static",
        "ffma-indexed", "ffma3", "shfl", "bar", "ffma-loop", "imma", "dmma", "hmma-sparse", "qmma-fp8", "opcode"};
    return computeOp < names.size() ? names[computeOp] : "?";
}

std::string OpcodeName(const uint64_t opcode)
{
    const std::vector<std::string> names = {"?", "fadd", "fmul", "ffma-rr", "dadd", "dmul", "dfma-rr", "iadd3", "lop3", "shf",
        "mufu-ex2", "mufu-lg2", "mufu-sin", "mufu-rcp", "mufu-sqrt", "mufu-tanh", "i2f", "f2i", "popc", "f2f", "ffma-same",
        "ffma-reuse1", "ffma-reuse2"};
    return opcode < names.size() ? names[opcode] : "?";
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
    for (const auto& [unit, name] : {std::pair{GiB, " GiB"}, std::pair{MiB, " MiB"}, std::pair{KiB, " KiB"}})
    {
        if (bytes >= unit && bytes % unit == 0)
        {
            return std::to_string(bytes / unit) + name;
        }
    }

    return std::to_string(bytes) + " B";
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

Line FitLine(const std::vector<std::pair<double, double>>& points)
{
    Line line;
    const double n = static_cast<double>(points.size());
    double meanX = 0.0;
    double meanY = 0.0;

    for (const auto& [x, y] : points)
    {
        meanX += x / n;
        meanY += y / n;
    }

    double sxx = 0.0;
    double sxy = 0.0;
    double syy = 0.0;

    for (const auto& [x, y] : points)
    {
        sxx += (x - meanX) * (x - meanX);
        sxy += (x - meanX) * (y - meanY);
        syy += (y - meanY) * (y - meanY);
    }

    if (points.size() < 2 || sxx <= 0.0)
    {
        return line;
    }

    line.slope = sxy / sxx;
    line.intercept = meanY - line.slope * meanX;
    line.r2 = points.size() >= 3 && syy > 0.0 ? sxy * sxy / (sxx * syy) : NaN;
    return line;
}
