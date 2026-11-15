#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "Common.h"

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
        "ffma-indexed", "ffma3", "shfl", "bar", "ffma-loop"};
    return computeOp < names.size() ? names[computeOp] : "?";
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
