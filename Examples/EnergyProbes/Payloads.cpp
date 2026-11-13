#include <algorithm>
#include <bitset>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "Payloads.h"

namespace
{

// The kind of a pattern and its number ("ones:0.25" -> "ones", 0.25; "sorted" -> "sorted", 0).
std::pair<std::string, double> Parse(const std::string& spec)
{
    const size_t colon = spec.find(':');

    if (colon == std::string::npos)
    {
        return {spec, 0.0};
    }

    try
    {
        return {spec.substr(0, colon), std::stod(spec.substr(colon + 1))};
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("--payload-bits: no number in " + spec);
    }
}

int OnesOf(const uint64_t word)
{
    return static_cast<int>(std::bitset<64>(word).count());
}

// A word with exactly `bits` one bits at random positions.
uint32_t WordWithBits(const int bits, std::mt19937_64& generator)
{
    std::vector<int> positions(32);

    for (int i = 0; i < 32; ++i)
    {
        positions[i] = i;
    }

    std::shuffle(positions.begin(), positions.end(), generator);
    uint32_t word = 0;

    for (int i = 0; i < bits; ++i)
    {
        word |= 1u << positions[i];
    }

    return word;
}

std::vector<uint32_t> Tile(const std::string& spec)
{
    const auto [kind, value] = Parse(spec);
    std::mt19937_64 generator(20261106);
    std::vector<uint32_t> tile(PatternTileWords);
    auto random = [&]() { return static_cast<uint32_t>(generator() >> 16); };

    if (kind == "ones")
    {
        // Each bit is 1 with probability value: compare 16 random bits per bit with the threshold.
        const uint64_t threshold = static_cast<uint64_t>(std::llround(value * 65536.0));

        for (auto& word : tile)
        {
            word = 0;

            for (int bit = 0; bit < 32; bit += 4)
            {
                const uint64_t r = generator();

                for (int k = 0; k < 4; ++k)
                {
                    word |= static_cast<uint32_t>(((r >> (16 * k)) & 0xffffu) < threshold) << (bit + k);
                }
            }
        }
    }
    else if (kind == "same")
    {
        std::fill(tile.begin(), tile.end(), WordWithBits(static_cast<int>(std::lround(value * 32)), generator));
    }
    else if (kind == "flip")
    {
        const int bits = static_cast<int>(std::lround(value * 32));
        tile[0] = random();

        for (size_t i = 1; i < tile.size(); ++i)
        {
            tile[i] = tile[i - 1] ^ WordWithBits(bits, generator);
        }
    }
    else if (kind == "unique")
    {
        std::vector<uint32_t> values(static_cast<size_t>(value));

        for (auto& v : values)
        {
            v = random();
        }

        for (auto& word : tile)
        {
            word = values[generator() % values.size()];
        }
    }
    else if (kind == "lsb0")
    {
        const int bits = static_cast<int>(value);
        const uint32_t mask = bits >= 32 ? 0u : ~((1u << bits) - 1u);

        for (auto& word : tile)
        {
            word = random() & mask;
        }
    }
    else if (kind == "sorted")
    {
        for (auto& word : tile)
        {
            word = random();
        }

        std::sort(tile.begin(), tile.end());
    }

    return tile;
}

struct Counter
{
    double ones = 0.0;
    double toggles = 0.0;
    double bits = 0.0;
    double toggleBits = 0.0;
    double lanes = 0.0;     // toggles between the same operand of neighbouring threads
    double laneBits = 0.0;

    void Word(const uint64_t word, const int width)
    {
        ones += OnesOf(word);
        bits += width;
    }

    void Pair(const uint64_t first, const uint64_t second, const int width)
    {
        toggles += OnesOf(first ^ second);
        toggleBits += width;
    }

    void Lane(const uint64_t first, const uint64_t second, const int width)
    {
        lanes += OnesOf(first ^ second);
        laneBits += width;
    }

    nlohmann::json Json() const
    {
        nlohmann::json result = {{"ones", ones / bits}, {"toggles", toggleBits > 0 ? toggles / toggleBits : 0.0}};

        if (laneBits > 0)
        {
            result["lanes"] = lanes / laneBits;
        }

        return result;
    }
};

uint64_t FloatOperand(const uint32_t word)
{
    return (word & 0x807fffffu) | 0x3f800000u;
}

uint64_t HalvesOperand(const uint32_t word)
{
    return (word & 0x83ff83ffu) | 0x3c003c00u;
}

uint64_t DoubleOperand(const uint32_t high, const uint32_t low)
{
    return (static_cast<uint64_t>((high & 0x800fffffu) | 0x3ff00000u) << 32) | low;
}

} // namespace

void CheckPayloadPattern(const std::string& spec)
{
    const auto [kind, value] = Parse(spec);
    const bool fraction = kind == "ones" || kind == "same" || kind == "flip";

    if ((fraction && (value < 0.0 || value > 1.0)) || (kind == "unique" && (value < 1.0 || value > 65536.0))
        || (kind == "lsb0" && (value < 0.0 || value > 32.0)) || (kind == "sorted" && spec != "sorted")
        || !(fraction || kind == "unique" || kind == "lsb0" || kind == "sorted"))
    {
        throw std::runtime_error("--payload-bits: " + spec + " is not ones:Q, same:Q, flip:T (0 to 1), unique:K, lsb0:N "
            "(0 to 32) or sorted");
    }
}

std::vector<uint32_t> PatternPayload(const std::string& spec, const size_t words)
{
    const std::vector<uint32_t> tile = Tile(spec);
    std::vector<uint32_t> result(words);

    for (size_t i = 0; i < words; ++i)
    {
        result[i] = tile[i % tile.size()];
    }

    return result;
}

nlohmann::json PayloadStatistics(const std::vector<uint32_t>& words)
{
    const size_t count = std::min(words.size(), PatternTileWords);
    Counter raw;
    // Toggles between words `lag` apart: 1 for neighbours, 8 for the beats of a 32-byte sector on a 256-bit path, 32
    // for the sectors of a 128-byte line.
    std::vector<size_t> lags = {2, 4, 8, 16, 32, 128};
    std::vector<Counter> lagged(lags.size());

    for (size_t i = 0; i < count; ++i)
    {
        raw.Word(words[i], 32);

        if (i > 0)
        {
            raw.Pair(words[i - 1], words[i], 32);
        }

        for (size_t k = 0; k < lags.size(); ++k)
        {
            if (i >= lags[k])
            {
                lagged[k].Word(words[i], 32);
                lagged[k].Pair(words[i - lags[k]], words[i], 32);
            }
        }
    }

    // The operands of probeOperands (kernels/Operands.cu): thread t reads from word 128 t; operand j of chain i in set s.
    Counter ffma;
    Counter dfma;
    Counter hmma;
    const size_t threads = count / 128;

    for (size_t t = 0; t < threads; ++t)
    {
        const uint32_t* w = words.data() + 128 * t;

        // The next thread's words, for the toggles between lanes (thread t + 1 of the same warp).
        const uint32_t* next = (t + 1) % 32 != 0 && t + 1 < threads ? w + 128 : nullptr;

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                const uint64_t first = FloatOperand(w[2 * (j + 3 * i)]);
                const uint64_t second = FloatOperand(w[2 * (j + 3 * i) + 1]);
                ffma.Word(first, 32);
                ffma.Word(second, 32);
                ffma.Pair(first, second, 32);

                if (next != nullptr)
                {
                    ffma.Lane(first, FloatOperand(next[2 * (j + 3 * i)]), 32);
                }
            }
        }

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                const uint64_t first = DoubleOperand(w[4 * (j + 3 * i) + 2], w[4 * (j + 3 * i)]);
                const uint64_t second = DoubleOperand(w[4 * (j + 3 * i) + 3], w[4 * (j + 3 * i) + 1]);
                dfma.Word(first, 64);
                dfma.Word(second, 64);
                dfma.Pair(first, second, 64);

                if (next != nullptr)
                {
                    dfma.Lane(first, DoubleOperand(next[4 * (j + 3 * i) + 2], next[4 * (j + 3 * i)]), 64);
                }
            }

            for (int r = 0; r < 10; ++r)
            {
                const bool halves = r < 6;
                const uint32_t a = w[2 * (r + 10 * i)];
                const uint32_t b = w[2 * (r + 10 * i) + 1];
                const uint64_t first = halves ? HalvesOperand(a) : FloatOperand(a);
                const uint64_t second = halves ? HalvesOperand(b) : FloatOperand(b);
                hmma.Word(first, 32);
                hmma.Word(second, 32);
                hmma.Pair(first, second, 32);

                if (next != nullptr)
                {
                    const uint32_t c = next[2 * (r + 10 * i)];
                    hmma.Lane(first, halves ? HalvesOperand(c) : FloatOperand(c), 32);
                }
            }
        }
    }

    nlohmann::json result = {{"words", raw.Json()}, {"ffma-bits", ffma.Json()}, {"dfma-bits", dfma.Json()},
        {"hmma-bits", hmma.Json()}};

    for (size_t k = 0; k < lags.size(); ++k)
    {
        result["words"]["toggles_lag" + std::to_string(lags[k])] = lagged[k].Json()["toggles"];
    }

    return result;
}

std::vector<std::string>& PayloadPatterns()
{
    static std::vector<std::string> patterns;
    return patterns;
}
