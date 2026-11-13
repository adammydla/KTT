// Payloads with controlled bits (--payload-bits): data with a chosen fraction of one bits and of bits that toggle between
// consecutive words, for a curve of energy over the data instead of only zero against random (VAMPIRE's model of DRAM
// energy, Ghose et al., SIGMETRICS 2018, is E = E0 + a x ones + b x toggles). Patterns:
//   ones:Q    every bit 1 with probability Q, independently (ones Q, toggles 2Q(1-Q)); ones:0.5 is the random payload
//   same:Q    one word with round(32 Q) one bits, repeated (ones Q, toggles 0)
//   flip:T    each word is the previous one with round(32 T) random bits flipped (ones 1/2, toggles T); flip:1 alternates
//             a word and its complement
//   unique:K  K distinct random words, each position one of them at random (few unique values, as Gregersen et al.)
//   lsb0:N    random words with the N low bits zeroed (as Gregersen et al.'s zeroed least significant bits)
//   sorted    random words in ascending order (per tile): neighbours share their high bits
// A tile of PatternTileWords words is generated from a fixed seed and repeated, so a pattern's statistics are the same
// in every part of the footprint.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <json.hpp>

inline constexpr size_t PatternTileWords = size_t{1} << 20;

// Throws for a pattern that is not one of the above.
void CheckPayloadPattern(const std::string& spec);

// The words of a pattern, `words` of them.
std::vector<uint32_t> PatternPayload(const std::string& spec, const size_t words);

// Bit statistics of a payload: the fraction of one bits and of bits that toggle between consecutive words, and the same
// for the operands that probeOperands builds from it (kernels/Operands.cu: FFMA, DFMA and HMMA operands with their fixed
// exponents, toggles between the two operand sets that alternate).
nlohmann::json PayloadStatistics(const std::vector<uint32_t>& words);

// The names of the --payload-bits patterns of this run, in the order of their PAYLOAD codes (2, 3, ...); PayloadName()
// reads them.
std::vector<std::string>& PayloadPatterns();
