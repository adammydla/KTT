#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <json.hpp>

#include "CounterTable.h"

namespace
{

std::map<std::string, std::string>& Canonical()
{
    static std::map<std::string, std::string> canonical;
    return canonical;
}

} // namespace

std::vector<std::string> ModelCounters(const std::string& tableFile, const bool study)
{
    std::ifstream input(tableFile);

    if (!input)
    {
        throw std::runtime_error("Unable to open the counter table " + tableFile + " (see --counter-table)");
    }

    nlohmann::json root;
    input >> root;
    std::vector<std::string> names;

    for (const auto& entry : root.at("counters"))
    {
        const std::string name = entry.at("name").get<std::string>();

        // Screening counters (NEXT_STEPS_PLAN M3, M4): only with --study-counters, so that other runs collect what they
        // did before (SASS counters are collected by patching the kernel and add profiling passes).
        if (entry.contains("study") && entry.at("study").get<bool>() && !study)
        {
            continue;
        }

        names.push_back(name);

        if (entry.contains("alias_of"))
        {
            Canonical()[name] = entry.at("alias_of").get<std::string>();
        }
    }

    if (names.empty())
    {
        throw std::runtime_error("The counter table " + tableFile + " lists no counters");
    }

    return names;
}

std::string CanonicalCounter(const std::string& name)
{
    const auto iterator = Canonical().find(name);
    return iterator != Canonical().cend() ? iterator->second : name;
}
