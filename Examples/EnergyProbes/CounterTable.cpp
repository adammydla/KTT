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

std::vector<std::string> ModelCounters(const std::string& tableFile, const double computeCapability)
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

        if (computeCapability > 0.0 && entry.contains("min_cc") && entry.at("min_cc").get<double>() > computeCapability + 1e-9)
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
