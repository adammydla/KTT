#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <json.hpp>

#include "EnergyModel.h"

namespace
{

constexpr int ModelVersion = 1;
constexpr double ClockTolerance = 0.05; // relative; beyond it a prediction is only good for ranking
constexpr double UnpricedShare = 0.01;  // of the warp instructions; from here an unpriced event gets a warning
const double NotANumber = std::numeric_limits<double>::quiet_NaN();

double NumberOr(const nlohmann::json& object, const char* key, const double fallback)
{
    return object.contains(key) && object.at(key).is_number() ? object.at(key).get<double>() : fallback;
}

std::string Percent(const double fraction)
{
    std::string text = std::to_string(std::round(1000.0 * fraction) / 10.0);
    return text.substr(0, text.find('.') + 2);
}

} // namespace

EnergyModel::EnergyModel(const std::string& file)
{
    std::ifstream input(file);

    if (!input)
    {
        throw std::runtime_error("Unable to open the model file " + file);
    }

    nlohmann::json root;
    input >> root;

    if (root.value("format", std::string()) != "energyprobes-model")
    {
        throw std::runtime_error(file + " is not an EnergyProbes model file");
    }

    if (root.value("version", 0) > ModelVersion)
    {
        throw std::runtime_error(file + " has a newer format version than this program knows");
    }

    const auto& device = root.at("device");
    m_Device = device.value("name", std::string());
    m_ClockMHz = NumberOr(device, "sm_clock_mhz", NotANumber);

    // The events in the order of energy_events.py, so that sums add up in the same order as there.
    for (const std::string name : {"inst alu", "inst fma (fp32, imad)", "inst fma lite (fp32)", "inst fp16", "inst fp64",
        "inst tensor", "inst xu (mufu)", "inst lsu shared", "inst lsu global", "inst lsu other", "inst tex", "inst other",
        "L1 sector", "L1 tex sector", "L2 read sector", "L2 write sector", "L2 read request", "L2 write request",
        "L2 read miss request", "L2 partial write miss request", "DRAM read sector", "DRAM write sector", "L2 atom sector",
        "L2 red sector", "shared ld wavefront", "shared st wavefront", "shared ld conflict", "shared st conflict",
        "shared atom wavefront", "const request", "const L1 miss", "DRAM busy ns"})
    {
        double energy = 0.0;

        if (root.at("events").contains(name))
        {
            const auto& entry = root.at("events").at(name);
            energy = entry.at("pj").get<double>();

            if (entry.value("status", std::string()) == "no probe")
            {
                m_Unpriced.push_back(name);
            }
        }

        m_Energies.emplace_back(name, energy);
    }

    for (const auto& item : root.at("events").items())
    {
        if (std::none_of(m_Energies.cbegin(), m_Energies.cend(), [&item](const auto& e) { return e.first == item.key(); }))
        {
            throw std::runtime_error(file + " has an event this program does not know: " + item.key());
        }
    }

    const auto& counters = root.at("counters");
    m_Required = counters.at("required").get<std::vector<std::string>>();
    m_LsuShared = counters.at("lsu_shared").get<std::vector<std::string>>();
    m_LsuGlobal = counters.at("lsu_global").get<std::vector<std::string>>();

    for (const auto& item : counters.at("aliases").items())
    {
        m_Aliases[item.value().get<std::string>()].push_back(item.key());
    }

    if (counters.contains("group_columns"))
    {
        for (const auto& item : counters.at("group_columns").items())
        {
            m_Groups.emplace_back(item.key(), item.value().get<std::string>());
        }
    }

    const auto& options = root.at("options");
    m_Busy = options.value("busy", true);
    m_BusyScaleGbs = NumberOr(options, "busy_scale_gbs", 95.0);
    m_FmaHalves = options.value("fma_halves", false);

    const auto& baseline = root.at("baseline");
    m_BaselineW = baseline.at("mean_w").get<double>();
    m_BaselineTemperatureC = baseline.at("mean_temperature_c").get<double>();
    m_BaselineSlope = baseline.at("slope_w_per_c").get<double>();

    const auto& uncertainty = root.at("uncertainty");
    m_RelativeSd = uncertainty.at("dynamic_relative_sd").get<double>();
    m_BaselineSdW = NumberOr(uncertainty, "baseline_sd_w", 0.0);
    m_UnknownTemperatureSdW = NumberOr(uncertainty, "baseline_unknown_temperature_sd_w", 0.0);
}

std::vector<std::string> EnergyModel::Counters() const
{
    std::vector<std::string> names = m_Required;

    for (const auto& group : m_Groups)
    {
        names.push_back(group.second);
    }

    for (const auto* list : {&m_LsuShared, &m_LsuGlobal})
    {
        names.insert(names.end(), list->cbegin(), list->cend());
    }

    // The lane counters and the two fp16 variants: optional, each GPU has the ones of its architecture.
    for (const char* name : {"sm__thread_inst_executed_pipe_alu_pred_on.sum", "sm__thread_inst_executed_pipe_fma_pred_on.sum",
        "sm__thread_inst_executed_pipe_fp16_pred_on.sum", "sm__thread_inst_executed_pipe_fma_type_fp16_pred_on.sum",
        "smsp__inst_executed_pipe_fp16.sum", "smsp__inst_executed_pipe_fma_type_fp16.sum",
        "sm__thread_inst_executed_pipe_fmalite_pred_on.sum"})
    {
        names.emplace_back(name);
    }

    const size_t own = names.size();

    for (size_t i = 0; i < own; ++i)
    {
        const auto aliases = m_Aliases.find(names[i]);

        if (aliases != m_Aliases.cend())
        {
            names.insert(names.end(), aliases->second.cbegin(), aliases->second.cend());
        }
    }

    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

double EnergyModel::Value(const CounterValues& counters, const std::string& name) const
{
    auto found = counters.find(name);

    if (found != counters.cend() && !std::isnan(found->second))
    {
        return found->second;
    }

    const auto aliases = m_Aliases.find(name);

    if (aliases != m_Aliases.cend())
    {
        for (const auto& alias : aliases->second)
        {
            found = counters.find(alias);

            if (found != counters.cend() && !std::isnan(found->second))
            {
                return found->second;
            }
        }
    }

    return NotANumber;
}

double EnergyModel::Count(const CounterValues& counters, const std::string& name) const
{
    const double value = Value(counters, name);
    return std::isnan(value) ? 0.0 : value;
}

std::vector<std::string> EnergyModel::MissingCounters(const CounterValues& counters) const
{
    std::vector<std::string> missing;

    for (const auto& name : m_Required)
    {
        if (std::isnan(Value(counters, name)))
        {
            missing.push_back(name);
        }
    }

    return missing;
}

double EnergyModel::BaselineW(const double temperatureC) const
{
    return std::isnan(temperatureC) ? m_BaselineW : m_BaselineW + m_BaselineSlope * (temperatureC - m_BaselineTemperatureC);
}

// The same as events() and add_busy() of tools/energy_events.py; the comments there say why each event is what it is.
std::vector<std::pair<std::string, double>> EnergyModel::Events(const CounterValues& c, const double kernelS) const
{
    const double total = Count(c, "smsp__inst_executed.sum");
    const double alu = Count(c, "smsp__inst_executed_pipe_alu.sum");
    const double fma = Count(c, "smsp__inst_executed_pipe_fma.sum");
    const bool separateFp16 = !std::isnan(Value(c, "smsp__inst_executed_pipe_fp16.sum"));
    const double fp16 = Count(c, separateFp16 ? "smsp__inst_executed_pipe_fp16.sum" : "smsp__inst_executed_pipe_fma_type_fp16.sum");
    const double fp64 = Count(c, "smsp__inst_executed_pipe_fp64.sum");
    const double tensor = Count(c, "smsp__inst_executed_pipe_tensor.sum");
    const double xu = Count(c, "smsp__inst_executed_pipe_xu.sum");
    const double lsu = Count(c, "smsp__inst_executed_pipe_lsu.sum");
    const double tex = Count(c, "smsp__inst_executed_pipe_tex.sum");

    const auto lanes = [this, &c](const std::string& name, const double warpCount)
    {
        const double value = Value(c, name);
        return std::isnan(value) ? warpCount : value / 32.0;
    };

    const double aluLanes = lanes("sm__thread_inst_executed_pipe_alu_pred_on.sum", alu);
    const double fmaLanes = lanes("sm__thread_inst_executed_pipe_fma_pred_on.sum", fma);
    const double fp16Lanes = lanes(separateFp16 ? "sm__thread_inst_executed_pipe_fp16_pred_on.sum"
        : "sm__thread_inst_executed_pipe_fma_type_fp16_pred_on.sum", fp16);
    const double ldConflicts = Count(c, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum");
    const double stConflicts = Count(c, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum");
    const double atomicMisses = Count(c, "lts__t_requests_srcunit_tex_op_atom_lookup_miss.sum")
        + Count(c, "lts__t_requests_srcunit_tex_op_red_lookup_miss.sum");
    double sharedInst = 0.0;
    double globalInst = 0.0;

    for (const auto& name : m_LsuShared)
    {
        sharedInst += Count(c, name);
    }

    for (const auto& name : m_LsuGlobal)
    {
        globalInst += Count(c, name);
    }

    const double fmaPipe = separateFp16 ? fmaLanes : std::max(fmaLanes - fp16Lanes, 0.0);
    const double liteValue = Value(c, "sm__thread_inst_executed_pipe_fmalite_pred_on.sum");
    const double lite = m_FmaHalves && !std::isnan(liteValue)
        ? std::min(std::max(liteValue / 32.0 - fp16Lanes, 0.0), fmaPipe) : 0.0;
    const double dramRead = Count(c, "dram__sectors_read.sum");
    const double dramWrite = Count(c, "dram__sectors_write.sum");
    double busyNs = 0.0;
    const double ns = kernelS * 1e9;

    if (m_Busy && ns > 0.0)
    {
        const double rate = (dramRead + dramWrite) * 32.0 / ns; // GB/s
        busyNs = ns * -std::expm1(-rate / m_BusyScaleGbs);
    }

    return {
        {"inst alu", aluLanes},
        {"inst fma (fp32, imad)", fmaPipe - lite},
        {"inst fma lite (fp32)", lite},
        {"inst fp16", fp16Lanes},
        {"inst fp64", fp64},
        {"inst tensor", tensor},
        {"inst xu (mufu)", xu},
        {"inst lsu shared", sharedInst},
        {"inst lsu global", globalInst},
        {"inst lsu other", std::max(lsu - sharedInst - globalInst, 0.0)},
        {"inst tex", tex},
        {"inst other", std::max(total - alu - fma - fp64 - tensor - xu - lsu - tex - (separateFp16 ? fp16 : 0.0), 0.0)},
        {"L1 sector", Count(c, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
            + Count(c, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
            + Count(c, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum")
            + Count(c, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")},
        {"L1 tex sector", Count(c, "l1tex__t_sectors_pipe_tex_mem_texture.sum")},
        {"L2 read sector", Count(c, "lts__t_sectors_srcunit_tex_op_read.sum")},
        {"L2 write sector", Count(c, "lts__t_sectors_srcunit_tex_op_write.sum")},
        {"L2 read request", Count(c, "lts__t_requests_srcunit_tex_op_read.sum")},
        {"L2 write request", Count(c, "lts__t_requests_srcunit_tex_op_write.sum")},
        {"L2 read miss request", Count(c, "lts__t_requests_srcunit_tex_op_read_lookup_miss.sum") + atomicMisses},
        {"L2 partial write miss request", std::max(Count(c, "lts__t_requests_srcunit_tex_op_write_lookup_miss.sum")
            + atomicMisses - dramWrite / 4.0, 0.0)},
        {"DRAM read sector", dramRead},
        {"DRAM write sector", dramWrite},
        {"L2 atom sector", Count(c, "lts__t_sectors_srcunit_tex_op_atom.sum")},
        {"L2 red sector", Count(c, "lts__t_sectors_srcunit_tex_op_red.sum")},
        {"shared ld wavefront", std::max(Count(c, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") - ldConflicts, 0.0)},
        {"shared st wavefront", std::max(Count(c, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum") - stConflicts, 0.0)},
        {"shared ld conflict", ldConflicts},
        {"shared st conflict", stConflicts},
        {"shared atom wavefront", Count(c, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum")},
        {"const request", Count(c, "idc__requests.sum")},
        {"const L1 miss", Count(c, "idc__requests_lookup_miss_tag_miss.sum")},
        {"DRAM busy ns", busyNs},
    };
}

EnergyPrediction EnergyModel::Predict(const CounterValues& counters, const double kernelS, const double temperatureC,
    const double smClockMHz, const bool strict) const
{
    if (!(kernelS > 0.0))
    {
        throw std::runtime_error("EnergyModel::Predict: the kernel's time must be positive");
    }

    EnergyPrediction prediction;
    prediction.kernelS = kernelS;
    const std::vector<std::string> missing = MissingCounters(counters);

    if (!missing.empty())
    {
        std::string text = "counters the model needs are missing:";

        for (const auto& name : missing)
        {
            text += " " + name;
        }

        if (strict)
        {
            throw std::runtime_error(text);
        }

        prediction.warnings.push_back(text + "; their events count as 0");
    }

    for (const auto& group : m_Groups)
    {
        if (std::isnan(Value(counters, group.second)))
        {
            const std::string text = "the model was fitted with the " + group.first + ", which these counters do not have";

            if (strict)
            {
                throw std::runtime_error(text);
            }

            prediction.warnings.push_back(text);
        }
    }

    prediction.events = Events(counters, kernelS);
    const double instructions = Count(counters, "smsp__inst_executed.sum");

    for (size_t i = 0; i < prediction.events.size(); ++i)
    {
        const auto& [name, count] = prediction.events[i];
        prediction.dynamicJ += count * m_Energies[i].second * 1e-12;

        if (count > 0.0 && std::find(m_Unpriced.cbegin(), m_Unpriced.cend(), name) != m_Unpriced.cend())
        {
            const bool instruction = name.rfind("inst", 0) == 0;

            if (!instruction)
            {
                prediction.warnings.push_back(name + " occurs, but no probe priced it (counts as free)");
            }
            else if (instructions > 0.0 && count / instructions >= UnpricedShare)
            {
                prediction.warnings.push_back(name + ": " + Percent(count / instructions)
                    + " % of the warp instructions, but no probe priced it (counts as free)");
            }
        }
    }

    if (!std::isnan(smClockMHz) && !std::isnan(m_ClockMHz) && std::abs(smClockMHz / m_ClockMHz - 1.0) > ClockTolerance)
    {
        prediction.warnings.push_back("SM clock " + std::to_string(std::lround(smClockMHz)) + " MHz, model calibrated at "
            + std::to_string(std::lround(m_ClockMHz)) + " MHz: the prediction is only good for ranking");
    }

    prediction.baselineJ = BaselineW(temperatureC) * kernelS;
    prediction.energyJ = prediction.baselineJ + prediction.dynamicJ;
    double baselineSd = m_BaselineSdW * kernelS;

    if (std::isnan(temperatureC))
    {
        baselineSd = std::hypot(baselineSd, m_UnknownTemperatureSdW * kernelS);
    }

    prediction.sdJ = std::hypot(m_RelativeSd * prediction.dynamicJ, baselineSd);

    if (!std::isnan(m_PowerLimitW) && prediction.energyJ > m_PowerLimitW * kernelS)
    {
        prediction.warnings.push_back("predicted power " + std::to_string(std::lround(prediction.energyJ / kernelS))
            + " W is above the GPU's limit of " + std::to_string(std::lround(m_PowerLimitW))
            + " W: the GPU will lower its clock; energy taken as the limit times the time");
        prediction.energyJ = m_PowerLimitW * kernelS;
        prediction.dynamicJ = std::max(prediction.energyJ - prediction.baselineJ, 0.0);
    }

    return prediction;
}

double ObjectiveValue(const EnergyPrediction& prediction, const EnergyObjective objective)
{
    switch (objective)
    {
    case EnergyObjective::Energy:
        return prediction.energyJ;
    case EnergyObjective::Edp:
        return prediction.energyJ * prediction.kernelS;
    case EnergyObjective::Ed2p:
        return prediction.energyJ * prediction.kernelS * prediction.kernelS;
    default:
        return prediction.kernelS;
    }
}

EnergyRecommendation EnergyModel::Recommend(const std::vector<EnergyPrediction>& predictions, const EnergyObjective objective,
    const double z, const double maxSlowdown, const double maxEnergyIncrease) const
{
    if (predictions.empty())
    {
        throw std::runtime_error("EnergyModel::Recommend: no predictions");
    }

    double fastest = predictions[0].kernelS;
    double cheapest = predictions[0].energyJ;

    for (const auto& p : predictions)
    {
        fastest = std::min(fastest, p.kernelS);
        cheapest = std::min(cheapest, p.energyJ);
    }

    EnergyRecommendation result;
    result.values.assign(predictions.size(), NotANumber);
    bool any = false;

    for (size_t i = 0; i < predictions.size(); ++i)
    {
        const auto& p = predictions[i];

        if ((std::isnan(maxSlowdown) || p.kernelS <= fastest * (1.0 + maxSlowdown))
            && (std::isnan(maxEnergyIncrease) || p.energyJ <= cheapest * (1.0 + maxEnergyIncrease)))
        {
            result.values[i] = ObjectiveValue(p, objective);

            if (!any || result.values[i] < result.values[result.index])
            {
                result.index = i;
                any = true;
            }
        }
    }

    if (objective == EnergyObjective::Time)
    {
        return result;
    }

    const double baselineSd = std::hypot(m_BaselineSdW, m_UnknownTemperatureSdW);
    const auto& best = predictions[result.index];
    const double scaleBest = result.values[result.index] / best.energyJ;

    for (size_t i = 0; i < predictions.size(); ++i)
    {
        if (i == result.index || std::isnan(result.values[i]))
        {
            continue;
        }

        // The dynamic errors of two configurations are independent; a baseline error is common to both and acts on the
        // difference of their times.
        const auto& p = predictions[i];
        const double scale = result.values[i] / p.energyJ;
        const double sd = std::sqrt(std::pow(m_RelativeSd * p.dynamicJ * scale, 2) + std::pow(m_RelativeSd * best.dynamicJ * scaleBest, 2)
            + std::pow(baselineSd * (p.kernelS * scale - best.kernelS * scaleBest), 2));

        if (result.values[i] - result.values[result.index] <= z * sd)
        {
            result.contenders.push_back(i);
        }
    }

    std::sort(result.contenders.begin(), result.contenders.end(), [&result](const size_t a, const size_t b)
    {
        return result.values[a] < result.values[b];
    });
    return result;
}
