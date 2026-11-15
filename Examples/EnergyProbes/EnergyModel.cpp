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
// How closely a GPU at its power limit holds it (energy_predictor.py LIMIT_RELATIVE_SD: ValFree1, sd 2.1 %).
constexpr double LimitRelativeSd = 0.021;
const double NotANumber = std::numeric_limits<double>::quiet_NaN();

// The keys a model file must have; the same list as energy_predictor.py REQUIRED_KEYS.
const std::vector<std::vector<std::string>> RequiredKeys = {{"device"}, {"options"}, {"events"}, {"baseline", "mean_w"},
    {"baseline", "mean_temperature_c"}, {"baseline", "slope_w_per_c"}, {"uncertainty", "dynamic_relative_sd"},
    {"counters", "required"}, {"counters", "lsu_shared"}, {"counters", "lsu_global"}, {"counters", "aliases"}};

// Optional counters for model files that do not list them (energy_predictor.py OPTIONAL_COUNTERS).
const std::vector<std::string> DefaultOptional = {"sm__thread_inst_executed_pipe_alu_pred_on.sum",
    "sm__thread_inst_executed_pipe_fma_pred_on.sum", "sm__thread_inst_executed_pipe_fp16_pred_on.sum",
    "sm__thread_inst_executed_pipe_fma_type_fp16_pred_on.sum", "smsp__inst_executed_pipe_fp16.sum",
    "smsp__inst_executed_pipe_fma_type_fp16.sum"};

// A counter value for a message, without trailing zeros.
std::string ValueText(const double value)
{
    if (std::isinf(value))
    {
        return value > 0.0 ? "inf" : "-inf";
    }

    std::string text = std::to_string(value);
    text.erase(text.find_last_not_of('0') + 1);
    return text.back() == '.' ? text.substr(0, text.size() - 1) : text;
}

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

EnergyModel::EnergyModel(const std::string& file) :
    m_File(file)
{
    std::ifstream input(file);

    if (!input)
    {
        throw std::runtime_error("Unable to open the model file " + file);
    }

    nlohmann::json root;

    try
    {
        input >> root;
    }
    catch (const nlohmann::json::exception& error)
    {
        throw std::runtime_error(file + ": not JSON (" + error.what() + ")");
    }

    if (!root.is_object() || root.value("format", std::string()) != "energyprobes-model")
    {
        throw std::runtime_error(file + " is not an EnergyProbes model file");
    }

    if (root.value("version", 0) > ModelVersion)
    {
        throw std::runtime_error(file + " has a newer format version than this program knows");
    }

    for (const auto& key : RequiredKeys)
    {
        const nlohmann::json* entry = &root;
        std::string path;

        for (const auto& part : key)
        {
            path += (path.empty() ? "" : "/") + part;

            if (!entry->is_object() || !entry->contains(part))
            {
                throw std::runtime_error(file + ": missing " + path);
            }

            entry = &entry->at(part);
        }
    }

    try
    {
        Load(root);
    }
    catch (const nlohmann::json::exception& error)
    {
        throw std::runtime_error(file + ": " + error.what());
    }
}

void EnergyModel::Load(const nlohmann::json& root)
{
    const std::string& file = m_File;

    const auto& device = root.at("device");
    m_Device = device.value("name", std::string());
    m_ClockMHz = NumberOr(device, "sm_clock_mhz", NotANumber);

    // The events in the order of energy_events.py (Events()), so that sums add up in the same order as there.
    for (const auto& [name, count] : Events(CounterValues{}, 0.0))
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

    m_BusyScaleGbs = NumberOr(root.at("options"), "busy_scale_gbs", 95.0);

    const auto& baseline = root.at("baseline");
    m_BaselineW = baseline.at("mean_w").get<double>();
    m_BaselineTemperatureC = baseline.at("mean_temperature_c").get<double>();
    m_BaselineSlope = baseline.at("slope_w_per_c").get<double>();

    const auto& uncertainty = root.at("uncertainty");
    m_RelativeSd = uncertainty.at("dynamic_relative_sd").get<double>();
    m_BaselineSdW = NumberOr(uncertainty, "baseline_sd_w", 0.0);
    m_UnknownTemperatureSdW = NumberOr(uncertainty, "baseline_unknown_temperature_sd_w", 0.0);
    m_BetweenKernelsSd = NumberOr(uncertainty, "dynamic_bias_sd_between_kernels", NotANumber);
    m_ContextIdleW = NumberOr(baseline, "context_idle_w", NotANumber);
    m_Optional = counters.contains("optional") && counters.at("optional").is_array()
        ? counters.at("optional").get<std::vector<std::string>>() : DefaultOptional;
}

double EnergyModel::IdlePowerW() const
{
    if (!std::isnan(m_IdleW))
    {
        return m_IdleW;
    }

    if (std::isnan(m_ContextIdleW))
    {
        throw std::runtime_error(m_File + ": no idle power in the file (rebuild it with tools/build_model.py, or give "
            "the idle power: --idle-w)");
    }

    return m_ContextIdleW;
}

double EnergyModel::BetweenKernelsSd() const
{
    return std::isnan(m_BetweenKernelsSd) ? m_RelativeSd : m_BetweenKernelsSd;
}

double EnergyModel::KernelScale(const std::string& group, const std::vector<EnergyPrediction>& predictions,
    const std::vector<double>& measuredDynamicJ, const size_t minimum)
{
    double sum = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < predictions.size() && i < measuredDynamicJ.size(); ++i)
    {
        if (predictions[i].unscaledDynamicJ > 0.0 && measuredDynamicJ[i] > 0.0)
        {
            sum += measuredDynamicJ[i] / predictions[i].unscaledDynamicJ;
            ++count;
        }
    }

    if (count < minimum || count == 0)
    {
        return NotANumber;
    }

    m_KernelScales[group] = {sum / static_cast<double>(count), count};
    return m_KernelScales[group].first;
}

EnergyAccounting AccountingOf(const std::string& name)
{
    if (name == "awake")
    {
        return EnergyAccounting::Awake;
    }

    if (name == "above-idle")
    {
        return EnergyAccounting::AboveIdle;
    }

    if (name == "dynamic")
    {
        return EnergyAccounting::Dynamic;
    }

    throw std::runtime_error("the accounting must be awake, above-idle or dynamic, not " + name);
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
    names.insert(names.end(), m_Optional.cbegin(), m_Optional.cend());

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
    // Negative values count as 0 (energy_events.counter): CUPTI reports the L2 reduction counters negative now and then.
    const double value = Value(counters, name);
    return std::isnan(value) || value < 0.0 ? 0.0 : value;
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

    const double dramRead = Count(c, "dram__sectors_read.sum");
    const double dramWrite = Count(c, "dram__sectors_write.sum");
    double busyNs = 0.0;
    const double ns = kernelS * 1e9;

    if (ns > 0.0)
    {
        const double rate = (dramRead + dramWrite) * 32.0 / ns; // GB/s
        busyNs = ns * -std::expm1(-rate / m_BusyScaleGbs);
    }

    return {
        {"inst alu", aluLanes},
        {"inst fma (fp32, imad)", separateFp16 ? fmaLanes : std::max(fmaLanes - fp16Lanes, 0.0)},
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

EnergyPrediction EnergyModel::Predict(const CounterValues& input, const double kernelS, const double temperatureC,
    const double smClockMHz, const bool strict, const std::string& group) const
{
    if (!(kernelS > 0.0) || std::isinf(kernelS))
    {
        throw std::runtime_error("EnergyModel::Predict: the kernel's time must be positive and finite");
    }

    EnergyPrediction prediction;
    prediction.kernelS = kernelS;
    prediction.group = group;

    // Negative counter values count as 0 with a warning (CUPTI reports some); values that are not finite are an error
    // with strict, else the same.
    CounterValues counters = input;
    std::vector<std::string> negative;
    std::vector<std::string> infinite;

    for (const auto& name : Counters())
    {
        const auto found = counters.find(name);

        if (found != counters.end() && (found->second < 0.0 || std::isinf(found->second)))
        {
            (std::isinf(found->second) ? infinite : negative).push_back(name);
        }
    }

    if (!infinite.empty() && strict)
    {
        std::string text = "counter values that are not finite:";

        for (const auto& name : infinite)
        {
            text += " " + name + " = " + ValueText(counters.at(name));
        }

        throw std::runtime_error(text);
    }

    if (!negative.empty() || !infinite.empty())
    {
        std::string text = "counter values that are negative or not finite count as 0:";

        for (const auto* list : {&negative, &infinite})
        {
            for (const auto& name : *list)
            {
                text += " " + name + " = " + ValueText(counters.at(name));
                counters[name] = 0.0;
            }
        }

        prediction.warnings.push_back(text);
    }

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
    const auto scaleEntry = m_KernelScales.find(group);
    const double scale = scaleEntry != m_KernelScales.cend() ? scaleEntry->second.first : 1.0;

    for (size_t i = 0; i < prediction.events.size(); ++i)
    {
        const auto& [name, count] = prediction.events[i];
        prediction.unscaledDynamicJ += count * m_Energies[i].second * 1e-12;
        prediction.dynamicJ += count * m_Energies[i].second * 1e-12 * scale;

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

    const double awakeJ = BaselineW(temperatureC) * kernelS;
    double baselineSd = m_BaselineSdW * kernelS;

    if (std::isnan(temperatureC))
    {
        baselineSd = std::hypot(baselineSd, m_UnknownTemperatureSdW * kernelS);
    }

    prediction.dynamicSdJ = m_RelativeSd * prediction.dynamicJ;

    if (!std::isnan(m_PowerLimitW) && awakeJ + prediction.dynamicJ > m_PowerLimitW * kernelS)
    {
        prediction.warnings.push_back("predicted power " + std::to_string(std::lround((awakeJ + prediction.dynamicJ) / kernelS))
            + " W is above the GPU's limit of " + std::to_string(std::lround(m_PowerLimitW))
            + " W: the GPU will lower its clock; energy taken as the limit times the time");
        prediction.dynamicJ = std::max(m_PowerLimitW * kernelS - awakeJ, 0.0);
        // The energy is the limit times the measured time: its uncertainty is how closely the GPU holds the limit.
        prediction.dynamicSdJ = LimitRelativeSd * m_PowerLimitW * kernelS;
        prediction.clamped = true;
    }

    // The baseline under the accounting, and the weight with which its error enters the energy (energy_predictor.py).
    const double subtract = m_Accounting == EnergyAccounting::Awake ? 0.0
        : m_Accounting == EnergyAccounting::AboveIdle ? IdlePowerW() * kernelS : awakeJ;
    prediction.baselineWeight = (prediction.clamped ? 0.0 : 1.0) - (m_Accounting == EnergyAccounting::Dynamic ? 1.0 : 0.0);
    prediction.baselineJ = awakeJ - subtract;
    prediction.energyJ = prediction.baselineJ + prediction.dynamicJ;
    prediction.sdJ = std::hypot(prediction.dynamicSdJ, prediction.baselineWeight * baselineSd);
    return prediction;
}

double EnergyModel::IndependentSd(const EnergyPrediction& prediction, const EnergyPrediction& other) const
{
    if (prediction.clamped)
    {
        return prediction.dynamicSdJ;
    }

    if (prediction.group == other.group)
    {
        return m_RelativeSd * prediction.dynamicJ;
    }

    // Against another kernel the bias of this one adds: the spread between kernels, or for a kernel with a measured
    // scale what is left of its bias, the scale's error from n measurements and the repeat of a measurement.
    const auto scaled = m_KernelScales.find(prediction.group);
    const double bias = scaled == m_KernelScales.cend() ? BetweenKernelsSd()
        : std::hypot(m_RelativeSd / std::sqrt(static_cast<double>(scaled->second.second)), ScaleRunSd);
    return std::hypot(m_RelativeSd, bias) * prediction.dynamicJ;
}

double EnergyModel::PairSd(const EnergyPrediction& a, const EnergyPrediction& b, const double scaleA, const double scaleB) const
{
    // The independent parts add in quadrature; a baseline error is common to both and acts on the difference of their
    // times, weighted by how the baseline enters each energy.
    const double baselineSd = std::hypot(m_BaselineSdW, m_UnknownTemperatureSdW);
    return std::sqrt(std::pow(IndependentSd(a, b) * scaleA, 2) + std::pow(IndependentSd(b, a) * scaleB, 2)
        + std::pow(baselineSd * (a.baselineWeight * a.kernelS * scaleA - b.baselineWeight * b.kernelS * scaleB), 2));
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
    size_t byTime = 0;
    size_t byEnergy = 0;
    bool any = false;

    for (size_t i = 0; i < predictions.size(); ++i)
    {
        const auto& p = predictions[i];
        const bool timeOk = std::isnan(maxSlowdown) || p.kernelS <= fastest * (1.0 + maxSlowdown);
        const bool energyOk = std::isnan(maxEnergyIncrease) || p.energyJ <= cheapest * (1.0 + maxEnergyIncrease);
        byTime += timeOk;
        byEnergy += energyOk;

        if (timeOk && energyOk)
        {
            result.values[i] = ObjectiveValue(p, objective);

            if (!any || result.values[i] < result.values[result.index])
            {
                result.index = i;
                any = true;
            }
        }
    }

    if (!any)
    {
        throw std::runtime_error("no configuration is admissible: the slowdown limit admits " + std::to_string(byTime)
            + " of " + std::to_string(predictions.size()) + ", the energy-increase limit " + std::to_string(byEnergy)
            + ", and none is in both");
    }

    if (objective == EnergyObjective::Time)
    {
        return result;
    }

    const auto& best = predictions[result.index];
    const double scaleBest = result.values[result.index] / best.energyJ;

    for (size_t i = 0; i < predictions.size(); ++i)
    {
        if (i == result.index || std::isnan(result.values[i]))
        {
            continue;
        }

        const auto& p = predictions[i];
        const double sd = PairSd(p, best, result.values[i] / p.energyJ, scaleBest);

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
