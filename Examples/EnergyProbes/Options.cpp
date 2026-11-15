#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "Options.h"
#include "ProbePlan.h"
#include "ValidationKernels.h"

namespace
{

// One command-line option: its name, the placeholder of its value ("" for a flag), its help text (lines separated by
// '\n') and what it sets.
struct OptionSpec
{
    std::string name;
    std::string value;
    std::string help;
    std::function<void(Options&, const std::string&)> apply;
};

std::vector<std::string> SplitList(const std::string& text)
{
    std::vector<std::string> items;
    std::stringstream list(text);
    std::string item;

    while (std::getline(list, item, ','))
    {
        items.push_back(item);
    }

    return items;
}

// A non-negative whole number: all of `text`, or an error that names the option (std::stoull alone accepts "16x" and
// "-5", and its own error message is just "stoull").
uint64_t Number(const std::string& text, const std::string& option)
{
    if (text.empty() || text.size() > 18 || text.find_first_not_of("0123456789") != std::string::npos)
    {
        throw std::runtime_error(option + " needs a non-negative whole number, not '" + text + "'");
    }

    return std::stoull(text);
}

double Fraction(const std::string& text, const std::string& option)
{
    size_t used = 0;
    double value = 0.0;

    try
    {
        value = std::stod(text, &used);
    }
    catch (const std::exception&)
    {
        used = 0;
    }

    if (text.empty() || used != text.size() || !(value >= 0.0))
    {
        throw std::runtime_error(option + " needs a non-negative number, not '" + text + "'");
    }

    return value;
}

// One of `choices`.
std::string Choice(const std::string& text, const std::string& option, const std::vector<std::string>& choices)
{
    if (std::find(choices.cbegin(), choices.cend(), text) == choices.cend())
    {
        std::string list;

        for (const std::string& choice : choices)
        {
            list += (list.empty() ? "" : ", ") + choice;
        }

        throw std::runtime_error(option + " must be one of: " + list + " (not '" + text + "')");
    }

    return text;
}

// A comma-separated list of names of `choices`; a name of `sets` stands for its members.
std::vector<std::string> NameList(const std::string& text, const std::string& option, std::vector<std::string> choices,
    const std::vector<std::pair<std::string, std::vector<std::string>>>& sets = {})
{
    std::vector<std::string> names;

    for (const auto& set : sets)
    {
        choices.push_back(set.first);
    }

    for (const std::string& item : SplitList(text))
    {
        Choice(item, option, choices);
        const auto set = std::find_if(sets.cbegin(), sets.cend(), [&item](const auto& s) { return s.first == item; });

        for (const std::string& name : set != sets.cend() ? set->second : std::vector<std::string>{item})
        {
            if (std::find(names.cbegin(), names.cend(), name) == names.cend())
            {
                names.push_back(name);
            }
        }
    }

    return names;
}

std::vector<std::string> KernelList(const std::string& text, const std::string& option)
{
    std::vector<std::string> kernels;

    for (const auto& set : ValidationKernelSets())
    {
        kernels.insert(kernels.end(), set.second.cbegin(), set.second.cend());
    }

    return NameList(text, option, kernels, ValidationKernelSets());
}

// Every option, in the order of the usage text. --profile and --quick are presets: they are applied first, so that an
// explicit option overrides them wherever it stands.
const std::vector<OptionSpec>& OptionSpecs()
{
    static const std::vector<OptionSpec> specs = {
        {"--platform", "N", "platform index (default 0)",
            [](Options& o, const std::string& v) { o.platform = static_cast<ktt::PlatformIndex>(Number(v, "--platform")); }},
        {"--device", "N", "device index (default 0)",
            [](Options& o, const std::string& v) { o.device = static_cast<ktt::DeviceIndex>(Number(v, "--device")); }},
        {"--mt4g", "FILE", "MT4G result JSON of the GPU, for the cache sizes (recommended)",
            [](Options& o, const std::string& v) { o.mt4gFile = v; }},
        {"--l1-kib", "N", "override the L1 size in KiB",
            [](Options& o, const std::string& v) { o.l1Override = Number(v, "--l1-kib") * KiB; }},
        {"--l2-kib", "N", "override the L2 size in KiB",
            [](Options& o, const std::string& v) { o.l2Override = Number(v, "--l2-kib") * KiB; }},
        {"--output", "PREFIX", "prefix of the output files (default EnergyProbes)",
            [](Options& o, const std::string& v) { o.outputPrefix = v; }},
        {"--lock-clock", "MHZ", "lock the SM clock for the run via NVML (needs root), reset at exit",
            [](Options& o, const std::string& v) { o.lockClockMHz = static_cast<uint32_t>(Number(v, "--lock-clock")); }},
        {"--clock-locked", "", "the SM clock is already locked (sudo nvidia-smi -i <gpu> -lgc F,F): rows at another\n"
            "clock than the baseline's are withheld",
            [](Options& o, const std::string&) { o.clockLockedExternally = true; }},

        {"--only", "LIST", "probe groups to run (default all): global, compute, shared, latency, banks, constant,\n"
            "locality, sub-sector, write-path, misaligned, writes",
            [](Options& o, const std::string& v) { o.only = NameList(v, "--only", ProbeGroupNames()); }},
        {"--profile", "", "the counter run: CUPTI counters of every probe, at duty 100 with short windows (--duty 100\n"
            "--min-time-ms 500 --max-time-ms 1000 --warmup-s 10 --idle-s 0); the counts are a property of\n"
            "the code, not of the measurement",
            [](Options& o, const std::string&)
        {
            o.profile = true;
            o.duties = {100};
            o.minTimeMs = 500;
            o.maxTimeMs = 1000;
            o.warmupS = 10;
            o.idleS = 0;
        }},
        {"--quick", "", "the 4-minute run that tools/check_model.py compares with a model file: the global, compute\n"
            "and shared probes at duty 100, 5-10 s each. Not a calibration: one duty gives no fit",
            [](Options& o, const std::string&)
        {
            o.only = {"global", "compute", "shared"};
            o.duties = {100};
            o.warmupS = 30;
            o.idleS = 0;
            o.minTimeMs = 5000;
            o.maxTimeMs = 10000;
        }},
        {"--validate", "LIST", "instead of the probes, measure random configurations of KTT example kernels: power and\n"
            "time, then CUPTI counters (for the model's uncertainty, tools/build_model.py --calibration).\n"
            "all = transpose, gemm, nbody, coulomb; heldout = reduction, convolution, pnpoly; heldout2 =\n"
            "bicg, ktconvolution",
            [](Options& o, const std::string& v) { o.validate = KernelList(v, "--validate"); }},
        {"--validate-configs", "N", "--validate, --tune: configurations per kernel (default 24)",
            [](Options& o, const std::string& v) { o.validateConfigs = Number(v, "--validate-configs"); }},
        {"--validate-ms", "N", "--validate, --tune-measure: power measurement per configuration (default 4000)",
            [](Options& o, const std::string& v) { o.validateMs = Number(v, "--validate-ms"); }},
        {"--tune", "LIST", "instead of the probes, tune example kernels (names as for --validate) with the energy\n"
            "model: time and CUPTI counters of random configurations, no power measurement; prints the\n"
            "configuration to use and the ones the model cannot tell apart from it",
            [](Options& o, const std::string& v) { o.tune = KernelList(v, "--tune"); }},
        {"--predict", "FILE", "the same choice from a --validate run's CSV (PREFIX_validation.csv), without a GPU",
            [](Options& o, const std::string& v) { o.predictCsv = v; }},
        {"--model-file", "FILE", "the energy model for --tune and --predict (tools/build_model.py writes it)",
            [](Options& o, const std::string& v) { o.modelFile = v; }},
        {"--objective", "NAME", "what --tune and --predict minimise: energy (default), edp, ed2p or time",
            [](Options& o, const std::string& v) { o.objective = Choice(v, "--objective", {"energy", "edp", "ed2p", "time"}); }},
        {"--max-slowdown", "PCT", "only configurations at most PCT % slower than the fastest",
            [](Options& o, const std::string& v) { o.maxSlowdownPct = Fraction(v, "--max-slowdown"); }},
        {"--max-energy-increase", "PCT", "only configurations at most PCT % above the lowest predicted energy (with\n"
            "--objective time: the fastest of the nearly cheapest)",
            [](Options& o, const std::string& v) { o.maxEnergyIncreasePct = Fraction(v, "--max-energy-increase"); }},
        {"--accounting", "NAME", "which baseline the energy includes: awake (default; kernels back to back), above-idle\n"
            "(the GPU idles with a CUDA context between runs of the work) or dynamic (no baseline)",
            [](Options& o, const std::string& v)
            {
                o.accounting = Choice(v, "--accounting", {"awake", "above-idle", "dynamic"});
            }},
        {"--idle-w", "W", "--accounting above-idle: the idle power to subtract (default: the model file's)",
            [](Options& o, const std::string& v) { o.idleW = Fraction(v, "--idle-w"); }},
        {"--contender-sd", "X", "configurations within X standard deviations of the best are contenders (default 2)",
            [](Options& o, const std::string& v) { o.contenderSd = Fraction(v, "--contender-sd"); }},
        {"--tune-measure", "WHAT", "--tune: after the model's choice, measure the power of its contenders (and of the\n"
            "fastest configuration) or of all configurations, and take the measured best; default none",
            [](Options& o, const std::string& v) { o.tuneMeasure = Choice(v, "--tune-measure", {"none", "contenders", "all"}); }},
        {"--measure-margin", "PCT", "--tune-measure: measurements within PCT % of the lowest are not told apart; among\n"
            "them the model's choice, else the fastest (default 1)",
            [](Options& o, const std::string& v) { o.measureMarginPct = Fraction(v, "--measure-margin"); }},
        {"--tune-scale-configs", "N", "--tune-measure: also measure N configurations spread over each kernel's predicted\n"
            "energy and take the kernel's scale from them (default 5 with more than one kernel, else 0)",
            [](Options& o, const std::string& v) { o.tuneScaleConfigs = Number(v, "--tune-scale-configs"); }},
        {"--lenient", "", "--tune, --predict: counters the model needs and does not find, and values that are not\n"
            "finite, count as 0 with a warning (default: an error)",
            [](Options& o, const std::string&) { o.lenient = true; }},
        {"--dry-run", "", "print the plan of the probe run from the --mt4g file and exit, no GPU needed",
            [](Options& o, const std::string&) { o.dryRun = true; }},
        {"--analyze", "PREFIX", "analyse PREFIX.csv of an earlier probe run again with the --mt4g file, no GPU needed;\n"
            "the outputs go to --output",
            [](Options& o, const std::string& v) { o.analyzePrefix = v; }},

        {"--duty", "LIST", "duties in % of the time issuing accesses (default 100,25); the energy per access is the\n"
            "slope of power over access rate across them",
            [](Options& o, const std::string& v)
            {
                o.duties.clear();

                for (const std::string& item : SplitList(v))
                {
                    o.duties.push_back(Number(item, "--duty"));
                }
            }},
        {"--fit-min-duty", "N", "fit only over duties of at least N % (default 25: below, power is not linear in the rate)",
            [](Options& o, const std::string& v) { o.fitMinDuty = Number(v, "--fit-min-duty"); }},
        {"--idle-s", "N", "idle power without a kernel for N s: before KTT starts (no CUDA context), before the\n"
            "warm-up and at the end (default 20, 0 = off)",
            [](Options& o, const std::string& v) { o.idleS = Number(v, "--idle-s"); }},
        {"--warmup-s", "N", "heat the GPU for N seconds before measuring (default 120, 0 = off)",
            [](Options& o, const std::string& v) { o.warmupS = Number(v, "--warmup-s"); }},
        {"--launch-ms", "N", "duration of one probe launch (default 200)",
            [](Options& o, const std::string& v) { o.launchMs = Number(v, "--launch-ms"); }},
        {"--min-time-ms", "N", "power measurement per configuration: KTT stops once the samples of the last N ms are\n"
            "within --max-power-diff of their mean (default 10000)",
            [](Options& o, const std::string& v) { o.minTimeMs = Number(v, "--min-time-ms"); }},
        {"--max-time-ms", "N", "... or after N ms (default 60000)",
            [](Options& o, const std::string& v) { o.maxTimeMs = Number(v, "--max-time-ms"); }},
        {"--max-power-diff", "X", "tolerated relative power variation, above 0 and at most 1 (default 0.02)",
            [](Options& o, const std::string& v) { o.maxPowerDiff = Fraction(v, "--max-power-diff"); }},
        {"--block-size", "N", "threads per block, a power of two from 32 to 1024 (default 256)",
            [](Options& o, const std::string& v) { o.blockSize = Number(v, "--block-size"); }},
        {"--blocks-per-sm", "N", "resident blocks per SM (default 4)",
            [](Options& o, const std::string& v) { o.blocksPerSm = Number(v, "--blocks-per-sm"); }},
        {"--unroll", "N", "accesses between timer checks, a multiple of 8 (default 16)",
            [](Options& o, const std::string& v) { o.unroll = Number(v, "--unroll"); }},
        {"--burst-accesses", "N", "accesses per thread between two sleeps, a multiple of --unroll (default 64)",
            [](Options& o, const std::string& v) { o.burstAccesses = Number(v, "--burst-accesses"); }},
        {"--shared-kib", "N", "shared memory per block of the shared probes, a power of two from 4 (default 8)",
            [](Options& o, const std::string& v) { o.sharedBytes = Number(v, "--shared-kib") * KiB; }},
        {"--fixed-order", "", "run the probes of a group in a fixed order (default: random, which decorrelates\n"
            "heating from the probe order)",
            [](Options& o, const std::string&) { o.randomOrder = false; }},
        {"--force-energy", "", "also compute withheld energies (power not above baseline, or clock off the baseline's)",
            [](Options& o, const std::string&) { o.forceEnergy = true; }},
        {"--trace-ms", "N", "NVML trace sampling interval (default 50)",
            [](Options& o, const std::string& v) { o.traceIntervalMs = Number(v, "--trace-ms"); }},
        {"--no-monitor", "", "do not use NVML directly (no clock lock, trace or energy counter)",
            [](Options& o, const std::string&) { o.monitor = false; }},
        {"--kernel-dir", "DIR", "directory of the probe kernels (default: kernels/ of this example)",
            [](Options& o, const std::string& v) { o.kernelDirectory = v.empty() || v.back() == '/' ? v : v + '/'; }},
        {"--counter-table", "FILE", "the CUPTI counters to collect (default: tools/counters.json of this example, which\n"
            "the Python tools read too)",
            [](Options& o, const std::string& v) { o.counterTable = v; }},
    };
    return specs;
}

// The checks that need all options.
void Check(Options& options)
{
    const auto require = [](const bool condition, const std::string& message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    };

    // The probes count in warps (the constant and the latency probe address by warp, a shared wavefront is a warp).
    require(IsPowerOfTwo(options.blockSize) && options.blockSize >= 32 && options.blockSize <= 1024,
        "--block-size must be a power of two from 32 to 1024");
    require(options.maxPowerDiff > 0.0 && options.maxPowerDiff <= 1.0, "--max-power-diff must be above 0 and at most 1");
    // The bank-conflict probe: 32 lanes, each on a word of its own in the same bank.
    require(IsPowerOfTwo(options.sharedBytes) && options.sharedBytes >= 4 * KiB, "--shared-kib must be a power of two from 4");
    require(options.blocksPerSm > 0 && options.launchMs > 0 && options.traceIntervalMs > 0,
        "--blocks-per-sm, --launch-ms and --trace-ms must be positive");
    // The sub-sector probes issue whole sectors of 4-byte words, the write probes groups of 4 (WRITE_GROUP in
    // kernels/Write.cu): with another --unroll their bursts would not be --burst-accesses long.
    require(options.unroll > 0 && options.unroll % (SectorBytes / 4) == 0, "--unroll must be a positive multiple of 8");
    require(options.burstAccesses > 0 && options.burstAccesses % options.unroll == 0,
        "--burst-accesses must be a positive multiple of --unroll");
    require(options.fitMinDuty >= 1 && options.fitMinDuty <= 100, "--fit-min-duty must be between 1 and 100");
    require(options.maxTimeMs >= options.minTimeMs, "--max-time-ms must not be smaller than --min-time-ms");
    require(!options.duties.empty() && std::all_of(options.duties.cbegin(), options.duties.cend(),
        [](const uint64_t duty) { return duty >= 1 && duty <= 100; }), "--duty values must be between 1 and 100");
    require((options.tune.empty() && options.predictCsv.empty()) || !options.modelFile.empty(),
        "--tune and --predict need --model-file");
    require(options.tune.empty() || options.predictCsv.empty(), "--tune and --predict exclude each other");
    require(!options.idleW.has_value() || options.accounting == "above-idle", "--idle-w goes with --accounting above-idle");
    require(!options.profile || options.validate.empty(), "--validate collects counters in a pass of its own, remove --profile");
    require(!options.lockClockMHz.has_value() || options.monitor, "--lock-clock needs the NVML monitor, remove --no-monitor");

    std::sort(options.duties.begin(), options.duties.end(), std::greater<uint64_t>());
    options.duties.erase(std::unique(options.duties.begin(), options.duties.end()), options.duties.end());
}

} // namespace

bool ClockIsLocked(const Options& options)
{
    return options.lockClockMHz.has_value() || options.clockLockedExternally;
}

void PrintUsage()
{
    std::cout << "Usage: EnergyProbesCuda [options]\n";

    for (const OptionSpec& spec : OptionSpecs())
    {
        std::stringstream help(spec.help);
        std::string line;
        std::getline(help, line);
        std::cout << "  " << std::left << std::setw(26) << (spec.name + (spec.value.empty() ? "" : " " + spec.value)) << line
            << "\n";

        while (std::getline(help, line))
        {
            std::cout << std::string(28, ' ') << line << "\n";
        }
    }
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    const auto& specs = OptionSpecs();

    for (const bool presets : {true, false})
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];

            if (argument == "--help" || argument == "-h")
            {
                PrintUsage();
                std::exit(0);
            }

            const auto spec = std::find_if(specs.cbegin(), specs.cend(), [&](const OptionSpec& s) { return s.name == argument; });

            if (spec == specs.cend())
            {
                throw std::runtime_error("Unknown option " + argument + " (use --help)");
            }

            if (!spec->value.empty() && i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for option " + argument);
            }

            const std::string value = spec->value.empty() ? std::string() : std::string(argv[++i]);

            if ((argument == "--profile" || argument == "--quick") == presets)
            {
                spec->apply(options, value);
            }
        }
    }

    Check(options);
    return options;
}
