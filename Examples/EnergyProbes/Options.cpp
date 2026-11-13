#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "Options.h"
#include "Payloads.h"
#include "ValidationKernels.h"

namespace
{

// One command-line option: its name, the placeholder of its value ("" for a flag), its help text (lines separated by
// '\n'; empty: not listed) and what it sets.
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
    uint64_t number = 0;

    try
    {
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        {
            throw std::invalid_argument(text);
        }

        number = std::stoull(text);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(option + " needs a non-negative whole number, not '" + text + "'");
    }

    return number;
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

    if (text.empty() || used != text.size())
    {
        throw std::runtime_error(option + " needs a number, not '" + text + "'");
    }

    return value;
}

// A comma-separated list of numbers, each accepted by `valid`.
std::vector<uint64_t> NumberList(const std::string& text, const std::string& option, const std::string& error,
    const std::function<bool(uint64_t)>& valid)
{
    std::vector<uint64_t> numbers;

    for (const std::string& item : SplitList(text))
    {
        const uint64_t number = Number(item, option);

        if (!valid(number))
        {
            throw std::runtime_error(error);
        }

        numbers.push_back(number);
    }

    if (numbers.empty())
    {
        throw std::runtime_error(option + " needs at least one value");
    }

    return numbers;
}

std::vector<uint64_t> PayloadList(const std::string& text)
{
    std::vector<uint64_t> payloads;

    for (const std::string& item : SplitList(text))
    {
        if (item != "random" && item != "zero")
        {
            throw std::runtime_error("--payload values must be random or zero");
        }

        payloads.push_back(item == "zero" ? PayloadZero : PayloadRandom);
    }

    return payloads;
}

// Kernels for --validate: names from ValidationKernelNames() and HeldOutKernelNames(), "all" (the example kernels),
// "heldout" (the held-out kernels) or the control kernel "mixed".
std::vector<std::string> ValidationList(const std::string& text)
{
    const std::vector<std::string> examples = ValidationKernelNames();
    std::vector<std::string> kernels;
    auto add = [&kernels](const std::string& name)
    {
        if (std::find(kernels.cbegin(), kernels.cend(), name) == kernels.cend())
        {
            kernels.push_back(name);
        }
    };

    for (const std::string& item : SplitList(text))
    {
        const std::vector<std::string> heldOut = HeldOutKernelNames();

        if (item == "all" || item == "heldout")
        {
            for (const auto& name : item == "all" ? examples : heldOut)
            {
                add(name);
            }
        }
        else if (item == MixedControlName || std::find(examples.cbegin(), examples.cend(), item) != examples.cend()
            || std::find(heldOut.cbegin(), heldOut.cend(), item) != heldOut.cend())
        {
            add(item);
        }
        else
        {
            throw std::runtime_error("--validate values must be transpose, gemm, nbody, coulomb, all, reduction, "
                "convolution, pnpoly, heldout or mixed");
        }
    }

    return kernels;
}

// Every option, in the order of the usage text.
const std::vector<OptionSpec>& OptionSpecs()
{
    static const std::vector<OptionSpec> specs = {
        {"--platform", "N", "platform index (default 0)",
            [](Options& o, const std::string& v) { o.platform = static_cast<ktt::PlatformIndex>(Number(v, "--platform")); }},
        {"--device", "N", "device index (default 0)",
            [](Options& o, const std::string& v) { o.device = static_cast<ktt::DeviceIndex>(Number(v, "--device")); }},
        {"--kernel-dir", "DIR", "directory of the probe kernels (kernels/ of this example)",
            [](Options& o, const std::string& v) { o.kernelDirectory = v.empty() || v.back() == '/' ? v : v + '/'; }},
        {"--counter-table", "FILE", "the CUPTI counters to collect with --profile and --validate (tools/counters.json\n"
            "of this example, which the Python tools read too)",
            [](Options& o, const std::string& v) { o.counterTable = v; }},
        {"--mt4g", "FILE", "MT4G result JSON used for cache sizes (recommended)",
            [](Options& o, const std::string& v) { o.mt4gFile = v; }},
        {"--l1-kib", "N", "override L1 size in KiB",
            [](Options& o, const std::string& v) { o.l1Override = Number(v, "--l1-kib") * KiB; }},
        {"--l2-kib", "N", "override L2 size in KiB",
            [](Options& o, const std::string& v) { o.l2Override = Number(v, "--l2-kib") * KiB; }},
        {"--output", "PREFIX", "prefix of output files (default EnergyProbes)",
            [](Options& o, const std::string& v) { o.outputPrefix = v; }},
        {"--lock-clock", "MHZ", "lock the SM clock for the run via NVML (needs root), reset at exit",
            [](Options& o, const std::string& v) { o.lockClockMHz = static_cast<uint32_t>(Number(v, "--lock-clock")); }},
        {"--clock-locked", "", "the SM clock is already locked (sudo nvidia-smi -i <gpu> -lgc F,F): strict mode",
            [](Options& o, const std::string&) { o.clockLockedExternally = true; }},
        {"--duty", "LIST", "probe intensities in % of time issuing accesses (default 100,50,25,10)",
            [](Options& o, const std::string& v)
            {
                o.duties = NumberList(v, "--duty", "--duty values must be between 1 and 100",
                    [](uint64_t d) { return d >= 1 && d <= 100; });
            }},
        {"--burst-accesses", "N", "accesses per thread between two sleeps, multiple of --unroll (default 64)",
            [](Options& o, const std::string& v) { o.burstAccesses = Number(v, "--burst-accesses"); }},
        {"--fit-min-duty", "N", "fit power vs access rate only over duties of at least N % (default 25)",
            [](Options& o, const std::string& v) { o.fitMinDuty = Number(v, "--fit-min-duty"); }},
        {"--payload", "LIST", "data of all probes: random, zero or random,zero (control vs datapath energy;\n"
            "default random)",
            [](Options& o, const std::string& v) { o.payloads = PayloadList(v); }},
        {"--payload-bits", "LIST", "also run 16-byte loads (L1, L2, DRAM) and FFMA, DFMA and HMMA (sm_80+) on payloads\n"
            "with controlled bits, besides random and zero: ones:Q (bits 1 with probability Q), same:Q (one\n"
            "word with 32Q one bits, repeated), flip:T (32T bits flip between consecutive words), unique:K,\n"
            "lsb0:N, sorted (Payloads.h); their bit statistics go to PREFIX_payloads.json\n"
            "(tools/fit_payload.py)",
            [](Options& o, const std::string& v)
            {
                o.payloadBits.clear();
                std::stringstream list(v);
                std::string item;

                while (std::getline(list, item, ','))
                {
                    CheckPayloadPattern(item);
                    o.payloadBits.push_back(item);
                }
            }},
        {"--widths", "LIST", "bytes per global load and thread, from 4,8,16 (default 16); several widths separate\n"
            "the energy per request from the energy per byte",
            [](Options& o, const std::string& v)
            {
                o.widths = NumberList(v, "--widths", "--widths values must be 4, 8 or 16",
                    [](uint64_t w) { return w == 4 || w == 8 || w == 16; });
            }},
        {"--compute", "", "also run the arithmetic probes (FFMA, DFMA, IMAD, HFMA2, HMMA on sm_80+, MUFU, FFMA with three\n"
            "register operands, empty loop)",
            [](Options& o, const std::string&) { o.compute = true; }},
        {"--atomics", "", "also run 32-bit atomic adds: red and atom at the L2 and DRAM footprints, compared with\n"
            "4-byte loads and stores there, and red.shared",
            [](Options& o, const std::string&) { o.atomics = true; }},
        {"--constant", "", "also run constant-memory loads (all threads of a warp read one word) at the constant L1\n"
            "and L1.5 footprints",
            [](Options& o, const std::string&) { o.constant = true; }},
        {"--texture", "", "also run texture fetches (tex.1d, 16-byte texels) at the L1, L2 and DRAM footprints",
            [](Options& o, const std::string&) { o.texture = true; }},
        {"--dram-locality", "", "also run DRAM loads and stores that stream or scatter runs of 32 B (one sector) to 8 KiB\n"
            "over the footprint (granularity and DRAM row locality), with an L2 control",
            [](Options& o, const std::string&) { o.dramLocality = true; }},
        {"--sub-sector", "", "also run 4- and 8-byte loads and stores in which every thread walks a sector of its own\n"
            "(a warp instruction writes one word of 32 sectors, which L2 merges), sectors adjacent or\n"
            "scattered",
            [](Options& o, const std::string&) { o.subSector = true; }},
        {"--indexing", "", "also run 32 FFMA accumulators with static and with runtime indices (a dynamically\n"
            "indexed register array: compares and guarded moves, as in the Coulomb example)",
            [](Options& o, const std::string&) { o.indexing = true; }},
        {"--bank-conflicts", "", "also run 4-, 8- and 16-byte shared loads and stores whose lanes fall on fewer banks, so\n"
            "that a warp instruction takes 1, 2, 4, ... times the wavefronts of the conflict-free access\n"
            "(up to 32 wavefronts)",
            [](Options& o, const std::string&) { o.bankConflicts = true; }},
        {"--mixed", "", "also run 16-byte loads with 0 to 1024 FFMAs per load at the L1, L2 and DRAM footprints\n"
            "(memory- to compute-bound, as Mixbench), at duty 100",
            [](Options& o, const std::string&) { o.mixed = true; }},
        {"--writes", "", "also run loads and stores together (copy: a read and a write stream; update: store back\n"
            "where loaded, so that L2 evicts dirty lines; local: a local-memory frame per thread, as\n"
            "spills) at the L2 and DRAM footprints (local: L1 and DRAM)",
            [](Options& o, const std::string&) { o.writes = true; }},
        {"--latency", "", "also run dependent loads (chains of random lines, one load in flight per chain) with 1-16\n"
            "chains per warp and 4-32 working warps per SM at the L2 and DRAM footprints, at duty 100:\n"
            "what a load costs when latency, not bandwidth, limits",
            [](Options& o, const std::string&) { o.latency = true; }},
        {"--sm-sweep", "LIST", "also run L1, L2, no-load, shared (and FFMA) probes and a baseline with only N SMs\n"
            "working, for each N of the list (power of awake and active SMs); with --latency also the\n"
            "dependent loads with 1 and 16 chains at L2 and DRAM (steady traffic at low rates)",
            [](Options& o, const std::string& v)
            {
                o.smSweep = NumberList(v, "--sm-sweep", "--sm-sweep values must be positive",
                    [](uint64_t n) { return n > 0; });
            }},
        {"--all", "", "--compute --atomics --constant --texture --dram-locality --sub-sector --indexing\n"
            "--bank-conflicts --mixed --writes --latency --payload random,zero --widths 4,8,16 and an SM\n"
            "sweep over all SMs, halves, ..., 1 (918 configurations on an RTX 3090)",
            [](Options& o, const std::string&)
        {
            o.compute = true;
            o.atomics = true;
            o.constant = true;
            o.texture = true;
            o.dramLocality = true;
            o.subSector = true;
            o.indexing = true;
            o.bankConflicts = true;
            o.mixed = true;
            o.writes = true;
            o.latency = true;
            o.payloads = {PayloadRandom, PayloadZero};
            o.widths = {4, 8, 16};
            o.smSweepAuto = true;
        }},
        {"--quick", "", "the short run that tools/check_model.py compares with a model file (about 5 min): the\n"
            "default probes and --compute at duty 100, 5-10 s per configuration, 30 s warm-up, no idle\n"
            "measurement. Not a calibration: one duty gives no fit",
            [](Options& o, const std::string&)
        {
            o.compute = true;
            o.duties = {100};
            o.warmupS = 30;
            o.idleS = 0;
            o.minTimeMs = 5000;
            o.maxTimeMs = 10000;
        }},
        {"--model", "", "what tools/energy_model.py uses, and nothing else: the probe families of --all on random data\n"
            "at duties 100,50,25, and the SM sweep with only its baselines and dependent loads (458\n"
            "configurations on an RTX 3090, against 918 for --all). It leaves out the zero payload, duty 10\n"
            "and the SM sweep's other probes, which only the studies use",
            [](Options& o, const std::string&)
        {
            o.compute = true;
            o.atomics = true;
            o.constant = true;
            o.texture = true;
            o.dramLocality = true;
            o.subSector = true;
            o.indexing = true;
            o.bankConflicts = true;
            o.mixed = true;
            o.writes = true;
            o.latency = true;
            o.payloads = {PayloadRandom};
            o.widths = {4, 8, 16};
            o.duties = {100, 50, 25};
            o.smSweepAuto = true;
            o.sweepStudies = false;
        }},
        {"--idle-s", "N", "measure the idle power (no kernel) for N seconds before KTT starts (no CUDA context),\n"
            "before the warm-up and at the end (default 20, 0 = off)",
            [](Options& o, const std::string& v) { o.idleS = Number(v, "--idle-s"); }},
        {"--warmup-s", "N", "heat the GPU for N seconds before measuring (default 120, 0 = off)",
            [](Options& o, const std::string& v) { o.warmupS = Number(v, "--warmup-s"); }},
        {"--launch-ms", "N", "duration of one kernel launch (default 200)",
            [](Options& o, const std::string& v) { o.launchMs = Number(v, "--launch-ms"); }},
        {"--min-time-ms", "N", "minimal measurement time per configuration (default 10000)",
            [](Options& o, const std::string& v) { o.minTimeMs = Number(v, "--min-time-ms"); }},
        {"--max-time-ms", "N", "maximal measurement time per configuration (default 60000)",
            [](Options& o, const std::string& v) { o.maxTimeMs = Number(v, "--max-time-ms"); }},
        {"--max-power-diff", "X", "tolerated relative power variation, 0-1 (default 0.02)",
            [](Options& o, const std::string& v) { o.maxPowerDiff = Fraction(v, "--max-power-diff"); }},
        {"--block-size", "N", "threads per block, power of two (default 256)",
            [](Options& o, const std::string& v) { o.blockSize = Number(v, "--block-size"); }},
        {"--blocks-per-sm", "N", "resident blocks per SM (default 4)",
            [](Options& o, const std::string& v) { o.blocksPerSm = Number(v, "--blocks-per-sm"); }},
        {"--unroll", "N", "accesses between timer checks (default 16)",
            [](Options& o, const std::string& v) { o.unroll = Number(v, "--unroll"); }},
        {"--shared-kib", "N", "shared memory per block for the shared probe, power of two (default 8)",
            [](Options& o, const std::string& v) { o.sharedBytes = Number(v, "--shared-kib") * KiB; }},
        {"--sweep", "", "sweep all power-of-two footprints (energy vs footprint curve)",
            [](Options& o, const std::string&) { o.sweep = true; }},
        {"--fixed-order", "", "run global probe configurations in a fixed order (default: random)",
            [](Options& o, const std::string&) { o.randomOrder = false; }},
        {"--force-energy", "", "also compute withheld energies (power not above baseline, or throttled below the lock)",
            [](Options& o, const std::string&) { o.forceEnergy = true; }},
        {"--trace-ms", "N", "NVML trace sampling interval (default 50)",
            [](Options& o, const std::string& v) { o.traceIntervalMs = Number(v, "--trace-ms"); }},
        {"--no-monitor", "", "do not use NVML directly (no clock lock, trace or energy counter)",
            [](Options& o, const std::string&) { o.monitor = false; }},
        {"--profile", "", "collect CUPTI counters (KTT built with --profiling=cupti)",
            [](Options& o, const std::string&) { o.profile = true; }},
        {"--validate", "LIST", "instead of the probes, measure random configurations of KTT example kernels\n"
            "(transpose, gemm, nbody, coulomb or all): power, time and CUPTI counters, for checking\n"
            "energy predictions (see energy_model.py); mixed adds the mixed-intensity probe, measured\n"
            "the same way, as a control of the measurement; heldout (reduction, convolution, pnpoly):\n"
            "kernels the model was not designed on",
            [](Options& o, const std::string& v) { o.validate = ValidationList(v); }},
        {"--validate-configs", "N", "configurations per example kernel (default 24)",
            [](Options& o, const std::string& v) { o.validateConfigs = Number(v, "--validate-configs"); }},
        {"--validate-ms", "N", "measurement time per configuration (default 4000)",
            [](Options& o, const std::string& v) { o.validateMs = Number(v, "--validate-ms"); }},
        {"--tune", "LIST", "instead of the probes, tune example kernels (names as for --validate, without mixed) with\n"
            "the energy model: time and CUPTI counters of random configurations, no power measurement;\n"
            "prints the configuration to use and the ones the model cannot tell apart from it",
            [](Options& o, const std::string& v) { o.tune = ValidationList(v); }},
        {"--model-file", "FILE", "the energy model for --tune and --predict (tools/build_model.py writes it)",
            [](Options& o, const std::string& v) { o.modelFile = v; }},
        {"--predict", "FILE", "the same choice from a --validate run's CSV (PREFIX_validation.csv), without a GPU",
            [](Options& o, const std::string& v) { o.predictCsv = v; }},
        {"--objective", "NAME", "what --tune and --predict minimise: energy (default), edp, ed2p or time",
            [](Options& o, const std::string& v) { o.objective = v; }},
        {"--max-slowdown", "PCT", "--tune, --predict: only configurations at most PCT % slower than the fastest",
            [](Options& o, const std::string& v) { o.maxSlowdownPct = Fraction(v, "--max-slowdown"); }},
        {"--contender-sd", "X", "--tune, --predict: configurations within X standard deviations of the best are\n"
            "contenders (default 2)",
            [](Options& o, const std::string& v) { o.contenderSd = Fraction(v, "--contender-sd"); }},
        {"--tune-measure", "WHAT", "--tune: after the model's choice, measure power (NVML, --validate-ms each) of its\n"
            "contenders (the choice, the configurations the model cannot tell apart from it and the\n"
            "fastest one) or of all configurations, and take the measured best; default none",
            [](Options& o, const std::string& v) { o.tuneMeasure = v; }},
        {"--lenient", "", "--tune, --predict: counters the model needs and does not find count as 0 with a warning\n"
            "(default: an error)",
            [](Options& o, const std::string&) { o.lenient = true; }},
        {"--thermal-curve", "", "instead of the probes, the sleep baseline's power over temperature (about 15 min):\n"
            "the GPU cools in a sleep after 2 min of compute (spin) and after 2 min of DRAM loads, then\n"
            "heats in steps with a sleep after each; traced every --trace-ms (tools/fit_thermal.py)",
            [](Options& o, const std::string&) { o.thermalCurve = true; }},
        {"--dry-run", "", "print the plan from the --mt4g file and exit, no GPU needed",
            [](Options& o, const std::string&) { o.dryRun = true; }},
        {"--analyze", "PREFIX", "analyse PREFIX.csv of an earlier run again (baseline, energies, fits, summary,\n"
            "studies) with the --mt4g file, no GPU needed; outputs go to --output",
            [](Options& o, const std::string& v) { o.analyzePrefix = v; }},
        // Accepted for compatibility, not listed: the default.
        {"--random-order", "", "", [](Options& o, const std::string&) { o.randomOrder = true; }},
    };
    return specs;
}

// The checks that need all options, and the defaults and orders the rest of the program relies on.
void CheckAndNormalize(Options& options)
{
    // The probes count in warps (the constant and the latency probe address by warp, a shared wavefront is a warp).
    if (!IsPowerOfTwo(options.blockSize) || options.blockSize < 32 || options.blockSize > 1024)
    {
        throw std::runtime_error("--block-size must be a power of two from 32 to 1024");
    }

    if (!(options.maxPowerDiff > 0.0) || options.maxPowerDiff > 1.0)
    {
        throw std::runtime_error("--max-power-diff must be above 0 and at most 1");
    }

    if (!IsPowerOfTwo(options.sharedBytes))
    {
        throw std::runtime_error("--shared-kib must be a power of two");
    }

    // 32 lanes, each on a word of its own in the same bank: 32 x 32 words.
    if (options.bankConflicts && options.sharedBytes < 32 * 32 * 4)
    {
        throw std::runtime_error("--bank-conflicts needs --shared-kib 4 or more");
    }

    if (options.unroll == 0 || options.blocksPerSm == 0 || options.launchMs == 0 || options.traceIntervalMs == 0)
    {
        throw std::runtime_error("--unroll, --blocks-per-sm, --launch-ms and --trace-ms must be positive");
    }

    // The mixed probe runs its FFMAs in groups of MixedFmaGroup after every --unroll loads.
    for (const uint64_t fmas : options.mixed ? MixedFmas : std::vector<uint64_t>{})
    {
        if (fmas * options.unroll % MixedFmaGroup != 0)
        {
            throw std::runtime_error("--mixed needs an --unroll that is a multiple of 16 (" + std::to_string(fmas)
                + " FFMAs per load must be whole groups of " + std::to_string(MixedFmaGroup) + " per --unroll loads)");
        }
    }

    // The write probe issues its loads and stores in groups of WriteGroup, the sub-sector probe whole sectors of 4-byte
    // words: with another --unroll their bursts would not be --burst-accesses long.
    if (options.writes && options.unroll % WriteGroup != 0)
    {
        throw std::runtime_error("--writes needs an --unroll that is a multiple of " + std::to_string(WriteGroup));
    }

    if (options.subSector && options.unroll % (SectorBytes / 4) != 0)
    {
        throw std::runtime_error("--sub-sector needs an --unroll that is a multiple of " + std::to_string(SectorBytes / 4));
    }

    if (options.fitMinDuty == 0 || options.fitMinDuty > 100)
    {
        throw std::runtime_error("--fit-min-duty must be between 1 and 100");
    }

    if (options.burstAccesses == 0 || options.burstAccesses % options.unroll != 0)
    {
        throw std::runtime_error("--burst-accesses must be a positive multiple of --unroll");
    }

    if (options.maxTimeMs < options.minTimeMs)
    {
        throw std::runtime_error("--max-time-ms must not be smaller than --min-time-ms");
    }

    if ((!options.tune.empty() || !options.predictCsv.empty()) && options.modelFile.empty())
    {
        throw std::runtime_error("--tune and --predict need --model-file");
    }

    if (!options.tune.empty() && !options.predictCsv.empty())
    {
        throw std::runtime_error("--tune and --predict exclude each other");
    }

    if (std::find(options.tune.cbegin(), options.tune.cend(), std::string(MixedControlName)) != options.tune.cend())
    {
        throw std::runtime_error("--tune takes example kernels, not the control kernel mixed");
    }

    if (options.objective != "energy" && options.objective != "edp" && options.objective != "ed2p"
        && options.objective != "time")
    {
        throw std::runtime_error("--objective must be energy, edp, ed2p or time");
    }

    if (options.tuneMeasure != "none" && options.tuneMeasure != "contenders" && options.tuneMeasure != "all")
    {
        throw std::runtime_error("--tune-measure must be none, contenders or all");
    }

    if (!(options.contenderSd >= 0.0) || (options.maxSlowdownPct.has_value() && !(options.maxSlowdownPct.value() >= 0.0)))
    {
        throw std::runtime_error("--contender-sd and --max-slowdown must not be negative");
    }

    if (options.profile && !options.validate.empty())
    {
        throw std::runtime_error("--validate collects counters in a pass of its own, remove --profile");
    }

    if (options.lockClockMHz.has_value() && !options.monitor)
    {
        throw std::runtime_error("--lock-clock needs the NVML monitor, remove --no-monitor");
    }

    // Several intensities give the fit across access rates (fixed power vs cost per access). Without a locked clock, the
    // lower ones also give the GPU a chance to keep the probe at the baseline's clock. L1 and shared memory stay saturated
    // down to a low duty (at 25 % their access rate was still 62 % of the full rate), hence 10 %.
    if (options.duties.empty())
    {
        options.duties = {100, 50, 25, 10};
    }

    std::sort(options.duties.begin(), options.duties.end(), std::greater<uint64_t>());
    options.duties.erase(std::unique(options.duties.begin(), options.duties.end()), options.duties.end());

    for (auto* list : {&options.payloads, &options.widths, &options.smSweep})
    {
        if (list == &options.payloads && list->empty())
        {
            throw std::runtime_error("--payload needs at least one value");
        }

        std::sort(list->begin(), list->end(), std::greater<uint64_t>());
        list->erase(std::unique(list->begin(), list->end()), list->end());
    }

    // The default width is always measured: path energies, fits and differences use it.
    if (std::find(options.widths.cbegin(), options.widths.cend(), DefaultWidth) == options.widths.cend())
    {
        options.widths.insert(options.widths.begin(), DefaultWidth);
    }

    std::sort(options.payloads.begin(), options.payloads.end()); // random first
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
        if (spec.help.empty())
        {
            continue;
        }

        std::stringstream help(spec.help);
        std::string line;
        std::getline(help, line);
        std::cout << "  " << std::left << std::setw(22) << (spec.name + (spec.value.empty() ? "" : " " + spec.value)) << line
            << "\n";

        while (std::getline(help, line))
        {
            std::cout << std::string(24, ' ') << line << "\n";
        }
    }
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    const auto& specs = OptionSpecs();

    // Two passes: the presets (--all, --model, --quick) first, then everything else, so that an explicit option overrides a preset
    // wherever it stands ("--duty 100 --model" used to end with the preset's duties).
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

            if ((argument == "--all" || argument == "--model" || argument == "--quick") == presets)
            {
                spec->apply(options, value);
            }
        }
    }

    CheckAndNormalize(options);
    return options;
}
