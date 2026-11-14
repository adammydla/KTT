#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <Ktt.h>

// Kernels of other KTT examples (Transpose, ClTuneGemm, Nbody, CoulombSum3d; held out: Reduction, ClTuneConvolution,
// KernelTunerPnpoly) with their tuning spaces, used to check
// energy predictions built from the probe results on real code. The kernel sources are read from those examples' folders.
// The control kernel (--validate mixed, see Validation.cpp) is a probe: it also has arguments to set for every
// configuration and counts of its own.
struct ValidationKernel
{
    std::string name;
    ktt::KernelId kernel;
    ktt::KernelDefinitionId definition;
    std::string compilerOptions; // the options the example compiles with; KTT's compiler options are per tuner
    // Control kernel only: sets the runtime arguments of a configuration before every launch, and the buffer of the
    // probe's own counts (accesses per block, then launches; `blocks` blocks), which the power pass reads.
    std::function<void(ktt::ComputeInterface&, const ktt::KernelConfiguration&)> configure = {};
    std::optional<ktt::ArgumentId> counts = std::nullopt;
    uint64_t blocks = 0;
};

// The control kernel of --validate: the mixed-intensity probe (16-byte loads at the L1, L2 and DRAM footprints with these
// FFMAs per load), in launches of these lengths.
inline constexpr const char* MixedControlName = "mixed";
inline const std::vector<uint64_t> MixedControlFmas = {0, 16, 128, 1024};
inline const std::vector<uint64_t> MixedControlLaunchMs = {5, 50};

// AccelWattch's microbenchmarks (Kandiah et al., MICRO 2021) as one kernel with a configuration per benchmark
// (--validate accelwattch, --accelwattch-dir; 2026-11-14): a held-out set written by others for power modelling. The
// kernel file comes from tools/accelwattch_import.py, which reads the published sources.
inline constexpr const char* AccelWattchName = "accelwattch";

// Names accepted by AddValidationKernel, in the order they run with --validate all.
std::vector<std::string> ValidationKernelNames();

// The held-out kernels (--validate heldout): Reduction, ClTuneConvolution and KernelTunerPnpoly, added after the model was
// built, so that none of its design decisions looked at them.
std::vector<std::string> HeldOutKernelNames();

// The second held-out set (--validate heldout2, 2026-11-13): Bicg and KernelTunerConvolution (the latter on
// ClTuneConvolution's task, a pair of kernels for one task). Not to be looked at per configuration while changing the model.
std::vector<std::string> HeldOut2KernelNames();

// The third held-out set (--validate heldout3, 2026-11-14, NEXT_STEPS_PLAN M6): PolyBench/GPU's gesummv, syrk, syr2k
// and 2-D and 3-D convolution with small tuning spaces (kernels/HeldOut3.cu). Run once with the unchanged model; not to
// be looked at per configuration while the model changes.
std::vector<std::string> HeldOut3KernelNames();

// What the example kernels compute on (--validate-data, --validate-size-scale; 2026-11-13, NEXT_STEPS_PLAN B12 and B14).
struct ValidationInputs
{
    std::string data = "random"; // random: uniform floats in each kernel's range (default); zero: all zeros; small:
                                 // integers 0 to 15 stored as floats (few bits set, as low-entropy data)
    double sizeScale = 1.0;      // gemm's m = n = k and convolution's image side times this (multiples of 1024 / 64 kept)
    std::string accelwattchDirectory; // --accelwattch-dir (AccelWattchName)
};

// Applies to the kernels added after it.
void SetValidationInputs(const ValidationInputs& inputs);

// A searcher that picks configurations at random from a fixed seed (--validate-seed), so that runs with the same seed and
// the same space measure the same configurations (KTT's RandomSearcher seeds from std::random_device).
class SeededSearcher : public ktt::Searcher
{
public:
    explicit SeededSearcher(uint64_t seed);
    void OnInitialize() override;
    bool CalculateNextConfiguration(const ktt::KernelResult& previousResult) override;
    ktt::KernelConfiguration GetCurrentConfiguration() const override;

private:
    uint64_t m_Seed;
    std::vector<uint64_t> m_Order;
    size_t m_Next = 0;
    ktt::KernelConfiguration m_Current;
};

// Adds the kernel definition, its arguments and the example's tuning space (sizes grown so that one launch takes at least a
// few milliseconds). The tuner must use GlobalSizeType::CUDA. The caller sets the launcher.
ValidationKernel AddValidationKernel(ktt::Tuner& tuner, const std::string& name, const std::string& examplesDirectory);
