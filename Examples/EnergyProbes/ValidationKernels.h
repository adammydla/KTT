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

// Names accepted by AddValidationKernel, in the order they run with --validate all.
std::vector<std::string> ValidationKernelNames();

// The held-out kernels (--validate heldout): Reduction, ClTuneConvolution and KernelTunerPnpoly, added after the model was
// built, so that none of its design decisions looked at them.
std::vector<std::string> HeldOutKernelNames();

// Adds the kernel definition, its arguments and the example's tuning space (sizes grown so that one launch takes at least a
// few milliseconds). The tuner must use GlobalSizeType::CUDA. The caller sets the launcher.
ValidationKernel AddValidationKernel(ktt::Tuner& tuner, const std::string& name, const std::string& examplesDirectory);
