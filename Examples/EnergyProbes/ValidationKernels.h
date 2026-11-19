// Kernels of other KTT examples with their tuning spaces, as real code for the energy model: --validate measures them
// for the model's uncertainty, --tune tunes them with the model. The kernel sources are read from those examples'
// folders; the tuning spaces are copied from the examples' programs (which define them in main).
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <Ktt.h>

struct ValidationKernel
{
    std::string name;
    ktt::KernelId kernel;
    ktt::KernelDefinitionId definition;
    std::string compilerOptions; // the options the example compiles with; KTT's compiler options are per tuner
};

// The kernels by set, in the order they run: "all" (Transpose, ClTuneGemm, Nbody, CoulombSum3d: the kernels the model
// was designed on), "heldout" (Reduction, ClTuneConvolution, KernelTunerPnpoly) and "heldout2" (Bicg,
// KernelTunerConvolution on ClTuneConvolution's task), which were added after the model was built. These nine calibrate the
// model's uncertainty. "heldout3" (five PolyBench/GPU kernels, kernels/HeldOut3.cu) was the first frozen evaluation and is
// reused since its biases led to the model change of 2026-11-17; "heldout4" (five more, kernels/HeldOut4.cu, written and
// measured before that change) is the frozen evaluation since (docs/PROBES.md, "Which data was used for what").
// "heldout5" (2026-11-17, after that model change) holds the CUDA kernels of KTT examples not used before: ATF's Gaussian
// convolution (AtfSamples), the first pass of Sort's radix sort and two kernels of Sort2's; it calibrates nothing either.
const std::vector<std::pair<std::string, std::vector<std::string>>>& ValidationKernelSets();

// The empty kernel --validate measures the power between launches with (AddValidationKernel accepts the name; no set has it).
inline const std::string LaunchGapKernel = "launchgap";

// The input data of the next kernels added (--validate-data): "example" (default: as each kernel below writes it), "zeros",
// or "random" (uniform floats in [0, 1), random 32-bit keys). Only some kernels support the other two; a check of whether
// the data values move power (2026-11-17).
void SetValidationData(const std::string& mode);
bool SupportsDataMode(const std::string& name);

// Adds the kernel definition, its arguments and the example's tuning space (sizes grown so that one launch takes at least a
// few milliseconds). The tuner must use GlobalSizeType::CUDA. The caller sets the launcher and the searcher.
ValidationKernel AddValidationKernel(ktt::Tuner& tuner, const std::string& name, const std::string& examplesDirectory);
