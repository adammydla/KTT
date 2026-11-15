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
// KernelTunerConvolution on ClTuneConvolution's task), which were added after the model was built.
const std::vector<std::pair<std::string, std::vector<std::string>>>& ValidationKernelSets();

// Adds the kernel definition, its arguments and the example's tuning space (sizes grown so that one launch takes at least a
// few milliseconds). The tuner must use GlobalSizeType::CUDA. The caller sets the launcher and the searcher.
ValidationKernel AddValidationKernel(ktt::Tuner& tuner, const std::string& name, const std::string& examplesDirectory);
