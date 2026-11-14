// The smallest program that tunes a kernel of its own for energy with the model (README, "Using the model in a tuner").
// Not part of the EnergyProbesCuda project; tests/test_tools.py compiles and links it. To build it by hand next to a KTT
// build (from KTT/Build/x86_64_Release; CUDA_PATH as for KTT):
//
//   g++ -std=c++17 -O2 -I../../Source -I../../Libraries/Json-3.9.1 -I$CUDA_PATH/include \
//       ../../Examples/EnergyProbes/example/MinimalTuner.cpp ../../Examples/EnergyProbes/EnergyModel.cpp \
//       ../../Examples/EnergyProbes/EnergyTuner.cpp -L. -lktt -Wl,-rpath,. -o MinimalTuner
//   ./MinimalTuner ../../Examples/EnergyProbes/models/rtx3090_1200mhz.model.json
//
// KTT must be built with --profiling=cupti (the counters) and the GPU should run at the model's clock.
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Ktt.h>

#include "../EnergyModel.h"
#include "../EnergyTuner.h"

// A kernel with a tuning space: y = a x + y with ITEMS elements per thread, in blocks of BLOCK threads.
const std::string source = R"(
extern "C" __global__ void saxpy(const float a, const float* x, float* y, const int n)
{
    const int first = (blockIdx.x * BLOCK + threadIdx.x) * ITEMS;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i)
    {
        if (first + i < n)
        {
            y[first + i] = a * x[first + i] + y[first + i];
        }
    }
}
)";

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: MinimalTuner MODEL.json [device]\n";
        return 1;
    }

    const int n = 1 << 26;
    ktt::Tuner tuner(0, argc > 2 ? std::stoul(argv[2]) : 0, ktt::ComputeApi::CUDA);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

    // The kernel, its arguments and its tuning space, as in any KTT program.
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinition("saxpy", source, ktt::DimensionVector(n),
        ktt::DimensionVector(1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Saxpy", definition);
    const std::vector<float> x(n, 1.0f);
    const std::vector<float> y(n, 2.0f);
    tuner.SetArguments(definition, {tuner.AddArgumentScalar(2.0f), tuner.AddArgumentVector(x, ktt::ArgumentAccessType::ReadOnly),
        tuner.AddArgumentVector(y, ktt::ArgumentAccessType::ReadWrite), tuner.AddArgumentScalar(n)});
    tuner.AddParameter(kernel, "BLOCK", std::vector<uint64_t>{64, 128, 256, 512, 1024});
    tuner.AddParameter(kernel, "ITEMS", std::vector<uint64_t>{1, 2, 4, 8, 16});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"BLOCK", "ITEMS"},
        [](const uint64_t size, const std::vector<uint64_t>& v) { return (size + v[0] * v[1] - 1) / (v[0] * v[1]); });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK",
        ktt::ModifierAction::Multiply);

    // Time every configuration, then collect the counters the model needs, then price and choose.
    EnergyModel model(argv[1]);
    EnergyTuner energy(tuner);
    energy.AddKernel("saxpy", kernel, definition);
    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
    energy.TimeConfigurations("saxpy", 25);
    std::vector<TunedConfiguration> rows = energy.CollectCounters(model.Counters());

    std::vector<EnergyPrediction> predictions;
    std::vector<const TunedConfiguration*> priced;

    for (const auto& row : rows)
    {
        if (row.kernelS > 0.0 && !row.counters.empty())
        {
            predictions.push_back(model.Predict(row.counters, row.kernelS, row.temperatureC, row.smClockMHz, true, row.kernel));
            priced.push_back(&row);
        }
    }

    if (predictions.empty())
    {
        std::cerr << "no configuration has a time and counters\n";
        return 1;
    }

    const EnergyRecommendation choice = model.Recommend(predictions, EnergyObjective::Energy);
    const EnergyPrediction& best = predictions[choice.index];
    std::cout << "use " << priced[choice.index]->configuration << ": " << best.energyJ << " J +- " << best.sdJ
              << " per launch, " << best.kernelS * 1e3 << " ms; " << choice.contenders.size()
              << " contender(s) to measure if the difference matters\n";
    return 0;
}
