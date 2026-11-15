// Tuning spaces copied from the Transpose, ClTuneGemm, Nbody and CoulombSum3d examples, and for the held-out kernels from
// the Reduction, ClTuneConvolution and KernelTunerPnpoly examples (CUDA variants, without reference kernels: the outputs
// are not checked). GEMM and Nbody use GlobalSizeType::OpenCL in their examples; here the global sizes
// are numbers of blocks (GlobalSizeType::CUDA, as in EnergyProbes), so their modifiers divide by the block size instead.
// Problem sizes are larger than in the examples where a launch would take only about a millisecond, because every launch
// through KTT is followed by a synchronisation and NVML queries, during which the GPU idles.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <sstream>
#include <random>
#include <stdexcept>

#include "ValidationKernels.h"

namespace
{

// Uniform floats in [low, high) from a seed.
std::vector<float> RandomFloats(const size_t count, const float low, const float high, const uint32_t seed)
{
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    std::generate(values.begin(), values.end(), [&]() { return distribution(engine); });
    return values;
}

bool IsMultiple(const uint64_t a, const uint64_t b)
{
    return b != 0 && a % b == 0;
}

// Transpose: 16384 x 16384 floats (1 GiB each way), memory-bound; tiles in shared memory or not, vector loads.
ValidationKernel AddTranspose(ktt::Tuner& tuner, const std::string& directory)
{
    const int width = 16384;
    const int height = 16384;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("mtran", directory + "Transpose/Transpose.cu",
        ktt::DimensionVector(width, height), ktt::DimensionVector(1, 1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Transpose", definition);

    std::vector<float> src = RandomFloats(static_cast<size_t>(width) * height, 0.0f, 10.0f, 1);
    std::vector<float> dst(src.size(), 0.0f);
    const ktt::ArgumentId srcId = tuner.AddArgumentVector(src, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId dstId = tuner.AddArgumentVector(dst, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId widthId = tuner.AddArgumentScalar(width);
    const ktt::ArgumentId heightId = tuner.AddArgumentScalar(height);
    tuner.SetArguments(definition, {dstId, srcId, widthId, heightId});

    const std::vector<uint64_t> sizes{1, 2, 4, 8, 16, 32, 64};
    tuner.AddParameter(kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4});
    tuner.AddParameter(kernel, "CR", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "LOCAL_MEM", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "PADD_LOCAL", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", sizes);
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", sizes);
    tuner.AddParameter(kernel, "TILE_SIZE_X", sizes);
    tuner.AddParameter(kernel, "TILE_SIZE_Y", sizes);
    tuner.AddParameter(kernel, "DIAGONAL_MAP", std::vector<uint64_t>{0, 1});

    tuner.AddConstraint(kernel, {"TILE_SIZE_X", "WORK_GROUP_SIZE_X"}, [](const std::vector<uint64_t>& v)
        { return v[0] == v[1]; });
    tuner.AddConstraint(kernel, {"TILE_SIZE_Y", "WORK_GROUP_SIZE_Y"}, [](const std::vector<uint64_t>& v)
        { return v[1] <= v[0] && v[0] % v[1] == 0; });
    tuner.AddConstraint(kernel, {"LOCAL_MEM", "TILE_SIZE_Y", "WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"},
        [](const std::vector<uint64_t>& v) { return !v[0] || v[1] <= v[2] * v[3]; });
    tuner.AddConstraint(kernel, {"LOCAL_MEM", "PADD_LOCAL"}, [](const std::vector<uint64_t>& v) { return v[0] || !v[1]; });
    tuner.AddConstraint(kernel, {"LOCAL_MEM", "VECTOR_TYPE"}, [](const std::vector<uint64_t>& v) { return !v[0] || v[1] == 1; });
    tuner.AddConstraint(kernel, {"TILE_SIZE_X", "VECTOR_TYPE"}, [](const std::vector<uint64_t>& v) { return v[0] * v[1] <= 64; });
    // The example only asks for at least 32 threads; blocks above 1024 threads cannot launch.
    tuner.AddConstraint(kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, [](const std::vector<uint64_t>& v)
        { return v[0] * v[1] >= 32 && v[0] * v[1] <= 1024; });

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"TILE_SIZE_X", "VECTOR_TYPE"}, [](const uint64_t size, const std::vector<uint64_t>& v) { return size / v[0] / v[1]; });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "TILE_SIZE_Y",
        ktt::ModifierAction::Divide);

    return {"transpose", kernel, definition, ""};
}

// GEMM (CLTune's kernel, reduced parameter set of the example): C = A * B for 4096 x 4096 floats, FFMA and shared memory.
ValidationKernel AddGemm(ktt::Tuner& tuner, const std::string& directory)
{
    const uint32_t m = 4096;
    const uint32_t n = 4096;
    const uint32_t k = 4096;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("gemm_fast",
        directory + "ClTuneGemm/ClTuneGemm.cu", ktt::DimensionVector(m, n), ktt::DimensionVector(1, 1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Gemm", definition);

    std::vector<float> a = RandomFloats(static_cast<size_t>(m) * k, -2.0f, 2.0f, 2);
    std::vector<float> b = RandomFloats(static_cast<size_t>(n) * k, -2.0f, 2.0f, 3);
    std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
    const ktt::ArgumentId mId = tuner.AddArgumentScalar(m);
    const ktt::ArgumentId nId = tuner.AddArgumentScalar(n);
    const ktt::ArgumentId kId = tuner.AddArgumentScalar(k);
    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId bId = tuner.AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId cId = tuner.AddArgumentVector(c, ktt::ArgumentAccessType::WriteOnly);
    tuner.SetArguments(definition, {mId, nId, kId, aId, bId, cId});

    tuner.AddParameter(kernel, "MWG", std::vector<uint64_t>{16, 32, 64});
    tuner.AddParameter(kernel, "NWG", std::vector<uint64_t>{16, 32, 64});
    tuner.AddParameter(kernel, "KWG", std::vector<uint64_t>{32});
    tuner.AddParameter(kernel, "MDIMC", std::vector<uint64_t>{8, 16, 32});
    tuner.AddParameter(kernel, "NDIMC", std::vector<uint64_t>{8, 16, 32});
    tuner.AddParameter(kernel, "MDIMA", std::vector<uint64_t>{8, 16, 32});
    tuner.AddParameter(kernel, "NDIMB", std::vector<uint64_t>{8, 16, 32});
    tuner.AddParameter(kernel, "KWI", std::vector<uint64_t>{2});
    tuner.AddParameter(kernel, "VWM", std::vector<uint64_t>{1, 2, 4});
    tuner.AddParameter(kernel, "VWN", std::vector<uint64_t>{1, 2, 4});
    tuner.AddParameter(kernel, "STRM", std::vector<uint64_t>{0});
    tuner.AddParameter(kernel, "STRN", std::vector<uint64_t>{0});
    tuner.AddParameter(kernel, "SA", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "SB", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "PRECISION", std::vector<uint64_t>{32});

    // MWG x NWG outputs per block of MDIMC x NDIMC threads.
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "MWG",
        ktt::ModifierAction::Divide);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "NWG",
        ktt::ModifierAction::Divide);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "MDIMC",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "NDIMC",
        ktt::ModifierAction::Multiply);

    auto multipleOfX = [](const std::vector<uint64_t>& v) { return IsMultiple(v[0], v[1]); };
    auto multipleOfXMulY = [](const std::vector<uint64_t>& v) { return IsMultiple(v[0], v[1] * v[2]); };
    auto multipleOfXMulYDivZ = [](const std::vector<uint64_t>& v) { return IsMultiple(v[0], v[1] * v[2] / v[3]); };
    tuner.AddConstraint(kernel, {"KWG", "KWI"}, multipleOfX);
    tuner.AddConstraint(kernel, {"MWG", "MDIMC", "VWM"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"NWG", "NDIMC", "VWN"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"MWG", "MDIMA", "VWM"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"NWG", "NDIMB", "VWN"}, multipleOfXMulY);
    tuner.AddConstraint(kernel, {"KWG", "MDIMC", "NDIMC", "MDIMA"}, multipleOfXMulYDivZ);
    tuner.AddConstraint(kernel, {"KWG", "MDIMC", "NDIMC", "NDIMB"}, multipleOfXMulYDivZ);

    return {"gemm", kernel, definition, ""};
}

// N-body: 128 Ki bodies, FFMA with one reciprocal square root (MUFU) per interaction, bodies optionally staged in shared
// memory. The example's default space adds the inner unroll factors twice; 0-8 is used here.
ValidationKernel AddNbody(ktt::Tuner& tuner, const std::string& directory)
{
    const int bodies = 128 * 1024;
    const float timeDelta = 0.001f;
    const float damping = 0.5f;
    const float softeningSqr = 0.1f * 0.1f;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("nbody_kernel", directory + "Nbody/Nbody.cu",
        ktt::DimensionVector(bodies, 1), ktt::DimensionVector(1, 1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Nbody", definition);

    std::vector<float> posX = RandomFloats(bodies, 0.0f, 20.0f, 4);
    std::vector<float> posY = RandomFloats(bodies, 0.0f, 20.0f, 5);
    std::vector<float> posZ = RandomFloats(bodies, 0.0f, 20.0f, 6);
    std::vector<float> mass = RandomFloats(bodies, 0.0f, 20.0f, 7);
    std::vector<float> velX = RandomFloats(bodies, 0.0f, 20.0f, 8);
    std::vector<float> velY = RandomFloats(bodies, 0.0f, 20.0f, 9);
    std::vector<float> velZ = RandomFloats(bodies, 0.0f, 20.0f, 10);
    std::vector<float> bodyInfo(4 * bodies);
    std::vector<float> bodyVel(4 * bodies, 0.0f);

    for (int i = 0; i < bodies; ++i)
    {
        bodyInfo[4 * i] = posX[i];
        bodyInfo[4 * i + 1] = posY[i];
        bodyInfo[4 * i + 2] = posZ[i];
        bodyInfo[4 * i + 3] = mass[i];
        bodyVel[4 * i] = velX[i];
        bodyVel[4 * i + 1] = velY[i];
        bodyVel[4 * i + 2] = velZ[i];
    }

    std::vector<float> newBodyInfo(4 * bodies, 0.0f);
    std::vector<float> newBodyVel(4 * bodies, 0.0f);
    const ktt::ArgumentId bodyInfoId = tuner.AddArgumentVector(bodyInfo, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId posXId = tuner.AddArgumentVector(posX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId posYId = tuner.AddArgumentVector(posY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId posZId = tuner.AddArgumentVector(posZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId massId = tuner.AddArgumentVector(mass, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId newBodyInfoId = tuner.AddArgumentVector(newBodyInfo, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId bodyVelId = tuner.AddArgumentVector(bodyVel, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId velXId = tuner.AddArgumentVector(velX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId velYId = tuner.AddArgumentVector(velY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId velZId = tuner.AddArgumentVector(velZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId newBodyVelId = tuner.AddArgumentVector(newBodyVel, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId timeDeltaId = tuner.AddArgumentScalar(timeDelta);
    const ktt::ArgumentId dampingId = tuner.AddArgumentScalar(damping);
    const ktt::ArgumentId softeningSqrId = tuner.AddArgumentScalar(softeningSqr);
    const ktt::ArgumentId bodiesId = tuner.AddArgumentScalar(bodies);
    tuner.SetArguments(definition, {timeDeltaId, bodyInfoId, posXId, posYId, posZId, massId, newBodyInfoId, bodyVelId, velXId,
        velYId, velZId, newBodyVelId, dampingId, softeningSqrId, bodiesId});

    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{16, 32, 64, 128, 256, 512});
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16});
    tuner.AddParameter(kernel, "OUTER_UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR1", std::vector<uint64_t>{0, 1, 2, 4, 8});
    tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR2", std::vector<uint64_t>{0, 1, 2, 4, 8});
    tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "LOCAL_MEM", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0});
    tuner.AddParameter(kernel, "VECTOR_TYPE", std::vector<uint64_t>{1, 2, 4});

    // One block row: (bodies / OUTER_UNROLL_FACTOR) threads in blocks of WORK_GROUP_SIZE_X x WORK_GROUP_SIZE_Y, where the
    // threads along y split the bodies they interact with.
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"OUTER_UNROLL_FACTOR", "WORK_GROUP_SIZE_X"},
        [](const uint64_t size, const std::vector<uint64_t>& v) { return size / v[0] / v[1]; });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::Multiply);

    tuner.AddConstraint(kernel, {"INNER_UNROLL_FACTOR2", "OUTER_UNROLL_FACTOR"}, [](const std::vector<uint64_t>& v)
        { return v[0] <= v[1]; });
    tuner.AddConstraint(kernel, {"INNER_UNROLL_FACTOR1", "INNER_UNROLL_FACTOR2"}, [](const std::vector<uint64_t>& v)
        { return v[0] * v[1] <= 256; });
    tuner.AddConstraint(kernel, {"VECTOR_TYPE", "USE_SOA"}, [](const std::vector<uint64_t>& v)
        { return (v[0] == 1 && v[1] == 0) || v[1] == 1; });
    tuner.AddConstraint(kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, [](const std::vector<uint64_t>& v)
        { return v[0] * v[1] >= 64 && v[0] * v[1] <= 1024; });

    return {"nbody", kernel, definition, "-use_fast_math"};
}

// Direct Coulomb summation on a 256^3 grid for 2048 atoms (the example uses 256): FFMA plus one reciprocal square root per
// atom and grid point, atom data broadcast from L1. Register limits (a compiler parameter in the example) are left out.
ValidationKernel AddCoulomb(ktt::Tuner& tuner, const std::string& directory)
{
    const int gridSize = 256;
    const int atoms = 2048;
    const float gridSpacing = 0.5f;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("directCoulombSum",
        directory + "CoulombSum3d/CoulombSum3d.cu", ktt::DimensionVector(gridSize, gridSize, gridSize),
        ktt::DimensionVector(1, 1, 1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("CoulombSum", definition);

    std::vector<float> atomX = RandomFloats(atoms, 0.0f, 20.0f, 11);
    std::vector<float> atomY = RandomFloats(atoms, 0.0f, 20.0f, 12);
    std::vector<float> atomZ = RandomFloats(atoms, 0.0f, 20.0f, 13);
    std::vector<float> atomW = RandomFloats(atoms, 0.0f, 0.5f, 14);
    std::vector<float> atomInfo(4 * atoms);

    for (int i = 0; i < atoms; ++i)
    {
        atomInfo[4 * i] = atomX[i];
        atomInfo[4 * i + 1] = atomY[i];
        atomInfo[4 * i + 2] = atomZ[i];
        atomInfo[4 * i + 3] = atomW[i];
    }

    std::vector<float> energyGrid(static_cast<size_t>(gridSize) * gridSize * gridSize, 0.0f);
    const ktt::ArgumentId atomInfoId = tuner.AddArgumentVector(atomInfo, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomXId = tuner.AddArgumentVector(atomX, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomYId = tuner.AddArgumentVector(atomY, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomZId = tuner.AddArgumentVector(atomZ, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomWId = tuner.AddArgumentVector(atomW, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId atomsId = tuner.AddArgumentScalar(atoms);
    const ktt::ArgumentId spacingId = tuner.AddArgumentScalar(gridSpacing);
    const ktt::ArgumentId gridSizeId = tuner.AddArgumentScalar(gridSize);
    const ktt::ArgumentId gridId = tuner.AddArgumentVector(energyGrid, ktt::ArgumentAccessType::WriteOnly);
    tuner.SetArguments(definition, {atomInfoId, atomXId, atomYId, atomZId, atomWId, atomsId, spacingId, gridSizeId, gridId});

    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{16, 32});
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8});
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_Z", std::vector<uint64_t>{1});
    tuner.AddParameter(kernel, "Z_ITERATIONS", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    tuner.AddParameter(kernel, "INNER_UNROLL_FACTOR", std::vector<uint64_t>{0, 1, 2, 4, 8, 16, 32});
    tuner.AddParameter(kernel, "USE_CONSTANT_MEMORY", std::vector<uint64_t>{0});
    tuner.AddParameter(kernel, "USE_SOA", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "VECTOR_SIZE", std::vector<uint64_t>{1});

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::DivideCeil);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, "WORK_GROUP_SIZE_Y",
        ktt::ModifierAction::DivideCeil);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Z, "Z_ITERATIONS",
        ktt::ModifierAction::DivideCeil);

    tuner.AddConstraint(kernel, {"INNER_UNROLL_FACTOR", "Z_ITERATIONS"}, [](const std::vector<uint64_t>& v)
        { return v[0] < v[1]; });
    tuner.AddConstraint(kernel, {"WORK_GROUP_SIZE_X", "WORK_GROUP_SIZE_Y"}, [](const std::vector<uint64_t>& v)
        { return v[0] * v[1] >= 64; });

    return {"coulomb", kernel, definition, "-use_fast_math"};
}

// The held-out kernels: added after the model was built, to test it on kernels that none of its design decisions looked
// at.

// Reduction (the example's kernel): the sum of 1 Gi floats (4 GiB), streamed from DRAM, partial sums in shared memory with
// a barrier per step, one atomic add per block. Only the single-pass variants (USE_ATOMICS 1): without atomics, the
// example's launcher runs further passes over the partial sums, which the validation launcher does not. WG_NUM blocks loop
// over the data (UNBOUNDED_WG 0) or one block per WORK_GROUP_SIZE_X elements (UNBOUNDED_WG 1).
ValidationKernel AddReduction(ktt::Tuner& tuner, const std::string& directory)
{
    const uint32_t n = 1024u * 1024 * 1024;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("reduce", directory + "Reduction/Reduction.cu",
        ktt::DimensionVector(n), ktt::DimensionVector(1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Reduction", definition);

    const std::vector<float> src = RandomFloats(n, 0.0f, 1000.0f, 11);
    const std::vector<float> dst(16, 0.0f);
    const ktt::ArgumentId srcId = tuner.AddArgumentVector(src, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId dstId = tuner.AddArgumentVector(dst, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId nId = tuner.AddArgumentScalar(n);
    const ktt::ArgumentId inOffsetId = tuner.AddArgumentScalar(0u);
    const ktt::ArgumentId outOffsetId = tuner.AddArgumentScalar(0u);
    tuner.SetArguments(definition, {srcId, dstId, nId, inOffsetId, outOffsetId});

    const uint64_t cus = tuner.GetCurrentDeviceInfo().GetMaxComputeUnits();
    tuner.AddParameter(kernel, "WORK_GROUP_SIZE_X", std::vector<uint64_t>{32, 64, 128, 256, 512});
    tuner.AddParameter(kernel, "UNBOUNDED_WG", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "WG_NUM", std::vector<uint64_t>{0, cus, cus * 2, cus * 4, cus * 8, cus * 16});
    tuner.AddParameter(kernel, "VECTOR_SIZE", std::vector<uint64_t>{1, 2, 4});
    tuner.AddParameter(kernel, "USE_ATOMICS", std::vector<uint64_t>{1});
    tuner.AddConstraint(kernel, {"UNBOUNDED_WG", "WG_NUM"}, [](const std::vector<uint64_t>& v)
        { return (v[0] == 1 && v[1] == 0) || (v[0] == 0 && v[1] > 0); });

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "WORK_GROUP_SIZE_X",
        ktt::ModifierAction::Multiply);
    // Blocks: WG_NUM, or one per WORK_GROUP_SIZE_X vectors of VECTOR_SIZE elements.
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"UNBOUNDED_WG", "WG_NUM", "WORK_GROUP_SIZE_X", "VECTOR_SIZE"},
        [](const uint64_t size, const std::vector<uint64_t>& v) { return v[0] == 1 ? (size / v[3] + v[2] - 1) / v[2] : v[1]; });

    return {"reduction", kernel, definition, ""};
}

// 2D convolution (CLTune's kernel, the example's default space): a 7 x 7 filter over 16384 x 16384 floats (the example's
// 4096 x 4096 takes about a millisecond); tiles in shared memory with or without halo threads, or none (through L1), vector
// loads and stores, unrolled filter loops. The example's modifiers give OpenCL work-items; here blocks, so the global size is
// divided by the outputs per block (TBX x WPTX).
ValidationKernel AddConvolution(ktt::Tuner& tuner, const std::string& directory)
{
    const int hfs = 3;
    const int fs = 2 * hfs + 1;
    const uint32_t sizeX = 16384;
    const uint32_t sizeY = 16384;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("conv",
        directory + "ClTuneConvolution/ClTuneConvolution.cu", ktt::DimensionVector(sizeX, sizeY), ktt::DimensionVector(1, 1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Convolution", definition);

    std::vector<float> src(static_cast<size_t>(sizeX + 2 * hfs) * (sizeY + 2 * hfs), 0.0f);
    const std::vector<float> values = RandomFloats(static_cast<size_t>(sizeX) * sizeY, -2.0f, 2.0f, 12);

    for (size_t y = 0; y < sizeY; ++y)
    {
        std::copy_n(values.cbegin() + static_cast<std::ptrdiff_t>(y * sizeX), sizeX,
            src.begin() + static_cast<std::ptrdiff_t>((y + hfs) * (sizeX + 2 * hfs) + hfs));
    }

    // The example's normalised Gaussian filter.
    std::vector<float> coeff(fs * fs);
    float sum = 0.0f;

    for (int x = 0; x < fs; ++x)
    {
        for (int y = 0; y < fs; ++y)
        {
            const float mean = fs / 2.0f;
            coeff[y * fs + x] = std::exp(-0.5f * ((x - mean) * (x - mean) + (y - mean) * (y - mean)));
            sum += coeff[y * fs + x];
        }
    }

    for (auto& value : coeff)
    {
        value /= sum;
    }

    const std::vector<float> dst(static_cast<size_t>(sizeX) * sizeY, 0.0f);
    const ktt::ArgumentId sizeXId = tuner.AddArgumentScalar(static_cast<int>(sizeX));
    const ktt::ArgumentId sizeYId = tuner.AddArgumentScalar(static_cast<int>(sizeY));
    const ktt::ArgumentId srcId = tuner.AddArgumentVector(src, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId coeffId = tuner.AddArgumentVector(coeff, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId dstId = tuner.AddArgumentVector(dst, ktt::ArgumentAccessType::WriteOnly);
    tuner.SetArguments(definition, {sizeXId, sizeYId, srcId, coeffId, dstId});

    const std::vector<uint64_t> blocks{8, 16, 32, 64};
    const std::vector<uint64_t> wpt{1, 2, 4, 8, 16};
    tuner.AddParameter(kernel, "TBX", blocks);
    tuner.AddParameter(kernel, "TBY", blocks);
    tuner.AddParameter(kernel, "LOCAL", std::vector<uint64_t>{0, 1, 2});
    tuner.AddParameter(kernel, "WPTX", wpt);
    tuner.AddParameter(kernel, "WPTY", wpt);
    tuner.AddParameter(kernel, "VECTOR", std::vector<uint64_t>{1, 2, 4});
    tuner.AddParameter(kernel, "UNROLL_FACTOR1", std::vector<uint64_t>{0, 1, static_cast<uint64_t>(fs)});
    tuner.AddParameter(kernel, "UNROLL_FACTOR2", std::vector<uint64_t>{0, 1, static_cast<uint64_t>(fs)});
    tuner.AddParameter(kernel, "PADDING", std::vector<uint64_t>{0, 1});
    const std::vector<uint64_t> threads{8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74};
    tuner.AddParameter(kernel, "TBX_XL", threads);
    tuner.AddParameter(kernel, "TBY_XL", threads);

    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"TBX", "WPTX"},
        [](const uint64_t size, const std::vector<uint64_t>& v) { return size / (v[0] * v[1]); });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"TBY", "WPTY"},
        [](const uint64_t size, const std::vector<uint64_t>& v) { return size / (v[0] * v[1]); });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "TBX_XL",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "TBY_XL",
        ktt::ModifierAction::Multiply);

    // The example's constraints: halo threads (LOCAL 2) are TBX + the halo per WPTX, otherwise TBX_XL = TBX; vectors divide
    // the outputs per thread (and the halo with LOCAL 2); padding only with shared memory; WPT divides the tile.
    auto halo = [hfs](const std::vector<uint64_t>& v)
    {
        return v[0] == 2 ? v[1] == v[2] + (2 * hfs + v[3] - 1) / v[3] : v[1] == v[2];
    };
    tuner.AddConstraint(kernel, {"LOCAL", "TBX_XL", "TBX", "WPTX"}, halo);
    tuner.AddConstraint(kernel, {"LOCAL", "TBY_XL", "TBY", "WPTY"}, halo);
    tuner.AddConstraint(kernel, {"LOCAL", "VECTOR", "WPTX"}, [hfs](const std::vector<uint64_t>& v)
        { return v[0] == 2 ? IsMultiple(v[2], v[1]) && IsMultiple(2 * hfs, v[1]) : IsMultiple(v[2], v[1]); });
    tuner.AddConstraint(kernel, {"LOCAL", "PADDING"}, [](const std::vector<uint64_t>& v) { return v[1] == 0 || v[0] != 0; });
    tuner.AddConstraint(kernel, {"TBX", "WPTX"}, [](const std::vector<uint64_t>& v) { return v[0] % v[1] == 0; });
    tuner.AddConstraint(kernel, {"TBY", "WPTY"}, [](const std::vector<uint64_t>& v) { return v[0] % v[1] == 0; });
    // Not in the example, whose space keeps configurations that cannot launch: at most 1024 threads per block, and a tile
    // that fits in 48 KiB of shared memory.
    tuner.AddConstraint(kernel, {"TBX_XL", "TBY_XL"}, [](const std::vector<uint64_t>& v) { return v[0] * v[1] <= 1024; });
    tuner.AddConstraint(kernel, {"LOCAL", "TBX", "WPTX", "TBY", "WPTY", "PADDING"}, [hfs](const std::vector<uint64_t>& v)
        { return v[0] == 0 || (v[3] * v[4] + 2 * hfs) * (v[1] * v[2] + 2 * hfs + v[5]) * sizeof(float) <= 48 * 1024; });

    return {"convolution", kernel, definition, ""};
}

// Point in polygon (Kernel Tuner's kernel): 20 M points against a polygon of 600 vertices, instruction-bound; the variants
// compute the crossing test with branches, compares and selects (BETWEEN_METHOD, USE_METHOD). The kernel reads the vertices
// from constant memory (d_vertices), which the example never fills (it notes that KTT would need to move constant data);
// here it is filled as a symbol argument, so that the kernel computes on the polygon.
ValidationKernel AddPnpoly(ktt::Tuner& tuner, const std::string& directory)
{
    constexpr size_t vertices = 600;
    const int points = 20000000;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("Pnpoly",
        directory + "KernelTunerPnpoly/KernelTunerPnpoly.cu", ktt::DimensionVector(points), ktt::DimensionVector(1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Pnpoly", definition);

    const std::vector<float> pointData = RandomFloats(2 * static_cast<size_t>(points), 0.0f, 1.0f, 13);
    const std::vector<float> vertexData = RandomFloats(2 * vertices, 0.0f, 1.0f, 14);
    std::array<float, 2 * vertices> vertexSymbol{};
    std::copy(vertexData.cbegin(), vertexData.cend(), vertexSymbol.begin());
    const std::vector<int> bitmap(points, 0);
    const ktt::ArgumentId bitmapId = tuner.AddArgumentVector(bitmap, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId pointsId = tuner.AddArgumentVector(pointData, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId verticesId = tuner.AddArgumentVector(vertexData, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId nId = tuner.AddArgumentScalar(points);
    const ktt::ArgumentId symbolId = tuner.AddArgumentSymbol(vertexSymbol, "", "d_vertices");
    tuner.SetArguments(definition, {bitmapId, pointsId, verticesId, nId, symbolId});

    std::vector<uint64_t> blockSizes;

    for (uint64_t size = 32; size <= 992; size += 32)
    {
        blockSizes.push_back(size);
    }

    tuner.AddParameter(kernel, "VERTICES", std::vector<uint64_t>{vertices});
    tuner.AddParameter(kernel, "BLOCK_SIZE_X", blockSizes);
    tuner.AddParameter(kernel, "TILE_SIZE", std::vector<uint64_t>{1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
    tuner.AddParameter(kernel, "BETWEEN_METHOD", std::vector<uint64_t>{0, 1, 2, 3});
    tuner.AddParameter(kernel, "USE_METHOD", std::vector<uint64_t>{0, 1, 2});
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"BLOCK_SIZE_X", "TILE_SIZE"}, [](const uint64_t size, const std::vector<uint64_t>& v)
        { return ((size + v[0] - 1) / v[0] + v[1] - 1) / v[1]; });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE_X",
        ktt::ModifierAction::Multiply);

    return {"pnpoly", kernel, definition, ""};
}

// The second held-out set (--validate heldout2): the remaining KTT examples that are single CUDA kernels with a tuning
// space.

// BiCG of PolyBench (Bicg example): y1 = A x1 and y2 = A^T x2 in one fused kernel. A is 32768 x 16384 floats (2 GiB, about
// 2.5 ms per launch). Only ATOMICS 1: with 0 the example runs two reduction kernels after the fused one.
ValidationKernel AddBicg(ktt::Tuner& tuner, const std::string& directory)
{
    const int m = 16384;
    const int n = 32768;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("bicgFused", directory + "Bicg/Bicg.cu",
        ktt::DimensionVector(1, 1), ktt::DimensionVector(1, 1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("Bicg", definition);

    const std::vector<float> a = RandomFloats(static_cast<size_t>(m) * n, 0.0f, 100.0f, 21);
    const std::vector<float> x1 = RandomFloats(static_cast<size_t>(m), 0.0f, 100.0f, 22);
    const std::vector<float> x2 = RandomFloats(static_cast<size_t>(n), 0.0f, 100.0f, 23);
    const std::vector<float> y1(static_cast<size_t>(n), 0.0f);
    const std::vector<float> y2(static_cast<size_t>(m), 0.0f);
    const ktt::ArgumentId aId = tuner.AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId x1Id = tuner.AddArgumentVector(x1, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId y1Id = tuner.AddArgumentVector(y1, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId x2Id = tuner.AddArgumentVector(x2, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId y2Id = tuner.AddArgumentVector(y2, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId mId = tuner.AddArgumentScalar(m);
    const ktt::ArgumentId nId = tuner.AddArgumentScalar(n);
    tuner.SetArguments(definition, {aId, x1Id, y1Id, x2Id, y2Id, mId, nId});

    tuner.AddParameter(kernel, "FUSED", std::vector<uint64_t>{2});
    tuner.AddParameter(kernel, "BICG_BATCH", std::vector<uint64_t>{1, 2, 4, 8, 16, 32, 64});
    tuner.AddParameter(kernel, "USE_SHARED_MATRIX", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "USE_SHARED_VECTOR_1", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "USE_SHARED_VECTOR_2", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "USE_SHARED_REDUCTION_1", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "USE_SHARED_REDUCTION_2", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "ATOMICS", std::vector<uint64_t>{1});
    tuner.AddParameter(kernel, "UNROLL_BICG_STEP", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "ROWS_PROCESSED", std::vector<uint64_t>{128, 256, 512, 1024});
    tuner.AddParameter(kernel, "TILE", std::vector<uint64_t>{16, 32, 64});

    // The example's thread space (OpenCL sizes there): m threads by n / ROWS_PROCESSED x TILE / BICG_BATCH in blocks of
    // TILE x TILE / BICG_BATCH, i.e. a grid of m / TILE by n / ROWS_PROCESSED blocks.
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, {"TILE"},
        [m](const uint64_t, const std::vector<uint64_t>& v) { return static_cast<uint64_t>(m) / v[0]; });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y, {"ROWS_PROCESSED"},
        [n](const uint64_t, const std::vector<uint64_t>& v) { return static_cast<uint64_t>(n) / v[0]; });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, {"TILE"},
        [](const uint64_t, const std::vector<uint64_t>& v) { return v[0]; });
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, {"TILE", "BICG_BATCH"},
        [](const uint64_t, const std::vector<uint64_t>& v) { return v[0] / v[1]; });
    tuner.AddConstraint(kernel, {"TILE", "BICG_BATCH"}, [](const std::vector<uint64_t>& v)
        { return v[0] * v[0] / v[1] <= 1024 && v[1] <= v[0]; });

    return {"bicg", kernel, definition, ""};
}

// Kernel Tuner's 2-D convolution (KernelTunerConvolution example) on the task of ClTuneConvolution here: a 16384 x 16384
// image and the same normalised 7 x 7 Gaussian filter, so that the two kernels form a pair for one task. The filter is read
// from constant memory (d_filter), which the example never fills; here it is filled as a symbol argument. The example's
// PADDING parameter is not read by the kernel, which tests use_padding: the parameter is use_padding here.
ValidationKernel AddKtConvolution(ktt::Tuner& tuner, const std::string& directory)
{
    constexpr int hfs = 3;
    constexpr int fs = 2 * hfs + 1;
    const uint32_t sizeX = 16384;
    const uint32_t sizeY = 16384;
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("Convolution",
        directory + "KernelTunerConvolution/KernelTunerConvolution.cu", ktt::DimensionVector(sizeX, sizeY),
        ktt::DimensionVector(1, 1));
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("KtConvolution", definition);

    // The same input as ClTuneConvolution (seed 12, values in [-2, 2], a halo of zeros).
    std::vector<float> input(static_cast<size_t>(sizeX + 2 * hfs) * (sizeY + 2 * hfs), 0.0f);
    const std::vector<float> values = RandomFloats(static_cast<size_t>(sizeX) * sizeY, -2.0f, 2.0f, 12);

    for (size_t y = 0; y < sizeY; ++y)
    {
        std::copy_n(values.cbegin() + static_cast<std::ptrdiff_t>(y * sizeX), sizeX,
            input.begin() + static_cast<std::ptrdiff_t>((y + hfs) * (sizeX + 2 * hfs) + hfs));
    }

    std::array<float, 33 * 33> filter{};
    float sum = 0.0f;

    for (int x = 0; x < fs; ++x)
    {
        for (int y = 0; y < fs; ++y)
        {
            const float mean = fs / 2.0f;
            filter[static_cast<size_t>(y * fs + x)] = std::exp(-0.5f * ((x - mean) * (x - mean) + (y - mean) * (y - mean)));
            sum += filter[static_cast<size_t>(y * fs + x)];
        }
    }

    for (auto& value : filter)
    {
        value /= sum;
    }

    const std::vector<float> filterVector(filter.cbegin(), filter.cend());
    const std::vector<float> output(static_cast<size_t>(sizeX) * sizeY, 0.0f);
    const ktt::ArgumentId outputId = tuner.AddArgumentVector(output, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId inputId = tuner.AddArgumentVector(input, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId filterId = tuner.AddArgumentVector(filterVector, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId symbolId = tuner.AddArgumentSymbol(filter, "", "d_filter");
    tuner.SetArguments(definition, {outputId, inputId, filterId, symbolId});

    tuner.AddParameter(kernel, "IMAGE_WIDTH", std::vector<uint64_t>{sizeX});
    tuner.AddParameter(kernel, "IMAGE_HEIGHT", std::vector<uint64_t>{sizeY});
    tuner.AddParameter(kernel, "HFS", std::vector<uint64_t>{hfs});
    tuner.AddParameter(kernel, "BLOCK_SIZE_X", std::vector<uint64_t>{1, 2, 4, 8, 16, 32, 48, 64, 96, 112, 128});
    tuner.AddParameter(kernel, "BLOCK_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    tuner.AddParameter(kernel, "TILE_SIZE_X", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8});
    tuner.AddParameter(kernel, "TILE_SIZE_Y", std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8});
    tuner.AddParameter(kernel, "use_padding", std::vector<uint64_t>{0, 1});
    tuner.AddParameter(kernel, "READ_ONLY", std::vector<uint64_t>{0, 1});

    const auto blocks = [](const uint64_t size, const std::vector<uint64_t>& v)
    {
        return ((size + v[0] - 1) / v[0] + v[1] - 1) / v[1];
    };
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
        {"BLOCK_SIZE_X", "TILE_SIZE_X"}, blocks);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::Y,
        {"BLOCK_SIZE_Y", "TILE_SIZE_Y"}, blocks);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE_X",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::Y, "BLOCK_SIZE_Y",
        ktt::ModifierAction::Multiply);
    tuner.AddConstraint(kernel, {"BLOCK_SIZE_X", "BLOCK_SIZE_Y"}, [](const std::vector<uint64_t>& v)
        { return v[0] * v[1] >= 64 && v[0] * v[1] <= 1024; });
    tuner.AddConstraint(kernel, {"TILE_SIZE_X", "TILE_SIZE_Y"}, [](const std::vector<uint64_t>& v) { return v[0] * v[1] < 30; });
    // Not in the example: the shared tile (with up to 31 padding columns) must fit in 48 KiB.
    tuner.AddConstraint(kernel, {"BLOCK_SIZE_X", "TILE_SIZE_X", "BLOCK_SIZE_Y", "TILE_SIZE_Y"}, [](const std::vector<uint64_t>& v)
        { return (v[2] * v[3] + 2 * hfs) * (v[0] * v[1] + 2 * hfs + 31) * sizeof(float) <= 48 * 1024; });

    return {"ktconvolution", kernel, definition, ""};
}

} // namespace

const std::vector<std::pair<std::string, std::vector<std::string>>>& ValidationKernelSets()
{
    static const std::vector<std::pair<std::string, std::vector<std::string>>> sets = {
        {"all", {"transpose", "gemm", "nbody", "coulomb"}}, {"heldout", {"reduction", "convolution", "pnpoly"}},
        {"heldout2", {"bicg", "ktconvolution"}}};
    return sets;
}

ValidationKernel AddValidationKernel(ktt::Tuner& tuner, const std::string& name, const std::string& examplesDirectory)
{
    using Add = ValidationKernel (*)(ktt::Tuner&, const std::string&);
    const std::map<std::string, Add> kernels = {{"transpose", AddTranspose}, {"gemm", AddGemm}, {"nbody", AddNbody},
        {"coulomb", AddCoulomb}, {"reduction", AddReduction}, {"convolution", AddConvolution}, {"pnpoly", AddPnpoly},
        {"bicg", AddBicg}, {"ktconvolution", AddKtConvolution}};
    const auto kernel = kernels.find(name);

    if (kernel == kernels.cend())
    {
        throw std::runtime_error("unknown validation kernel " + name);
    }

    return kernel->second(tuner, examplesDirectory);
}
