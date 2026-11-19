// Probes of what an application does besides its kernels, measured by tools/app_energy.py probes: the GPU idle between
// kernels (how fast its power falls from the awake level to the idle level with a context), and copies between the host
// and the GPU. The kernel model prices neither.
//
//   app_probes gap KERNEL_US GAP_US SECONDS   sleep kernels of KERNEL_US on every SM (the sleep baseline of the probes),
//                                             each followed by a host wait of GAP_US (0: back to back)
//   app_probes copy KIND MEMORY MB SECONDS    cudaMemcpy of MB megabytes again and again; KIND h2d, d2h or d2d, MEMORY
//                                             pinned or pageable (host side)
//   app_probes context BEFORE_S AFTER_S [MB]  creates the CUDA context, waits BEFORE_S, runs one sleep kernel of 0.5 s,
//                                             waits AFTER_S and exits (when the GPU wakes up and falls asleep again);
//                                             with MB, allocates and clears that much device memory first
//
// Prints one line "result key=value ..." (launches or copies, their GPU time from CUDA events, the wall time, bytes).
// The device is the current one (apps/device_select.cpp sets it from APP_CUDA_DEVICE).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        const cudaError_t status_ = (call);                                                                            \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_));                              \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;

namespace
{

__device__ unsigned long long GlobalTimer()
{
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// As probeSleep in kernels/Baseline.cu: resident warps on every SM that sleep, so the GPU is awake at its clock but
// issues almost nothing.
__global__ void SleepKernel(unsigned int* sink, const unsigned long long durationNs)
{
    const unsigned long long start = GlobalTimer();
    unsigned int iterations = 0;

    while (GlobalTimer() - start < durationNs)
    {
        asm volatile("nanosleep.u32 %0;" : : "r"(20000u));
        ++iterations;
    }

    if (iterations == 0xffffffffu)
    {
        sink[threadIdx.x] = iterations;
    }
}

double Seconds(const Clock::time_point a, const Clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

// Waits on the host: sleeps for long waits, spins for the last 200 us (sleep_for overshoots by tens of us).
void Wait(const double seconds)
{
    const auto end = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));

    if (seconds > 400e-6)
    {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds - 200e-6));
    }

    while (Clock::now() < end)
    {
    }
}

int Gap(const double kernelUs, const double gapUs, const double seconds)
{
    int device = 0;
    cudaDeviceProp properties;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaGetDeviceProperties(&properties, device));
    unsigned int* sink = nullptr;
    CHECK(cudaMalloc(&sink, 1024 * sizeof(unsigned int)));
    cudaEvent_t begin;
    cudaEvent_t end;
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&end));
    const int blocks = properties.multiProcessorCount * 4;
    const auto durationNs = static_cast<unsigned long long>(kernelUs * 1e3);

    SleepKernel<<<blocks, 256>>>(sink, durationNs);
    CHECK(cudaDeviceSynchronize());

    const auto start = Clock::now();
    long launches = 0;
    double kernelS = 0.0;

    while (Seconds(start, Clock::now()) < seconds)
    {
        CHECK(cudaEventRecord(begin));
        SleepKernel<<<blocks, 256>>>(sink, durationNs);
        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, begin, end));
        kernelS += ms * 1e-3;
        ++launches;

        if (gapUs > 0)
        {
            Wait(gapUs * 1e-6);
        }
    }

    const double wallS = Seconds(start, Clock::now());
    std::printf("result probe=gap kernel_us=%g gap_us=%g launches=%ld kernel_s=%.6f wall_s=%.6f\n", kernelUs, gapUs,
        launches, kernelS, wallS);
    CHECK(cudaFree(sink));
    return 0;
}

int Copy(const std::string& kind, const std::string& memory, const double megabytes, const double seconds)
{
    const size_t bytes = static_cast<size_t>(megabytes * 1024.0 * 1024.0);
    const bool pinned = memory == "pinned";

    if (!pinned && memory != "pageable")
    {
        std::fprintf(stderr, "MEMORY must be pinned or pageable\n");
        return 2;
    }

    void* host = nullptr;

    if (pinned)
    {
        CHECK(cudaMallocHost(&host, bytes));
    }
    else
    {
        host = std::malloc(bytes);
    }

    std::memset(host, 0x5a, bytes);
    void* device = nullptr;
    void* device2 = nullptr;
    CHECK(cudaMalloc(&device, bytes));
    CHECK(cudaMalloc(&device2, bytes));
    CHECK(cudaMemset(device, 0xa5, bytes));
    void* destination = nullptr;
    const void* source = nullptr;
    cudaMemcpyKind direction;

    if (kind == "h2d")
    {
        destination = device;
        source = host;
        direction = cudaMemcpyHostToDevice;
    }
    else if (kind == "d2h")
    {
        destination = host;
        source = device;
        direction = cudaMemcpyDeviceToHost;
    }
    else if (kind == "d2d")
    {
        destination = device2;
        source = device;
        direction = cudaMemcpyDeviceToDevice;
    }
    else
    {
        std::fprintf(stderr, "KIND must be h2d, d2h or d2d\n");
        return 2;
    }

    cudaEvent_t begin;
    cudaEvent_t end;
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaMemcpy(destination, source, bytes, direction));
    CHECK(cudaDeviceSynchronize());

    const auto start = Clock::now();
    long copies = 0;
    double copyS = 0.0;

    while (Seconds(start, Clock::now()) < seconds)
    {
        CHECK(cudaEventRecord(begin));
        CHECK(cudaMemcpy(destination, source, bytes, direction));
        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, begin, end));
        copyS += ms * 1e-3;
        ++copies;
    }

    const double wallS = Seconds(start, Clock::now());
    std::printf("result probe=copy kind=%s memory=%s bytes=%zu copies=%ld copy_s=%.6f wall_s=%.6f\n", kind.c_str(),
        memory.c_str(), bytes, copies, copyS, wallS);

    if (pinned)
    {
        CHECK(cudaFreeHost(host));
    }
    else
    {
        std::free(host);
    }

    CHECK(cudaFree(device));
    CHECK(cudaFree(device2));
    return 0;
}

int Context(const double beforeS, const double afterS, const double megabytes)
{
    const auto start = Clock::now();
    CHECK(cudaFree(nullptr));
    void* block = nullptr;

    if (megabytes > 0)
    {
        const size_t bytes = static_cast<size_t>(megabytes * 1024.0 * 1024.0);
        CHECK(cudaMalloc(&block, bytes));
        CHECK(cudaMemset(block, 0, bytes));
        CHECK(cudaDeviceSynchronize());
    }

    const auto created = Clock::now();
    Wait(beforeS);
    int device = 0;
    cudaDeviceProp properties;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaGetDeviceProperties(&properties, device));
    unsigned int* sink = nullptr;
    CHECK(cudaMalloc(&sink, 1024 * sizeof(unsigned int)));
    const auto launched = Clock::now();
    SleepKernel<<<properties.multiProcessorCount * 4, 256>>>(sink, 500000000ull);
    CHECK(cudaDeviceSynchronize());
    const auto finished = Clock::now();
    Wait(afterS);
    std::printf("result probe=context created_s=%.6f launched_s=%.6f finished_s=%.6f exit_s=%.6f\n", Seconds(start, created),
        Seconds(start, launched), Seconds(start, finished), Seconds(start, Clock::now()));
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if ((argc == 4 || argc == 5) && std::string(argv[1]) == "context")
    {
        return Context(std::atof(argv[2]), std::atof(argv[3]), argc == 5 ? std::atof(argv[4]) : 0.0);
    }

    if (argc == 5 && std::string(argv[1]) == "gap")
    {
        return Gap(std::atof(argv[2]), std::atof(argv[3]), std::atof(argv[4]));
    }

    if (argc == 6 && std::string(argv[1]) == "copy")
    {
        return Copy(argv[2], argv[3], std::atof(argv[4]), std::atof(argv[5]));
    }

    std::fprintf(stderr, "usage: %s gap KERNEL_US GAP_US SECONDS | copy h2d|d2h|d2d pinned|pageable MB SECONDS | "
        "context BEFORE_S AFTER_S [MB]\n", argv[0]);
    return 2;
}
