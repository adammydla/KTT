// Runs an application on the CUDA device that APP_CUDA_DEVICE names, whatever device the application picks itself.
// With CUDA_DEVICE_ORDER=PCI_BUS_ID the number is the one EnergyProbes takes with --device. Linked into every application
// that tools/app_energy.py measures (apps/build_rodinia.sh), with -Xlinker --wrap=cudaSetDevice,--wrap=main: the
// application's own cudaSetDevice calls are redirected, and the device is selected before main for applications that
// never call it. Not in a constructor: CUDA calls before main hide the application from Nsight Systems' trace. Without
// APP_CUDA_DEVICE the application behaves as built.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" cudaError_t __real_cudaSetDevice(int device);
extern "C" int __real_main(int argc, char** argv, char** environment);

namespace
{

int WantedDevice()
{
    const char* text = std::getenv("APP_CUDA_DEVICE");
    return text != nullptr && *text != '\0' ? std::atoi(text) : -1;
}

void SelectDevice()
{
    const int wanted = WantedDevice();

    if (wanted < 0)
    {
        return;
    }

    const cudaError_t status = __real_cudaSetDevice(wanted);

    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "device_select: cudaSetDevice(%d) failed: %s\n", wanted, cudaGetErrorString(status));
        std::exit(1);
    }
}

} // namespace

extern "C" cudaError_t __wrap_cudaSetDevice(int device)
{
    const int wanted = WantedDevice();
    return __real_cudaSetDevice(wanted >= 0 ? wanted : device);
}

extern "C" int __wrap_main(int argc, char** argv, char** environment)
{
    SelectDevice();
    return __real_main(argc, argv, environment);
}
