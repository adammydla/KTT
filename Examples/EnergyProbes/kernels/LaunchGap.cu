// An empty kernel (--validate, 2026-11-17). Launched back to back through KTT, the GPU spends almost all of the wall time
// between launches, in KTT's NVML hook, so its power is the power between the launches of every validation kernel.
extern "C" __global__ void launchGap(const int unused)
{
}
