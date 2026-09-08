#include "cuda_runtime.h"

#include <cstdio>
#include <cstdlib>

// A device-to-host copy into PINNED memory is a Metal blit -- a raw byte copy in
// the stream's command buffer -- so the embedded-pointer restore the staged
// host-function path performs in line did not run on it, and the device
// addresses in the copied bytes reached the CPU intact. PhysX's GPU narrowphase
// blits its contact-manager output back to pinned host memory exactly this way;
// host code reading contact points (PxContactPair::extractContacts) then
// dereferenced a GPU virtual address and died with EXC_BAD_ACCESS on the first
// step with a touch. This test is that crash without PhysX: a device buffer
// holding a device pointer, blitted back, then read.
//
// It only means anything with CUMETAL_USE_METAL_DEVICE_ADDRESSES=1, where the
// two address spaces actually differ; main() sets it before the runtime starts.

// The runtime's own view of an allocation's CPU mapping. With
// CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 a public CuMetal pointer IS the GPU
// virtual address, so this is what a restored pointer has to become.
extern "C" void* cumetalRuntimeGetHostPointer(const void* ptr, size_t count);

namespace {

bool check(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", what);
    return ok;
}

bool cuda_ok(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
        return false;
    }
    return true;
}

}  // namespace

int main() {
    // before any runtime call: the flag is read once, at initialization
    setenv("CUMETAL_USE_METAL_DEVICE_ADDRESSES", "1", 1);

    void* payload = nullptr;
    void* record = nullptr;
    void* pinned = nullptr;
    if (!cuda_ok(cudaMalloc(&payload, 256), "cudaMalloc(payload)") ||
        !cuda_ok(cudaMalloc(&record, sizeof(void*)), "cudaMalloc(record)") ||
        !cuda_ok(cudaHostAlloc(&pinned, sizeof(void*), cudaHostAllocDefault),
                 "cudaHostAlloc(pinned)")) {
        return 1;
    }

    // Under the flag, cudaMalloc hands back the GPU virtual address, so seeding
    // the device buffer with `payload` puts there exactly what a kernel writing
    // a device pointer would: PhysX's narrowphase filling in
    // PxContactPair::contactPatches / contactPoints.
    if (!cuda_ok(cudaMemcpy(record, &payload, sizeof(void*), cudaMemcpyHostToDevice),
                 "seed record with a device pointer")) {
        return 1;
    }

    void* const host_view = cumetalRuntimeGetHostPointer(payload, 256);
    if (!check(host_view != nullptr, "the runtime has no CPU mapping for payload")) return 1;
    if (host_view == payload) {
        std::printf("SKIP: device and host addresses coincide here; "
                    "the restore is unobservable\n");
        return 0;
    }

    // A raw copy (host-to-host: no relocation either way) confirms the device
    // buffer really holds the device address, so the checks below are about the
    // restore and nothing else.
    if (!cuda_ok(cudaMemcpy(pinned, record, sizeof(void*), cudaMemcpyHostToHost),
                 "read record raw")) {
        return 1;
    }
    if (!check(*static_cast<void* const*>(pinned) == payload,
               "the seeded device buffer does not hold the device address")) {
        return 1;
    }

    // (1) the blit, drained by a stream synchronization
    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) return 1;
    *static_cast<void**>(pinned) = nullptr;
    if (!cuda_ok(cudaMemcpyAsync(pinned, record, sizeof(void*), cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync D2H into pinned memory") ||
        !cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
        return 1;
    }
    void* seen = *static_cast<void**>(pinned);
    if (!check(seen == host_view,
               "the D2H blit left a device address in pinned memory "
               "(the CPU would dereference a GPU virtual address, as PhysX did)")) {
        std::fprintf(stderr, "  after stream sync: got %p, want the host mapping %p "
                             "(the device address is %p)\n", seen, host_view, payload);
        return 1;
    }

    // (2) destroying the stream synchronizes, so it must drain the restore too
    *static_cast<void**>(pinned) = nullptr;
    if (!cuda_ok(cudaMemcpyAsync(pinned, record, sizeof(void*), cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync D2H before destroy") ||
        !cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy")) {
        return 1;
    }
    seen = *static_cast<void**>(pinned);
    if (!check(seen == host_view, "cudaStreamDestroy did not drain the pending restore")) {
        std::fprintf(stderr, "  after stream destroy: got %p, want the host mapping %p\n",
                     seen, host_view);
        return 1;
    }

    // (3) the legacy stream, drained by a device synchronization
    *static_cast<void**>(pinned) = nullptr;
    if (!cuda_ok(cudaMemcpyAsync(pinned, record, sizeof(void*), cudaMemcpyDeviceToHost, nullptr),
                 "cudaMemcpyAsync D2H on the legacy stream") ||
        !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        return 1;
    }
    seen = *static_cast<void**>(pinned);
    if (!check(seen == host_view, "cudaDeviceSynchronize did not drain the pending restore")) {
        std::fprintf(stderr, "  after device sync: got %p, want the host mapping %p\n",
                     seen, host_view);
        return 1;
    }

    cudaFreeHost(pinned);
    cudaFree(record);
    cudaFree(payload);

    std::printf("PASS: a device-to-host blit into pinned memory restores its "
                "embedded pointers (stream sync, stream destroy, device sync)\n");
    return 0;
}
