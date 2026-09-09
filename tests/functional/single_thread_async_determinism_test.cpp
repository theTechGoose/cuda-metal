#include "cuda_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// One thread, several streams, dependent work, read back and fed forward --
// PhysX's GRB pipeline shape, which drifts run to run on this runtime and stops
// drifting entirely under CUMETAL_SYNC_EACH_LAUNCH=1 (a 2 800-body pile: six
// runs, six end states with PhysX's dispatcher running every task on the calling
// thread, so exactly ONE thread ever issues; six bit-identical runs when every
// launch synchronises). Serialising the async path removing the difference makes
// this an ordering bug in that path, not arithmetic: the same kernels are
// bit-identical when serialised.
//
// This is that shape without PhysX. Each iteration launches across several
// streams whose results depend on the previous iteration's, joins them with
// events, blits the result back and uses it to seed the next iteration -- so any
// launch that observes a predecessor's buffer early changes every value after
// it. The whole run is reduced to one checksum, and the test's claim is that
// the checksum does not depend on how the runtime scheduled the work: the same
// program, run twice in one process, must produce it twice.
//
// It is a race detector, so a single pass is weak: it runs many trials, and
// reports the first trial whose checksum leaves the others.
//
// HONEST LIMIT: this does NOT currently reproduce the reported drift -- it
// passes. It is kept as a regression guard for the shape and as a record of
// what has been ruled out: four streams, event joins, dependent readback and
// feed-forward from one thread are not on their own enough. The reported scene
// differs in ways this does not yet model, and the reporter's own data says
// which are likely to matter: scale (564 bodies is deterministic, 2 800 is
// not), enough launches to cross max_batch_dispatches() and force a mid-run
// flush, and above all buffers SUBALLOCATED from a shared pool -- PhysX carves
// its arrays out of a few large allocations, and encode_submission_waits calls
// out aliasing suballocations as its own case, which separate cudaMalloc'd
// buffers never exercise. Widen it there before trusting a pass.

namespace {

constexpr std::size_t kElements = 1u << 12;
constexpr std::size_t kThreadsPerBlock = 256;
constexpr int kStreams = 4;
constexpr int kIterations = 24;
constexpr int kTrials = 12;

bool cuda_ok(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }
    const std::string metallib_path = argv[1];
    if (!std::filesystem::exists(metallib_path)) {
        std::fprintf(stderr, "SKIP: metallib not found at %s\n", metallib_path.c_str());
        return 77;
    }
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "vector_add",
        .arg_count = 3,
        .arg_info = kArgInfo,
    };

    const std::size_t bytes = kElements * sizeof(float);
    const dim3 block(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid(static_cast<unsigned int>((kElements + kThreadsPerBlock - 1) /
                                              kThreadsPerBlock), 1, 1);

    // One trial: the whole dependent pipeline, reduced to a checksum.
    auto run_trial = [&](double* out_checksum) -> bool {
        cudaStream_t streams[kStreams] = {};
        cudaEvent_t events[kStreams] = {};
        void* dev_a[kStreams] = {};
        void* dev_b[kStreams] = {};
        void* dev_c[kStreams] = {};
        void* pinned = nullptr;

        for (int s = 0; s < kStreams; ++s) {
            if (!cuda_ok(cudaStreamCreate(&streams[s]), "cudaStreamCreate") ||
                !cuda_ok(cudaEventCreate(&events[s]), "cudaEventCreate") ||
                !cuda_ok(cudaMalloc(&dev_a[s], bytes), "cudaMalloc a") ||
                !cuda_ok(cudaMalloc(&dev_b[s], bytes), "cudaMalloc b") ||
                !cuda_ok(cudaMalloc(&dev_c[s], bytes), "cudaMalloc c")) {
                return false;
            }
        }
        if (!cuda_ok(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault), "cudaHostAlloc")) {
            return false;
        }

        std::vector<float> seed(kElements);
        for (std::size_t i = 0; i < kElements; ++i) seed[i] = static_cast<float>(i % 31);

        double checksum = 0.0;
        for (int iter = 0; iter < kIterations; ++iter) {
            for (int s = 0; s < kStreams; ++s) {
                // b carries the previous iteration's readback: every launch
                // depends on what came before it.
                std::vector<float> b(kElements);
                for (std::size_t i = 0; i < kElements; ++i) {
                    b[i] = seed[i] + static_cast<float>(s);
                }
                if (!cuda_ok(cudaMemcpyAsync(dev_a[s], seed.data(), bytes,
                                             cudaMemcpyHostToDevice, streams[s]), "H2D a") ||
                    !cuda_ok(cudaMemcpyAsync(dev_b[s], b.data(), bytes,
                                             cudaMemcpyHostToDevice, streams[s]), "H2D b")) {
                    return false;
                }
                void* arg_a = dev_a[s]; void* arg_b = dev_b[s]; void* arg_c = dev_c[s];
                void* args[] = {&arg_a, &arg_b, &arg_c};
                if (!cuda_ok(cudaLaunchKernel(&kernel, grid, block, args, 0, streams[s]),
                             "launch")) {
                    return false;
                }
                if (!cuda_ok(cudaEventRecord(events[s], streams[s]), "eventRecord")) {
                    return false;
                }
            }
            // Join every stream onto stream 0, then read stream 0's result and
            // feed it forward -- so a launch that ran early is visible here.
            for (int s = 1; s < kStreams; ++s) {
                if (!cuda_ok(cudaStreamWaitEvent(streams[0], events[s], 0), "streamWaitEvent")) {
                    return false;
                }
            }
            if (!cuda_ok(cudaMemcpyAsync(pinned, dev_c[0], bytes,
                                         cudaMemcpyDeviceToHost, streams[0]), "D2H") ||
                !cuda_ok(cudaStreamSynchronize(streams[0]), "streamSynchronize")) {
                return false;
            }
            const float* got = static_cast<const float*>(pinned);
            for (std::size_t i = 0; i < kElements; ++i) {
                checksum += static_cast<double>(got[i]);
                seed[i] = got[i] * 0.5f;   // next iteration depends on this one
            }
        }

        cudaFreeHost(pinned);
        for (int s = 0; s < kStreams; ++s) {
            cudaFree(dev_c[s]); cudaFree(dev_b[s]); cudaFree(dev_a[s]);
            cudaEventDestroy(events[s]);
            cudaStreamDestroy(streams[s]);
        }
        *out_checksum = checksum;
        return true;
    };

    double reference = 0.0;
    if (!run_trial(&reference)) return 1;

    int diverged = 0;
    for (int trial = 1; trial < kTrials; ++trial) {
        double checksum = 0.0;
        if (!run_trial(&checksum)) return 1;
        if (checksum != reference) {
            if (diverged < 5) {
                std::fprintf(stderr,
                             "FAIL: trial %d checksum %.17g, trial 0 gave %.17g "
                             "(same program, one thread, no input changed)\n",
                             trial, checksum, reference);
            }
            ++diverged;
        }
    }

    if (diverged != 0) {
        std::fprintf(stderr,
                     "FAIL: %d of %d trials disagreed -- the asynchronous path is "
                     "ordering-dependent\n", diverged, kTrials - 1);
        return 1;
    }

    std::printf("PASS: %d trials of %d iterations over %d streams agree exactly\n",
                kTrials, kIterations, kStreams);
    return 0;
}
