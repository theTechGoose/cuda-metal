#include "cuda_runtime.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

// PhysX drives its GPU pipeline from several worker threads at once, each with
// its own stream, and the 2 800-body pile reported against this runtime explodes
// in most runs with 4 workers while being stable in every run with 1
// (scripts/physx-patches/README.md). "Explodes" is wrong numbers, not a crash,
// which points at the runtime's stream-ordered data paths rather than a missing
// lock somewhere obvious.
//
// This is that shape without PhysX: N threads, each owning a stream and its own
// buffers, each looping launch -> async D2H -> synchronize and checking its own
// results byte for byte. Every thread's data is private, so ANY mismatch is the
// runtime letting one worker's work land in or disturb another's.
//
// It is a race detector, so it is a repetition test: a single clean pass proves
// nothing, which is why it runs many rounds and reports the first failure with
// the round and thread that saw it.

namespace {

constexpr std::size_t kElements = 1u << 12;
constexpr std::size_t kThreadsPerBlock = 256;
constexpr int kWorkers = 4;      // the count that made the pile explode
constexpr int kRounds = 64;

float expected_c(std::size_t i, int worker) {
    return static_cast<float>(i % 97) + static_cast<float>(worker) * 1000.0f;
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

    std::atomic<bool> start{false};
    std::atomic<int> ready{0};
    std::atomic<int> failures{0};
    std::vector<std::string> reports(kWorkers);

    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    for (int w = 0; w < kWorkers; ++w) {
        workers.emplace_back([&, w]() {
            auto note = [&](const std::string& text) {
                if (reports[w].empty()) reports[w] = text;
                failures.fetch_add(1, std::memory_order_relaxed);
            };

            cudaStream_t stream = nullptr;
            if (cudaStreamCreate(&stream) != cudaSuccess) {
                note("cudaStreamCreate failed");
                ready.fetch_add(1, std::memory_order_release);
                return;
            }

            // a[i] = i % 97, b[i] = worker * 1000, so c[i] is unique per worker
            std::vector<float> host_a(kElements), host_b(kElements), host_c(kElements);
            for (std::size_t i = 0; i < kElements; ++i) {
                host_a[i] = static_cast<float>(i % 97);
                host_b[i] = static_cast<float>(w) * 1000.0f;
            }

            void *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr, *pinned = nullptr;
            const std::size_t bytes = kElements * sizeof(float);
            if (cudaMalloc(&dev_a, bytes) != cudaSuccess ||
                cudaMalloc(&dev_b, bytes) != cudaSuccess ||
                cudaMalloc(&dev_c, bytes) != cudaSuccess ||
                cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault) != cudaSuccess) {
                note("allocation failed");
                ready.fetch_add(1, std::memory_order_release);
                return;
            }
            if (cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                note("seed H2D failed");
                ready.fetch_add(1, std::memory_order_release);
                return;
            }

            void* arg_a = dev_a; void* arg_b = dev_b; void* arg_c = dev_c;
            void* launch_args[] = {&arg_a, &arg_b, &arg_c};
            const dim3 block(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
            const dim3 grid(static_cast<unsigned int>((kElements + kThreadsPerBlock - 1) /
                                                     kThreadsPerBlock), 1, 1);

            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();

            for (int round = 0; round < kRounds && reports[w].empty(); ++round) {
                // PhysX's per-step shape: clear the output, launch, blit it back,
                // wait, read. The clear is stream-ordered and must not land early.
                std::memset(pinned, 0xEE, bytes);
                if (cudaMemsetAsync(dev_c, 0, bytes, stream) != cudaSuccess) {
                    note("cudaMemsetAsync failed at round " + std::to_string(round));
                    break;
                }
                if (cudaLaunchKernel(&kernel, grid, block, launch_args, 0, stream) != cudaSuccess) {
                    note("cudaLaunchKernel failed at round " + std::to_string(round));
                    break;
                }
                if (cudaMemcpyAsync(pinned, dev_c, bytes, cudaMemcpyDeviceToHost, stream)
                        != cudaSuccess) {
                    note("cudaMemcpyAsync D2H failed at round " + std::to_string(round));
                    break;
                }
                if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    note("cudaStreamSynchronize failed at round " + std::to_string(round));
                    break;
                }
                const float* got = static_cast<const float*>(pinned);
                for (std::size_t i = 0; i < kElements; ++i) {
                    const float want = expected_c(i, w);
                    if (got[i] != want) {
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                                      "round %d element %zu: got %.3f want %.3f "
                                      "(worker %d's own stream and buffers)",
                                      round, i, got[i], want, w);
                        note(buf);
                        break;
                    }
                }
            }

            // Phase 2: a cross-stream dependency, which is how PhysX orders its
            // pipeline -- the launch is recorded on one stream and a second
            // stream waits the event before reading the result. The event
            // machinery is also what carries the deferred pointer restore
            // (an event remembers its stream's restore sequence at record
            // time), so it is the path most likely to misbehave under threads.
            cudaStream_t consumer = nullptr;
            cudaEvent_t produced = nullptr;
            if (reports[w].empty()) {
                if (cudaStreamCreate(&consumer) != cudaSuccess ||
                    cudaEventCreate(&produced) != cudaSuccess) {
                    note("cudaStreamCreate/cudaEventCreate for the event phase failed");
                }
            }
            for (int round = 0; round < kRounds && reports[w].empty(); ++round) {
                std::memset(pinned, 0xEE, bytes);
                if (cudaMemsetAsync(dev_c, 0, bytes, stream) != cudaSuccess ||
                    cudaLaunchKernel(&kernel, grid, block, launch_args, 0, stream)
                        != cudaSuccess ||
                    cudaEventRecord(produced, stream) != cudaSuccess ||
                    cudaStreamWaitEvent(consumer, produced, 0) != cudaSuccess ||
                    cudaMemcpyAsync(pinned, dev_c, bytes, cudaMemcpyDeviceToHost, consumer)
                        != cudaSuccess ||
                    cudaStreamSynchronize(consumer) != cudaSuccess) {
                    note("event-ordered round " + std::to_string(round) + " failed");
                    break;
                }
                const float* got = static_cast<const float*>(pinned);
                for (std::size_t i = 0; i < kElements; ++i) {
                    const float want = expected_c(i, w);
                    if (got[i] != want) {
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                                      "event-ordered round %d element %zu: got %.3f want %.3f "
                                      "(recorded on one stream, read after waiting it on another)",
                                      round, i, got[i], want);
                        note(buf);
                        break;
                    }
                }
            }

            // Phase 3: the same work with allocation churn alongside it. PhysX
            // allocates and frees every step, so the allocation table and Metal
            // buffer creation are contended by every worker while their streams
            // are in flight -- a resolve that races an insert or erase would
            // hand a launch the wrong buffer.
            for (int round = 0; round < kRounds && reports[w].empty(); ++round) {
                void* churn = nullptr;
                if (cudaMalloc(&churn, bytes) != cudaSuccess) {
                    note("churn cudaMalloc failed at round " + std::to_string(round));
                    break;
                }
                std::memset(pinned, 0xEE, bytes);
                if (cudaMemsetAsync(dev_c, 0, bytes, stream) != cudaSuccess ||
                    cudaLaunchKernel(&kernel, grid, block, launch_args, 0, stream)
                        != cudaSuccess ||
                    cudaMemcpyAsync(pinned, dev_c, bytes, cudaMemcpyDeviceToHost, stream)
                        != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess) {
                    note("churn round " + std::to_string(round) + " failed");
                    cudaFree(churn);
                    break;
                }
                const float* got = static_cast<const float*>(pinned);
                for (std::size_t i = 0; i < kElements; ++i) {
                    const float want = expected_c(i, w);
                    if (got[i] != want) {
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                                      "churn round %d element %zu: got %.3f want %.3f "
                                      "(concurrent cudaMalloc/cudaFree on every worker)",
                                      round, i, got[i], want);
                        note(buf);
                        break;
                    }
                }
                cudaFree(churn);
            }

            // Phase 4: every worker on the SAME legacy (null) stream at once,
            // still with its own buffers. A context's null stream is shared by
            // whatever threads touch it, and "stable with one worker, explodes
            // with four" is the signature of shared-stream state that is not
            // safe under threads. Host functions ride here too: the async
            // memsets are enqueued as host functions now, so this contends
            // that machinery as well.
            for (int round = 0; round < kRounds && reports[w].empty(); ++round) {
                std::memset(pinned, 0xEE, bytes);
                if (cudaMemsetAsync(dev_c, 0, bytes, nullptr) != cudaSuccess ||
                    cudaLaunchKernel(&kernel, grid, block, launch_args, 0, nullptr)
                        != cudaSuccess ||
                    cudaMemcpyAsync(pinned, dev_c, bytes, cudaMemcpyDeviceToHost, nullptr)
                        != cudaSuccess ||
                    cudaStreamSynchronize(nullptr) != cudaSuccess) {
                    note("legacy-stream round " + std::to_string(round) + " failed");
                    break;
                }
                const float* got = static_cast<const float*>(pinned);
                for (std::size_t i = 0; i < kElements; ++i) {
                    const float want = expected_c(i, w);
                    if (got[i] != want) {
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                                      "legacy-stream round %d element %zu: got %.3f want %.3f "
                                      "(all %d workers sharing the null stream)",
                                      round, i, got[i], want, kWorkers);
                        note(buf);
                        break;
                    }
                }
            }

            if (produced != nullptr) cudaEventDestroy(produced);
            if (consumer != nullptr) cudaStreamDestroy(consumer);
            cudaFreeHost(pinned);
            cudaFree(dev_c); cudaFree(dev_b); cudaFree(dev_a);
            cudaStreamDestroy(stream);
        });
    }

    while (ready.load(std::memory_order_acquire) < kWorkers) std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (std::thread& t : workers) t.join();

    if (failures.load() != 0) {
        std::fprintf(stderr, "FAIL: %d worker(s) diverged with %d threads over %d rounds\n",
                     failures.load(), kWorkers, kRounds);
        for (int w = 0; w < kWorkers; ++w) {
            if (!reports[w].empty()) {
                std::fprintf(stderr, "  worker %d: %s\n", w, reports[w].c_str());
            }
        }
        return 1;
    }

    std::printf("PASS: %d workers x %d rounds x 4 phases (own stream, "
                "cross-stream event, allocation churn, shared legacy stream) agree exactly\n",
                kWorkers, kRounds);
    return 0;
}
