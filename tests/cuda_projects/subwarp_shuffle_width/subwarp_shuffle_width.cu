#include <cuda_runtime.h>

#include <cstdio>

// CUDA's shuffles take a WIDTH, and a shuffle narrower than the warp confines
// each lane to its own segment of that width: a lane may never read outside it.
// PTX encodes this in the shuffle's control operand as
//
//   segmask = c[12:8]                     (32 - width)
//   cval    = c[4:0]                      (the clamp)
//   maxLane = (lane & segmask) | (cval & ~segmask)
//   pred    = requested <= maxLane        (false -> the lane reads itself)
//
// Dropping the `& ~segmask` from maxLane makes the bound too large, the
// predicate passes when it should not, and a lane reads a neighbouring
// segment's data. It is invisible at width 32, where segmask is 0 and the two
// forms agree -- which is every shuffle the rest of this suite performs. PhysX's
// GJK/EPA narrowphase shuffles within sub-warp groups, and a lane reading the
// next group's vertex is how a pair of hulls metres apart came back reported as
// metres deep.
//
// Each lane writes what it read for widths 2/4/8/16/32 in all four modes. The
// expected values are computed on the host from the CUDA definition, so this
// test states the semantics rather than blessing whatever the backend does.

constexpr unsigned kWarp = 32u;
constexpr unsigned kWidths = 5u;   // 2, 4, 8, 16, 32
constexpr unsigned kModes = 4u;    // idx, up, down, xor
constexpr unsigned kSlots = kWidths * kModes;

__global__ void subwarp_shuffle_width(unsigned* out) {
    const unsigned lane = threadIdx.x;
    // Each lane carries a value that identifies it unambiguously.
    const unsigned value = 1000u + lane;
    unsigned slot = 0u;
    for (unsigned width = 2u; width <= 32u; width *= 2u) {
        out[slot++ * kWarp + lane] = __shfl_sync(0xffffffffu, value, 1u, width);
        out[slot++ * kWarp + lane] = __shfl_up_sync(0xffffffffu, value, 1u, width);
        out[slot++ * kWarp + lane] = __shfl_down_sync(0xffffffffu, value, 1u, width);
        out[slot++ * kWarp + lane] = __shfl_xor_sync(0xffffffffu, value, 1u, width);
    }
}

int main() {
    unsigned expected[kSlots * kWarp];
    unsigned slot = 0u;
    for (unsigned width = 2u; width <= 32u; width *= 2u) {
        for (unsigned mode = 0u; mode < kModes; ++mode) {
            for (unsigned lane = 0u; lane < kWarp; ++lane) {
                const unsigned base = (lane / width) * width;   // the lane's segment
                const unsigned local = lane - base;
                unsigned source = lane;                         // default: read self
                switch (mode) {
                    case 0:  // __shfl_sync(v, 1, width): every lane reads local 1
                        source = base + (1u % width);
                        break;
                    case 1:  // __shfl_up_sync(v, 1, width): read local-1, clamped at 0
                        if (local >= 1u) source = lane - 1u;
                        break;
                    case 2:  // __shfl_down_sync(v, 1, width): read local+1, clamped at width-1
                        if (local + 1u < width) source = lane + 1u;
                        break;
                    case 3:  // __shfl_xor_sync(v, 1, width): read local^1 inside the segment
                        if ((local ^ 1u) < width) source = base + (local ^ 1u);
                        break;
                    default:
                        break;
                }
                expected[(slot + mode) * kWarp + lane] = 1000u + source;
            }
        }
        slot += kModes;
    }

    unsigned* device_out = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_out), sizeof(expected)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        return 1;
    }
    if (cudaMemset(device_out, 0, sizeof(expected)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset\n");
        return 1;
    }

    subwarp_shuffle_width<<<1, kWarp>>>(device_out);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    unsigned host_out[kSlots * kWarp] = {};
    const cudaError_t copy_status =
        cudaMemcpy(host_out, device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
    cudaFree(device_out);
    if (sync_status != cudaSuccess || copy_status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(sync_status), cudaGetErrorString(copy_status));
        return 1;
    }

    static const char* kModeName[kModes] = {"idx", "up", "down", "xor"};
    unsigned failures = 0u;
    for (unsigned s = 0u; s < kSlots; ++s) {
        const unsigned width = 2u << (s / kModes);
        const char* mode = kModeName[s % kModes];
        for (unsigned lane = 0u; lane < kWarp; ++lane) {
            const unsigned got = host_out[s * kWarp + lane];
            const unsigned want = expected[s * kWarp + lane];
            if (got != want) {
                if (failures < 40u) {
                    std::fprintf(stderr,
                                 "FAIL: width %2u %-4s lane %2u: read lane %d, expected lane %u"
                                 "%s\n",
                                 width, mode, lane,
                                 got >= 1000u ? static_cast<int>(got - 1000u) : -1,
                                 want - 1000u,
                                 (got >= 1000u && (got - 1000u) / width != lane / width)
                                     ? "  (outside its segment)"
                                     : "");
                }
                ++failures;
            }
        }
    }
    if (failures != 0u) {
        std::fprintf(stderr, "FAIL: %u of %u sub-warp shuffle reads were wrong\n",
                     failures, kSlots * kWarp);
        return 1;
    }

    std::printf("PASS: sub-warp shuffles stay inside their segment at widths 2/4/8/16/32\n");
    return 0;
}
