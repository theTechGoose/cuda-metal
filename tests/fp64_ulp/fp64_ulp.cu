// Quantifies how far CuMetal's emulated FP64 lands from true binary64.
//
// The device runs each operation under whichever --fp64 policy the binary was
// compiled with; the host runs the identical operation in hardware binary64 and
// the two are compared in ULPs. That turns "will my results drift from CUDA?"
// -- previously a guess -- into a number per operation, which is what a
// narrowing or precision-mode change has to be judged against.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define N 4096

__global__ void fp64_ops(const double* a, const double* b,
                         double* add, double* sub, double* mul,
                         double* div, double* fma_out, double* sqrt_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double x = a[i];
    const double y = b[i];
    add[i] = x + y;
    sub[i] = x - y;
    mul[i] = x * y;
    div[i] = x / y;
    fma_out[i] = fma(x, y, x);
    sqrt_out[i] = sqrt(x < 0.0 ? -x : x);
}

// Distance in representable binary64 steps. Monotone integer ordering trick:
// map the sign-magnitude bit pattern onto a two's-complement number line so
// adjacent doubles differ by exactly 1, including across zero.
static uint64_t ulp_distance(double lhs, double rhs) {
    if (std::isnan(lhs) && std::isnan(rhs)) return 0;
    if (std::isnan(lhs) || std::isnan(rhs)) return UINT64_MAX;
    if (lhs == rhs) return 0;
    int64_t li, ri;
    std::memcpy(&li, &lhs, sizeof(li));
    std::memcpy(&ri, &rhs, sizeof(ri));
    if (li < 0) li = INT64_MIN - li;
    if (ri < 0) ri = INT64_MIN - ri;
    return static_cast<uint64_t>(li > ri ? li - ri : ri - li);
}

struct Op {
    const char* name;
    double* device_result;
    double (*reference)(double, double);
};

int main(int argc, char** argv) {
    const uint64_t budget = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : UINT64_MAX;

    double *ha = (double*)malloc(N * sizeof(double));
    double *hb = (double*)malloc(N * sizeof(double));
    // Spread across exponents so the comparison exercises more than one binade.
    for (int i = 0; i < N; ++i) {
        const double t = (i + 1) * 0.7548776662;
        ha[i] = t * std::pow(2.0, (i % 61) - 30);
        hb[i] = (t + 1.3) * std::pow(2.0, (i % 37) - 18);
    }

    double *da, *db, *dr[6];
    cudaMalloc(&da, N * sizeof(double));
    cudaMalloc(&db, N * sizeof(double));
    for (int k = 0; k < 6; ++k) cudaMalloc(&dr[k], N * sizeof(double));
    cudaMemcpy(da, ha, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(double), cudaMemcpyHostToDevice);

    fp64_ops<<<(N + 255) / 256, 256>>>(da, db, dr[0], dr[1], dr[2], dr[3], dr[4], dr[5]);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL: kernel launch/sync failed\n");
        return 1;
    }

    double* got[6];
    for (int k = 0; k < 6; ++k) {
        got[k] = (double*)malloc(N * sizeof(double));
        cudaMemcpy(got[k], dr[k], N * sizeof(double), cudaMemcpyDeviceToHost);
    }

    const char* names[6] = {"add", "sub", "mul", "div", "fma", "sqrt"};
    uint64_t worst_overall = 0;
    std::printf("%-6s %14s %14s\n", "op", "max_ulp", "mean_ulp");
    for (int k = 0; k < 6; ++k) {
        uint64_t worst = 0;
        long double total = 0.0L;
        for (int i = 0; i < N; ++i) {
            double want;
            switch (k) {
                case 0: want = ha[i] + hb[i]; break;
                case 1: want = ha[i] - hb[i]; break;
                case 2: want = ha[i] * hb[i]; break;
                case 3: want = ha[i] / hb[i]; break;
                case 4: want = std::fma(ha[i], hb[i], ha[i]); break;
                default: want = std::sqrt(std::fabs(ha[i])); break;
            }
            const uint64_t d = ulp_distance(got[k][i], want);
            if (d > worst) worst = d;
            total += static_cast<long double>(d);
        }
        std::printf("%-6s %14llu %14.2Lf\n", names[k],
                    (unsigned long long)worst, total / N);
        if (worst > worst_overall) worst_overall = worst;
    }

    std::printf("max_ulp_overall=%llu budget=%llu\n",
                (unsigned long long)worst_overall, (unsigned long long)budget);
    if (worst_overall > budget) {
        std::printf("FAIL: max ULP %llu exceeds budget %llu\n",
                    (unsigned long long)worst_overall, (unsigned long long)budget);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
