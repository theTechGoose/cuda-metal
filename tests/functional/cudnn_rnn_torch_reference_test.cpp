// The cuDNN v8 LSTM checked against PyTorch's numbers, not against itself.
//
// runtime/rt/cudnn.cpp claims its weight space holds, per (layer, direction),
// W_ih then W_hh then b_ih then b_hh with gates in i,f,g,o order -- "which is
// both cuDNN's linLayerID order and PyTorch's chunk order". Every other test
// here compares CuMetal to CuMetal, and a wrong gate order is exactly the
// defect that survives that: both sides are wrong the same way and agree.
//
// So this test consumes a fixture produced by a real torch.nn.LSTM on CPU
// (tests/data/lstm_ref_small.bin, written by scripts/gen_lstm_reference.py):
// the module's own parameters in ITS layout, the input, and the output it
// produced. Those parameters are placed into the weight space THROUGH
// cudnnGetRNNWeightParams -- the same query a framework would use -- and the
// forward has to reproduce torch's y, hy and cy.
//
// CUMETAL_LSTM_REF overrides the fixture path, so the same binary checks the
// 640/640/2 Parakeet shape without committing a 26 MB file.

#include "cudnn.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kMagic = 0x4C53544D;

bool check(cudnnStatus_t s, const char* what) {
    if (s != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: %s -> %d\n", what, static_cast<int>(s));
        return false;
    }
    return true;
}

bool read_all(std::FILE* f, void* dst, std::size_t bytes) {
    return std::fread(dst, 1, bytes, f) == bytes;
}

// Largest absolute difference, and where it was, so a failure names a gate
// rather than only a magnitude.
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b,
                    std::size_t* where) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > worst) { worst = d; if (where) *where = i; }
    }
    return worst;
}

}  // namespace

int main() {
    const char* path = std::getenv("CUMETAL_LSTM_REF");
    if (!path || !*path) {
        std::fprintf(stderr, "FAIL: CUMETAL_LSTM_REF is not set\n");
        return 1;
    }
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "FAIL: cannot open fixture %s\n", path);
        return 1;
    }

    std::int32_t header[6] = {0, 0, 0, 0, 0, 0};
    if (!read_all(f, header, sizeof(header)) || header[0] != kMagic) {
        std::fprintf(stderr, "FAIL: %s is not an LSTM reference fixture\n", path);
        std::fclose(f);
        return 1;
    }
    const int kInput = header[1], kHidden = header[2], kLayers = header[3];
    const int kSeq = header[4], kBatch = header[5];
    std::printf("fixture %s: input=%d hidden=%d layers=%d seq=%d batch=%d\n",
                path, kInput, kHidden, kLayers, kSeq, kBatch);

    // torch's parameters, in torch's layout: per layer w_ih [4H, in],
    // w_hh [4H, H], b_ih [4H], b_hh [4H].
    struct LayerParams {
        std::vector<float> w_ih, w_hh, b_ih, b_hh;
    };
    std::vector<LayerParams> params(static_cast<std::size_t>(kLayers));
    for (int layer = 0; layer < kLayers; ++layer) {
        const int in = (layer == 0) ? kInput : kHidden;
        LayerParams& p = params[static_cast<std::size_t>(layer)];
        p.w_ih.resize(static_cast<std::size_t>(4) * kHidden * in);
        p.w_hh.resize(static_cast<std::size_t>(4) * kHidden * kHidden);
        p.b_ih.resize(static_cast<std::size_t>(4) * kHidden);
        p.b_hh.resize(static_cast<std::size_t>(4) * kHidden);
        if (!read_all(f, p.w_ih.data(), p.w_ih.size() * sizeof(float)) ||
            !read_all(f, p.w_hh.data(), p.w_hh.size() * sizeof(float)) ||
            !read_all(f, p.b_ih.data(), p.b_ih.size() * sizeof(float)) ||
            !read_all(f, p.b_hh.data(), p.b_hh.size() * sizeof(float))) {
            std::fprintf(stderr, "FAIL: fixture truncated in layer %d\n", layer);
            std::fclose(f);
            return 1;
        }
    }

    std::vector<float> x(static_cast<std::size_t>(kSeq) * kBatch * kInput);
    std::vector<float> y_ref(static_cast<std::size_t>(kSeq) * kBatch * kHidden);
    const std::size_t state_elems = static_cast<std::size_t>(kLayers) * kBatch * kHidden;
    std::vector<float> hy_ref(state_elems), cy_ref(state_elems);
    if (!read_all(f, x.data(), x.size() * sizeof(float)) ||
        !read_all(f, y_ref.data(), y_ref.size() * sizeof(float)) ||
        !read_all(f, hy_ref.data(), hy_ref.size() * sizeof(float)) ||
        !read_all(f, cy_ref.data(), cy_ref.size() * sizeof(float))) {
        std::fprintf(stderr, "FAIL: fixture truncated in the tensors\n");
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    cudnnHandle_t handle = nullptr;
    cudnnRNNDescriptor_t rnn = nullptr;
    if (!check(cudnnCreate(&handle), "cudnnCreate") ||
        !check(cudnnCreateRNNDescriptor(&rnn), "cudnnCreateRNNDescriptor") ||
        !check(cudnnSetRNNDescriptor_v8(rnn, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                        CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                        CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT,
                                        CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH,
                                        kInput, kHidden, /*projSize=*/kHidden, kLayers, nullptr, 0),
               "cudnnSetRNNDescriptor_v8")) {
        return 1;
    }

    size_t weight_bytes = 0;
    if (!check(cudnnGetRNNWeightSpaceSize(handle, rnn, &weight_bytes),
               "cudnnGetRNNWeightSpaceSize")) {
        return 1;
    }
    // torch holds 4 matrices and 4 biases of each kind per layer; if the space
    // is not that size the placement below would write out of bounds.
    std::size_t expect = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::size_t in = (layer == 0) ? kInput : kHidden;
        expect += (4u * kHidden * in + 4u * kHidden * kHidden + 8u * kHidden) * sizeof(float);
    }
    if (weight_bytes != expect) {
        std::fprintf(stderr,
                     "FAIL: weight space is %zu bytes, torch's parameters need %zu\n",
                     weight_bytes, expect);
        return 1;
    }
    std::vector<float> weights(weight_bytes / sizeof(float), 0.0f);

    // Place torch's tensors through the weight-params query, exactly as a
    // framework would. linLayerID 0..3 are the input-side gates in i,f,g,o
    // order, 4..7 the recurrent ones -- which is the claim being tested.
    for (int layer = 0; layer < kLayers; ++layer) {
        const LayerParams& p = params[static_cast<std::size_t>(layer)];
        const int in = (layer == 0) ? kInput : kHidden;
        for (int id = 0; id < 8; ++id) {
            cudnnTensorDescriptor_t mDesc = nullptr, bDesc = nullptr;
            void* mAddr = nullptr; void* bAddr = nullptr;
            if (!check(cudnnCreateTensorDescriptor(&mDesc), "createTensor m") ||
                !check(cudnnCreateTensorDescriptor(&bDesc), "createTensor b") ||
                !check(cudnnGetRNNWeightParams(handle, rnn, layer, weight_bytes,
                                               weights.data(), id, mDesc, &mAddr,
                                               bDesc, &bAddr),
                       "cudnnGetRNNWeightParams")) {
                return 1;
            }
            const bool recurrent = id >= 4;
            const int gate = recurrent ? id - 4 : id;
            const int cols = recurrent ? kHidden : in;
            const std::vector<float>& src_m = recurrent ? p.w_hh : p.w_ih;
            const std::vector<float>& src_b = recurrent ? p.b_hh : p.b_ih;
            std::memcpy(mAddr,
                        src_m.data() + static_cast<std::size_t>(gate) * kHidden * cols,
                        static_cast<std::size_t>(kHidden) * cols * sizeof(float));
            std::memcpy(bAddr,
                        src_b.data() + static_cast<std::size_t>(gate) * kHidden,
                        static_cast<std::size_t>(kHidden) * sizeof(float));
            cudnnDestroyTensorDescriptor(mDesc);
            cudnnDestroyTensorDescriptor(bDesc);
        }
    }

    cudnnRNNDataDescriptor_t xDesc = nullptr, yDesc = nullptr;
    std::vector<int> lengths(static_cast<std::size_t>(kBatch), kSeq);
    if (!check(cudnnCreateRNNDataDescriptor(&xDesc), "createRNNData x") ||
        !check(cudnnCreateRNNDataDescriptor(&yDesc), "createRNNData y") ||
        !check(cudnnSetRNNDataDescriptor(xDesc, CUDNN_DATA_FLOAT,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                                         kSeq, kBatch, kInput, lengths.data(), nullptr),
               "setRNNData x") ||
        !check(cudnnSetRNNDataDescriptor(yDesc, CUDNN_DATA_FLOAT,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                                         kSeq, kBatch, kHidden, lengths.data(), nullptr),
               "setRNNData y")) {
        return 1;
    }

    cudnnTensorDescriptor_t hDesc = nullptr, cDesc = nullptr;
    const int state_dims[3] = {kLayers, kBatch, kHidden};
    const int state_strides[3] = {kBatch * kHidden, kHidden, 1};
    if (!check(cudnnCreateTensorDescriptor(&hDesc), "createTensor h") ||
        !check(cudnnCreateTensorDescriptor(&cDesc), "createTensor c") ||
        !check(cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, state_dims,
                                          state_strides), "setTensorNd h") ||
        !check(cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, state_dims,
                                          state_strides), "setTensorNd c")) {
        return 1;
    }

    size_t work_bytes = 0, reserve_bytes = 0;
    if (!check(cudnnGetRNNTempSpaceSizes(handle, rnn, CUDNN_FWD_MODE_INFERENCE, xDesc,
                                         &work_bytes, &reserve_bytes),
               "cudnnGetRNNTempSpaceSizes")) {
        return 1;
    }
    std::vector<char> work(work_bytes ? work_bytes : 1);
    std::vector<char> reserve(reserve_bytes ? reserve_bytes : 1);

    // torch was called without an initial state, which is zeros.
    std::vector<float> h0(state_elems, 0.0f), c0(state_elems, 0.0f);
    std::vector<float> y(y_ref.size(), 0.0f);
    std::vector<float> hy(state_elems, 0.0f), cy(state_elems, 0.0f);

    if (!check(cudnnRNNForward(handle, rnn, CUDNN_FWD_MODE_INFERENCE, lengths.data(),
                               xDesc, x.data(), yDesc, y.data(),
                               hDesc, h0.data(), hy.data(),
                               cDesc, c0.data(), cy.data(),
                               weight_bytes, weights.data(),
                               work_bytes, work.data(),
                               reserve_bytes, reserve.data()),
               "cudnnRNNForward")) {
        return 1;
    }

    // FP32 across a 640-wide matmul and 5 timesteps: the two implementations
    // accumulate in different orders, so this is an agreement threshold, not
    // an equality one. A gate-order error is not a small number -- it is O(1).
    const double tol = 2e-4;
    struct { const char* what; const std::vector<float>& got; const std::vector<float>& want; }
        checks[3] = {{"y", y, y_ref}, {"hy", hy, hy_ref}, {"cy", cy, cy_ref}};
    int failures = 0;
    for (const auto& c : checks) {
        std::size_t where = 0;
        const double d = max_abs_diff(c.got, c.want, &where);
        std::printf("  max |cumetal - torch| over %-2s = %.3e (at %zu: %.6f vs %.6f)\n",
                    c.what, d, where, c.got[where], c.want[where]);
        if (!(d <= tol)) {
            std::fprintf(stderr, "FAIL: %s disagrees with torch by %.3e (tol %.1e)\n",
                         c.what, d, tol);
            ++failures;
        }
    }

    cudnnDestroyRNNDataDescriptor(xDesc);
    cudnnDestroyRNNDataDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(hDesc);
    cudnnDestroyTensorDescriptor(cDesc);
    cudnnDestroyRNNDescriptor(rnn);
    cudnnDestroy(handle);

    if (failures) return 1;
    std::printf("PASS: the v8 LSTM reproduces torch.nn.LSTM through its own "
                "weight-params query\n");
    return 0;
}
