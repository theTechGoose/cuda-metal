#include "cudnn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The cuDNN v8 RNN path, exercised in the shape NVIDIA Parakeet's RNNT/TDT
// prediction network actually uses. From the shipped parakeet-tdt-0.6b config:
// pred_hidden 640, pred_rnn_layers 2, no rnn_hidden_size, so NeMo's
//   proj_size = pred_n_hidden if pred_n_hidden < rnn_hidden_size else 0
// leaves proj_size 0 -- an ordinary 2-layer unidirectional LSTM. Greedy decode
// then calls it ONE TIMESTEP AT A TIME carrying (h, c) forward, which is the
// part the v7 full-sequence API never had to serve.
//
// The test's claim is that stepping the sequence one timestep at a time with
// externally carried state produces exactly what one full-sequence call
// produces. That is the property a decoder depends on, and it is checkable
// without any model in the loop.

namespace {

// Small by default so the equivalence check is quick, but the shape that
// matters is Parakeet's prediction network: input 640, hidden 640, 2 layers,
// which is a 26 MB weight space. CUMETAL_RNN_V8_REAL_SHAPE=1 runs that instead,
// so sizing and indexing are exercised at the dimensions a consumer actually
// uses rather than only at dimensions that fit in a cache line.
const int kInput = std::getenv("CUMETAL_RNN_V8_REAL_SHAPE") ? 640 : 8;
const int kHidden = std::getenv("CUMETAL_RNN_V8_REAL_SHAPE") ? 640 : 8;
const int kLayers = 2;
const int kBatch = 3;
const int kSeq = 5;

bool check(cudnnStatus_t s, const char* what) {
    if (s != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: %s -> %d\n", what, static_cast<int>(s));
        return false;
    }
    return true;
}

float deterministic(int i) {
    return 0.05f * static_cast<float>((i * 37) % 21 - 10);
}

}  // namespace

int main() {
    cudnnHandle_t handle = nullptr;
    if (!check(cudnnCreate(&handle), "cudnnCreate")) return 1;

    cudnnRNNDescriptor_t rnn = nullptr;
    if (!check(cudnnCreateRNNDescriptor(&rnn), "cudnnCreateRNNDescriptor")) return 1;
    if (!check(cudnnSetRNNDescriptor_v8(rnn, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                        CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                        CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT,
                                        CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH,
                                        kInput, kHidden, /*projSize=*/kHidden, kLayers, nullptr, 0),
               "cudnnSetRNNDescriptor_v8")) {
        return 1;
    }

    // cuDNN 8.9.7 documents that cudnnGetRNNWeightParams reports nbDims = 0 when
    // a weight matrix or bias does not exist -- the first layer's input GEMMs
    // under CUDNN_SKIP_INPUT, and the absent biases under the non-double bias
    // modes. CuMetal does not implement that reporting, so the only thing
    // keeping it honest is that those configurations are refused outright. If
    // one is ever accepted without the nbDims = 0 path, a caller sizing from
    // this query reads a matrix that is not there.
    {
        const struct { const char* what; cudnnRNNInputMode_t im; cudnnRNNBiasMode_t bm; }
            unsupported[4] = {
                {"CUDNN_SKIP_INPUT", CUDNN_SKIP_INPUT, CUDNN_RNN_DOUBLE_BIAS},
                {"CUDNN_RNN_NO_BIAS", CUDNN_LINEAR_INPUT, CUDNN_RNN_NO_BIAS},
                {"CUDNN_RNN_SINGLE_INP_BIAS", CUDNN_LINEAR_INPUT, CUDNN_RNN_SINGLE_INP_BIAS},
                {"CUDNN_RNN_SINGLE_REC_BIAS", CUDNN_LINEAR_INPUT, CUDNN_RNN_SINGLE_REC_BIAS},
            };
        for (const auto& u : unsupported) {
            cudnnRNNDescriptor_t probe = nullptr;
            if (!check(cudnnCreateRNNDescriptor(&probe), "createRNNDescriptor probe")) {
                return 1;
            }
            const cudnnStatus_t st = cudnnSetRNNDescriptor_v8(
                probe, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, u.bm, CUDNN_UNIDIRECTIONAL,
                u.im, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH,
                kInput, kHidden, /*projSize=*/kHidden, kLayers, nullptr, 0);
            cudnnDestroyRNNDescriptor(probe);
            if (st == CUDNN_STATUS_SUCCESS) {
                std::fprintf(stderr,
                             "FAIL: %s was accepted, but the weight-params query has no "
                             "nbDims = 0 path for the tensors it makes absent\n", u.what);
                return 1;
            }
        }
    }

    size_t weight_bytes = 0;
    if (!check(cudnnGetRNNWeightSpaceSize(handle, rnn, &weight_bytes),
               "cudnnGetRNNWeightSpaceSize")) {
        return 1;
    }
    std::vector<float> weights(weight_bytes / sizeof(float));
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = deterministic(static_cast<int>(i));
    }

    // The weight-params query has to describe the same layout the forward reads,
    // because a framework fills the space through it. Walk every pseudo-layer
    // and gate and confirm each reported slice lands inside the space.
    for (int layer = 0; layer < kLayers; ++layer) {
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
            // The rank a framework reads back has to be the rank cuDNN reports:
            // these are 3-D descriptors, and a getter that answers 4 regardless
            // of what was set hands out a shape the caller's parameters do not
            // have. Assert the rank and the dims, not just that the slice fits.
            // Only layer 0 consumes the network input; a stacked unidirectional
            // layer takes the layer below's hidden state.
            const int layer_in = (layer == 0) ? kInput : kHidden;
            const int expect_cols = (id < 4) ? layer_in : kHidden;
            const int expect_m[3] = {1, kHidden, expect_cols};
            const int expect_b[3] = {1, kHidden, 1};
            const struct { cudnnTensorDescriptor_t d; const int* want; const char* what; }
                probes[2] = {{mDesc, expect_m, "matrix"}, {bDesc, expect_b, "bias"}};
            for (const auto& probe : probes) {
                cudnnDataType_t dt = CUDNN_DATA_FLOAT;
                int rank = 0;
                int dims[4] = {0, 0, 0, 0};
                int strides[4] = {0, 0, 0, 0};
                if (!check(cudnnGetTensorNdDescriptor(probe.d, 4, &dt, &rank, dims,
                                                      strides),
                           "cudnnGetTensorNdDescriptor")) {
                    return 1;
                }
                if (rank != 3) {
                    std::fprintf(stderr,
                                 "FAIL: layer %d id %d %s descriptor reports rank %d, "
                                 "expected 3\n", layer, id, probe.what, rank);
                    return 1;
                }
                for (int d = 0; d < 3; ++d) {
                    if (dims[d] != probe.want[d]) {
                        std::fprintf(stderr,
                                     "FAIL: layer %d id %d %s dims[%d] = %d, expected "
                                     "%d\n", layer, id, probe.what, d, dims[d],
                                     probe.want[d]);
                        return 1;
                    }
                }
            }

            const float* base = weights.data();
            const float* m = static_cast<const float*>(mAddr);
            const float* b = static_cast<const float*>(bAddr);
            if (m < base || m >= base + weights.size() ||
                b < base || b >= base + weights.size()) {
                std::fprintf(stderr,
                             "FAIL: layer %d id %d reported a slice outside the weight "
                             "space\n", layer, id);
                return 1;
            }
            cudnnDestroyTensorDescriptor(mDesc);
            cudnnDestroyTensorDescriptor(bDesc);
        }
    }

    std::vector<float> x(static_cast<std::size_t>(kSeq) * kBatch * kInput);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = deterministic(static_cast<int>(i) + 3);
    const std::size_t state_elems = static_cast<std::size_t>(kLayers) * kBatch * kHidden;

    cudnnTensorDescriptor_t hDesc = nullptr, cDesc = nullptr;
    if (!check(cudnnCreateTensorDescriptor(&hDesc), "createTensor h") ||
        !check(cudnnCreateTensorDescriptor(&cDesc), "createTensor c")) {
        return 1;
    }
    const int state_dims[3] = {kLayers, kBatch, kHidden};
    const int state_strides[3] = {kBatch * kHidden, kHidden, 1};
    if (!check(cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, state_dims,
                                          state_strides), "setTensorNd h") ||
        !check(cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, state_dims,
                                          state_strides), "setTensorNd c")) {
        return 1;
    }

    auto make_data = [&](int seq, int vector, cudnnRNNDataDescriptor_t* out) {
        std::vector<int> lengths(kBatch, seq);
        return cudnnCreateRNNDataDescriptor(out) == CUDNN_STATUS_SUCCESS &&
               cudnnSetRNNDataDescriptor(*out, CUDNN_DATA_FLOAT,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                                         seq, kBatch, vector, lengths.data(),
                                         nullptr) == CUDNN_STATUS_SUCCESS;
    };

    // (a) one full-sequence call
    cudnnRNNDataDescriptor_t xFull = nullptr, yFull = nullptr;
    if (!make_data(kSeq, kInput, &xFull) || !make_data(kSeq, kHidden, &yFull)) {
        std::fprintf(stderr, "FAIL: building the full-sequence data descriptors\n");
        return 1;
    }
    std::vector<float> y_full(static_cast<std::size_t>(kSeq) * kBatch * kHidden, 0.0f);
    std::vector<float> hy_full(state_elems, 0.0f), cy_full(state_elems, 0.0f);
    if (!check(cudnnRNNForward(handle, rnn, CUDNN_FWD_MODE_INFERENCE, nullptr,
                               xFull, x.data(), yFull, y_full.data(),
                               hDesc, nullptr, hy_full.data(),
                               cDesc, nullptr, cy_full.data(),
                               weight_bytes, weights.data(), 0, nullptr, 0, nullptr),
               "cudnnRNNForward (full sequence)")) {
        return 1;
    }

    // (b) the same sequence one timestep at a time, carrying state -- what a
    //     greedy RNNT decoder does.
    cudnnRNNDataDescriptor_t xStep = nullptr, yStep = nullptr;
    if (!make_data(1, kInput, &xStep) || !make_data(1, kHidden, &yStep)) {
        std::fprintf(stderr, "FAIL: building the single-step data descriptors\n");
        return 1;
    }
    std::vector<float> h_state(state_elems, 0.0f), c_state(state_elems, 0.0f);
    std::vector<float> h_next(state_elems, 0.0f), c_next(state_elems, 0.0f);
    std::vector<float> y_step(static_cast<std::size_t>(kSeq) * kBatch * kHidden, 0.0f);
    for (int t = 0; t < kSeq; ++t) {
        std::vector<float> y_one(static_cast<std::size_t>(kBatch) * kHidden, 0.0f);
        if (!check(cudnnRNNForward(handle, rnn, CUDNN_FWD_MODE_INFERENCE, nullptr,
                                   xStep, x.data() + static_cast<std::size_t>(t) * kBatch * kInput,
                                   yStep, y_one.data(),
                                   hDesc, h_state.data(), h_next.data(),
                                   cDesc, c_state.data(), c_next.data(),
                                   weight_bytes, weights.data(), 0, nullptr, 0, nullptr),
                   "cudnnRNNForward (single step)")) {
            return 1;
        }
        std::memcpy(y_step.data() + static_cast<std::size_t>(t) * kBatch * kHidden,
                    y_one.data(), y_one.size() * sizeof(float));
        h_state = h_next;
        c_state = c_next;
    }

    int mismatches = 0;
    for (std::size_t i = 0; i < y_full.size(); ++i) {
        if (std::fabs(y_full[i] - y_step[i]) > 1e-5f) {
            if (mismatches < 5) {
                std::fprintf(stderr,
                             "FAIL: element %zu full=%.7f stepped=%.7f\n",
                             i, y_full[i], y_step[i]);
            }
            ++mismatches;
        }
    }
    for (std::size_t i = 0; i < state_elems; ++i) {
        if (std::fabs(hy_full[i] - h_state[i]) > 1e-5f ||
            std::fabs(cy_full[i] - c_state[i]) > 1e-5f) {
            if (mismatches < 5) {
                std::fprintf(stderr,
                             "FAIL: final state %zu h full=%.7f stepped=%.7f, "
                             "c full=%.7f stepped=%.7f\n",
                             i, hy_full[i], h_state[i], cy_full[i], c_state[i]);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr,
                     "FAIL: %d values differ between one full-sequence call and %d "
                     "single-timestep calls carrying state\n", mismatches, kSeq);
        return 1;
    }
    // projSize is not a boolean. cuDNN: "It is legal to set projSize equal to
    // hiddenSize, however, in this case, the recurrent projection feature is
    // disabled." So the legal range is 1..hiddenSize, DISABLED is encoded as
    // projSize == hiddenSize, and real cuDNN returns BAD_PARAM for 0 (measured
    // on 9.20 across the range).
    //
    // The load-bearing row is projSize == hiddenSize, because PyTorch's
    // aten/src/ATen/cudnn/Descriptors.h passes `proj_size ? proj_size :
    // hidden_size` -- an ordinary non-projected torch.nn.LSTM arrives with
    // projSize == hiddenSize. A guard refusing every nonzero projSize refuses
    // every PyTorch LSTM, which is what 0.6.6 through 0.6.9 did. Keep that row
    // asserted as ACCEPTED; it is the one that was broken.
    const struct { int proj; cudnnStatus_t want; const char* why; } proj_rows[] = {
        {0,            CUDNN_STATUS_BAD_PARAM,     "0 is outside the legal range"},
        {kHidden,      CUDNN_STATUS_SUCCESS,       "== hiddenSize disables projection"},
        {1,            CUDNN_STATUS_NOT_SUPPORTED, "a genuine projection"},
        {kHidden / 2,  CUDNN_STATUS_NOT_SUPPORTED, "a genuine projection"},
        {kHidden - 1,  CUDNN_STATUS_NOT_SUPPORTED, "a genuine projection"},
        {kHidden + 1,  CUDNN_STATUS_NOT_SUPPORTED, "wider than hiddenSize"},
    };
    for (const auto& row : proj_rows) {
        const cudnnStatus_t got = cudnnSetRNNDescriptor_v8(
            rnn, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS,
            CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT,
            CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, kInput, kHidden, row.proj,
            kLayers, nullptr, 0);
        if (got != row.want) {
            std::fprintf(stderr,
                         "FAIL: projSize %d returned %d, expected %d (%s)\n",
                         row.proj, (int)got, (int)row.want, row.why);
            return 1;
        }
    }

    cudnnDestroyRNNDataDescriptor(yStep);
    cudnnDestroyRNNDataDescriptor(xStep);
    cudnnDestroyRNNDataDescriptor(yFull);
    cudnnDestroyRNNDataDescriptor(xFull);
    cudnnDestroyTensorDescriptor(cDesc);
    cudnnDestroyTensorDescriptor(hDesc);
    cudnnDestroyRNNDescriptor(rnn);
    cudnnDestroy(handle);

    std::printf("PASS: v8 RNN input=%d hidden=%d layers=%d batch=%d — %d single-timestep "
                "calls with carried state match one full-sequence call, weight space %zu B, "
                "weight params land in it, the projSize contract honoured\n",
                kInput, kHidden, kLayers, kBatch, kSeq, weight_bytes);
    return 0;
}
