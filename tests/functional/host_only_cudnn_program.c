/* A host-only C program that calls a CUDA *library* API and contains no device
 * code -- the shape of a consumer's tool that links -lcudnn to inspect layout,
 * dump weights, or probe a descriptor.
 *
 * Real toolchains build this with the host compiler plus -lcudnn. cumetalc has
 * to accept the same shape, because porting a build line means replacing the
 * compiler and keeping the rest. Until 0.6.8 it rejected both halves: -lcudnn
 * was "unknown option", and a .c input was "unsupported input extension". */

#include <stdio.h>

#include "cudnn.h"

int main(void) {
    const size_t version = cudnnGetVersion();
    if (version == 0) {
        fprintf(stderr, "FAIL: cudnnGetVersion returned 0\n");
        return 1;
    }

    cudnnHandle_t handle = NULL;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS || handle == NULL) {
        fprintf(stderr, "FAIL: cudnnCreate\n");
        return 1;
    }

    /* Exercise one real query so the link is proven to reach the library, not
     * merely to resolve a symbol that returns a constant. */
    cudnnRNNDescriptor_t rnn = NULL;
    if (cudnnCreateRNNDescriptor(&rnn) != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: cudnnCreateRNNDescriptor\n");
        return 1;
    }
    if (cudnnSetRNNDescriptor_v8(rnn, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                 CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                 CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT,
                                 CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH,
                                 8, 8, /*projSize=*/8, 2, NULL, 0) != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: cudnnSetRNNDescriptor_v8\n");
        return 1;
    }
    size_t weight_bytes = 0;
    if (cudnnGetRNNWeightSpaceSize(handle, rnn, &weight_bytes) != CUDNN_STATUS_SUCCESS ||
        weight_bytes == 0) {
        fprintf(stderr, "FAIL: cudnnGetRNNWeightSpaceSize\n");
        return 1;
    }

    cudnnDestroyRNNDescriptor(rnn);
    cudnnDestroy(handle);
    printf("PASS: host-only program linked and ran (cudnn %zu, weights %zu)\n",
           version, weight_bytes);
    return 0;
}
