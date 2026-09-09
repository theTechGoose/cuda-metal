"""Dump a PyTorch nn.LSTM reference for the cuDNN v8 path to check itself against.

The claim under test is the gate order. CuMetal states that its weight space
holds, per (layer, direction), W_ih then W_hh then b_ih then b_hh with LSTM
gates in i,f,g,o order -- "which is both cuDNN's linLayerID order and PyTorch's
chunk order". Nothing in this project ever checked the second half of that
sentence against PyTorch; every existing test compares CuMetal to CuMetal.

So: run a real nn.LSTM on CPU, write out its parameters in ITS layout together
with the input and the outputs it produced. The consumer places those same
parameters through cudnnGetRNNWeightParams and runs cudnnRNNForward. If the
gate order claim is wrong, the numbers diverge -- and a wrong gate order is
invisible to a self-consistency test, because both sides would be wrong the
same way.
"""
import struct
import sys

import torch

MAGIC = 0x4C53544D  # 'LSTM'


def dump(path, input_size, hidden, layers, seq, batch, seed):
    torch.manual_seed(seed)
    lstm = torch.nn.LSTM(input_size, hidden, num_layers=layers,
                         batch_first=False, bidirectional=False, bias=True)
    lstm.eval()

    # Deterministic, and spread wider than the default init so a gate swap
    # cannot hide inside near-identical values.
    with torch.no_grad():
        for name, p in lstm.named_parameters():
            g = torch.Generator().manual_seed(seed + sum(ord(c) for c in name))
            p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=g))

    gx = torch.Generator().manual_seed(seed + 977)
    x = torch.empty(seq, batch, input_size).uniform_(-1.0, 1.0, generator=gx)

    with torch.no_grad():
        y, (hy, cy) = lstm(x)

    blobs = []
    for layer in range(layers):
        blobs.append(getattr(lstm, f"weight_ih_l{layer}"))
        blobs.append(getattr(lstm, f"weight_hh_l{layer}"))
        blobs.append(getattr(lstm, f"bias_ih_l{layer}"))
        blobs.append(getattr(lstm, f"bias_hh_l{layer}"))
    blobs += [x, y, hy, cy]

    with open(path, "wb") as f:
        f.write(struct.pack("<6i", MAGIC, input_size, hidden, layers, seq, batch))
        for b in blobs:
            f.write(b.detach().contiguous().float().numpy().tobytes())

    print(f"{path}: torch {torch.__version__} "
          f"input={input_size} hidden={hidden} layers={layers} seq={seq} batch={batch}")
    print(f"  y  [0,0,:4] = {[round(v, 6) for v in y[0, 0, :4].tolist()]}")
    print(f"  hy [0,0,:4] = {[round(v, 6) for v in hy[0, 0, :4].tolist()]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    if which == "small":
        dump(sys.argv[2], 7, 5, 2, 4, 3, 20260909)
    else:
        # Parakeet's prediction network: input 640, hidden 640, 2 layers.
        dump(sys.argv[2], 640, 640, 2, 5, 2, 20260909)
