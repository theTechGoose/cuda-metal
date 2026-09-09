# Platform and legal boundaries

[Known-gaps index](../known-gaps.md)

These are durable boundaries unless the canonical specification changes.

- macOS 14+ on Apple Silicon only.
- No Linux container on a Mac. Docker Desktop, OrbStack, Colima and the like run
  a Linux VM, and that VM has no Metal passthrough -- there is no device to pass
  through and no Linux Metal driver to use it. So a Linux container cannot reach
  CuMetal even when the container runs on Apple Silicon, and no image, flag or
  runtime changes that. CuMetal runs on the macOS host, natively. A project that
  needs one identical container everywhere should plan on CPU (or a remote CUDA
  host) for the Mac leg rather than expecting the container to find the GPU.
- No Windows, Linux ARM, non-Apple discrete GPU, or Thunderbolt eGPU target.
- One Apple GPU device; no multi-GPU or peer-to-peer execution.
- No OpenGL, Vulkan, or DirectX interop.
- No SASS execution, decompilation, or SASS-only binary support.
- No private Apple APIs.
- CUDA-facing headers remain clean-room; NVIDIA headers are not shipped.

The optional `libcuda.dylib` alias is a bounded PTX-bearing compatibility path,
not the project architecture. Closed-source drop-in use has legal and technical
risk distinct from source recompilation. See [the legal notice](../legal-notice.md).

Apple's public Metal compiler rejects native `double` arithmetic on the tested
targets. CuMetal software modes are explicit alternatives, not evidence of
native Metal FP64.
