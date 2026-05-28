# Edge Inference Engine for Time Series Anomaly Detection

A from-scratch implementation of memory-efficient transformer attention, demonstrating up to **9.5x speedup** over a naive baseline through custom CUDA kernels, deployed end-to-end to a Raspberry Pi 4 at **128 ms** per inference window.

The project builds three things: a PatchTST transformer trained in PyTorch, three CUDA kernels written from scratch (naive attention, tiled FlashAttention-style attention with online softmax, and a fused QKV projection), and a pure-numpy inference runtime running on a Pi without PyTorch, ONNX, or any other framework.

The point is to show, concretely and with numbers, why transformer inference is bound by memory bandwidth rather than compute, and how the FlashAttention idea actually makes that limit go away.

---

## Why This Matters

Transformer attention as naively written materializes a full N×N score matrix in GPU high-bandwidth memory (HBM). For an N=1024 sequence with batch 32, that is roughly 130 MB written and read three times: once for the matmul output, once for the softmax pass, once for the final P@V multiply. The compute itself is cheap. The round-trips to HBM are not. This is the bottleneck FlashAttention solved by tiling the computation so the N×N matrix never leaves the GPU's small but fast on-chip SRAM.

This project reproduces that idea from scratch, measures the impact, and then deploys the resulting model to an edge device, closing the GPU-training-to-edge-deployment loop that real industrial ML infrastructure teams build.

The application context is electrical transformer monitoring: the model watches oil-temperature sensors at a power substation and flags anomalies. The Pi plays the role of an edge node at a remote substation that can't rely on cloud connectivity.

---

## Architecture

```
ETTh1 dataset (7 channels, 17420 hourly readings)
        |
        v
PatchTST: patch length 16, stride 8, d_model 64, 2 transformer layers, 4 heads
        |
        +----> [Colab T4 training + CUDA kernel benchmarking]
        |
        +----> [Pi deployment: numpy-only runtime, sliding window inference]
```

**Model.** PatchTST is a 2023 architecture for time series forecasting. It splits each input series into patches of timesteps that act as tokens, drastically shortening the sequence length the attention layer has to handle (336 timesteps becomes 41 patches in this config). With 2 layers and d_model=64, the trained model has roughly 220K parameters and exports to about 1 MB of `.npy` weights.

**Kernels** (`naive_attention.cu`, `tiled_attention.cu`, `fused_qkv.cu`):
- *Naive attention* materializes the full N×N matrix and runs three separate kernels (QK^T, row-wise softmax, P@V).
- *Tiled attention* is a FlashAttention-1 style implementation. Blocks of 32 queries are processed against tiles of 32 keys/values at a time, with a single-pass online softmax that updates the running max, normalizer, and output accumulator as each tile is streamed through SRAM. The N×N matrix is never written to HBM.
- *Fused QKV projection* combines three input-projection matmuls (X @ W_Q, X @ W_K, X @ W_V) into a single kernel pass over X, eliminating two redundant HBM reads of the input.

**Edge runtime.** `infer.py` and `stream.py` on the Pi load the exported numpy weights and reproduce the forward pass using only `numpy.matmul` and `numpy.exp`. No PyTorch, no ONNX, no graph compiler. Every operation is visible.

---

## Results

### Tiled attention vs naive attention (Colab T4, batch 32, dk=16)

| N    | Naive (ms) | Tiled (ms) | Speedup | HBM ratio |
|------|-----------:|-----------:|--------:|----------:|
| 64   | 0.188      | 0.034      | 5.56x   | 4.0x      |
| 128  | 0.506      | 0.068      | 7.42x   | 8.0x      |
| 256  | 0.992      | 0.105      | 9.48x   | 16.0x     |
| 512  | 2.754      | 0.338      | 8.14x   | 32.0x     |
| 1024 | 10.081     | 1.270      | 7.94x   | 64.0x     |

Speedup peaks near N=256 at 9.5x and plateaus around 8x as N grows. HBM traffic reduction grows linearly in N (since naive scales as O(N²) and tiled as O(N·d)) but realized speedup grows sub-linearly because the tiled kernel becomes shared-memory and compute bound at larger N rather than fully HBM bound. This matches the FlashAttention paper's own measurements for similar configurations on T4-class hardware.

### Fused QKV projection (Colab T4)

| Variant                | Time (ms) | Speedup |
|------------------------|----------:|--------:|
| Three separate matmuls | 0.084     | 1.00x   |
| Fused single kernel    | 0.038     | 2.21x   |

Numerical agreement with the unfused reference: max abs error ≈ 9.5e-6 across Q, K, V outputs. HBM savings of 0.67 MB per pass from eliminating two redundant reads of the input tensor.

### Pi 4 deployment (numpy-only runtime, 4GB Pi 4B, OpenBLAS)

| Metric                    | Value         |
|---------------------------|---------------|
| Mean inference latency    | 128 ms        |
| P95 inference latency     | 128 ms        |
| Sliding window length     | 336 timesteps |
| Forecast horizon          | 96 timesteps  |
| Memory footprint          | ~30 MB        |

The flat P95-to-mean ratio is a useful signal. It indicates the Pi runtime has no occasional stalls (no garbage collection pauses, no thermal throttling visible at this rate, no I/O contention).

### PyTorch reference baseline

PyTorch's built-in scaled-dot-product attention on the same inputs runs in 0.080 ms, within ~5x of cuBLAS performance. That is expected, since PyTorch dispatches to highly optimized vendor primitives that use tensor cores and warp-level matmul instructions this project deliberately does not implement. The tiled kernel is meant to demonstrate the algorithmic idea, not to beat cuBLAS.

---

## Repository Structure

```
inference_engine/
├── README.md                 # this file
├── notebooks/
│   └── cuda_kernels.ipynb    # Colab notebook: training, kernels, benchmarks
├── kernels/
│   ├── naive_attention.cu    # baseline: 3 kernels, materializes N×N
│   ├── tiled_attention.cu    # online softmax, single-pass, float4 loads
│   └── fused_qkv.cu          # combined input projection
├── pi/
│   ├── infer.py              # numpy-only forward pass
│   └── stream.py             # sliding window streaming inference
├── data/
│   ├── ETTh1_anomaly.csv     # injected synthetic anomalies (point spikes + level shifts)
│   └── labels.npy            # binary anomaly labels
└── weights/                  # exported model weights (~23 .npy files, ~1 MB)
```

---

## How to Run

### GPU side (Colab T4 or any Ampere+ NVIDIA GPU)

1. Open `notebooks/cuda_kernels.ipynb` in Colab. Set runtime to T4 GPU.
2. Run cells sequentially. Phase 0 sets up data and injects anomalies. Phase 1 trains PatchTST. Phase 2 benchmarks the PyTorch attention baseline. Phases 3 through 5 build, verify, and benchmark the three CUDA kernels.
3. Weights export to `weights/` as `.npy` files at the end of Phase 6.

### Pi side (Raspberry Pi 4, 4GB+, Pi OS Lite 64-bit / Trixie)

```bash
sudo apt update && sudo apt install -y libopenblas-dev python3-numpy python3-pandas
mkdir -p ~/inference_engine/{weights,data,src}
# transfer weights/, ETTh1_anomaly.csv, labels.npy, infer.py, stream.py via scp
cd ~/inference_engine/src
python3 stream.py
```

The stream script processes 200 sliding windows from the held-out test region and prints per-window latency plus mean and P95 at the end.

---

## Key Implementation Notes

**Online softmax.** The tiled kernel uses the recurrence `m' = max(m, s); α = exp(m - m'); o = α·o + exp(s - m')·v; l = α·l + exp(s - m')` to fold the softmax max-subtraction, normalization, and value-weighting into a single streaming pass over each key/value tile. This is what allows the N×N matrix to be eliminated. At no point in the kernel does any thread hold more than one tile's worth of intermediates.

**Block layout.** 32 queries per block, 32 keys per tile, one full warp per block. The earlier draft of this project used 16 threads per block which wasted half the warp. Switching to a full warp plus a single-pass softmax (rather than the two-pass max-then-accumulate version) is what moved the N=1024 speedup from 1.56x to 7.94x.

**Vectorized loads.** Q, K, V rows are loaded as `float4` (4 vector loads per row instead of 16 scalar) since the head dimension is exactly 16 floats. This is a small but real win on memory-bound kernels.

**No tensor cores, no warp MMA.** The kernel uses scalar FMAs in registers. A real production implementation would use `wmma` or `mma.sync` instructions for an additional several-x speedup. Implementing that was out of scope for a first-principles project.

---

## Limitations and Honest Assessment

The project demonstrates the FlashAttention algorithmic idea cleanly, but it is not a production attention kernel. Three things a real implementation would have that this one doesn't:

1. **Tensor cores via `wmma`.** Would push the kernel from 8x to roughly 30-50x over naive on T4 hardware.
2. **Multi-head fusion.** This kernel runs heads serially across the batch dimension. Production kernels parallelize heads in the same launch.
3. **Variable sequence length and causal masking.** The kernel assumes square attention with no mask. Real LLM inference needs both.

The anomaly detection task itself shows limited signal. In the 80/20 train-test split, only about 13 windows in the test set fell on injected anomalies (out of 3053), so the F1 score on the anomaly class is low. This is a data construction issue (anomalies were injected uniformly across the full series rather than ensured to land in the test split) rather than a model issue, and the project's contribution is the inference engine itself rather than detection accuracy.

---
