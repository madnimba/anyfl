This folder contains **high-throughput defaults** intended for ~32GB VRAM GPUs.

- The original configs in `experiments/clean_accuracy/configs/` are kept as a **fallback** baseline.
- Runners will **try VRAM32 first**, and if CUDA OOM occurs, automatically retry using the fallback config (or an 8GB-safe override).

