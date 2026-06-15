# LongCat Avatar Patch Notes

Keep patch files generic. Do not include machine IPs, user names, private home
directories, passwords, or generated benchmark artifacts.

The OmniRT adapter expects the external LongCat checkout to provide a worker
entrypoint compatible with the protocol documented in the parent README. Keep
CUDA and Ascend patches separate where possible.

A practical shared patch set usually contains these changes:

1. Add a worker script such as `run_ascend_avatar_cp_worker.py` that accepts:
   `--input-json`, `--checkpoint-dir`, `--base-checkpoint-dir`,
   `--request-file`, `--response-file`, `--save-mp4`, `--cp-split-hw`,
   `--model-type`, `--stage`, `--resolution`, `--frames`, `--steps`, `--seed`,
   `--output-type`, `--placement`, and the cache/save flags emitted by OmniRT.
2. Add a CUDA worker script such as `run_cuda_avatar_worker.py` or alias the
   official CUDA entrypoint to the same request/response protocol.
3. Use HCCL for distributed launch when `LONGCAT_DIST_BACKEND=hccl`, and NCCL
   when `LONGCAT_DIST_BACKEND=nccl`.
4. Respect `ASCEND_RT_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, `GPU_NUM`, and
   `NPROC_PER_NODE`.
5. Route CUDA attention through `LONGCAT_ATTENTION_BACKEND=flash_attn` or
   `sdpa`.
6. Route dense Ascend attention through `LONGCAT_ATTENTION_BACKEND=npu_fusion`.
7. Route preview Ascend self-attention through `LONGCAT_ATTENTION_BACKEND=ascend_bsa`
   while keeping cross-attention on `LONGCAT_CROSS_ATTENTION_BACKEND=npu_fusion`.
8. Replace brittle fp32 assertions in modulation/final-layer time embeddings
   with explicit casts to fp32.
9. Fuse or merge Avatar LoRA before the hot generation path when
   `LONGCAT_MERGE_LORA=1` or `--merge-lora` is set.
10. Support streamed VAE save and direct copy-mux export for MP4 output.

Suggested patch file names:

- `longcat-video-avatar-ascend-worker.patch`
- `longcat-video-avatar-cuda-worker.patch`
- `longcat-video-avatar-attention-backends.patch`
- `longcat-video-avatar-stream-save.patch`
