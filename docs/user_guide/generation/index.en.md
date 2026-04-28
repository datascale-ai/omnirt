# Generation Tasks

This section is organized by task surface. Every page follows the same structure: **minimal example** (Python / CLI / HTTP) → **key parameters** → **supported models** → **common combinations** → **troubleshooting**.

| Task | Shape | Typical models | Page |
|---|---|---|---|
| `text2image` | text → single image | `sd15`, `sdxl-base-1.0`, `flux2.dev`, `qwen-image` | [Text to Image](text2image.md) |
| `text2audio` | text + reference audio → speech | `cosyvoice3-triton-trtllm` | [Text to Audio](text2audio.md) |
| `image2image` | image + prompt → image | `sd15`, `sdxl-base-1.0` | [Image to Image](image2image.md) |
| `text2video` | text → video | `wan2.2-t2v-14b`, `cogvideox-2b`, `hunyuan-video` | [Text to Video](text2video.md) |
| `image2video` | first frame + prompt → video | `svd-xt`, `wan2.2-i2v-14b` | [Image to Video](image2video.md) |
| `audio2video` | audio + portrait → video | `soulx-flashtalk-14b` | [Talking Head](talking_head.md) |

!!! tip "Not sure where to start?"
    Read [Text to Image](text2image.md) first — it's the cheapest task to get running end-to-end on OmniRT, and it's what `omnirt validate` / `omnirt generate` defaults to when teaching the tool.
