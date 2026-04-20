from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import GenerateRequest
from omnirt.models.sdxl.image2image import SDXLImageToImagePipeline


class FakeCudaRuntime(BackendRuntime):
    name = "cuda"
    device_name = "cpu"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module


class FakeDiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.text_encoder = object()
        self.text_encoder_2 = object()
        self.unet = object()
        self.vae = object()
        self.scheduler = None
        self.calls = []

    @classmethod
    def from_pretrained(cls, source, torch_dtype=None, use_safetensors=True, variant=None):
        pipeline = cls()
        pipeline.source = source
        pipeline.torch_dtype = torch_dtype
        pipeline.variant = variant
        cls.created.append(pipeline)
        return pipeline

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        image=None,
        num_inference_steps=None,
        guidance_scale=None,
        strength=None,
        generator=None,
        num_images_per_prompt=1,
        output_type="pil",
        **kwargs,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "generator": generator,
                "num_images_per_prompt": num_images_per_prompt,
                "output_type": output_type,
                **kwargs,
            }
        )
        return SimpleNamespace(images=[Image.new("RGB", image.size, color="white")])


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="sdxl-base-1.0",
        task="image2image",
        pipeline_cls=SDXLImageToImagePipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
    )


def test_sdxl_image2image_pipeline_passes_strength(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    Image.new("RGB", (64, 64), color="red").save(input_image)

    monkeypatch.setattr(SDXLImageToImagePipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sdxl.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="image2image",
        model="sdxl-base-1.0",
        backend="cuda",
        inputs={"image": str(input_image), "prompt": "cinematic concept art"},
        config={"output_dir": str(tmp_path), "strength": 0.7},
    )
    pipeline = SDXLImageToImagePipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    created = FakeDiffusersPipeline.created[-1]
    call = created.calls[-1]
    assert Path(result.outputs[0].path).suffix == ".png"
    assert call["image"].size == (64, 64)
    assert call["strength"] == 0.7
    assert created.scheduler == {"name": "euler"}
