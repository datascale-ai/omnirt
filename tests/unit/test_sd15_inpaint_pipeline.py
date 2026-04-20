from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import GenerateRequest
from omnirt.models.sd15.inpaint import SD15InpaintPipeline


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
        mask_image=None,
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
                "mask_image": mask_image,
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
        id="sd15",
        task="inpaint",
        pipeline_cls=SD15InpaintPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 6, "dtype": "fp16"},
    )


def test_sd15_inpaint_pipeline_uses_image_and_mask(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    mask_image = tmp_path / "mask.png"
    Image.new("RGB", (40, 20), color="green").save(input_image)
    Image.new("L", (40, 20), color=255).save(mask_image)

    monkeypatch.setattr(SD15InpaintPipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sd15.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="inpaint",
        model="sd15",
        backend="cuda",
        inputs={"image": str(input_image), "mask": str(mask_image), "prompt": "repair the sky"},
        config={"output_dir": str(tmp_path), "strength": 1.0},
    )
    pipeline = SD15InpaintPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    created = FakeDiffusersPipeline.created[-1]
    call = created.calls[-1]
    assert Path(result.outputs[0].path).suffix == ".png"
    assert call["image"].size == (40, 20)
    assert call["mask_image"].size == (40, 20)
    assert call["strength"] == 1.0
    assert created.scheduler == {"name": "euler"}
