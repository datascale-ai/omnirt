"""QuickTalk realtime audio2video runtime support."""

__all__ = ["default_quicktalk_model_root", "quicktalk_checkpoint_path"]


def default_quicktalk_model_root(*args, **kwargs):
    from omnirt.models.quicktalk.converter import default_quicktalk_model_root as _default_quicktalk_model_root

    return _default_quicktalk_model_root(*args, **kwargs)


def quicktalk_checkpoint_path(*args, **kwargs):
    from omnirt.models.quicktalk.converter import quicktalk_checkpoint_path as _quicktalk_checkpoint_path

    return _quicktalk_checkpoint_path(*args, **kwargs)
