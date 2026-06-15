"""IndexTTS realtime text-to-audio runtime."""

__all__ = ["IndexTTSStreamingRuntime", "create_indextts_runtime_from_env"]


def __getattr__(name: str):
    if name in __all__:
        from omnirt.models.indextts.runtime import IndexTTSStreamingRuntime, create_indextts_runtime_from_env

        values = {
            "IndexTTSStreamingRuntime": IndexTTSStreamingRuntime,
            "create_indextts_runtime_from_env": create_indextts_runtime_from_env,
        }
        return values[name]
    raise AttributeError(name)
