"""Public data-engineering API."""

from importlib import import_module

# Preserve the public API without importing torch/audio libraries for metadata work.
_MODULES = {
    "audio_io": ("AudioData", "load_audio", "save_audio"),
    "metadata": ("ManifestRecord", "MANIFEST_COLUMNS", "append_manifest"),
    "pipeline": ("process_audio_file",),
    "preprocessing": ("convert_to_mono", "resample_waveform", "segment_waveform"),
    "splits": ("generator_disjoint_split", "speaker_disjoint_split"),
    "validation": ("ValidationReport", "validate_waveform"),
}


def __getattr__(name: str) -> object:
    """Load an existing public symbol only when requested."""
    for module, names in _MODULES.items():
        if name in names:
            value = getattr(import_module(f"{__name__}.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(name)

__all__ = [
    "AudioData",
    "MANIFEST_COLUMNS",
    "ManifestRecord",
    "ValidationReport",
    "append_manifest",
    "convert_to_mono",
    "generator_disjoint_split",
    "load_audio",
    "process_audio_file",
    "resample_waveform",
    "save_audio",
    "segment_waveform",
    "speaker_disjoint_split",
    "validate_waveform",
]
