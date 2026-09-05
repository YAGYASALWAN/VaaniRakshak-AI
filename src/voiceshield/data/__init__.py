"""Public data-engineering API."""

from voiceshield.data.audio_io import AudioData, load_audio, save_audio
from voiceshield.data.metadata import ManifestRecord, MANIFEST_COLUMNS, append_manifest
from voiceshield.data.pipeline import process_audio_file
from voiceshield.data.preprocessing import convert_to_mono, resample_waveform, segment_waveform
from voiceshield.data.splits import generator_disjoint_split, speaker_disjoint_split
from voiceshield.data.validation import ValidationReport, validate_waveform

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
