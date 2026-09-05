"""Project-specific exceptions.

Library code raises these instead of swallowing errors. Scripts may catch them
and continue over a directory of files after logging the structured reason.
"""


class VoiceShieldError(Exception):
    """Base error for recoverable VoiceShield failures."""


class AudioLoadError(VoiceShieldError):
    """The file exists but cannot be decoded into a waveform."""


class AudioValidationError(VoiceShieldError):
    """Validation failed and the caller asked for a hard failure."""


class ManifestSchemaError(VoiceShieldError):
    """A table is missing required columns or uses an illegal label."""


class SplitError(VoiceShieldError):
    """Speaker- or generator-disjoint split constraints cannot be satisfied."""
