"""twinscribe: offline transcription with speaker labels for recordings that will be relied on.

Two engines run. The transducer's transcript is published because it cannot write into
silence. The Whisper engine is never published; it marks the spans where it heard speech and
the published engine heard nothing, and those marks are what a person listens to.
"""

__version__ = "0.0.1"
