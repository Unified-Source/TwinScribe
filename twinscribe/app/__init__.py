"""Desktop screens for twinscribe.

PySide6 is imported only under this package, so the rest of the library imports and tests
without a GUI toolkit present. The application window lives in `main`: a library of
recordings, a player with the transcript following the audio, and the batch that produces the
outputs. The verification screen for the review list lives in `verify`. The theme, the
painted icons and the timeline widget are shared.
"""
