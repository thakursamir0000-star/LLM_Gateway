"""Router lab: benchmark data collection, router training and evaluation."""


def safe_console() -> None:
    """Never crash on printing non-ASCII question text to a Windows console or a redirected file."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
