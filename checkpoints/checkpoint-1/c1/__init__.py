"""WeightWatch C1 package."""
import sys

# Windows consoles/redirects often use a legacy code page; never crash on a print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
