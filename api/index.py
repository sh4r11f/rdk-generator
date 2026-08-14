"""Vercel serverless entrypoint.

Vercel's Python runtime looks for a WSGI callable named `app` in this module. The Flask
app is already serverless-safe: it creates no directories at import and streams every
export straight back in the response rather than storing it between requests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The repo root holds the `rdk_generator` package; Vercel only puts this file's
# directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdk_generator.webapp.app import app  # noqa: E402

__all__ = ["app"]
