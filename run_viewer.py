"""
Launch the 3-D trajectory viewer.

    python run_viewer.py

Then open http://127.0.0.1:8000 in a browser. Edit any solver file under
src/gnc/problems/ and the server reloads automatically.
"""

import sys
import webbrowser
from pathlib import Path
from threading import Timer

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import uvicorn  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    open_browser = "--no-browser" not in sys.argv
    reload = "--no-reload" not in sys.argv

    if open_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()

    uvicorn.run(
        "gnc.server:app",
        host=HOST,
        port=PORT,
        reload=reload,
        reload_dirs=[str(SRC)] if reload else None,
    )


if __name__ == "__main__":
    main()
