"""Portable PyCharm/command-line entry point; run in the installed environment."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent


def main():
    user_cache = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.cache'))) / 'Delft3DSectionViewer'
    os.environ.setdefault('MPLCONFIGDIR', str(user_cache / 'matplotlib'))
    os.environ.setdefault('QT_API', 'pyside6')
    sys.path.insert(0, str(HERE))
    from interactive_sections import main as run
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
