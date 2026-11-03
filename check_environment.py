"""Print portable environment diagnostics; optionally test a real OpenGL frame."""
from __future__ import annotations

import argparse
import importlib
from importlib.metadata import version
import os
import platform
from pathlib import Path
import struct
import sys


PACKAGES = {'numpy': 'numpy', 'scipy': 'scipy', 'matplotlib': 'matplotlib',
            'cigvis': 'cigvis', 'vispy': 'vispy', 'PySide6': 'PySide6', 'pillow': 'PIL'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true', help='Test an off-screen OpenGL frame using the desktop display.')
    parser.add_argument('--app-root', type=Path, default=Path(__file__).parent,
                        help='Application root; defaults to this script directory.')
    args = parser.parse_args(argv)
    os.environ.setdefault('QT_API', 'pyside6')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(args.app_root.resolve() / '.cache' / 'matplotlib'))
    print(f'Python: {platform.python_version()} ({struct.calcsize("P")*8}-bit)')
    print(f'Executable: {sys.executable}')
    print(f'OS: {platform.platform()}')
    errors = []
    if struct.calcsize('P') != 8:
        errors.append('64-bit Python is required.')
    if sys.version_info[:2] != (3, 12):
        print('NOTE: the supplied environment is tested on Python 3.12.')
    for distribution, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            print(f'{distribution}: {version(distribution)}')
        except Exception as exc:
            errors.append(f'{distribution}: {type(exc).__name__}: {exc}')
    entry = args.app_root / 'outputs/time_stratigraphy_preview/run_interactive.py'
    if not entry.is_file():
        errors.append(f'Missing app entry point: {entry}')
    if args.render and not errors:
        canvas = None
        try:
            import numpy as np
            from vispy import app, scene
            app.use_app('pyside6')
            canvas = scene.SceneCanvas(size=(128, 128), show=False, bgcolor='white')
            view = canvas.central_widget.add_view()
            scene.visuals.Line(pos=np.array([[0., 0.], [1., 1.]], dtype='f4'), color='red', parent=view.scene)
            view.camera = 'panzoom'
            view.camera.set_range(x=(0, 1), y=(0, 1))
            frame = canvas.render()
            if frame.shape[:2] != (128, 128):
                raise RuntimeError(f'Unexpected frame shape: {frame.shape}')
            if not ((frame[..., 0] > 150) & (frame[..., 1] < 100)).any():
                raise RuntimeError('No rendered red line found in the OpenGL frame.')
            print(f'OpenGL frame: OK {frame.shape}')
        except Exception as exc:
            errors.append(f'OpenGL render: {type(exc).__name__}: {exc}')
        finally:
            if canvas is not None:
                canvas.close()
    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print('Environment checks passed.' + (' OpenGL was not tested; use --render on the target desktop.' if not args.render else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
