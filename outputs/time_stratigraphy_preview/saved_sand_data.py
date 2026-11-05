"""Read-only, memory-mapped access to saved physical-coordinate sand models."""
from pathlib import Path
import json
import math

import numpy as np


def sampled_indices(shape, budget, max_axis=None, preserve_depth=False):
    steps = [max(1, math.ceil(n / max_axis)) if max_axis else 1 for n in shape]
    def indices():
        return [np.unique(np.r_[np.arange(0, n, step), n - 1]).astype('i4') for n, step in zip(shape, steps)]
    result = indices()
    while math.prod(len(a) for a in result) > budget:
        choices = [i for i in range(len(shape)) if len(result[i]) > 2]
        if preserve_depth and any(i < 2 for i in choices):
            choices = [i for i in choices if i < 2]
        if not choices:
            break
        axis = max(choices, key=lambda i: len(result[i]))
        steps[axis] += 1; result = indices()
    return result


class SavedSandModel:
    """A saved [X,Y,depth] model, independent of native DAT/DEF input files."""
    def __init__(self, path, progress=None):
        path = Path(path).expanduser().resolve()
        self.folder = path if path.is_dir() else path.parent
        self.path = self.folder / 'sand_fraction.npy' if path.is_dir() else path
        self._maps = []
        self.closed = False
        emit = progress or (lambda _: None)
        try:
            if self.path.suffix.lower() != '.npy' or not self.path.is_file():
                raise ValueError('请选择保存模型根目录下的 sand_fraction.npy，或选择其文件夹。')
            self.q = self._open(self.path)
            if self.q.ndim != 3 or min(self.q.shape) < 2:
                raise ValueError('含砂率须为 [X,Y,深度] 三维数组；请勿选择 xz/yz 内的二维切片文件。')
            if self.q.dtype.kind != 'f':
                raise ValueError('含砂率须为普通浮点数 NPY 数组，数值为0～1。')
            self.shape = tuple(int(n) for n in self.q.shape)
            self.axes = []
            for name, size in zip(('x_m', 'y_m', 'depth_m'), self.shape):
                axis = np.load(self.folder / f'{name}.npy', allow_pickle=False)
                if (axis.ndim != 1 or len(axis) != size or axis.dtype.kind not in 'fiu' or
                        not np.isfinite(axis).all() or np.any(np.diff(axis.astype('f8')) <= 0)):
                    raise ValueError(f'{name}.npy 的长度、数值或坐标顺序与三维数组不一致。')
                self.axes.append(axis.astype('f8'))
            if self.axes[2][0] < 0:
                raise ValueError('depth_m.npy 须采用向下为正的非负深度。')
            self.occupancy = self._open(self.folder / 'occupancy.npy')
            self.valid = self._open(self.folder / 'valid_xy.npy')
            if self.occupancy.shape != self.shape or self.occupancy.dtype.kind not in 'fb':
                raise ValueError('occupancy.npy 必须与三维含砂率同形状。')
            if self.valid.shape != self.shape[:2] or self.valid.dtype != np.dtype(bool) or not self.valid.any():
                raise ValueError('valid_xy.npy 须为匹配X/Y网格且含有效列的布尔数组。')
            metadata_path = self.folder / 'model_parameters.json'
            self.metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
            if self.metadata.get('complete') is False:
                raise ValueError('该模型未保存完整，请选择已完成保存的模型目录。')
            if 'grid_shape_xyz' in self.metadata and list(self.shape) != self.metadata['grid_shape_xyz']:
                raise ValueError('参数文件中的三维网格大小与含砂率数组不一致。')
            emit('分块检查含砂率、占据率和坐标…')
            block = max(1, (32 * 1024**2) // (self.shape[1] * self.shape[2] * max(4, self.q.dtype.itemsize)))
            for start in range(0, self.shape[0], block):
                for label, array in [('含砂率', self.q), ('占据率', self.occupancy)]:
                    values = array[start:start + block]
                    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                        raise ValueError(f'{label}数组须为有限的0～1数值。')
        except Exception:
            self.close()
            raise

    def _open(self, path):
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        self._maps.append(array)
        return array

    def close(self):
        for array in self._maps:
            mapping = getattr(array, '_mmap', None)
            if mapping is not None:
                mapping.close()
        self._maps = []
        self.closed = True

    def initial_indices(self):
        if self.closed:
            raise ValueError('三维模型已关闭，请重新读取。')
        result = [n // 2 for n in self.shape]
        for cut in self.metadata.get('sections', []):
            if cut.get('axis') in ('x', 'y') and 'actual_fixed_m' in cut:
                fixed_axis = 1 if cut['axis'] == 'x' else 0
                result[fixed_axis] = int(np.argmin(abs(self.axes[fixed_axis] - cut['actual_fixed_m'])))
        if not self.valid[result[0], result[1]]:
            valid = np.argwhere(self.valid)
            nearest = valid[np.argmin(np.sum((valid - np.array(result[:2]))**2, axis=1))]
            result[:2] = nearest.tolist()
        occupied = np.flatnonzero(self.occupancy[result[0], result[1], :] > 0)
        if occupied.size:
            result[2] = int(occupied[len(occupied) // 2])
        return result

    def plane(self, axis, index, budget=250_000):
        if self.closed:
            raise ValueError('三维模型已关闭，请重新读取。')
        if axis not in (0, 1, 2) or not 0 <= int(index) < self.shape[axis]:
            raise ValueError('切片位置超出三维网格范围。')
        remaining = [i for i in range(3) if i != axis]
        picks = sampled_indices([self.shape[i] for i in remaining], budget)
        full_picks = []; k = 0
        for i in range(3):
            if i == axis:
                full_picks.append(np.array([index], dtype='i4'))
            else:
                full_picks.append(picks[k]); k += 1
        selection = np.ix_(*full_picks)
        values = np.take(self.q[selection], 0, axis=axis).astype('f4')
        occ = np.take(self.occupancy[selection], 0, axis=axis)
        valid_xy = self.valid[np.ix_(full_picks[0], full_picks[1])]
        valid = (np.broadcast_to(valid_xy[..., None], tuple(len(p) for p in full_picks)))
        valid = np.take(valid, 0, axis=axis).copy() & (occ > 0)
        a, b = np.meshgrid(self.axes[remaining[0]][picks[0]], self.axes[remaining[1]][picks[1]], indexing='ij')
        coordinates = np.empty(values.shape + (3,), dtype='f4')
        coordinates[..., axis] = self.axes[axis][int(index)]
        coordinates[..., remaining[0]] = a; coordinates[..., remaining[1]] = b
        coordinates[..., :2] /= 1000.
        coordinates[..., 2] *= -.001  # Depth down -> world Z up.
        return {'vertices': coordinates.reshape(-1, 3), 'values': values.ravel(),
                'valid': valid, 'shape': list(values.shape), 'axis': axis, 'index': int(index)}

    def surface_grid(self, budget=2_000_000, threshold_percent=0.):
        if self.closed:
            raise ValueError('三维模型已关闭，请重新读取。')
        if not 0 <= threshold_percent <= 100:
            raise ValueError('体表面的含砂率阈值须在0～100%之间。')
        picks = sampled_indices(self.shape, budget, max_axis=320, preserve_depth=True)
        values = self.q[np.ix_(*picks)].astype('f4')
        occ = self.occupancy[np.ix_(*picks)]
        valid = self.valid[np.ix_(picks[0], picks[1])]
        inside = (occ > 0) & valid[..., None] & (values >= threshold_percent / 100.)
        return values, inside, picks


def _index_to_world(vertices, axes):
    result = np.empty_like(vertices, dtype='f4')
    for i, axis in enumerate(axes):
        index = vertices[:, i]
        values = np.interp(index, np.arange(len(axis)), axis)
        below, above = index < 0, index > len(axis) - 1
        values[below] = axis[0] + index[below] * (axis[1] - axis[0])
        values[above] = axis[-1] + (index[above] - len(axis) + 1) * (axis[-1] - axis[-2])
        result[:, i] = values * (-.001 if i == 2 else .001)
    return result


def surface_mesh(model, budget=2_000_000, threshold_percent=0., progress=None):
    """Sampled display mesh only; source arrays remain memory-mapped and intact."""
    from skimage.measure import marching_cubes
    from scipy.ndimage import map_coordinates
    emit = progress or (lambda _: None)
    while True:
        emit('按显示分辨率重建三维砂体表面…')
        values, inside, picks = model.surface_grid(budget, threshold_percent)
        if not inside.any():
            return {'vertices': np.empty((0, 3), dtype='f4'), 'faces': np.empty((0, 3), dtype='u4'),
                    'values': np.empty(0, dtype='f4'), 'sample_shape': list(inside.shape), 'empty': True}
        vertices, faces, _, _ = marching_cubes(np.pad(inside.astype('f4'), 1), level=.5, allow_degenerate=False)
        vertices -= 1
        if len(faces) <= 1_000_000 or budget <= 32_000:
            break
        budget //= 2
    sand = map_coordinates(values, vertices.T, order=1, mode='nearest').astype('f4')
    world = _index_to_world(vertices, [axis[pick] for axis, pick in zip(model.axes, picks)])
    return {'vertices': world, 'faces': faces.astype('u4'), 'values': sand,
            'sample_shape': list(inside.shape), 'empty': False}
