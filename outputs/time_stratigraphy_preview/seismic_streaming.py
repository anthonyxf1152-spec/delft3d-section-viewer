"""Bounded-workspace 3-D NPY generation, independent of PSF size limits."""
from pathlib import Path
import json
import math
import shutil
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from seismic_forward import (MODEL_VOLUME_NAMES, _current_grid, estimate_current_grid,
    _normal_field, _build_grid_model, regional_datum, _line)

TILE_WORK_BYTES = 256 * 1024**2


def streaming_plan(shape, record, native_nx, native_ny, params):
    nx, ny, nz = shape
    # Native-layer interpolation buffers dominate for fine XY/coarse Z grids.
    per_column = max(1, (record + 1) * 96 + nz * 128)
    max_columns = max(1, TILE_WORK_BYTES // per_column)
    bx = min(nx, 64, max(1, int(np.sqrt(max_columns))))
    by = min(ny, 128, max(1, max_columns // bx))
    radius = [int(4 * width / params.spacing_xy_m + .5) for width in (500., 750.)]
    halo_x, halo_y = min(nx, bx + 2 * radius[0]), min(ny, by + 2 * radius[1])
    fill_bytes = ((record + 1) * native_nx * native_ny * 24 if params.sand_mode == 'filled' else 0)
    peak = (bx * by * per_column + halo_x * halo_y * nz * 12
            + min(16, halo_x) * halo_y * nz * 64 + fill_bytes)
    return {'tile_shape_xy': [int(bx), int(by)], 'noise_halo_samples_xy': radius,
            'tile_count': math.ceil(nx / bx) * math.ceil(ny / by),
            'estimated_peak_bytes': int(peak), 'convolution_performed': False,
            'method': 'XY tiles written to full-shape numeric NPY files; Gaussian halos preserve spatial filtering'}


def _source_support(model, x, y):
    ix0 = max(0, int(np.searchsorted(model.x, x[0])) - 1)
    ix1 = min(len(model.x), int(np.searchsorted(model.x, x[-1])) + 2)
    iy0 = max(0, int(np.searchsorted(model.y, y[0])) - 1)
    iy1 = min(len(model.y), int(np.searchsorted(model.y, y[-1])) + 2)
    valid = model.valid[ix0:ix1, iy0:iy1]
    if not valid.any():
        raise ValueError('当前保存范围没有有效模型单元。')
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return ix0, iy0, nearest[0] + ix0, nearest[1] + iy0


def _filtered_tile_noise(x, y, z, x_slice, y_slice, model, params):
    sigma = np.array([500., 750., 10.]) / [params.spacing_xy_m, params.spacing_xy_m, params.spacing_z_m]
    rx, ry = [int(4 * s + .5) for s in sigma[:2]]
    xa, xb = max(0, x_slice.start - rx), min(len(x), x_slice.stop + rx)
    ya, yb = max(0, y_slice.start - ry), min(len(y), y_slice.stop + ry)
    noise = _normal_field(x[xa:xb], y[ya:yb], z, params.spacing_xy_m,
                           (model.x[0], model.y[0]), params.seed + 13)
    gaussian_filter(noise, sigma=sigma, output=noise, mode='reflect', truncate=4.)
    return noise[x_slice.start - xa:x_slice.stop - xa,
                 y_slice.start - ya:y_slice.stop - ya].copy()


def _mapped_write(path, target_slice, values):
    mapped = np.load(path, mmap_mode='r+', allow_pickle=False)
    try:
        mapped[target_slice] = values
        mapped.flush()
    finally:
        mapped._mmap.close()


def save_3d_model_streaming(output, model, sections, params, progress=None, filled_hint=None,
                            tile_shape_xy=None):
    """Write full 3-D properties without a full in-memory physical volume.

    Uses exactly the same local geometry and rock laws as the forward model.
    Filtering halos are clipped only at full-ROI boundaries, so tile joins
    retain the same Gaussian realization as the whole-ROI computation.
    """
    record, x, y, z, margin = _current_grid(model, sections, params)
    estimate = estimate_current_grid(model, sections, params)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'模型保存目录已存在，请选择新目录：{output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < estimate['estimated_npy_payload_bytes'] + 32 * 1024**2:
        raise OSError('保存位置剩余空间不足，请更换目录。')
    emit = progress or (lambda _: None)
    emit('准备三维分块保存：检查坐标、有效区与初始参考面…')
    support = _source_support(model, x, y)
    coefficients = regional_datum(model)
    if params.sand_mode == 'filled' and filled_hint is None:
        from sand_completion import build_completed_volume
        emit('准备当前时刻的源含砂率补全缓存…')
        filled_hint = build_completed_volume(model, record)
    plan = dict(estimate['streaming_save'])
    bx, by = tuple(tile_shape_xy or plan['tile_shape_xy'])
    if not isinstance(bx, int) or not isinstance(by, int) or min(bx, by) < 1:
        raise ValueError('分块尺寸必须为正整数。')
    total = math.ceil(len(x) / bx) * math.ceil(len(y) / by)
    plan.update(tile_shape_xy=[bx, by], tile_count=total)
    shape = (len(x), len(y), len(z))
    staging = Path(tempfile.mkdtemp(prefix='.model-writing-', dir=output.parent))
    arrays = {}

    def describe(relative, array_shape, dtype, order):
        arrays[relative] = {'shape': list(array_shape), 'dtype': str(np.dtype(dtype)), 'axis_order': order}

    def save(relative, array, order):
        path = staging / relative; path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        describe(relative, array.shape, array.dtype, order)

    def create(relative, array_shape, dtype, order):
        mapped = np.lib.format.open_memmap(staging / relative, mode='w+', dtype=dtype, shape=array_shape)
        try:
            mapped.flush()
        finally:
            mapped._mmap.close()
        describe(relative, array_shape, dtype, order)

    try:
        for name, array in [('x_m', x), ('y_m', y), ('depth_m', z)]:
            save(f'{name}.npy', array, name)
        for name in (*MODEL_VOLUME_NAMES, 'impedance_kg_m2_s'):
            create(f'{name}.npy', shape, 'f4', 'X,Y,depth')
        for name, dtype in [('valid_xy', bool), ('fan_top_depth_m', 'f8'), ('fan_base_depth_m', 'f8')]:
            create(f'{name}.npy', shape[:2], dtype, 'X,Y')
        done = 0; metadata = None
        for i in range(0, len(x), bx):
            xs = slice(i, min(i + bx, len(x)))
            for j in range(0, len(y), by):
                ys = slice(j, min(j + by, len(y)))
                emit(f'分块生成并写入三维 NPY：{done + 1}/{total}；X {xs.start + 1}–{xs.stop}，Y {ys.start + 1}–{ys.stop}…')
                prepared = _build_grid_model(model, record, x[xs], y[ys], z, params, estimate, margin,
                    lambda _: None, filled_hint, coefficients=coefficients, global_support=support,
                    noise_loader=lambda: _filtered_tile_noise(x, y, z, xs, ys, model, params))
                if metadata is None:
                    metadata = prepared['metadata']
                for name, array in prepared['volumes'].items():
                    _mapped_write(staging / f'{name}.npy', (xs, ys, slice(None)), array)
                vp, rho = prepared['volumes']['vp_m_s'], prepared['volumes']['density_kg_m3']
                _mapped_write(staging / 'impedance_kg_m2_s.npy', (xs, ys, slice(None)), vp * rho)
                for name, array in prepared['surfaces'].items():
                    _mapped_write(staging / f'{name}.npy', (xs, ys), array)
                done += 1
                del prepared, vp, rho
        metadata = dict(metadata, grid_shape_xyz=list(shape), grid_estimate=estimate,
                        streaming_save=plan, model_dimension='3d', convolution_performed=False)
        # Native interpolation support overlaps at tile edges; do not sum its
        # per-tile composition counts as if they were unique source samples.
        metadata.pop('filled_layer_samples_in_native_roi', None)
        emit('三维数组已写入，提取当前切片及坐标…')
        descriptions = []
        for sec in sections:
            axis, fixed = sec['axis'], float(sec['actual_fixed_m'])
            h_grid = x if axis == 'x' else y
            h = h_grid[(h_grid >= sec['horizontal_m'][0] - 1e-6) & (h_grid <= sec['horizontal_m'][-1] + 1e-6)]
            prefix = axis + 'z'
            save(f'{prefix}/horizontal_m.npy', h, 'horizontal')
            save(f'{prefix}/depth_m.npy', z, 'depth')
            for name in (*MODEL_VOLUME_NAMES, 'impedance_kg_m2_s', 'valid_xy', 'fan_top_depth_m', 'fan_base_depth_m'):
                full = np.load(staging / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                try:
                    # A bool validity map is small; conversion preserves the
                    # continuous validity threshold used by existing exports.
                    cut = _line(full.astype('f4') if name == 'valid_xy' else full, x, y, axis, fixed, h)
                finally:
                    full._mmap.close()
                target_name = 'valid' if name == 'valid_xy' else name
                if target_name == 'valid':
                    cut = cut >= .999
                save(f'{prefix}/{target_name}.npy', cut, 'horizontal,depth' if cut.ndim == 2 else 'horizontal')
            descriptions.append({'axis': axis, 'record': record, 'actual_fixed_m': fixed,
                                 'requested_fixed_m': float(sec['requested_fixed_m'])})
        report = dict(metadata, arrays=arrays, sections=descriptions,
            array_units={'x_m': 'm', 'y_m': 'm', 'depth_m': 'm, positive down',
                'fan_top_depth_m': 'm', 'fan_base_depth_m': 'm', 'sand_fraction': '0..1 per whole voxel',
                'occupancy': '0..1 fan occupancy', 'vp_m_s': 'm/s', 'density_kg_m3': 'kg/m3',
                'impedance_kg_m2_s': 'kg/(m2*s)', 'porosity_delta': 'absolute porosity change',
                'reflectivity': 'dimensionless', 'baseline_reflectivity': 'dimensionless'},
            invalid_columns='Use valid_xy.npy / cut valid.npy; invalid columns are computational support.',
            depth_convention='Positive down; x_m/y_m are original project coordinates in metres.',
            model_save_ignores_forward_voxel_limit=True)
        (staging / 'model_parameters.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
