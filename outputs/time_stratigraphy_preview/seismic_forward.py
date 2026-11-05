"""Current-record 3D image-domain convolution, adapted from the user's reference.

The input is inferred geometry/composition, not measured impedance. Physical
thickness scaling is explicit and independent of the viewer's display VE.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import shutil
import tempfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt
from scipy.special import ndtri
from scipy.fft import next_fast_len

from seismic_reference import rock_model, psf_image, voxelize_fan

MAX_GRID_VOXELS = 22_000_000
MODEL_VOLUME_NAMES = ('sand_fraction', 'occupancy', 'vp_m_s', 'density_kg_m3',
                      'porosity_delta', 'reflectivity', 'baseline_reflectivity')


@dataclass(frozen=True)
class SeismicParameters:
    frequency_hz: float = 30.
    vertical_scale: float = 16.
    burial_depth_m: float = 1400.
    depth_max_m: float = 1800.
    spacing_xy_m: float = 125.
    spacing_z_m: float = 5.
    reference_velocity_m_s: float = 2800.
    porosity_strength: float = 1.
    seed: int = 20261028
    sand_mode: str = 'filled'
    model_dimension: str = '3d'

    def validate(self):
        if self.model_dimension not in ('2d', '3d'):
            raise ValueError('正演维度须为二维或三维。')
        for name in ('frequency_hz', 'vertical_scale', 'burial_depth_m', 'depth_max_m',
                     'spacing_xy_m', 'spacing_z_m', 'reference_velocity_m_s'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} 必须是有限正数。')
        if not np.isfinite(self.porosity_strength) or not 0 <= self.porosity_strength <= 2:
            raise ValueError('内部非均质强度须在 0～2 之间。')
        if self.sand_mode not in ('filled', 'mixed'):
            raise ValueError('正演输入请选择分期补全或当前混合床平均。')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError('随机种子必须是非负整数。')
        if self.spacing_z_m > self.reference_velocity_m_s / (8 * self.frequency_hz):
            raise ValueError('垂向采样过粗，无法稳定采样该主频的子波；请减小 Z 采样间距。')
        if self.depth_max_m / self.spacing_z_m > 4000:
            raise ValueError('深度样点过多，请增大 Z 采样间距或缩小深度范围。')


def regional_datum(model):
    """Fit one initial regional plane. Never flatten the selected final top."""
    xx, yy = np.meshgrid(model.x, model.y, indexing='ij')
    initial = np.asarray(model.raw[0], dtype='f8')
    valid = model.valid & np.isfinite(initial)
    design = np.column_stack((np.ones(valid.sum()), xx[valid], yy[valid]))
    coefficients = np.linalg.lstsq(design, initial[valid], rcond=None)[0]
    return coefficients


def _grid_window(axis, lower, upper, spacing):
    # Every ROI shares the same global lattice, so moving a cut does not
    # change spatial phases or the realization of internal heterogeneity.
    count = int(np.floor((axis[-1] - axis[0]) / spacing + 1e-9)) + 1
    lo = max(0, int(np.floor((lower - axis[0]) / spacing)))
    hi = min(count - 1, int(np.ceil((upper - axis[0]) / spacing)))
    if hi <= lo:
        raise ValueError('选中范围内不足两个正演网格点，请增大剖面半宽或减小 XY 采样间距。')
    return axis[0] + np.arange(lo, hi + 1) * spacing


def _normal_field(x, y, z, spacing, origin, seed):
    """Counter-based spatial normals; stable at the same XYZ across ROI sizes."""
    yi = np.rint((y - origin[1]) / spacing).astype('u8')[None, :, None]
    zi = np.arange(len(z), dtype='u8')[None, None, :]
    field = np.empty((len(x), len(y), len(z)), dtype='f4')
    for start in range(0, len(x), 16):
        xi = np.rint((x[start:start + 16] - origin[0]) / spacing).astype('u8')[:, None, None]
        with np.errstate(over='ignore'):
            value = (xi * np.uint64(0xD6E8FEB86659FD93) ^ yi * np.uint64(0xA5A3564E27F8862D)
                     ^ zi * np.uint64(0x9E3779B97F4A7C15) ^ np.uint64(seed))
            value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            value ^= value >> np.uint64(31)
        uniform = ((value >> np.uint64(11)).astype('f8') + .5) / 2**53
        field[start:start + 16] = ndtri(np.clip(uniform, 1e-12, 1 - 1e-12))
    return field


def correlated_porosity(q, occupancy, top, base, x, y, z, origin, params, filtered_noise=None):
    """Same bounded bed/porosity scenario as the reference; ROI-stable noise."""
    spacing = (params.spacing_xy_m, params.spacing_xy_m, params.spacing_z_m)
    sigma = np.array([500., 750., 10.]) / spacing
    if filtered_noise is None:
        field = _normal_field(x, y, z, params.spacing_xy_m, origin, params.seed + 13)
        gaussian_filter(field, sigma=sigma, output=field, mode='reflect', truncate=4.)
    else:
        field = filtered_noise
        if field.shape != q.shape or field.dtype != np.dtype('f4'):
            raise ValueError('分块孔隙度噪声与模型网格不一致。')
    # Fixed analytic variance, rather than ROI-dependent standard deviation.
    variance = 1.
    for width in sigma:
        radius = int(4 * width + .5)
        kernel = np.exp(-.5 * (np.arange(-radius, radius + 1) / width)**2)
        kernel /= kernel.sum()
        variance *= float(np.sum(kernel * kernel))
    field /= max(np.sqrt(variance), 1e-8)
    np.clip(field, -2.5, 2.5, out=field)
    rng = np.random.default_rng(params.seed + 13)
    edges = [0.]
    while edges[-1] < 1.1:
        edges.append(edges[-1] + float(rng.uniform(.022, .055)))
    edges = np.array(edges)
    levels = gaussian_filter1d(rng.normal(size=len(edges) + 1), .35)
    levels = np.clip(levels / max(levels.std(), 1e-8), -2, 2).astype('f4')
    xx, yy = np.meshgrid(x - origin[0], y - origin[1], indexing='ij')
    warp = .015 * np.sin(2 * np.pi * xx / 2800 + .8 * np.cos(2 * np.pi * yy / 4500))
    warp += .010 * np.cos(2 * np.pi * (xx / 5000 + yy / 3300))
    modulation = .8 + .2 * np.sin(2 * np.pi * xx / 4300) * np.cos(2 * np.pi * yy / 6200)
    thickness = np.maximum(base - top, 5.)
    for start in range(0, len(x), 16):
        stop = min(start + 16, len(x))
        u = (z[None, None, :] - top[start:stop, :, None]) / thickness[start:stop, :, None]
        strata = np.searchsorted(edges, u + warp[start:stop, :, None])
        lamina = levels[strata] * modulation[start:stop, :, None]
        phi = .011 * lamina + .011 * field[start:stop]
        taper = np.clip(np.minimum(u, 1 - u) / .025, 0, 1)
        phi *= np.sqrt(q[start:stop]) * occupancy[start:stop] * taper
        field[start:stop] = np.clip(phi, -.025, .025) * params.porosity_strength
    return field, {'normalized_bed_edges': edges.tolist(), 'seed': params.seed + 13,
                   'delta_phi_bound': .025 * params.porosity_strength,
                   'correlation_lengths_m': [500, 750, 10],
                   'random_field': 'XYZ counter-based Gaussian field; analytic variance; fixed global lattice',
                   'reference_difference': 'Stochastic realization is ROI-stable, not pixel-identical to the older full-volume random field.'}


def reflection(vp, density):
    impedance = vp * density
    result = np.zeros_like(impedance)
    result[..., :-1] = (impedance[..., 1:] - impedance[..., :-1]) / (impedance[..., 1:] + impedance[..., :-1])
    return result


def _line(array, x, y, axis, fixed, horizontal):
    """Cut at the exact current native fixed coordinate, without shifting it."""
    points = (np.column_stack((horizontal, np.full(len(horizontal), fixed))) if axis == 'x'
              else np.column_stack((np.full(len(horizontal), fixed), horizontal)))
    return RegularGridInterpolator((x, y), array, bounds_error=True)(points).astype('f4')


def _current_grid(model, sections, params):
    """Resolve the same lattice for estimation, model saving and convolution."""
    params = params or SeismicParameters()
    params.validate()
    if not sections or len({s['record'] for s in sections}) != 1:
        raise ValueError('请选择同一时刻的有效 XZ / YZ 切片。')
    record = int(sections[0]['record'])
    model._record(record)
    if record < 1:
        raise ValueError('当前为初始时刻，没有沉积增量层段；请选择更晚的保存时刻。')
    bounds = {a: [] for a in ('x', 'y')}
    for sec in sections:
        if sec['axis'] not in ('x', 'y'):
            raise ValueError('切片方向必须为 XZ 或 YZ。')
        if len(sec['horizontal_m']) < 2:
            raise ValueError('切片至少需要两个原始网格点。')
        axis = sec['axis']; other = 'y' if axis == 'x' else 'x'
        bounds[axis].extend([float(sec['horizontal_m'][0]), float(sec['horizontal_m'][-1])])
        bounds[other].append(float(sec['actual_fixed_m']))
    # Four Gaussian correlation lengths plus PSF guard support.
    margin = (3000., 4000.)
    x = _grid_window(model.x, min(bounds['x']) - margin[0], max(bounds['x']) + margin[0], params.spacing_xy_m)
    y = _grid_window(model.y, min(bounds['y']) - margin[1], max(bounds['y']) + margin[1], params.spacing_xy_m)
    z = np.arange(0, params.depth_max_m + params.spacing_z_m * .5, params.spacing_z_m)
    for sec in sections:
        grid = x if sec['axis'] == 'x' else y
        horizontal = grid[(grid >= sec['horizontal_m'][0] - 1e-6) & (grid <= sec['horizontal_m'][-1] + 1e-6)]
        if len(horizontal) < 2:
            raise ValueError('当前切片的正演采样不足两道，请减小 XY 采样间距。')
        fixed_grid = y if sec['axis'] == 'x' else x
        if not fixed_grid[0] <= sec['actual_fixed_m'] <= fixed_grid[-1]:
            raise ValueError('切片固定坐标超出正演网格，请减小 XY 采样间距。')
    return record, x, y, z, margin


def estimate_current_grid(model, sections, params=None):
    """Cheap exact grid/disk counts; approximate working RAM, no 3-D allocation."""
    params = params or SeismicParameters()
    if params.model_dimension == '2d':
        from seismic_2d import estimate_2d_grid
        return estimate_2d_grid(model, sections, params)
    record, x, y, z, margin = _current_grid(model, sections, params)
    shape = (len(x), len(y), len(z))
    voxels = math.prod(shape)
    fft_shape = tuple(next_fast_len(shape[i] + 2 * (16, 16, 48)[i]) for i in range(3))
    fft_voxels = math.prod(fft_shape)
    spectrum_voxels = fft_shape[0] * fft_shape[1] * (fft_shape[2] // 2 + 1)
    native_x = min(len(model.x), int(np.searchsorted(model.x, x[-1])) + 2) - max(0, int(np.searchsorted(model.x, x[0])) - 1)
    native_y = min(len(model.y), int(np.searchsorted(model.y, y[-1])) + 2) - max(0, int(np.searchsorted(model.y, y[0])) - 1)
    # Account for simultaneous native histories, interpolation and global fill
    # workspaces. These are estimates, not an OS memory availability check.
    native_bytes = (record + 1) * (native_x * native_y * 64 + len(x) * len(y) * 32)
    fill_bytes = ((record + 1) * len(model.x) * len(model.y) * 24
                  if params.sand_mode == 'filled' else 0)
    model_peak = int(voxels * 96 + native_bytes + fill_bytes)
    forward_peak = int(model_peak + fft_voxels * 16 + spectrum_voxels * 64)
    cuts = []
    for sec in sections:
        grid = x if sec['axis'] == 'x' else y
        n = int(np.sum((grid >= sec['horizontal_m'][0] - 1e-6) & (grid <= sec['horizontal_m'][-1] + 1e-6)))
        cuts.append({'axis': sec['axis'], 'shape_horizontal_depth': [n, len(z)]})
    # 8 float32 volumes (7 stored + impedance), XY surfaces/mask, vectors,
    # and matching cuts. NPY headers and JSON add a small extra amount.
    disk_bytes = voxels * 32 + len(x) * len(y) * 17 + (len(x) + len(y) + len(z)) * 8
    for cut in cuts:
        n, nz = cut['shape_horizontal_depth']
        disk_bytes += n * nz * 32 + n * 25 + nz * 8
    from seismic_streaming import streaming_plan
    streaming = streaming_plan(shape, record, len(model.x), len(model.y), params)
    return {'grid_shape_xyz': list(shape), 'voxel_count': voxels,
        'grid_bounds_m': {'x': [float(x[0]), float(x[-1])], 'y': [float(y[0]), float(y[-1])],
                          'depth': [float(z[0]), float(z[-1])]},
        'spacing_m_xyz': [params.spacing_xy_m, params.spacing_xy_m, params.spacing_z_m],
        'fft_shape_xyz': list(fft_shape), 'single_float32_array_bytes': voxels * 4,
        'estimated_model_peak_bytes': model_peak, 'estimated_forward_peak_bytes': forward_peak,
        'estimated_npy_payload_bytes': int(disk_bytes), 'sections': cuts,
        'streaming_save': streaming,
        'maximum_voxels': MAX_GRID_VOXELS, 'within_grid_limit': voxels <= MAX_GRID_VOXELS,
        'roi_guard_m_xy': list(margin),
        'notes': ['RAM is an approximate working-set estimate, not measured free memory; keep additional headroom.',
                  'Grid counts are exact. Geometry fit/composition are checked during model building.',
                  'Disk estimate covers raw NPY payloads; headers and JSON are additional.']}


def build_current_model(model, sections, params=None, progress=None, filled_hint=None):
    """Generate physical property volumes without running a PSF convolution."""
    params = params or SeismicParameters()
    if params.model_dimension == '2d':
        from seismic_2d import build_2d_model
        return build_2d_model(model, sections, params, progress, filled_hint)
    record, x, y, z, margin = _current_grid(model, sections, params)
    estimate = estimate_current_grid(model, sections, params)
    if not estimate['within_grid_limit']:
        raise ValueError('当前范围的三维正演网格过大；请缩小剖面半宽，或增大 XY 采样间距。')
    emit = progress or (lambda _: None)
    return _build_grid_model(model, record, x, y, z, params, estimate, margin, emit, filled_hint)


def _build_grid_model(model, record, x, y, z, params, estimate, margin, emit,
                      filled_hint=None, coefficients=None, global_support=None, noise_loader=None):
    """Shared physical laws for whole-volume forward and bounded tile export."""
    emit('拟合统一初始参考面，保留所选时刻的真实冲淤起伏…')
    coefficients = regional_datum(model) if coefficients is None else coefficients
    # Include source nodes bracketing the regular ROI; no extrapolation.
    ix0 = max(0, int(np.searchsorted(model.x, x[0])) - 1)
    ix1 = min(len(model.x), int(np.searchsorted(model.x, x[-1])) + 2)
    iy0 = max(0, int(np.searchsorted(model.y, y[0])) - 1)
    iy1 = min(len(model.y), int(np.searchsorted(model.y, y[-1])) + 2)
    sx, sy = slice(ix0, ix1), slice(iy0, iy1)
    valid = model.valid[sx, sy]
    if not valid.any() and global_support is None:
        raise ValueError('当前计算范围没有有效模型单元。')
    history = np.asarray(model.raw[:record + 1, sx, sy])
    interfaces = np.minimum.accumulate(history[::-1], axis=0)[::-1].copy()
    # Initial 4.5 m substrate is excluded, as in the reference PSF scenario.
    native_z = np.moveaxis(interfaces[::-1], 0, -1)
    native_h = np.maximum(0, -np.diff(native_z, axis=-1))
    emit('读取当前时刻的含砂率输入及缺测补全来源…')
    if params.sand_mode == 'filled':
        if filled_hint is None:
            from sand_completion import build_completed_volume
            fractions, provenance = build_completed_volume(model, record)
        else:
            fractions, provenance = filled_hint
            if fractions.shape != (record + 1, len(model.x), len(model.y)):
                raise ValueError('含砂率快照与当前选择不一致。')
        q_native = np.moveaxis(fractions[1:record + 1, sx, sy][::-1], 0, -1).copy()
        sources = provenance[1:record + 1, sx, sy]
        visible = np.moveaxis(native_h[..., ::-1], -1, 0) >= .001
        filled_count = int(np.sum(visible & (sources >= 2)))
    else:
        model._load_sediment_inventory()
        sand = np.asarray(model._sand_equivalent[record, sx, sy], dtype='f8')
        mud = np.asarray(model._mud_equivalent[record, sx, sy], dtype='f8')
        ratio = np.divide(sand, sand + mud, out=np.full_like(sand, np.nan), where=valid & ((sand + mud) > 1e-7))
        q_native = np.broadcast_to(ratio[:, :, None], native_h.shape).copy()
        filled_count = 0
    unresolved = valid[:, :, None] & (native_h >= .001) & ~np.isfinite(q_native)
    if unresolved.any():
        raise ValueError('当前存在有厚度但含砂率未知的层段，不能把未知值当泥岩正演；请选择分期补全输入。')
    if np.any((q_native[np.isfinite(q_native)] < 0) | (q_native[np.isfinite(q_native)] > 1)):
        raise ValueError('含砂率必须在 0～1 之间。')
    q_native = np.nan_to_num(q_native, nan=0.)
    if not valid.all():
        if global_support is None:
            nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
            native_z = native_z[tuple(nearest)]
        else:
            # Use the full ROI's nearest valid source, not tile-local nearest
            # cells, so invalid support remains identical across tile seams.
            gx0, gy0, nearest_m, nearest_n = global_support
            mm = nearest_m[ix0 - gx0:ix1 - gx0, iy0 - gy0:iy1 - gy0][~valid]
            nn = nearest_n[ix0 - gx0:ix1 - gx0, iy0 - gy0:iy1 - gy0][~valid]
            support_history = np.asarray(model.raw[:record + 1, mm, nn])
            support_interfaces = np.minimum.accumulate(support_history[::-1], axis=0)[::-1]
            native_z[~valid] = support_interfaces[::-1].T
        q_native[~valid] = 0.
        native_h[~valid] = 0.
    xx, yy = np.meshgrid(x, y, indexing='ij')
    points = np.column_stack((xx.ravel(), yy.ravel()))
    axes = (model.x[sx], model.y[sy])
    interp = lambda a: RegularGridInterpolator(axes, a, bounds_error=True)(points).astype('f4')
    zi = interp(native_z)
    thickness = np.maximum(0, -np.diff(zi, axis=1))
    sand_volume = interp(q_native * native_h)
    layers = np.divide(sand_volume, thickness, out=np.zeros_like(sand_volume), where=thickness > 1e-7)
    np.clip(layers, 0, 1, out=layers)
    good = interp(valid.astype('f4')).reshape(len(x), len(y)) >= .999
    datum = coefficients[0] + coefficients[1] * xx + coefficients[2] * yy
    # Preflight the physical scenario before allocating rock-property volumes.
    depths = params.burial_depth_m - params.vertical_scale * (zi - datum.ravel()[:, None])
    active_depths = depths[good.ravel()]
    if ((not active_depths.size and global_support is None) or
            (active_depths.size and (np.min(active_depths) < 30 or np.max(active_depths) > z[-1] - 50))):
        raise ValueError('当前层厚倍率／埋深使扇体超出深度范围；请减小层厚倍率、调整埋深或增大最大深度。')
    # Inactive interpolation support cannot fail voxelizer's global fit check.
    zi[~good.ravel()] = datum.ravel()[~good.ravel(), None]
    emit(f'体素化当前地层情景：{len(x)} × {len(y)} × {len(z)}；物理层厚 ×{params.vertical_scale:g}…')
    q, occupancy, top, base, _ = voxelize_fan(zi, layers, x, y, z, params.vertical_scale,
        'original_relief', 0, datum_elevation=datum, burial_depth=params.burial_depth_m)
    q[~good] = 0.; occupancy[~good] = 0.
    # Release large native/interpolation workspaces before rock-property arrays.
    del history, interfaces, native_z, native_h, q_native, zi, thickness, sand_volume, layers, depths, active_depths, unresolved
    emit('构建速度、岩石密度和原岩石参数反射系数…')
    vp, rho, _, background_edges = rock_model(q, xx - model.x[0], yy - model.y[0], z, params.seed)
    baseline_reflectivity = reflection(vp, rho)
    emit('应用内部孔隙度情景，更新速度、密度和反射系数…')
    noise = noise_loader() if noise_loader is not None else None
    phi, phi_meta = correlated_porosity(q, occupancy, top, base, x, y, z, (model.x[0], model.y[0]), params, noise)
    vp -= 6000 * phi; rho -= 1650 * phi
    if not np.isfinite(vp).all() or not np.isfinite(rho).all() or np.min(vp) <= 0 or np.min(rho) <= 0:
        raise ValueError('岩石参数无效，请检查正演情景设置。')
    reflectivity = reflection(vp, rho)
    metadata = {'complete': True, 'parameters': asdict(params), 'selected_record': record,
        'selected_hydro_days': float(model.hydro_days[record]), 'selected_morph_days': float(model.morph_days[record]),
        'source_project': model.source_info, 'source_cache': str(model.volume_dir),
        'geometry': {'formula': 'depth = burial_depth - vertical_scale * (elevation - regional_initial_plane)',
            'plane_coefficients_a_b_c': coefficients.tolist(), 'physical_thickness_scale': params.vertical_scale,
            'initial_substrate': 'Excluded; base is the surviving earliest bed surface, as in the reference scenario.',
            'final_top_flattened': False, 'depth_positive_down': True},
        'product': 'seismic_input_model', 'convolution_performed': False,
        'method': 'Physical property model for 3D image-domain PSF convolution',
        'full_wave_equation': False, 'aperture_degrees': [25, 40], 'agc': False,
        'baseline_law': 'Vp=2200+.35*z+400*q+dv_layer*(1-q); rho=2050+.12*z+160*q+dr_layer*(1-q)',
        'heterogeneity_law': 'Vp_new=Vp_base-6000*delta_phi; rho_new=rho_base-1650*delta_phi',
        'porosity_scenario': phi_meta, 'background_layer_edges_m': background_edges.tolist(),
        'grid_shape_xyz': [len(x), len(y), len(z)], 'roi_guard_m_xy': list(margin),
        'grid_estimate': estimate,
        'filled_layer_samples_in_native_roi': filled_count,
        'sand_fraction_changed_by_heterogeneity': False,
        'reference_style': 'User supplied whole_fan_seismic_20261028 figures, column 3',
        'reference_implementation_sha256': hashlib.sha256(Path(__file__).with_name('seismic_reference.py').read_bytes()).hexdigest(),
        'limitations': ['Inferred layer compositions and uncalibrated rock properties form a research scenario.',
            'Physical scaling changes thickness; it is not the unscaled Delft3D response.',
            'ROI calculation and a stable spatial random realization need not reproduce the older volume pixel-for-pixel.',
            'No multiples, acquisition modelling, full scattering or wave-equation propagation.'],
        'source_original_files_modified': False}
    return {'coordinates': {'x_m': x, 'y_m': y, 'depth_m': z},
        'volumes': dict(zip(MODEL_VOLUME_NAMES, (q, occupancy, vp, rho, phi, reflectivity, baseline_reflectivity))),
        'surfaces': {'valid_xy': good, 'fan_top_depth_m': top, 'fan_base_depth_m': base},
        'metadata': metadata}


def save_current_model(output, model, sections, params=None, progress=None, filled_hint=None):
    """Model saving is independent of the convolution voxel limit."""
    params = params or SeismicParameters()
    if params.model_dimension == '3d':
        from seismic_streaming import save_3d_model_streaming
        return save_3d_model_streaming(output, model, sections, params, progress, filled_hint)
    prepared = build_current_model(model, sections, params, progress, filled_hint)
    return export_current_model(output, prepared, sections, progress)


def model_sections(prepared, sections, extra_volumes=None):
    """Sample the physical model and optional seismic at the exact native cut."""
    x, y, z = [prepared['coordinates'][key] for key in ('x_m', 'y_m', 'depth_m')]
    volumes = dict(prepared['volumes'], **(extra_volumes or {}))
    surfaces = prepared['surfaces']
    results = []
    for sec in sections:
        axis, fixed = sec['axis'], sec['actual_fixed_m']
        grid = x if axis == 'x' else y
        horizontal = grid[(grid >= sec['horizontal_m'][0] - 1e-6) & (grid <= sec['horizontal_m'][-1] + 1e-6)]
        extract = lambda a: _line(a, x, y, axis, fixed, horizontal)
        row = {'axis': axis, 'horizontal_m': horizontal, 'depth_m': z,
               'actual_fixed_m': float(fixed), 'requested_fixed_m': float(sec['requested_fixed_m']),
               'record': int(sec['record']), 'valid': extract(surfaces['valid_xy'].astype('f4')) >= .999,
               'fan_top_depth_m': extract(surfaces['fan_top_depth_m']),
               'fan_base_depth_m': extract(surfaces['fan_base_depth_m'])}
        for key, array in volumes.items():
            row[key] = extract(array)
        results.append(row)
    return results


def forward_current(model, sections, params=None, progress=None, filled_hint=None):
    params = params or SeismicParameters()
    if params.model_dimension == '2d':
        from seismic_2d import forward_2d
        return forward_2d(model, sections, params, progress, filled_hint)
    emit = progress or (lambda _: None)
    prepared = build_current_model(model, sections, params, emit, filled_hint)
    spacing = [params.spacing_xy_m, params.spacing_xy_m, params.spacing_z_m]
    volumes = prepared['volumes']
    emit('原岩石参数反射系数：执行三维 PSF 褶积…')
    baseline, _, _ = psf_image(volumes['baseline_reflectivity'], spacing, params.reference_velocity_m_s, params.frequency_hz)
    emit('内部非均质反射系数：执行三维 PSF 褶积…')
    improved, _, _ = psf_image(volumes['reflectivity'], spacing, params.reference_velocity_m_s, params.frequency_hz)
    emit('提取当前 XZ / YZ 位置，准备显示与可复现导出…')
    results = model_sections(prepared, sections, {'seismic_baseline': baseline, 'seismic': improved})
    if not all(np.isfinite(row[key]).all() for row in results for key in ('seismic', 'vp_m_s', 'density_kg_m3')):
        raise ValueError('地震正演出现非有限样点。')
    metadata = dict(prepared['metadata'], product='seismic_sections', convolution_performed=True,
        method='3D constant-reference finite-band finite-angle image-domain PSF convolution of normal-incidence impedance reflectivity')
    return {'sections': results, 'metadata': metadata}


def export_current_model(output, prepared, sections, progress=None):
    """Publish a complete, pickle-free NPY bundle; never overwrite an old run."""
    if prepared['metadata'].get('model_dimension') == '2d':
        from seismic_2d import export_2d_model
        return export_2d_model(output, prepared, progress)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'模型保存目录已存在，请选择新目录：{output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    estimate = prepared['metadata']['grid_estimate']
    if shutil.disk_usage(output.parent).free < estimate['estimated_npy_payload_bytes'] + 16 * 1024**2:
        raise OSError('保存位置剩余空间不足，请更换目录或减小网格。')
    emit = progress or (lambda _: None)
    staging = Path(tempfile.mkdtemp(prefix='.model-writing-', dir=output.parent))
    arrays = {}
    def save(relative, array, order):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, array, allow_pickle=False)
        arrays[relative] = {'shape': list(array.shape), 'dtype': str(array.dtype), 'axis_order': order}
    try:
        for name, array in prepared['coordinates'].items():
            save(f'{name}.npy', array, name)
        for name, array in prepared['volumes'].items():
            emit(f'保存三维模型：{name}.npy…')
            save(f'{name}.npy', array, 'X,Y,depth')
        vp, rho = prepared['volumes']['vp_m_s'], prepared['volumes']['density_kg_m3']
        # Write one X slab at a time, avoiding another full impedance volume.
        impedance = np.lib.format.open_memmap(staging / 'impedance_kg_m2_s.npy', mode='w+', dtype='f4', shape=vp.shape)
        try:
            for i in range(len(vp)):
                np.multiply(vp[i], rho[i], out=impedance[i])
            impedance.flush()
        finally:
            impedance._mmap.close()
            del impedance
        arrays['impedance_kg_m2_s.npy'] = {'shape': list(vp.shape), 'dtype': 'float32', 'axis_order': 'X,Y,depth'}
        for name, array in prepared['surfaces'].items():
            save(f'{name}.npy', array, 'X,Y')
        cuts = model_sections(prepared, sections)
        descriptions = []
        for cut in cuts:
            prefix = cut['axis'] + 'z'
            # Interpolate impedance itself, rather than multiply interpolated
            # Vp and density (which would differ at off-lattice cut locations).
            full = np.load(staging / 'impedance_kg_m2_s.npy', mmap_mode='r', allow_pickle=False)
            try:
                cut['impedance_kg_m2_s'] = _line(full, prepared['coordinates']['x_m'],
                    prepared['coordinates']['y_m'], cut['axis'], cut['actual_fixed_m'], cut['horizontal_m'])
            finally:
                full._mmap.close()
            for name, array in cut.items():
                if isinstance(array, np.ndarray):
                    save(f'{prefix}/{name}.npy', array, 'horizontal,depth' if array.ndim == 2 else
                         ('depth' if name == 'depth_m' else 'horizontal'))
            descriptions.append({k: v for k, v in cut.items() if not isinstance(v, np.ndarray)})
        report = dict(prepared['metadata'], arrays=arrays, sections=descriptions,
            array_units={'sand_fraction': '0..1; sand volume per whole voxel, including occupancy',
                         'occupancy': 'fan fraction of voxel 0..1', 'vp_m_s': 'm/s',
                         'density_kg_m3': 'kg/m3', 'impedance_kg_m2_s': 'kg/(m2*s)',
                         'porosity_delta': 'absolute porosity change', 'reflectivity': 'dimensionless',
                         'baseline_reflectivity': 'dimensionless, before internal heterogeneity'},
            invalid_columns='Use valid_xy.npy (or cut valid.npy); invalid columns contain computational support, not observations.',
            depth_convention='Positive down; x_m/y_m are original project coordinates in metres.')
        (staging / 'model_parameters.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def export_seismic(output, result, figure, display):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / 'seismic_sections.png', dpi=170, facecolor='white')
    descriptions = []
    for section in result['sections']:
        arrays = {key: value for key, value in section.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(output / f"{section['axis']}z_seismic.npz", **arrays)
        descriptions.append({key: value for key, value in section.items() if not isinstance(value, np.ndarray)})
    report = dict(result['metadata'], display=display, sections=descriptions,
                  array_order='horizontal, depth; depth positive down; seismic amplitudes are relative')
    (output / 'seismic_parameters.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return output
