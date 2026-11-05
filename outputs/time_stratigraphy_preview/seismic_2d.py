"""Independent horizontal-depth models and true 2-D finite-angle PSF images."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import shutil
import tempfile

import numpy as np
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq, next_fast_len
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt

from seismic_forward import (SeismicParameters, MAX_GRID_VOXELS, regional_datum,
                             _grid_window, _normal_field, reflection)
from seismic_reference import voxelize_fan, rock_model

FIELDS = ('sand_fraction', 'occupancy', 'vp_m_s', 'density_kg_m3',
          'impedance_kg_m2_s', 'porosity_delta', 'reflectivity', 'baseline_reflectivity')


def _grids(model, sections, params):
    params.validate()
    if not sections or len({s['record'] for s in sections}) != 1:
        raise ValueError('请选择同一时刻的有效 XZ / YZ 切片。')
    record = int(sections[0]['record']); model._record(record)
    if record < 1:
        raise ValueError('初始时刻没有沉积增量，请选择更晚的保存时刻。')
    if len({s['axis'] for s in sections}) != len(sections):
        raise ValueError('每个方向只能选择一条切片。')
    z = np.arange(0, params.depth_max_m + params.spacing_z_m * .5, params.spacing_z_m)
    rows = []
    for sec in sections:
        axis = sec['axis']
        if axis not in ('x', 'y') or len(sec['horizontal_m']) < 2:
            raise ValueError('请选择至少两个网格点的 XZ / YZ 切片。')
        native = model.x if axis == 'x' else model.y
        fixed_native = model.y if axis == 'x' else model.x
        fixed = float(sec['actual_fixed_m'])
        index = int(np.argmin(abs(fixed_native - fixed)))
        if not np.isclose(fixed_native[index], fixed, atol=1e-5, rtol=0):
            raise ValueError('二维建模须采用当前切片的实际原始网格坐标。')
        margin = 3000. if axis == 'x' else 4000.
        h = _grid_window(native, float(sec['horizontal_m'][0]) - margin,
                         float(sec['horizontal_m'][-1]) + margin, params.spacing_xy_m)
        select = (h >= sec['horizontal_m'][0] - 1e-6) & (h <= sec['horizontal_m'][-1] + 1e-6)
        if np.sum(select) < 2:
            raise ValueError('二维切片不足两道，请减小水平采样间距。')
        lo = max(0, int(np.searchsorted(native, h[0])) - 1)
        hi = min(len(native), int(np.searchsorted(native, h[-1])) + 2)
        rows.append({'source': sec, 'h': h, 'depth': z, 'select': select,
                     'fixed_index': index, 'native_slice': slice(lo, hi), 'margin': margin})
    return record, rows


def estimate_2d_grid(model, sections, params):
    """No physical volume allocation; exact padded and displayed 2-D grids."""
    record, grids = _grids(model, sections, params)
    count = disk = 0; model_peak = forward_peak = 0; descriptions = []
    fill_bytes = ((record + 1) * len(model.x) * len(model.y) * 24
                  if params.sand_mode == 'filled' else 0)
    for row in grids:
        nh, nz = len(row['h']), len(row['depth']); nshow = int(row['select'].sum())
        n = nh * nz; count += n
        fft_shape = [next_fast_len(nh + 32), next_fast_len(nz + 96)]
        padded = math.prod(fft_shape); spectrum = fft_shape[0] * (fft_shape[1] // 2 + 1)
        native_n = row['native_slice'].stop - row['native_slice'].start
        working = n * 128 + (record + 1) * (native_n + nh) * 64
        model_peak = max(model_peak, working)
        forward_peak = max(forward_peak, working + padded * 16 + spectrum * 64)
        # Guarded physics model plus displayed cut: 8 f4 fields, four 1-D
        # vectors/masks and a depth vector per saved grid; PNG/JSON extra.
        disk += (nh + nshow) * nz * 32 + (nh + nshow) * 25 + nz * 16
        descriptions.append({'axis': row['source']['axis'], 'shape_horizontal_depth': [nshow, nz],
            'computational_shape_horizontal_depth': [nh, nz], 'fft_shape_hz': fft_shape,
            'horizontal_bounds_m': [float(row['h'][0]), float(row['h'][-1])],
            'depth_bounds_m': [float(row['depth'][0]), float(row['depth'][-1])],
            'actual_fixed_m': float(row['source']['actual_fixed_m']), 'guard_m': row['margin']})
    # Completed cuts remain resident while the next direction is constructed.
    retained = sum(int(np.prod(s['computational_shape_horizontal_depth'])) * 32 for s in descriptions)
    return {'model_dimension': '2d', 'sections': descriptions, 'voxel_count': count,
        'single_float32_array_bytes': max(int(np.prod(s['computational_shape_horizontal_depth'])) * 4 for s in descriptions),
        'estimated_model_peak_bytes': int(model_peak + retained + fill_bytes),
        'estimated_forward_peak_bytes': int(forward_peak + retained + fill_bytes),
        'estimated_npy_payload_bytes': int(disk), 'maximum_voxels': MAX_GRID_VOXELS,
        'within_grid_limit': count <= MAX_GRID_VOXELS,
        'spacing_m_hz': [params.spacing_xy_m, params.spacing_z_m],
        'notes': ['Independent 2-D horizontal/depth models; no orthogonal physical volume is built.',
                  'Counts include horizontal guards. RAM includes native composition preparation and is approximate.',
                  'Disk is NPY payload estimate; PNGs, headers and JSON are additional.']}


def psf_image_2d(reflectivity, spacing, vref, frequency, angle0=25., angle1=40.):
    """2-D exploding-reflector transfer, with zero-phase unit-peak Ricker law.

    Apply H(kh,kz), not a 3-D image slice and not independent 1-D trace filters.
    The equivalent frequency is vref*sqrt(kh**2+kz**2)/2. The aperture tapers
    from angle0 to angle1; normalization matches the reference depth Ricker.
    """
    if reflectivity.ndim != 2 or min(reflectivity.shape) < 2:
        raise ValueError('二维 PSF 输入须为 [水平,深度] 数值数组。')
    shape = (next_fast_len(reflectivity.shape[0] + 32), next_fast_len(reflectivity.shape[1] + 96))
    slices = (slice(16, 16 + reflectivity.shape[0]), slice(48, 48 + reflectivity.shape[1]))
    work = np.zeros(shape, dtype='f4'); work[slices] = reflectivity
    spectrum = rfftn(work, workers=2); del work
    kh = fftfreq(shape[0], spacing[0]).astype('f4')[:, None]
    kz = rfftfreq(shape[1], spacing[1]).astype('f4')[None, :]
    equivalent_hz = vref / 2 * np.sqrt(kh * kh + kz * kz)
    angle = np.degrees(np.arctan2(abs(kh), np.maximum(kz, 1e-20)))
    aperture = .5 * (1 + np.cos(np.pi * np.clip((angle - angle0) / (angle1 - angle0), 0, 1)))
    u = (equivalent_hz / frequency)**2
    scale = vref / (np.sqrt(np.pi) * frequency * spacing[1] * np.e)
    transfer = (np.e * u * np.exp(-u) * aperture * scale).astype('f4')
    return irfftn(spectrum * transfer, s=shape, workers=2)[slices].copy().astype('f4')


def _porosity_2d(q, occ, top, base, h, z, xx, yy, model, axis, params):
    if axis == 'x':
        noise = _normal_field(h, np.array([yy[0]]), z, params.spacing_xy_m,
                              (model.x[0], model.y[0]), params.seed + 13)[:, 0, :]
    else:
        noise = _normal_field(np.array([xx[0]]), h, z, params.spacing_xy_m,
                              (model.x[0], model.y[0]), params.seed + 13)[0, :, :]
    correlation = 500. if axis == 'x' else 750.
    sigma = [correlation / params.spacing_xy_m, 10. / params.spacing_z_m]
    gaussian_filter(noise, sigma=sigma, output=noise, mode='reflect', truncate=4.)
    variance = 1.
    for width in sigma:
        radius = int(4 * width + .5)
        kernel = np.exp(-.5 * (np.arange(-radius, radius + 1) / width)**2)
        kernel /= kernel.sum(); variance *= float(np.sum(kernel * kernel))
    noise /= max(np.sqrt(variance), 1e-8); np.clip(noise, -2.5, 2.5, out=noise)
    rng = np.random.default_rng(params.seed + 13); edges = [0.]
    while edges[-1] < 1.1:
        edges.append(edges[-1] + float(rng.uniform(.022, .055)))
    levels = gaussian_filter1d(rng.normal(size=len(edges) + 1), .35)
    levels = np.clip(levels / max(levels.std(), 1e-8), -2, 2).astype('f4')
    xr, yr = xx - model.x[0], yy - model.y[0]
    warp = .015 * np.sin(2 * np.pi * xr / 2800 + .8 * np.cos(2 * np.pi * yr / 4500))
    warp += .010 * np.cos(2 * np.pi * (xr / 5000 + yr / 3300))
    modulation = .8 + .2 * np.sin(2 * np.pi * xr / 4300) * np.cos(2 * np.pi * yr / 6200)
    u = (z[None, :] - top[:, None]) / np.maximum(base - top, 5.)[:, None]
    lamina = levels[np.searchsorted(edges, u + warp[:, None])] * modulation[:, None]
    phi = (.011 * lamina + .011 * noise) * np.sqrt(q) * occ
    phi *= np.clip(np.minimum(u, 1 - u) / .025, 0, 1)
    return (np.clip(phi, -.025, .025) * params.porosity_strength).astype('f4'), {
        'dimension': 'horizontal,depth', 'delta_phi_bound': .025 * params.porosity_strength,
        'correlation_lengths_m_hz': [correlation, 10.], 'normalized_bed_edges': edges,
        'seed': params.seed + 13, 'realization': '2-D correlated field; not a slice of the 3-D stochastic field'}


def build_2d_model(model, sections, params, progress=None, filled_hint=None):
    record, grids = _grids(model, sections, params)
    estimate = estimate_2d_grid(model, sections, params)
    if not estimate['within_grid_limit']:
        raise ValueError('二维计算网格过大，请增大采样间距或缩小剖面范围。')
    emit = progress or (lambda _: None)
    coefficients = regional_datum(model)
    fractions = provenance = None
    if params.sand_mode == 'filled':
        emit('读取当前时刻的含砂率输入…')
        if filled_hint is None:
            from sand_completion import build_completed_volume
            fractions, provenance = build_completed_volume(model, record)
        else:
            fractions, provenance = filled_hint
        expected = (record + 1, len(model.x), len(model.y))
        if fractions.shape != expected or provenance.shape != expected:
            raise ValueError('含砂率快照与当前时刻不一致。')
    else:
        model._load_sediment_inventory()
    computed, displayed, scenarios = [], [], []
    for grid in grids:
        sec, h, z = grid['source'], grid['h'], grid['depth']
        axis, fixed = sec['axis'], float(sec['actual_fixed_m'])
        emit(f'生成 {axis.upper()}Z 二维模型：{len(h)} × {len(z)}（含水平保护区）…')
        sl, fi = grid['native_slice'], grid['fixed_index']
        indices = (sl, fi) if axis == 'x' else (fi, sl)
        native = (model.x if axis == 'x' else model.y)[sl]
        valid = model.valid[indices]
        if not valid.any():
            raise ValueError('当前二维切片范围没有有效模型单元。')
        history = np.asarray(model.raw[(slice(0, record + 1),) + indices])
        interfaces = np.minimum.accumulate(history[::-1], axis=0)[::-1]
        native_z = interfaces[::-1].T.copy()
        native_h = np.maximum(0, -np.diff(native_z, axis=-1))
        if fractions is not None:
            q_native = fractions[(slice(1, record + 1),) + indices][::-1].T.copy()
        else:
            sand = np.asarray(model._sand_equivalent[(record,) + indices], dtype='f8')
            mud = np.asarray(model._mud_equivalent[(record,) + indices], dtype='f8')
            ratio = np.divide(sand, sand + mud, out=np.full_like(sand, np.nan), where=valid & (sand + mud > 1e-7))
            q_native = np.broadcast_to(ratio[:, None], native_h.shape).copy()
        if np.any(valid[:, None] & (native_h >= .001) & ~np.isfinite(q_native)):
            raise ValueError('二维模型存在有厚度但含砂率未知的层段，请使用分期补全。')
        if np.any((q_native[np.isfinite(q_native)] < 0) | (q_native[np.isfinite(q_native)] > 1)):
            raise ValueError('含砂率必须在 0～1 之间。')
        q_native = np.nan_to_num(q_native, nan=0.)
        if not valid.all():
            nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)[0]
            native_z[~valid] = native_z[nearest[~valid]]
            native_h[~valid] = 0.; q_native[~valid] = 0.
        interp = lambda values: interp1d(native, values, axis=0, bounds_error=True)(h).astype('f4')
        zi = interp(native_z)
        thickness = np.maximum(0, -np.diff(zi, axis=1))
        sand_volume = interp(q_native * native_h)
        layers = np.divide(sand_volume, thickness, out=np.zeros_like(sand_volume), where=thickness > 1e-7)
        np.clip(layers, 0, 1, out=layers)
        good = interp(valid.astype('f4')) >= .999
        xx, yy = (h, np.full_like(h, fixed)) if axis == 'x' else (np.full_like(h, fixed), h)
        datum = coefficients[0] + coefficients[1] * xx + coefficients[2] * yy
        depths = params.burial_depth_m - params.vertical_scale * (zi - datum[:, None])
        active = depths[good]
        if not active.size or np.min(active) < 30 or np.max(active) > z[-1] - 50:
            raise ValueError('当前层厚倍率／埋深使扇体超出深度范围，请调整参数。')
        zi[~good] = datum[~good, None]
        # Singleton helper dimension is never expanded into an orthogonal ROI.
        q3, occ3, top2, base2, _ = voxelize_fan(zi, layers, h, np.array([fixed]), z,
            params.vertical_scale, 'original_relief', 0, datum_elevation=datum[:, None], burial_depth=params.burial_depth_m)
        q, occ = q3[:, 0, :], occ3[:, 0, :]
        top, base = top2[:, 0], base2[:, 0]
        q[~good] = 0.; occ[~good] = 0.
        vp3, rho3, _, background = rock_model(q[:, None, :], (xx - model.x[0])[:, None],
                                             (yy - model.y[0])[:, None], z, params.seed)
        vp, rho = vp3[:, 0, :], rho3[:, 0, :]
        baseline_r = reflection(vp, rho)
        phi, phi_meta = _porosity_2d(q, occ, top, base, h, z, xx, yy, model, axis, params)
        vp -= 6000 * phi; rho -= 1650 * phi
        if not np.isfinite(vp).all() or not np.isfinite(rho).all() or np.min(vp) <= 0 or np.min(rho) <= 0:
            raise ValueError('二维岩石参数无效，请检查参数。')
        row = {'axis': axis, 'record': record, 'actual_fixed_m': fixed,
            'requested_fixed_m': float(sec['requested_fixed_m']), 'horizontal_m': h, 'depth_m': z,
            'valid': good, 'fan_top_depth_m': top.astype('f4'), 'fan_base_depth_m': base.astype('f4'),
            'sand_fraction': q, 'occupancy': occ, 'vp_m_s': vp, 'density_kg_m3': rho,
            'impedance_kg_m2_s': vp * rho, 'porosity_delta': phi,
            'reflectivity': reflection(vp, rho), 'baseline_reflectivity': baseline_r}
        computed.append(row)
        displayed.append(_crop(row, grid['select']))
        scenarios.append(dict(axis=axis, porosity=phi_meta, background_layer_edges_m=background.tolist()))
    metadata = {'complete': True, 'product': 'seismic_input_model', 'convolution_performed': False,
        'model_dimension': '2d', 'parameters': asdict(params), 'grid_estimate': estimate,
        'selected_record': record, 'selected_hydro_days': float(model.hydro_days[record]),
        'selected_morph_days': float(model.morph_days[record]), 'source_project': model.source_info,
        'source_cache': str(model.volume_dir), 'method': 'Independent 2-D physical property sections; no orthogonal physics volume',
        'geometry': {'formula': 'depth = burial_depth - vertical_scale * (elevation - initial_regional_plane)',
                     'plane_coefficients_a_b_c': coefficients.tolist(), 'depth_positive_down': True,
                     'initial_substrate': 'Excluded', 'final_top_flattened': False},
        'porosity_scenarios': scenarios, 'full_wave_equation': False, 'agc': False,
        'aperture_degrees': [25, 40], 'sand_fraction_changed_by_heterogeneity': False,
        'baseline_law': 'Vp=2200+.35*z+400*q+dv_layer*(1-q); rho=2050+.12*z+160*q+dr_layer*(1-q)',
        'heterogeneity_law': 'Vp_new=Vp_base-6000*delta_phi; rho_new=rho_base-1650*delta_phi',
        'limitations': ['2-D PSF has no out-of-plane propagation/response.',
                       'Each cut has its own 2-D correlated porosity realization; intersections need not match.',
                       'Uncalibrated research rock properties and inferred layer compositions.'],
        'source_original_files_modified': False}
    return {'sections': displayed, 'computational_sections': computed, 'metadata': metadata}


def _crop(row, select):
    result = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            result[key] = value.copy() if key == 'depth_m' else value[select].copy()
        else:
            result[key] = value
    return result


def forward_2d(model, sections, params, progress=None, filled_hint=None):
    emit = progress or (lambda _: None)
    prepared = build_2d_model(model, sections, params, emit, filled_hint)
    result = []
    for full, display in zip(prepared['computational_sections'], prepared['sections']):
        emit(f"{full['axis'].upper()}Z：二维 PSF 褶积…")
        select = (full['horizontal_m'] >= display['horizontal_m'][0] - 1e-6) & (full['horizontal_m'] <= display['horizontal_m'][-1] + 1e-6)
        cut = dict(display)
        for source, target in [('baseline_reflectivity', 'seismic_baseline'), ('reflectivity', 'seismic')]:
            image = psf_image_2d(full[source], [params.spacing_xy_m, params.spacing_z_m],
                                 params.reference_velocity_m_s, params.frequency_hz)
            cut[target] = image[select].copy()
            if not np.isfinite(cut[target]).all():
                raise ValueError('二维 PSF 结果存在非有限数值。')
        result.append(cut)
    metadata = dict(prepared['metadata'], product='seismic_sections', convolution_performed=True,
        method='2-D horizontal-depth finite-band finite-angle image-domain PSF convolution of impedance reflectivity')
    return {'sections': result, 'metadata': metadata}


def export_2d_model(output, prepared, progress=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'模型保存目录已存在：{output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < prepared['metadata']['grid_estimate']['estimated_npy_payload_bytes'] + 16 * 1024**2:
        raise OSError('模型保存位置剩余空间不足。')
    emit = progress or (lambda _: None)
    staging = Path(tempfile.mkdtemp(prefix='.model-writing-', dir=output.parent))
    descriptions, arrays = [], {}
    try:
        for rows, suffix in [(prepared['sections'], ''), (prepared['computational_sections'], '/computational_grid')]:
            for cut in rows:
                prefix = cut['axis'] + 'z' + suffix
                emit(f'保存二维模型：{prefix}…')
                for name, array in cut.items():
                    if not isinstance(array, np.ndarray):
                        continue
                    relative = f'{prefix}/{name}.npy'
                    target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
                    np.save(target, array, allow_pickle=False)
                    arrays[relative] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                        'axis_order': 'horizontal,depth' if array.ndim == 2 else ('depth' if name == 'depth_m' else 'horizontal')}
                if not suffix:
                    descriptions.append({k: v for k, v in cut.items() if not isinstance(v, np.ndarray)})
        report = dict(prepared['metadata'], arrays=arrays, sections=descriptions,
            array_units={'horizontal_m': 'm, original project coordinates', 'depth_m': 'm, positive down',
                'fan_top_depth_m': 'm', 'fan_base_depth_m': 'm', 'sand_fraction': '0..1 per whole voxel',
                'occupancy': '0..1 fan occupancy', 'vp_m_s': 'm/s', 'density_kg_m3': 'kg/m3',
                'impedance_kg_m2_s': 'kg/(m2*s)', 'porosity_delta': 'absolute porosity change',
                'reflectivity': 'dimensionless', 'baseline_reflectivity': 'dimensionless'},
            invalid_columns='Use valid.npy; invalid columns are computational support, not observations.',
            computational_grid='Includes horizontal guards used by the 2-D convolution; main cut folder is the displayed window.')
        (staging / 'model_parameters.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
