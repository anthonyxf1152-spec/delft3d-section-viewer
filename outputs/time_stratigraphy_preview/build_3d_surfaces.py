"""Cache native bed-top histories and inferred surviving 3D stage interfaces.

Read-only NEFIS input; metres, original positive-up elevation and nonuniform
Cartesian native XY axes are retained. No rock physics or composition is made.
For a selected final record k, reconstruct from bed_top[:k+1], not from the
final-run surviving stack, because later erosion must not erase an earlier view.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs' / 'delft3d_viewer'))
from nefis_reader import NefisReader

HERE = Path(__file__).resolve().parent


def _stat(path):
    s = path.stat()
    return {'size_bytes': int(s.st_size), 'mtime_ns': int(s.st_mtime_ns)}


def _text(value):
    return bytes(value).decode('latin1').strip(' \x00')


def native_axes(reader):
    coord = _text(reader.read('map-const', 'COORDINATES').flat[0])
    if coord.upper() != 'CARTESIAN':
        raise ValueError(f'Expected Cartesian metre coordinates, found {coord!r}.')
    valid = reader.read('map-const', 'KCS') == 1
    xz, yz = reader.read('map-const', 'XZ'), reader.read('map-const', 'YZ')
    if valid.ndim != 2 or xz.shape != valid.shape or yz.shape != valid.shape or not valid.any():
        raise ValueError('Invalid native coordinate/mask dimensions.')
    if not np.isfinite(xz[valid]).all() or not np.isfinite(yz[valid]).all():
        raise ValueError('Nonfinite active native coordinates.')
    mi, ni = np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0))
    x = np.array([np.median(xz[m, valid[m]]) for m in mi])
    y = np.array([np.median(yz[valid[:, n], n]) for n in ni])
    ix, iy = np.argsort(x), np.argsort(y)
    x, y, mi, ni = x[ix], y[iy], mi[ix], ni[iy]
    if min(len(x), len(y)) < 2 or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError('Native XY axes must be increasing and unique.')
    mask = valid[np.ix_(mi, ni)]
    expected_x = np.broadcast_to(x[:, None], mask.shape)
    expected_y = np.broadcast_to(y[None, :], mask.shape)
    if not (np.allclose(xz[np.ix_(mi, ni)][mask], expected_x[mask], atol=1e-4, rtol=0)
            and np.allclose(yz[np.ix_(mi, ni)][mask], expected_y[mask], atol=1e-4, rtol=0)):
        raise ValueError('Native grid is curvilinear/rotated, not axis-aligned Cartesian rectilinear.')
    return x, y, mi.astype('i4'), ni.astype('i4'), mask


def surviving_until(raw_tops, final_record=None):
    """Infer interfaces only through the selected zero-based final record."""
    if final_record is None:
        final_record = len(raw_tops) - 1
    if not isinstance(final_record, (int, np.integer)) or not 0 <= final_record < len(raw_tops):
        raise ValueError('final_record must be an available zero-based record index.')
    selected = np.asarray(raw_tops[:int(final_record) + 1])
    return np.minimum.accumulate(selected[::-1], axis=0)[::-1].copy()


def compare_six_sections(raw, surviving, mi, ni, valid, hydro, morph, section_dir):
    m_lookup = {int(v): i for i, v in enumerate(mi)}
    n_lookup = {int(v): i for i, v in enumerate(ni)}
    checks = []
    for zone in ('inner', 'middle', 'outer'):
        for direction in ('x', 'y'):
            path = section_dir / f'{zone}_{direction}_geometry.npz'
            if not path.is_file():
                raise ValueError(f'Missing required six-section QA file: {path}')
            with np.load(path) as s:
                sm = np.array([m_lookup[int(v)] for v in s['native_M_indices']])
                sn = np.array([n_lookup[int(v)] for v in s['native_N_indices']])
                records = s['records_zero_based']
                nt = len(records)
                if nt > len(raw) or not np.array_equal(records, np.arange(nt)):
                    raise ValueError(f'{path.name}: records are not an available contiguous prefix.')
                if not np.array_equal(hydro[:nt], s['hydro_days']) or not np.array_equal(morph[:nt], s['morph_days']):
                    raise ValueError(f'{path.name}: cached section times differ.')
                if not np.array_equal(valid[sm, sn], s['valid']):
                    raise ValueError(f'{path.name}: cached section validity differs.')
                source_tops = raw[:nt, sm, sn]
                source_surviving = (surviving[:, sm, sn] if nt == len(raw)
                                    else surviving_until(source_tops))
                expected = s['inferred_surviving_interfaces_elevation_m']
                if not np.array_equal(source_tops, s['bed_top_elevation_m'], equal_nan=True):
                    raise AssertionError(f'{path.name}: raw bed tops do not exactly match.')
                if not np.array_equal(source_surviving, expected, equal_nan=True):
                    raise AssertionError(f'{path.name}: inferred interfaces do not exactly match.')
                finite = np.isfinite(expected)
                checks.append({'section': path.name, 'records': nt,
                               'native_columns': len(sm), 'raw_tops_exact': True,
                               'surviving_interfaces_exact': True,
                               'maximum_interface_error_m': float(np.max(abs(source_surviving[finite] - expected[finite]))),
                               'finite_interface_values_compared': int(finite.sum())})
    return checks


def build_cache(dat_file, output=HERE/'volume', section_dir=None, def_file=None, progress=None):
    started = time.monotonic()
    dat_file, output = Path(dat_file).resolve(), Path(output).resolve()
    section_dir = Path(section_dir).resolve() if section_dir is not None else None
    def_file = Path(def_file).resolve() if def_file is not None else dat_file.with_suffix('.def')
    stat_before = {'dat': _stat(dat_file), 'def': _stat(def_file)}
    with NefisReader(dat_file, def_file) as reader:
        x, y, mi, ni, valid = native_axes(reader)
        last = reader.latest_complete_record('map-sed-series', 'map-info-series', 'map-infsed-serie')
        if last < 0:
            raise ValueError('At least one synchronized record is required.')
        records = np.arange(last+1, dtype='i4')
        raw = np.full((len(records), len(x), len(y)), np.nan, dtype='f4')
        hydro, morph = np.empty(len(records)), np.empty(len(records))
        steps = np.empty(len(records), dtype='i8')
        dt_days = reader.scalar('map-const', 'DT') * reader.scalar('map-const', 'TUNIT') / 86400.
        base_min = np.full(valid.shape, np.inf)
        base_max = np.full(valid.shape, -np.inf)
        original_shape = list(reader.elements['DPS'].shape)
        stored_layers = int(reader.elements['DP_BEDLYR'].shape[0]-1)
        if stored_layers != 1:
            raise ValueError('当前时间层序方法支持一个保存的混合床层；多床层输出需要独立的原生地层读取流程。')
        for record in records:
            i = int(record)
            step = reader.scalar('map-info-series', 'ITMAPC', i)
            sedstep = reader.scalar('map-infsed-serie', 'ITMAPS', i)
            mt = reader.scalar('map-infsed-serie', 'MORFT', i)
            if step != sedstep or (i > 0 and step <= 0) or not np.isfinite(mt):
                raise ValueError(f'Record {i} has incomplete or unsynchronized time markers.')
            dps = reader.read('map-sed-series', 'DPS', i)
            cumulative = reader.read('map-sed-series', 'DP_BEDLYR', i)
            if dps.shape != tuple(original_shape) or cumulative.shape[1:] != dps.shape:
                raise ValueError(f'Record {i} has inconsistent bed array dimensions.')
            top = -dps[np.ix_(mi, ni)]
            depths = cumulative[:, mi, :][:, :, ni]
            if not np.isfinite(top[valid]).all() or not np.isfinite(depths[:, valid]).all():
                raise ValueError(f'Record {i} has nonfinite active bed geometry.')
            if np.max(abs(depths[0, valid])) > 1e-5 or np.any(np.diff(depths, axis=0)[:, valid] < -1e-5):
                raise ValueError(f'Record {i}: DP_BEDLYR is not cumulative depth below bed.')
            # Keep the same double-precision subtraction as the six section
            # reconstruction; the source bed top itself is exactly float32.
            base = top.astype('f8') - depths[-1].astype('f8')
            raw[i] = np.where(valid, top, np.nan)
            if i == 0:
                initial_base = np.where(valid, base, np.nan)
            if i == last:
                final_base = np.where(valid, base, np.nan)
                final_top = raw[i].copy()
            base_min[valid] = np.minimum(base_min[valid], base[valid])
            base_max[valid] = np.maximum(base_max[valid], base[valid])
            # Exact second reads catch changed/preallocated records. Together
            # with source stat checks this is consistency evidence, not an
            # atomic NEFIS transaction guarantee for a live writer.
            if not np.array_equal(dps, reader.read('map-sed-series', 'DPS', i), equal_nan=True):
                raise ValueError(f'Record {i}: DPS changed while reading.')
            if not np.array_equal(cumulative, reader.read('map-sed-series', 'DP_BEDLYR', i), equal_nan=True):
                raise ValueError(f'Record {i}: DP_BEDLYR changed while reading.')
            repeated = (reader.scalar('map-info-series', 'ITMAPC', i),
                        reader.scalar('map-infsed-serie', 'ITMAPS', i),
                        reader.scalar('map-infsed-serie', 'MORFT', i))
            if repeated != (step, sedstep, mt):
                raise ValueError(f'Record {i}: time markers changed while reading.')
            steps[i], hydro[i], morph[i] = step, step*dt_days, mt
            if i % 40 == 0 or i == last:
                message = f'床面缓存：读取并复核记录 {i + 1}/{last + 1}'
                if progress is not None:
                    progress(message)
                else:
                    print(message, flush=True)
    stat_after = {'dat': _stat(dat_file), 'def': _stat(def_file)}
    if stat_before != stat_after:
        raise ValueError('NEFIS source size/mtime changed; wait for completed output and rebuild.')
    if not np.isfinite(hydro).all() or np.any(np.diff(hydro) <= 0) or np.any(np.diff(morph) < 0):
        raise ValueError('Output times must be finite and increasing (morphology may repeat during spin-up).')
    surviving = surviving_until(raw)
    delta = np.diff(surviving, axis=0)[:, valid]
    if np.any(delta < 0):
        raise AssertionError('Surviving interfaces cross in an active column.')
    if not np.array_equal(surviving[-1], final_top, equal_nan=True):
        raise AssertionError('Final surviving interface differs from the final raw bed top.')
    if not np.isnan(raw[:, ~valid]).all() or not np.isnan(surviving[:, ~valid]).all():
        raise AssertionError('Inactive/unmeasured columns must remain NaN.')
    if np.any(surviving[:, valid] > raw[:, valid]):
        raise AssertionError('An erosion-truncated interface exceeds its original saved bed top.')
    base_variation = base_max[valid] - base_min[valid]
    base_error = float(base_variation.max())
    if base_error > 1e-4:
        raise ValueError(f'Source bed base changes materially ({base_error} m); review initial substrate interpretation.')
    checks = (compare_six_sections(raw, surviving, mi, ni, valid, hydro, morph, section_dir)
              if section_dir is not None else [])
    metadata = {
        'build_complete': True, 'dat_file': str(dat_file), 'def_file': str(def_file),
        'source_stat_before': stat_before, 'source_stat_after': stat_after,
        'source_size_mtime_unchanged': True,
        'read_consistency': 'Every DPS/DP_BEDLYR record and all time markers passed exact repeated reads. NEFIS has no atomic cross-element commit marker.',
        'record_count': len(records), 'records_zero_based': records.tolist(),
        'shape_time_x_y': list(raw.shape), 'axis_order': 'time,X,Y; elevation positive up in metres',
        'original_nefis_shape_m_n_including_ghosts': original_shape,
        'native_active_axis_shape_x_y': list(valid.shape), 'active_KCS1_columns': int(valid.sum()),
        'stored_bed_layers': stored_layers,
        'native_index_basis': 'Original zero-based M,N indices including source ghost rows/columns',
        'x_spacing_unique_m': np.unique(np.round(np.diff(x), 7)).tolist(),
        'y_spacing_unique_m': np.unique(np.round(np.diff(y), 7)).tolist(),
        'hydrodynamic_days_range': [float(hydro[0]), float(hydro[-1])],
        'morphological_days_range': [float(morph[0]), float(morph[-1])],
        'hydrodynamic_interval_hours_unique': np.unique(np.round(np.diff(hydro)*24, 8)).tolist(),
        'method': 'b_t=-DPS_t; S_t=min(b_k for k>=t) through the selected final record; nonnegative surviving time intervals are S_t-S_(t-1).',
        'arbitrary_final_time_method': 'For selected final record k use raw=bed_top_elevation_m.npy[:k+1]; S=minimum.accumulate(raw[::-1],axis=0)[::-1]. Never truncate the final-run S cache to show an earlier time.',
        'base_formula': 'z_base=-DPS-DP_BEDLYR[-1]; initial and final actual bases are retained without assuming 4.5 m everywhere.',
        'qa': {'noncrossing_all_active_columns': True, 'minimum_interval_thickness_m': float(delta.min()) if delta.size else 0.,
               'final_top_exact': True, 'final_top_max_error_m': 0., 'inactive_columns_all_NaN': True,
               'surviving_interfaces_never_above_original_bed_tops': True,
               'fixed_base_max_all_time_range_m': base_error,
               'final_minus_initial_base_range_m': [float(np.min((final_base-initial_base)[valid])), float(np.max((final_base-initial_base)[valid]))],
               'six_section_comparisons': checks},
        'files': {'native_geometry': 'native_geometry.npz',
                  'raw_bed_top': {'path': 'bed_top_elevation_m.npy', 'dtype': 'float32', 'shape': list(raw.shape)},
                  'final_run_surviving_interfaces': {'path': 'surviving_interfaces_elevation_m.npy', 'dtype': 'float32', 'shape': list(surviving.shape)}},
        'interpretation': 'Time-series-inferred surviving geometric stages; not native stored multi-bed stratigraphy, exact formation ages, recovered lithology or a Vp/rho model.',
        'composition_assigned': False, 'rock_physics_or_seismic_generated': False,
        'vertical_geometry_transformation': 'None: original metre elevations and thicknesses; display exaggeration belongs to the viewer only.',
        'limitations': ['Processes between saved stages are unresolved.', 'Later erosion truncates old stages only when the selected viewing time includes that erosion.', 'NaN outside KCS=1 is unmeasured/inactive, not a zero surface.'],
        'elapsed_seconds': float(time.monotonic()-started),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/'bed_top_elevation_m.npy', raw)
    np.save(output/'surviving_interfaces_elevation_m.npy', surviving)
    np.savez_compressed(output/'native_geometry.npz', x_m=x, y_m=y,
        native_m_indices=mi, native_n_indices=ni, valid=valid,
        records_zero_based=records, hydrodynamic_steps=steps,
        hydro_days=hydro, morph_days=morph,
        initial_base_elevation_m=initial_base, final_base_elevation_m=final_base,
        initial_bed_top_elevation_m=raw[0], final_top_elevation_m=final_top)
    # Metadata is written last, after cache and QA have completed.
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    message = f'床面缓存完成：{raw.shape}；原始坐标、时间轴、层面与数据一致性检查通过。'
    if progress is not None:
        progress(message)
    else:
        print(message, flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dat', type=Path, required=True)
    parser.add_argument('--def', dest='def_file', type=Path)
    parser.add_argument('--output', type=Path, default=HERE/'volume')
    parser.add_argument('--section-dir', type=Path, help='可选：与已有六条剖面文件做额外回归核对。')
    args = parser.parse_args()
    build_cache(args.dat, args.output, args.section_dir, args.def_file)


if __name__ == '__main__':
    main()
