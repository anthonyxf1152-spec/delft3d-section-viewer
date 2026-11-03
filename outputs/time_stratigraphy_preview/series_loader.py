"""Join verified continuation runs without modifying NEFIS source files."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import uuid

import numpy as np

import project_loader as single
from stratigraphy_model import StratigraphyModel

SERIES_VERSION = 1
ARRAYS = ('bed_top_elevation_m.npy', 'sand_equivalent_thickness_m.npy',
          'mud_equivalent_thickness_m.npy')


def close_model(model):
    for name in ('raw', '_sand_equivalent', '_mud_equivalent'):
        value = getattr(model, name, None)
        mapping = getattr(value, '_mmap', None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def inspect_series(sources):
    if len(sources) < 2:
        raise ValueError('合并至少需要两组 DAT / DEF。')
    infos = [single.inspect_source(**source) for source in sources]
    paths = [info['dat_file'] for info in infos]
    if len(set(paths)) != len(paths):
        raise ValueError('同一个 DAT 文件不能重复加入。')
    if any(info['sediment_names'] != infos[0]['sediment_names'] for info in infos[1:]):
        raise ValueError('各期泥沙组分名称或顺序不同，不能使用同一砂泥配置合并。')
    return infos


def _max_error(a, b, valid):
    return float(np.max(np.abs(np.asarray(a, dtype='f8')[valid] -
                               np.asarray(b, dtype='f8')[valid])))


def plan_series(models, time_mode='absolute'):
    """Return [source, local record, hydro day, morph day] and seam audit.

    Absolute: a later run replaces the older tail only at an equal-time,
    equal-state anchor. Reset: the next initial state must equal the old end.
    Unobserved absolute-time gaps retain their real duration.
    """
    if len(models) < 2 or time_mode not in ('absolute', 'reset'):
        raise ValueError('至少选择两期；时间模式必须为累计时间或各期从零计时。')
    first = models[0]
    for model in models:
        model._load_sediment_inventory()
        if (not np.array_equal(first.x, model.x) or not np.array_equal(first.y, model.y)
                or not np.array_equal(first.valid, model.valid)
                or not np.array_equal(first.native_m_indices, model.native_m_indices)
                or not np.array_equal(first.native_n_indices, model.native_n_indices)):
            raise ValueError('各期网格坐标、有效单元或原生网格索引不一致，不能直接合并。')
        if _max_error(first.initial_base, model.initial_base, first.valid) > 1e-4:
            raise ValueError('各期床层底界不一致，可能只传递了地形而重置了底床库存；不能直接合并含砂率。')
    rows = [(0, i, float(h), float(m)) for i, (h, m) in
            enumerate(zip(first.hydro_days, first.morph_days))]
    seams = []
    for source_index, model in enumerate(models[1:], 1):
        hydro = model.hydro_days.copy()
        morph = model.morph_days.copy()
        shift_h = shift_m = 0.
        if time_mode == 'reset':
            shift_h, shift_m = rows[-1][2] - hydro[0], rows[-1][3] - morph[0]
            hydro += shift_h
            morph += shift_m
        start = float(hydro[0])
        audit = {'source_index': source_index, 'source_name': Path(model.source_info.get('dat_file', model.volume_dir)).name,
                 'hydro_shift_days': float(shift_h), 'morph_shift_days': float(shift_m),
                 'discarded_previous_tail_records': 0, 'duplicate_anchor_removed': False}
        if start <= rows[-1][2] + 1e-9:
            hits = [i for i, row in enumerate(rows) if abs(row[2] - start) <= 1e-9]
            if not hits or start <= rows[0][2] + 1e-9:
                raise ValueError(f'第{source_index + 1}期起始时间没有可用接续锚点；请检查期次顺序，或选择各期从零计时。')
            anchor = hits[-1]
            old_source, old_record, _, old_morph = rows[anchor]
            old = models[old_source]
            errors = {name: _max_error(getattr(old, attr)[old_record], getattr(model, attr)[0], first.valid)
                      for name, attr in [('bed_m', 'raw'), ('sand_equivalent_m', '_sand_equivalent'),
                                         ('mud_equivalent_m', '_mud_equivalent')]}
            audit['anchor_max_errors'] = errors
            if max(errors.values()) > 1e-4 or abs(old_morph - morph[0]) > 1e-5:
                raise ValueError(f'第{source_index + 1}期接续锚点的床面、砂泥库存或形态时间不一致：{errors}。请检查是否为同一条接续模拟。')
            audit.update(discarded_previous_tail_records=len(rows) - anchor - 1,
                         duplicate_anchor_removed=True, anchor_hydro_days=start)
            rows = rows[:anchor]  # New run owns the verified shared anchor.
        else:
            if morph[0] < rows[-1][3] - 1e-5:
                raise ValueError('下一期的形态时间倒退，请检查期次顺序和时间模式。')
            interval = {
                'previous_hydro_days': rows[-1][2], 'next_hydro_days': start,
                'duration_hydro_days': start - rows[-1][2],
                'previous_morph_days': rows[-1][3], 'next_morph_days': float(morph[0]),
                'meaning': 'No intermediate records supplied; no interpolation or restart-state continuity claim.'}
            audit['boundary_interval'] = interval
            cadence = []
            if len(rows) > 1:
                cadence.append(rows[-1][2] - rows[-2][2])
            if len(hydro) > 1:
                cadence.append(float(hydro[1] - hydro[0]))
            if not cadence or interval['duration_hydro_days'] > max(cadence) * 1.5:
                audit['unobserved_interval'] = interval
        rows.extend((source_index, i, float(h), float(m)) for i, (h, m) in enumerate(zip(hydro, morph)))
        seams.append(audit)
    times = np.array([[row[2], row[3]] for row in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times[:, 0]) <= 0) or np.any(np.diff(times[:, 1]) < -1e-5):
        raise ValueError('合并后的时间轴不单调，请检查期次顺序。')
    # Tolerated roundoff must not produce a decreasing model time axis.
    times[:, 1] = np.maximum.accumulate(times[:, 1])
    rows = [(s, r, h, float(times[i, 1])) for i, (s, r, h, _) in enumerate(rows)]
    return rows, seams


def _build(output, models, rows, info, emit):
    first, last = models[0], models[rows[-1][0]]
    shape = (len(rows), len(first.x), len(first.y))
    hydro = np.array([row[2] for row in rows])
    morph = np.array([row[3] for row in rows])
    for filename, attr in zip(ARRAYS, ('raw', '_sand_equivalent', '_mud_equivalent')):
        dest = np.lib.format.open_memmap(output / filename, mode='w+', dtype='f4', shape=shape)
        try:
            for i, (source, record, _, _) in enumerate(rows):
                dest[i] = getattr(models[source], attr)[record]
                if i % 40 == 0 or i == len(rows) - 1:
                    emit(f'合并 {filename}：{i + 1}/{len(rows)}')
            dest.flush()
        finally:
            dest._mmap.close()
    raw = np.load(output / ARRAYS[0], mmap_mode='r')
    surviving = np.lib.format.open_memmap(output / 'surviving_interfaces_elevation_m.npy',
                                         mode='w+', dtype='f4', shape=shape)
    try:
        current = np.array(raw[-1])
        for i in range(len(rows) - 1, -1, -1):
            np.minimum(current, raw[i], out=current)
            surviving[i] = current
        surviving.flush()
        with np.load(last.volume_dir / 'native_geometry.npz') as g:
            final_base = g['final_base_elevation_m']
        np.savez_compressed(output / 'native_geometry.npz', x_m=first.x, y_m=first.y,
            valid=first.valid, native_m_indices=first.native_m_indices, native_n_indices=first.native_n_indices,
            records_zero_based=np.arange(len(rows), dtype='i4'), hydro_days=hydro, morph_days=morph,
            source_indices=np.array([row[0] for row in rows], dtype='i4'),
            source_records_zero_based=np.array([row[1] for row in rows], dtype='i4'),
            initial_base_elevation_m=first.initial_base, final_base_elevation_m=final_base,
            initial_bed_top_elevation_m=raw[0], final_top_elevation_m=raw[-1])
    finally:
        raw._mmap.close()
        surviving._mmap.close()
    bed = json.loads((first.volume_dir / 'metadata.json').read_text(encoding='utf-8'))
    bed.update(build_complete=True, record_count=len(rows), shape_time_x_y=list(shape),
               records_zero_based=list(range(len(rows))),
               hydrodynamic_days_range=[float(hydro[0]), float(hydro[-1])],
               morphological_days_range=[float(morph[0]), float(morph[-1])])
    # Single-source stat and QA fields must not describe the combined dataset.
    for key in ('dat_file', 'def_file', 'source_stat_before', 'source_stat_after', 'source_size_mtime_unchanged',
                'qa', 'files', 'elapsed_seconds', 'hydrodynamic_interval_hours_unique'):
        bed.pop(key, None)
    bed.update(source_kind='continuation_series', series=info,
               qa={'shared_grid_and_base_verified': True, 'anchors_bed_and_inventory_verified': True,
                   'strictly_increasing_hydro_days': True, 'erosion_reconstructed_across_all_periods': True})
    sand = json.loads((first.volume_dir / 'sand_cache_metadata.json').read_text(encoding='utf-8'))
    for key in ('source', 'def_file', 'source_stat_before', 'source_stat_after'):
        sand.pop(key, None)
    sand.update(records=len(rows), checked_records=len(rows), shape_time_xy=list(shape),
                fraction_vs_LYRFRAC_max_error=max(json.loads((m.volume_dir / 'sand_cache_metadata.json').read_text(encoding='utf-8'))['fraction_vs_LYRFRAC_max_error'] for m in models),
                height_vs_DP_BEDLYR_max_error_m=max(json.loads((m.volume_dir / 'sand_cache_metadata.json').read_text(encoding='utf-8'))['height_vs_DP_BEDLYR_max_error_m'] for m in models),
                source_kind='continuation_series', files=list(ARRAYS[1:]))
    for name, value in [('metadata.json', bed), ('sand_cache_metadata.json', sand)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_series(sources, *, time_mode='absolute', sand_index=0, mud_index=1,
                sand_density=1600., mud_density=500., cache_root=None, progress=None):
    single.validate_configuration(sand_index, mud_index, sand_density, mud_density)
    emit = progress or (lambda message: print(message, flush=True))
    infos = inspect_series(sources)
    config = dict(sand_index=int(sand_index), mud_index=int(mud_index),
                  sand_density=float(sand_density), mud_density=float(mud_density))
    identity = {'cache_version': single.CACHE_VERSION, 'series_version': SERIES_VERSION,
                'time_mode': time_mode, 'overlap_policy': 'verified_anchor_then_later_run',
                'source_signatures': [i['source_signature'] for i in infos], 'configuration': config}
    root = Path(cache_root or single.DEFAULT_CACHE_ROOT).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = 'series-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    output = single._assert_cache_child(root / key, root)
    models, staging = [], None
    reused = single._cache_complete(output, identity)
    try:
        if not reused:
            for i, source in enumerate(sources):
                emit(f'读取第{i + 1}/{len(sources)}期：{Path(source["dat_file"]).name}')
                models.append(single.load_project(**source, **config, cache_root=root, progress=emit))
                if models[-1].source_info['source_signature'] != infos[i]['source_signature']:
                    raise ValueError('源文件在合并准备期间变化，请待模拟结束后重试。')
            rows, seams = plan_series(models, time_mode)
            info = dict(kind='continuation_series', records=len(rows), time_mode=time_mode, **config,
                        sediment_names=infos[0]['sediment_names'], sources=infos, seams=seams, cache_key=key,
                        record_sources=[{'source_index': s, 'source_record': r} for s, r, _, _ in rows])
            staging = single._assert_cache_child(Path(tempfile.mkdtemp(prefix='build-', dir=root)), root)
            _build(staging, models, rows, info, emit)
            descriptor = {'complete': True, 'identity': identity, 'source_info': info}
            (staging / 'project.json').write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding='utf-8')
            if not single._cache_complete(staging, identity):
                raise ValueError('合并缓存完整性检查失败，未切换当前项目。')
            for source, before in zip(sources, infos):
                if single.source_signature(**source) != before['source_signature']:
                    raise ValueError('源文件在合并期间变化，未切换当前项目。')
            if output.exists():
                if single._cache_complete(output, identity):
                    reused = True
                else:
                    output.rename(single._assert_cache_child(root / f'{key}.invalid-{uuid.uuid4().hex[:8]}', root))
            if not reused:
                staging.rename(output)
        for source, before in zip(sources, infos):
            if single.source_signature(**source) != before['source_signature']:
                raise ValueError('源文件在载入前变化，未切换当前项目。')
    finally:
        for model in models:
            close_model(model)
        if staging is not None:
            staging = single._assert_cache_child(staging, root)
            if staging.exists() and staging.name.startswith('build-'):
                shutil.rmtree(staging)
    model = StratigraphyModel(output)
    model.cache_reused = reused
    emit(f'多期次合并完成：{len(sources)}期，{model.record_count}个时刻。')
    return model
