"""Cache configured sand/mud inventories, verifying fractions and bed closure."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'outputs' / 'delft3d_viewer'))
from nefis_reader import NefisReader
from build_3d_surfaces import _stat


def validate_configuration(sand_index, mud_index, sand_density, mud_density):
    if (isinstance(sand_index, bool) or isinstance(mud_index, bool) or
            not isinstance(sand_index, (int, np.integer)) or not isinstance(mud_index, (int, np.integer))):
        raise ValueError('砂、泥组分索引必须为0、1。')
    if {sand_index, mud_index} != {0, 1}:
        raise ValueError('当前算法支持两个互异组分：一个砂组分、一个泥组分。')
    for value in (sand_density, mud_density):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('干床容重必须为正数，单位kg/m³，并应与模型.sed配置一致。')


def build(volume=None, *, dat_file=None, def_file=None, sand_index=0, mud_index=1,
          sand_density=1600., mud_density=500., progress=None):
    """Build in volume; caller uses a staging directory for transactional loading."""
    validate_configuration(sand_index, mud_index, sand_density, mud_density)
    volume = Path(volume or HERE / 'volume').resolve()
    with np.load(volume / 'native_geometry.npz') as source:
        mi, ni = source['native_m_indices'], source['native_n_indices']
        valid = source['valid'].astype(bool)
        hydro, morph = source['hydro_days'], source['morph_days']
    raw = np.load(volume / 'bed_top_elevation_m.npy', mmap_mode='r')
    source_meta = json.loads((volume / 'metadata.json').read_text(encoding='utf-8'))
    dat = Path(dat_file or source_meta['dat_file']).resolve()
    defn = Path(def_file or source_meta.get('def_file') or dat.with_suffix('.def')).resolve()
    before = {'dat': _stat(dat), 'def': _stat(defn)}
    if before != source_meta.get('source_stat_after'):
        raise ValueError('原始.dat/.def与床面缓存不一致，请重建该项目缓存。')
    paths = [volume / 'sand_equivalent_thickness_m.npy', volume / 'mud_equivalent_thickness_m.npy']
    report_path = volume / 'sand_cache_metadata.json'
    report_path.unlink(missing_ok=True)
    arrays = [np.lib.format.open_memmap(path, mode='w+', dtype='f4', shape=raw.shape) for path in paths]
    fraction_error, height_error = 0., 0.
    try:
        with NefisReader(dat, defn) as reader:
            if reader.latest_complete_record('map-sed-series', 'map-info-series', 'map-infsed-serie') != len(raw)-1:
                raise ValueError('缓存时间轴与原始结果不一致，请重建床面缓存。')
            names = [bytes(v).decode('latin1').strip(' \x00') for v in reader.read('map-const', 'NAMSED').flat]
            if len(names) != 2:
                raise ValueError(f'当前含砂率方法支持两个组分；文件中为{len(names)}个：{names}')
            for t in range(len(raw)):
                step = reader.scalar('map-info-series', 'ITMAPC', t)
                if step != reader.scalar('map-infsed-serie', 'ITMAPS', t) or not np.isclose(
                        float(reader.scalar('map-infsed-serie', 'MORFT', t)), morph[t], atol=1e-6):
                    raise ValueError(f'时刻{t}的床层及泥沙输出不同步。')
                mass_source = reader.read('map-sed-series', 'MSED', t)
                if mass_source.shape[0:2] != (2, 1):
                    raise ValueError('当前算法支持两个泥沙组分、一个保存床层。')
                mass = mass_source[:, 0, mi, :][:, :, ni].astype('f8')
                if not np.isfinite(mass[:, valid]).all() or (mass[:, valid] < 0).any():
                    raise ValueError(f'时刻{t}存在非有限或负的泥沙库存。')
                volumes = (mass[sand_index] / sand_density, mass[mud_index] / mud_density)
                for dest, values in zip(arrays, volumes):
                    dest[t] = np.where(valid, values, np.nan)
                depth = reader.read('map-sed-series', 'DP_BEDLYR', t)[-1, mi, :][:, ni]
                frac = reader.read('map-sed-series', 'LYRFRAC', t)[sand_index, 0, mi, :][:, ni]
                height = volumes[0] + volumes[1]
                positive = valid & (height > 1e-7)
                q = np.divide(volumes[0], height, out=np.zeros_like(height), where=positive)
                if not np.isfinite(depth[valid]).all() or not np.isfinite(frac[positive]).all():
                    raise ValueError(f'时刻{t}的床厚或体积分数存在无效值。')
                if positive.any():
                    fraction_error = max(fraction_error, float(np.max(abs((q - frac)[positive]))))
                height_error = max(height_error, float(np.max(abs((height - depth)[valid]))))
                if fraction_error > 1e-5 or height_error > 1e-4:
                    raise ValueError('配置的干床容重/砂泥组分与LYRFRAC或DP_BEDLYR不闭合。'
                                     f'请核对.sed；体积分数最大误差{fraction_error:.3g}，床厚误差{height_error:.3g}m。')
                if not np.array_equal(mass_source, reader.read('map-sed-series', 'MSED', t), equal_nan=True):
                    raise ValueError(f'时刻{t}的组分库存读取期间发生变化。')
                if progress is not None and (t % 40 == 0 or t == len(raw)-1):
                    progress(f'泥沙缓存：读取并复核记录 {t + 1}/{len(raw)}')
        after = {'dat': _stat(dat), 'def': _stat(defn)}
        if before != after:
            raise ValueError('原始.dat/.def正在变化，请完成模拟后重试。')
        for array in arrays:
            array.flush()
        report = {'complete': True, 'source': str(dat), 'def_file': str(defn), 'records': len(raw),
                  'source_stat_before': before, 'source_stat_after': after,
                  'shape_time_xy': list(raw.shape), 'sediment_names': names,
                  'component_indices_zero_based': {'sand': int(sand_index), 'mud': int(mud_index)},
                  'dry_bulk_density_kg_m3': {'sand': float(sand_density), 'mud': float(mud_density)},
                  'fraction_vs_LYRFRAC_max_error': fraction_error,
                  'height_vs_DP_BEDLYR_max_error_m': height_error,
                  'checked_records': len(raw), 'files': [p.name for p in paths],
                  'meaning': 'Single mixed-bed mass inventories converted to equivalent bed thickness; not stratigraphic layers.'}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        for array in arrays:
            array._mmap.close()
        raw._mmap.close()
    if progress is not None:
        progress('泥沙库存缓存完成；所有记录的库存、体积分数和床厚闭合检查通过。')
    else:
        print('Actual sand/mud inventory cache complete:', report, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume', type=Path, default=HERE / 'volume')
    parser.add_argument('--sand-index', type=int, default=0)
    parser.add_argument('--mud-index', type=int, default=1)
    parser.add_argument('--sand-density', type=float, default=1600.)
    parser.add_argument('--mud-density', type=float, default=500.)
    args = parser.parse_args()
    build(args.volume, sand_index=args.sand_index, mud_index=args.mud_index,
          sand_density=args.sand_density, mud_density=args.mud_density)
