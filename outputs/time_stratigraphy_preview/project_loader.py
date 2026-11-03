"""Inspect NEFIS pairs and build independent, transactional project caches."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile

import numpy as np

from build_3d_surfaces import NefisReader, _stat, native_axes, build_cache
from build_sand_cache import build as build_sand_cache, validate_configuration
from stratigraphy_model import StratigraphyModel

HERE = Path(__file__).resolve().parent
CACHE_VERSION = 2
DEFAULT_CACHE_ROOT = HERE / 'projects'


def resolve_pair(dat_file, def_file=None):
    dat = Path(dat_file).expanduser().resolve()
    defn = Path(def_file).expanduser().resolve() if def_file else dat.with_suffix('.def')
    if dat.suffix.lower() != '.dat' or defn.suffix.lower() != '.def':
        raise ValueError('请选择NEFIS的.dat数据文件及对应的.def定义文件。')
    if not dat.is_file() or not defn.is_file():
        raise FileNotFoundError(f'未找到完整.dat/.def文件对：\n{dat}\n{defn}')
    return dat, defn


def source_signature(dat_file, def_file=None):
    """Identity uses both resolved paths, size/mtime and bounded content hashes.

    This deliberately avoids hashing a multi-GB .dat on every dialog opening.
    The .def is hashed in full; .dat leading/trailing 64KiB are sampled.
    """
    dat, defn = resolve_pair(dat_file, def_file)
    result = {}
    for kind, path in (('dat', dat), ('def', defn)):
        before = _stat(path)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            if kind == 'def':
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            else:
                digest.update(handle.read(65536))
                handle.seek(max(0, before['size_bytes'] - 65536))
                digest.update(handle.read(65536))
        if _stat(path) != before:
            raise ValueError('选择的结果文件正在变化，请待模拟完成后再载入。')
        result[kind] = {'path': os.path.normcase(str(path)), **before,
                        'sha256' if kind == 'def' else 'header_tail_sha256': digest.hexdigest()}
    return result


def inspect_source(dat_file, def_file=None):
    """Return supported source geometry, sediment choices and manual defaults."""
    dat, defn = resolve_pair(dat_file, def_file)
    signature = source_signature(dat, defn)
    with NefisReader(dat, defn) as reader:
        required = {
            'map-const': ('COORDINATES', 'KCS', 'XZ', 'YZ', 'DT', 'TUNIT', 'NAMSED'),
            'map-info-series': ('ITMAPC',),
            'map-infsed-serie': ('ITMAPS', 'MORFT'),
            'map-sed-series': ('DPS', 'DP_BEDLYR', 'MSED', 'LYRFRAC'),
        }
        for group, elements in required.items():
            if group not in reader.groups:
                raise ValueError(f'所选文件缺少{group}，需要Delft3D FLOW床层泥沙map结果。')
            missing = set(elements) - set(reader.groups[group]['elements'])
            if missing:
                raise ValueError(f'{group}缺少必要变量：{", ".join(sorted(missing))}')
        x, y, mi, ni, valid = native_axes(reader)
        names = [bytes(item).decode('latin1').strip(' \x00')
                 for item in reader.read('map-const', 'NAMSED').flat]
        bed_shape = reader.elements['DP_BEDLYR'].shape
        mass_shape = reader.elements['MSED'].shape
        frac_shape = reader.elements['LYRFRAC'].shape
        if len(names) != 2 or mass_shape[:2] != (2, 1) or frac_shape[:2] != (2, 1) or bed_shape[0] != 2:
            raise ValueError('当前时间层序与含砂率算法支持两个泥沙组分、一个保存的混合床层。'
                             f'当前文件：组分{names}，MSED形状{mass_shape}，DP_BEDLYR形状{bed_shape}。')
        last = reader.latest_complete_record('map-sed-series', 'map-info-series', 'map-infsed-serie')
        for record in {0, last}:
            if reader.scalar('map-info-series', 'ITMAPC', record) != reader.scalar('map-infsed-serie', 'ITMAPS', record):
                raise ValueError(f'记录{record}的水动力与泥沙时间标记不同步。')
            # Read complete cells now to catch a mismatched/truncated pair early.
            reader.read('map-sed-series', 'DPS', record)
            reader.read('map-sed-series', 'DP_BEDLYR', record)
        sand_guess = next((i for i, name in enumerate(names) if 'sand' in name.lower() or '砂' in name), None)
        mud_guess = next((i for i, name in enumerate(names) if any(word in name.lower() for word in ('mud', 'silt', 'clay', '泥'))), None)
        if sand_guess is not None and mud_guess is None:
            mud_guess = 1 - sand_guess
        elif mud_guess is not None and sand_guess is None:
            sand_guess = 1 - mud_guess
        info = {
            'dat_file': str(dat), 'def_file': str(defn), 'sediment_names': names,
            'records': last + 1, 'grid_shape_xy': [len(x), len(y)],
            'active_columns': int(valid.sum()), 'stored_bed_layers': 1,
            'x_range_m': [float(x[0]), float(x[-1])], 'y_range_m': [float(y[0]), float(y[-1])],
            'sand_index': sand_guess, 'mud_index': mud_guess,
            'sand_density': 1600., 'mud_density': 500.,
            'density_source': 'Manual defaults; confirm dry-bed densities from the model .sed configuration.',
            'source_signature': signature,
        }
    if source_signature(dat, defn) != signature:
        raise ValueError('结果文件在检查期间发生变化，请待模拟完成后重试。')
    return info


def _assert_cache_child(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path.parent != root or path == root:
        raise ValueError('缓存目标必须是指定缓存根目录的直接子目录。')
    return path


def _cache_complete(path, identity):
    try:
        project = json.loads((path / 'project.json').read_text(encoding='utf-8'))
        bed = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        sand = json.loads((path / 'sand_cache_metadata.json').read_text(encoding='utf-8'))
        if project.get('identity') != identity or not project.get('complete'):
            return False
        if not bed.get('build_complete') or not sand.get('complete'):
            return False
        expected = tuple(bed['shape_time_x_y'])
        if list(expected) != sand['shape_time_xy'] or expected[0] != project['source_info']['records']:
            return False
        for name in ('bed_top_elevation_m.npy', 'surviving_interfaces_elevation_m.npy',
                     'sand_equivalent_thickness_m.npy', 'mud_equivalent_thickness_m.npy'):
            array = np.load(path / name, mmap_mode='r', allow_pickle=False)
            matches = array.shape == expected and array.dtype == np.dtype('f4')
            array._mmap.close()
            if not matches:
                return False
        with np.load(path / 'native_geometry.npz', allow_pickle=False) as geometry:
            if geometry['valid'].shape != expected[1:] or len(geometry['morph_days']) != expected[0]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return False


def load_project(dat_file, def_file=None, *, sand_index=0, mud_index=1,
                 sand_density=1600., mud_density=500., cache_root=None, progress=None):
    """Return a ready model, using only complete caches tied to this file/config pair.

    Failed builds stay isolated from previously loaded projects and are removed.
    Each callback receives one human-readable progress string; no GUI dependency.
    """
    validate_configuration(sand_index, mud_index, sand_density, mud_density)
    emit = progress or (lambda message: print(message, flush=True))
    emit('检查所选.dat/.def结构、坐标和时间轴…')
    info = inspect_source(dat_file, def_file)
    dat, defn = Path(info['dat_file']), Path(info['def_file'])
    config = {'sand_index': int(sand_index), 'mud_index': int(mud_index),
              'sand_density': float(sand_density), 'mud_density': float(mud_density)}
    identity = {'cache_version': CACHE_VERSION, 'source_signature': info['source_signature'], 'configuration': config}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:24]
    root = Path(cache_root or DEFAULT_CACHE_ROOT).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = _assert_cache_child(root / key, root)
    reused = _cache_complete(output, identity)
    if reused:
        emit('所选文件与参数未变化，复用已验证的独立缓存。')
    else:
        staging = _assert_cache_child(Path(tempfile.mkdtemp(prefix='build-', dir=root)), root)
        try:
            build_cache(dat, staging, def_file=defn, progress=emit)
            build_sand_cache(staging, dat_file=dat, def_file=defn, progress=emit, **config)
            if source_signature(dat, defn) != info['source_signature']:
                raise ValueError('结果文件在构建缓存期间发生变化，请待模拟完成后重试。')
            info.update(config)
            info['cache_key'] = key
            descriptor = {'complete': True, 'identity': identity, 'source_info': info}
            (staging / 'project.json').write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding='utf-8')
            if not _cache_complete(staging, identity):
                raise ValueError('新缓存完整性检查失败，未切换当前项目。')
            if output.exists():
                if _cache_complete(output, identity):
                    reused = True  # Another instance completed the same build.
                else:
                    invalid = _assert_cache_child(root / f'{key}.invalid-{uuid.uuid4().hex[:8]}', root)
                    output.rename(invalid)
            if not reused:
                staging.rename(output)
        finally:
            # Only the exact staging child created above can be recursively removed.
            staging = _assert_cache_child(staging, root)
            if staging.exists() and staging.name.startswith('build-'):
                shutil.rmtree(staging)
    descriptor = json.loads((output / 'project.json').read_text(encoding='utf-8'))
    if source_signature(dat, defn) != info['source_signature']:
        raise ValueError('结果文件在载入前发生变化，未切换当前项目。')
    model = StratigraphyModel(output)
    model.source_info = descriptor['source_info']
    model.cache_reused = reused
    emit(f'项目已就绪：{dat.name}，{model.record_count}个保存时刻。')
    return model
