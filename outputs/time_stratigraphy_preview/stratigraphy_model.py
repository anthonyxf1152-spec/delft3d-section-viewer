"""Native-coordinate engine for arbitrary-time geometric stage sections.

Uses the cached raw bed history, not only the final surviving interfaces.
XY is the original nonuniform Cartesian cell-centre grid in metres. Interface
elevation is positive upwards; stages are inferred from saved bed surfaces,
not recovered sediment compositions or exact formation ages.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np


MIN_STAGE_THICKNESS_M = .001


class StratigraphyModel:
    """Memory-mapped bed history with native XY section extraction.

    ``axis='x'`` gives an XZ section at fixed Y; ``axis='y'`` gives a YZ
    section at fixed X. Fixed positions snap to the nearest native coordinate.
    The selected saved record is inclusive; ``record=-1`` means the latest.
    Stage counts include surviving deposition intervals >=1 mm and exclude
    the initial bed. Invalid geometry and invalid counts are NaN.
    """

    def __init__(self, volume_dir=None):
        self.volume_dir = (Path(volume_dir).expanduser().resolve() if volume_dir is not None
                           else Path(__file__).resolve().parent / 'volume')
        geometry = self.volume_dir / 'native_geometry.npz'
        raw_file = self.volume_dir / 'bed_top_elevation_m.npy'
        if not geometry.is_file() or not raw_file.is_file():
            raise FileNotFoundError(
                f'Missing native bed cache in {self.volume_dir}. '
                'Run build_3d_surfaces.py first.'
            )
        with np.load(geometry, allow_pickle=False) as cache:
            self.x = np.asarray(cache['x_m'], dtype=float).copy()
            self.y = np.asarray(cache['y_m'], dtype=float).copy()
            self.valid = np.asarray(cache['valid'], dtype=bool).copy()
            self.morph_days = np.asarray(cache['morph_days'], dtype=float).copy()
            self.hydro_days = np.asarray(cache['hydro_days'], dtype=float).copy()
            self.initial_base = np.asarray(cache['initial_base_elevation_m'], dtype=float).copy()
            self.native_m_indices = np.asarray(cache['native_m_indices'], dtype='i4').copy()
            self.native_n_indices = np.asarray(cache['native_n_indices'], dtype='i4').copy()
        self.raw = np.load(raw_file, mmap_mode='r', allow_pickle=False)
        if self.raw.dtype != np.dtype('f4') or self.raw.ndim != 3:
            raise ValueError('Expected float32 raw bed history [time,X,Y].')
        self.record_count = int(self.raw.shape[0])
        shape = (len(self.x), len(self.y))
        if self.raw.shape[1:] != shape or self.valid.shape != shape or self.initial_base.shape != shape:
            raise ValueError('Native axes, validity, base and raw history shapes disagree.')
        if self.record_count < 1 or self.morph_days.shape != (self.record_count,) or self.hydro_days.shape != (self.record_count,):
            raise ValueError('Cached time axis does not match raw history.')
        if self.native_m_indices.shape != self.x.shape or self.native_n_indices.shape != self.y.shape:
            raise ValueError('Original native M/N index vectors have inconsistent lengths.')
        if min(len(self.x), len(self.y)) < 2 or not self.valid.any():
            raise ValueError('Expected at least two samples per native axis and active model columns.')
        if (not np.isfinite(self.x).all() or not np.isfinite(self.y).all() or
                np.any(np.diff(self.x) <= 0) or np.any(np.diff(self.y) <= 0)):
            raise ValueError('Native coordinate vectors must be finite and strictly increasing.')
        if (not np.isfinite(self.morph_days).all() or not np.isfinite(self.hydro_days).all() or
                np.any(np.diff(self.morph_days) < 0) or np.any(np.diff(self.hydro_days) <= 0)):
            raise ValueError('Invalid cached output times.')
        if not np.isfinite(self.initial_base[self.valid]).all():
            raise ValueError('Nonfinite initial bed base at active columns.')
        self.initial_base[~self.valid] = np.nan
        for value in (self.x, self.y, self.valid, self.morph_days, self.hydro_days,
                      self.initial_base, self.native_m_indices, self.native_n_indices):
            value.flags.writeable = False
        self._cached_record = None
        self._cached_interfaces = None
        self._sand_equivalent = None
        self._mud_equivalent = None
        self._filled_record = None
        self._filled_values = None
        self._filled_sources = None
        self._m_lookup = {int(value): i for i, value in enumerate(self.native_m_indices)}
        self._n_lookup = {int(value): i for i, value in enumerate(self.native_n_indices)}
        self.source_info = {}
        project_descriptor = self.volume_dir / 'project.json'
        if project_descriptor.is_file():
            self.source_info = json.loads(project_descriptor.read_text(encoding='utf-8')).get('source_info', {})
        elif (self.volume_dir / 'metadata.json').is_file():
            metadata = json.loads((self.volume_dir / 'metadata.json').read_text(encoding='utf-8'))
            self.source_info = {key: metadata[key] for key in ('dat_file', 'def_file') if key in metadata}

    def _load_sediment_inventory(self):
        if self._sand_equivalent is not None:
            return
        report = self.volume_dir / 'sand_cache_metadata.json'
        if not report.is_file():
            from build_sand_cache import build
            keys = ('sand_index', 'mud_index', 'sand_density', 'mud_density', 'dat_file', 'def_file')
            configured = {key: self.source_info[key] for key in keys if key in self.source_info}
            build(self.volume_dir, **configured)
        import json
        meta = json.loads(report.read_text(encoding='utf-8'))
        if not meta.get('complete') or meta.get('shape_time_xy') != list(self.raw.shape):
            raise ValueError('泥沙组分缓存与当前床面缓存不一致。')
        arrays = [np.load(self.volume_dir / name, mmap_mode='r') for name in meta['files']]
        if any(array.shape != self.raw.shape or array.dtype != np.dtype('f4') for array in arrays):
            raise ValueError('泥沙组分缓存形状或精度不一致。')
        self._sand_equivalent, self._mud_equivalent = arrays

    def sand_fraction_for_section(self, section, mode):
        """Return [saved-stage,horizontal] fractions and original-support mask.

        ``mixed`` is the actual selected record's vertically mixed bed fraction.
        ``increment`` estimates the fraction of positive net inventory growth;
        exchanges/erosion are NaN, not clipped into a fabricated lithology.
        ``filled`` completes missing increments; support remains False for fills.
        """
        if mode == 'filled':
            values, source = self.completed_sand_for_section(section)
            return values, source == 1
        if mode not in ('mixed', 'increment'):
            raise ValueError('含砂率模式应为 mixed、increment 或 filled。')
        self._load_sediment_inventory()
        idx_m = np.asarray([self._m_lookup[int(v)] for v in section['native_m_indices']])
        idx_n = np.asarray([self._n_lookup[int(v)] for v in section['native_n_indices']])
        count = section['record'] + 1
        sand = np.asarray(self._sand_equivalent[:count, idx_m, idx_n], dtype='f8')
        mud = np.asarray(self._mud_equivalent[:count, idx_m, idx_n], dtype='f8')
        total = sand + mud
        if mode == 'mixed':
            final = np.divide(sand[-1], total[-1], out=np.full(total.shape[1], np.nan), where=total[-1] > 1e-7)
            values = np.broadcast_to(final, total.shape).copy()
            supported = section['valid'][None] & np.isfinite(values)
        else:
            values = np.full(total.shape, np.nan, dtype='f8')
            initial = np.divide(sand[0], total[0], out=np.full(total.shape[1], np.nan), where=total[0] > 1e-7)
            values[0] = initial
            if count > 1:
                ds, dm = np.diff(sand, axis=0), np.diff(mud, axis=0)
                dh = np.diff(section['interfaces'], axis=0).astype('f8')
                net = ds + dm
                eligible = ((dh >= .001) & (ds >= 0) & (dm >= 0) &
                            (abs(net - dh) <= 1e-4) & section['valid'][None])
                values[1:] = np.divide(ds, net, out=np.full_like(net, np.nan), where=eligible)
            supported = section['valid'][None] & np.isfinite(values)
        values[:, ~section['valid']] = np.nan
        return values.astype('f4'), supported

    def completed_sand_for_section(self, section):
        """Return completed fractions and 0/1/2/3 provenance on any native cut."""
        record = self._record(section['record'])
        if self._filled_record != record:
            from sand_completion import build_completed_volume
            self._filled_values, self._filled_sources = build_completed_volume(self, record)
            self._filled_record = record
        idx_m = np.asarray([self._m_lookup[int(v)] for v in section['native_m_indices']])
        idx_n = np.asarray([self._n_lookup[int(v)] for v in section['native_n_indices']])
        return self._filled_values[:, idx_m, idx_n].copy(), self._filled_sources[:, idx_m, idx_n].copy()

    def _record(self, record):
        if isinstance(record, (bool, np.bool_)) or not isinstance(record, (int, np.integer)):
            raise ValueError('record must be a zero-based integer; -1 means latest.')
        record = int(record)
        if record == -1:
            record = self.record_count - 1
        if not 0 <= record < self.record_count:
            raise ValueError(f'record must be -1 or 0..{self.record_count - 1}.')
        return record

    @staticmethod
    def _coordinate(value, axis, name):
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite coordinate in metres.') from exc
        if not np.isfinite(result) or not axis[0] <= result <= axis[-1]:
            raise ValueError(f'{name} must lie within [{axis[0]:.6f}, {axis[-1]:.6f}] m.')
        return result

    def interfaces_until(self, record):
        """Full native [0..record,X,Y] suffix-minimum; cache only the last call.

        Returned geometry is read-only. For efficient interactive 2D cuts,
        ``section`` applies the minimum to the small selected cut directly.
        """
        record = self._record(record)
        if record != self._cached_record:
            raw = self.raw[:record + 1]
            if not np.isfinite(raw[:, self.valid]).all():
                raise ValueError('Nonfinite raw bed tops at active model columns.')
            result = np.minimum.accumulate(raw[::-1], axis=0)[::-1].copy()
            result[:, ~self.valid] = np.nan
            result.flags.writeable = False
            self._cached_interfaces = result
            self._cached_record = record
        return self._cached_interfaces

    def section(self, record, axis, position_m, centre_m=None, half_width_m=6000):
        """Extract an arbitrary saved-time native XZ or YZ section.

        Fixed position and horizontal centre must be inside the native axis
        extents. The horizontal window is clipped to existing native samples,
        without interpolation/extrapolation. ``fixed_index`` addresses the
        trimmed model's fixed axis; native_m/n_indices retain original NEFIS
        zero-based indices including the source ghost-row/column convention.
        """
        record = self._record(record)
        if not isinstance(axis, str) or axis.lower() not in ('x', 'y'):
            raise ValueError("axis must be 'x' (XZ at fixed Y) or 'y' (YZ at fixed X).")
        axis = axis.lower()
        horizontal = self.x if axis == 'x' else self.y
        fixed = self.y if axis == 'x' else self.x
        requested_fixed = self._coordinate(position_m, fixed, 'position_m')
        centre = ((float(horizontal[0]) + float(horizontal[-1])) / 2 if centre_m is None
                  else self._coordinate(centre_m, horizontal, 'centre_m'))
        try:
            half_width = float(half_width_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('half_width_m must be finite and positive.') from exc
        if not np.isfinite(half_width) or half_width <= 0:
            raise ValueError('half_width_m must be finite and positive.')
        fixed_index = int(np.argmin(abs(fixed - requested_fixed)))
        select = np.flatnonzero(abs(horizontal - centre) <= half_width + 1e-7)
        if not len(select):
            raise ValueError('The horizontal window contains no native samples; increase half_width_m.')
        if axis == 'x':
            valid = self.valid[select, fixed_index].copy()
            history = np.asarray(self.raw[:record + 1, select, fixed_index]).copy()
            base = self.initial_base[select, fixed_index].copy()
            native_m = self.native_m_indices[select].copy()
            native_n = np.full(len(select), self.native_n_indices[fixed_index], dtype='i4')
        else:
            valid = self.valid[fixed_index, select].copy()
            history = np.asarray(self.raw[:record + 1, fixed_index, select]).copy()
            base = self.initial_base[fixed_index, select].copy()
            native_m = np.full(len(select), self.native_m_indices[fixed_index], dtype='i4')
            native_n = self.native_n_indices[select].copy()
        if not valid.any():
            raise ValueError('The selected native section window has no active (KCS=1) model columns.')
        if history.shape != (record + 1, len(select)):
            raise ValueError('Unexpected native section indexing shape.')
        if not np.isfinite(history[:, valid]).all():
            raise ValueError('Nonfinite raw bed history at active section columns.')
        history[:, ~valid] = np.nan
        base[~valid] = np.nan
        interfaces = np.minimum.accumulate(history[::-1], axis=0)[::-1].copy()
        stage_count = np.sum(np.diff(interfaces, axis=0) >= MIN_STAGE_THICKNESS_M, axis=0).astype(float)
        stage_count[~valid] = np.nan
        return {
            'axis': axis, 'horizontal_m': horizontal[select].copy(), 'valid': valid,
            'interfaces': interfaces, 'bed_top_history': history,
            'top_elevation_m': history[-1].copy(), 'base_elevation_m': base,
            'morph_days': self.morph_days[:record + 1].copy(),
            'hydro_days': self.hydro_days[:record + 1].copy(),
            'record': record, 'requested_fixed_m': requested_fixed,
            'actual_fixed_m': float(fixed[fixed_index]), 'fixed_index': fixed_index,
            'centre_m': centre, 'half_width_m': half_width,
            'native_m_indices': native_m, 'native_n_indices': native_n,
            'stage_count': stage_count,
            'minimum_counted_interval_thickness_m': MIN_STAGE_THICKNESS_M,
        }

    def extract_sections(self, record, x_m, y_m, direction='both', half_width_m=6000):
        """Extract sections centred on requested XY; XZ precedes YZ."""
        record = self._record(record)
        x_m = self._coordinate(x_m, self.x, 'x_m')
        y_m = self._coordinate(y_m, self.y, 'y_m')
        if not isinstance(direction, str) or direction.lower() not in ('x', 'y', 'both'):
            raise ValueError("direction must be 'x', 'y' or 'both'.")
        direction = direction.lower()
        result = []
        if direction in ('x', 'both'):
            result.append(self.section(record, 'x', y_m, x_m, half_width_m))
        if direction in ('y', 'both'):
            result.append(self.section(record, 'y', x_m, y_m, half_width_m))
        return result
