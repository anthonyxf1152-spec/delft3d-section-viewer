"""Physical-coordinate time-surface nodes for the CIGVis VisCanvas.

No window or event loop is created here. Vertices are global X/Y kilometres
and elevation kilometres (positive up), without an origin shift or physical
vertical exaggeration. The viewer may use camera axis_scales=(1,1,500).
"""
from pathlib import Path
import numpy as np


def _indices(length, stride):
    return np.unique(np.r_[np.arange(0, length, stride, dtype=int), length - 1])


def _triangles(shape, native_valid, ix, iy):
    """Two native-XY triangles per coarse quad, never bridging inactive holes."""
    nx, ny = shape
    corner = np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]
    first = np.stack((corner, corner + ny, corner + ny + 1), axis=-1).reshape(-1, 3)
    second = np.stack((corner, corner + ny + 1, corner + 1), axis=-1).reshape(-1, 3)
    # Integral image checks EVERY original cell between decimated endpoints.
    bad = np.pad((~native_valid).astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    i0, i1 = ix[:-1, None], ix[1:, None] + 1
    j0, j1 = iy[None, :-1], iy[None, 1:] + 1
    quad_valid = (bad[i1, j1] - bad[i0, j1] - bad[i1, j0] + bad[i0, j0]) == 0
    faces = np.concatenate((first, second))
    return faces[np.tile(quad_valid.ravel(), 2)]


def _compact_mesh(x, y, elevation, vertex_valid, faces):
    keep = np.all(vertex_valid.ravel()[faces], axis=1)
    faces = faces[keep]
    if not len(faces):
        return None, None, np.empty(0, dtype=int)
    used = np.unique(faces)
    inverse = np.full(elevation.size, -1, dtype=np.int32)
    inverse[used] = np.arange(len(used), dtype=np.int32)
    xx, yy = np.meshgrid(x / 1000., y / 1000., indexing='ij')
    vertices = np.column_stack((xx.ravel()[used], yy.ravel()[used], elevation.ravel()[used] / 1000.)).astype(np.float32)
    return vertices, inverse[faces], used


def _initial_base(volume_dir, geometry, record, ix, iy, fallback):
    path = volume_dir / 'bed_base_elevation_m.npy'
    if path.exists():
        array = np.load(path, mmap_mode='r')
        base = array[0] if array.ndim == 3 else array
        return np.asarray(base[np.ix_(ix, iy)], dtype=float), 'raw bed_base_elevation_m.npy record 0'
    for key in ('initial_base_m', 'initial_bed_base_elevation_m', 'initial_base_elevation_m'):
        if key in geometry:
            return np.asarray(geometry[key][np.ix_(ix, iy)], dtype=float), key
    return fallback.copy(), 'Unavailable: initial interface shown as geometric boundary only'


def build_surface_nodes(volume_dir, record=-1, max_surfaces=10, xy_stride=2,
                        opacity=.65, surface_records=None,
                        min_interval_thickness_m=.001,
                        min_surface_separation_m=.05,
                        include_current_top=True, shading=None, wireframe=False,
                        color_limits=None):
    """Return ``(nodes, meta)`` for selected-time surviving interfaces.

    Required cache files: native_geometry.npz with x_m/y_m/valid/morph_days;
    bed_top_elevation_m.npy [record,X,Y]. S is ALWAYS recomputed from raw bed
    tops only through ``record``; the final-time S cache is never reused for an
    earlier time. Automatic surfaces are approximately evenly spaced in saved
    morphological time. ``surface_records`` explicitly selects original record
    IDs. Returned ``drawn_interface_record_ids`` identifies actual meshes.

    Surviving stage thickness <1mm is masked. From younger to older, surfaces
    separated by <5cm are pruned at shared vertices (display thinning only),
    and triangles require three valid vertices. Current bed top is optionally
    retained as an OBSERVED GEOMETRIC BOUNDARY, including where no material was
    added in the last interval; its colour is not a formation-age assignment.
    This avoids pretending an eroded/zero-thickness saved stage is a layer.
    """
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from vispy.scene.visuals import Mesh

    volume_dir = Path(volume_dir)
    if max_surfaces < 1 or xy_stride < 1 or not isinstance(xy_stride, (int, np.integer)):
        raise ValueError('max_surfaces and xy_stride must be positive integers.')
    if not 0 < opacity <= 1 or min_interval_thickness_m < 0 or min_surface_separation_m < 0:
        raise ValueError('Opacity must be (0,1]; thickness and separation cannot be negative.')
    with np.load(volume_dir / 'native_geometry.npz') as source:
        geometry = {key: source[key] for key in source.files}
    x, y = np.asarray(geometry['x_m']), np.asarray(geometry['y_m'])
    native_valid = np.asarray(geometry['valid'], dtype=bool)
    morph = np.asarray(geometry['morph_days'], dtype=float)
    hydro = np.asarray(geometry.get('hydro_days', np.full(len(morph), np.nan)), dtype=float)
    raw = np.load(volume_dir / 'bed_top_elevation_m.npy', mmap_mode='r')
    if raw.ndim != 3 or raw.shape != (len(morph), len(x), len(y)) or native_valid.shape != raw.shape[1:]:
        raise ValueError('Raw time/X/Y bed tops, physical axes, times and validity must align.')
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0) or not np.isfinite(morph).all() or np.any(np.diff(morph) < 0):
        raise ValueError('Axes must increase and morphological times must be finite/nondecreasing.')
    selected = raw.shape[0] - 1 if record == -1 else int(record)
    if not 0 <= selected < len(morph):
        raise ValueError(f'record must be -1 or 0..{len(morph) - 1}.')
    ix, iy = _indices(len(x), xy_stride), _indices(len(y), xy_stride)
    sampled_valid = native_valid[np.ix_(ix, iy)]
    tops = np.array(raw[:selected + 1][:, ix, :][:, :, iy], dtype=np.float32)
    if not np.isfinite(tops[:, sampled_valid]).all():
        raise ValueError('Selected raw bed surface contains nonfinite active vertices.')
    tops[:, ~sampled_valid] = np.nan
    interfaces = np.minimum.accumulate(tops[::-1], axis=0)[::-1]
    base, base_source = _initial_base(volume_dir, geometry, selected, ix, iy, interfaces[0])
    if surface_records is None:
        targets = np.linspace(morph[0], morph[selected], min(int(max_surfaces), selected + 1))
        chosen = sorted({int(np.argmin(abs(morph[:selected + 1] - time))) for time in targets})
    else:
        chosen = sorted({int(value) for value in surface_records})
        if any(value < 0 or value > selected for value in chosen):
            raise ValueError('Every requested surface record must be within the selected time prefix.')
    if include_current_top and selected not in chosen:
        chosen.append(selected)
    face_candidates = _triangles(sampled_valid.shape, native_valid, ix, iy)
    next_kept = np.full(sampled_valid.shape, np.inf)
    active_limits = tuple(color_limits or (0., max(float(morph[-1]), 1.)))
    if len(active_limits) != 2 or not np.isfinite(active_limits).all() or active_limits[1] <= active_limits[0]:
        raise ValueError('Invalid stage colorbar limits.')
    norm = Normalize(*active_limits, clip=True)
    cmap = colormaps['turbo']
    reversed_nodes, entries = [], []
    for index in reversed(chosen):
        elevation = interfaces[index]
        current_top = bool(index == selected and include_current_top)
        thickness = elevation - (interfaces[index - 1] if index else base)
        # S0 can still be a useful geometric boundary if initial-base cache is
        # absent, without asserting that any surviving substrate thickness exists.
        stage_nonzero = thickness >= min_interval_thickness_m
        if index == 0 and base_source.startswith('Unavailable'):
            stage_nonzero = sampled_valid.copy()
        eligible = sampled_valid & np.isfinite(elevation)
        if not current_top:
            eligible &= stage_nonzero
        separated = next_kept - elevation >= min_surface_separation_m
        eligible &= separated
        vertices, faces, used = _compact_mesh(x[ix], y[iy], elevation, eligible, face_candidates)
        entry = {'record': int(index), 'morph_days': float(morph[index]),
                 'hydro_days': float(hydro[index]) if np.isfinite(hydro[index]) else None,
                 'role': 'current observed bed-top boundary' if current_top else 'surviving saved-stage interface',
                 'active_sampled_vertices': int(sampled_valid.sum()),
                 'zero_or_eroded_stage_vertices': int(np.count_nonzero(sampled_valid & ~stage_nonzero)),
                 'too_close_to_younger_surface_vertices': int(np.count_nonzero(sampled_valid & ~separated)),
                 'kept_vertices': int(len(used)), 'triangles': int(len(faces)) if faces is not None else 0}
        if vertices is None:
            entry['omitted_reason'] = 'No valid triangles after inactive/zero-stage/near-coincident surface pruning.'
        else:
            color = tuple(float(value) for value in cmap(norm(morph[index]))[:3]) + (float(opacity),)
            mesh = Mesh(vertices=vertices, faces=faces, color=color, shading=shading)
            mesh.set_gl_state('translucent' if opacity < 1 else 'opaque', depth_test=True,
                              depth_mask=bool(opacity >= 1), cull_face=False)
            if wireframe:
                from vispy.visuals.filters import WireframeFilter
                mesh.attach(WireframeFilter(color=(.08, .08, .08, .35), width=.6))
            reversed_nodes.append(mesh)
            next_kept.ravel()[used] = elevation.ravel()[used]
        entries.append(entry)
    entries.reverse()
    active_z = interfaces[:, sampled_valid]
    finite_base = base[sampled_valid & np.isfinite(base)]
    zlow = min(float(np.nanmin(active_z)), float(finite_base.min()) if len(finite_base) else np.inf)
    meta = {'selected_record': int(selected), 'selected_morph_days': float(morph[selected]),
            'selected_hydro_days': float(hydro[selected]) if np.isfinite(hydro[selected]) else None,
            'requested_interface_record_ids': chosen,
            'drawn_interface_record_ids': [row['record'] for row in entries if row['triangles'] > 0],
            'omitted_interface_record_ids': [row['record'] for row in entries if row['triangles'] == 0],
            'omitted_interface_count': int(sum(row['triangles'] == 0 for row in entries)),
            'interfaces': entries, 'coordinate_units': 'km for global X/Y and elevation Z; Z positive up',
            'axis_ranges_km': {'x': [float(x[0] / 1000), float(x[-1] / 1000)],
                               'y': [float(y[0] / 1000), float(y[-1] / 1000)],
                               'z': [zlow / 1000., float(np.nanmax(active_z)) / 1000.]},
            'full_time_color_limits_morph_days': [0., max(float(morph[-1]), 1.)],
            'active_color_limits_morph_days': list(active_limits),
            'colormap': 'turbo', 'xy_stride_display_only': int(xy_stride),
            'opacity_display_only': float(opacity), 'minimum_surviving_interval_m': float(min_interval_thickness_m),
            'minimum_surface_separation_display_m': float(min_surface_separation_m),
            'initial_base_source': base_source,
            'geometry_method': 'S_t=min(b_k, k=t..selected_record); recomputed from raw bed tops for the selected prefix.',
            'recommended_cigvis_axis_reversed': [False, False, False],
            'recommended_camera_axis_scales': [1., 1., 500.],
            'renderer': 'VisPy Mesh nodes accepted by CIGVis VisCanvas; native physical nonuniform XY retained.',
            'interpretation': 'Time-series-inferred geometry only; colours identify saved times, not precise formation ages or recovered sand/mud composition.'}
    # Old-to-young order helps transparent surfaces when viewed from above.
    return list(reversed(reversed_nodes)), meta
