"""CIGVis physical-coordinate surfaces and exact selected-time section curtains."""
from pathlib import Path
import json

import numpy as np


def _axis_nodes(ranges):
    """World coordinates are km; tick text explicitly reports elevation in m."""
    from vispy.scene.visuals import Line, Text
    x0, x1 = ranges['x']; y0, y1 = ranges['y']; z0, z1 = ranges['z']
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    positions, labels = [], []
    segments = [[x0, y0, z0], [x1, y0, z0], [x0, y0, z0], [x0, y1, z0],
                [x0, y0, z0], [x0, y0, z1]]
    for value in np.linspace(x0, x1, 4):
        segments.extend([[value, y0, z0], [value, y0 - dy * .018, z0]])
        positions.append([value, y0 - dy * .045, z0]); labels.append(f'{value:.1f}')
    for value in np.linspace(y0, y1, 4):
        segments.extend([[x0, value, z0], [x0 - dx * .018, value, z0]])
        positions.append([x0 - dx * .045, value, z0]); labels.append(f'{value:.1f}')
    for value in np.linspace(z0, z1, 4):
        segments.extend([[x0, y0, value], [x0 - dx * .018, y0, value]])
        positions.append([x0 - dx * .07, y0, value]); labels.append(f'{value * 1000:.1f}')
    line = Line(pos=np.asarray(segments, dtype='f4'), connect='segments', color='#344154', width=1)
    text = Text(text=labels, pos=np.asarray(positions, dtype='f4'), color='#344154', font_size=8)
    title = Text(text=['X / km', 'Y / km', 'Elevation / m'],
                 pos=np.asarray([[(x0+x1)/2, y0 - dy*.10, z0],
                                 [x0 - dx*.10, (y0+y1)/2, z0],
                                 [x0 - dx*.10, y0, z1 + dz*.06]], dtype='f4'),
                 color='#26384e', font_size=9)
    return [line, text, title]


def _curtain_nodes(sections, morph_max, color_limits=None, sand_values=None):
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from vispy.scene.visuals import Mesh, Line
    nodes, stats = [], []
    cmap = colormaps['turbo']
    norm = Normalize(*(color_limits or (0., morph_max)), clip=True)
    for sec in sections:
        q = None if sand_values is None else sand_values[sec['axis']]
        sand_cmap = colormaps['YlOrBr']
        h, interfaces = sec['horizontal_m'] / 1000, sec['interfaces'] / 1000
        valid = sec['valid']
        fixed = sec['actual_fixed_m'] / 1000
        vertices, faces, colors = [], [], []
        n_triangles = 0
        for stage in range(len(interfaces)):
            bottom = sec['base_elevation_m'] / 1000 if stage == 0 else interfaces[stage - 1]
            top = interfaces[stage]
            for j in range(len(h) - 1):
                if not (valid[j] and valid[j+1]):
                    continue
                if max(top[j] - bottom[j], top[j+1] - bottom[j+1]) <= 1e-10:
                    continue
                if sec['axis'] == 'x':
                    points = [[h[j], fixed, bottom[j]], [h[j+1], fixed, bottom[j+1]],
                              [h[j+1], fixed, top[j+1]], [h[j], fixed, top[j]]]
                else:
                    points = [[fixed, h[j], bottom[j]], [fixed, h[j+1], bottom[j+1]],
                              [fixed, h[j+1], top[j+1]], [fixed, h[j], top[j]]]
                start = len(vertices); vertices.extend(points)
                faces.extend([[start, start+1, start+2], [start, start+2, start+3]])
                if q is None:
                    rgb = (.74, .76, .79, 1.) if stage == 0 else cmap(norm(sec['morph_days'][stage]))
                    colors.extend([rgb, rgb])
                else:
                    for k in (j, j+1):
                        value = q[stage, k]
                        if not np.isfinite(value) and top[k] - bottom[k] <= 1e-10:
                            value = q[stage, j+1 if k == j else j]
                        colors.append(sand_cmap(float(value)) if np.isfinite(value) else (.62, .64, .67, 1.))
                n_triangles += 2
        if vertices:
            mesh = Mesh(vertices=np.asarray(vertices, dtype='f4'), faces=np.asarray(faces, dtype=np.uint32),
                        face_colors=np.asarray(colors, dtype='f4'), shading=None)
            mesh.set_gl_state('opaque', depth_test=True, cull_face=False,
                              polygon_offset_fill=True, polygon_offset=(1., 1.))
            nodes.append(mesh)
        # Explicit interval boundaries make the inferred layering readable,
        # including where similar adjacent stage colours would look continuous.
        previous = None
        for stage in np.unique(np.linspace(0, len(interfaces)-1, min(len(interfaces), 48)).round().astype(int)):
            boundary = interfaces[stage]
            if previous is not None and np.nanmax(abs(boundary - previous)) < .000001:
                continue
            points = (np.column_stack((h, np.full(len(h), fixed), boundary)) if sec['axis'] == 'x'
                      else np.column_stack((np.full(len(h), fixed), h, boundary)))
            connected = valid[:-1] & valid[1:]
            if connected.any():
                seg = np.stack((points[:-1][connected], points[1:][connected]), axis=1).reshape(-1, 3)
                line = Line(pos=seg.astype('f4'), connect='segments', color=(.08, .11, .17, .48), width=.65)
                line.set_gl_state('translucent', depth_test=True, depth_func='lequal', depth_mask=False)
                nodes.append(line)
            previous = boundary
        for boundary in (sec['top_elevation_m'] / 1000, sec['base_elevation_m'] / 1000):
            points = (np.column_stack((h, np.full(len(h), fixed), boundary)) if sec['axis'] == 'x'
                      else np.column_stack((np.full(len(h), fixed), h, boundary)))
            connected = valid[:-1] & valid[1:]
            if connected.any():
                seg = np.stack((points[:-1][connected], points[1:][connected]), axis=1).reshape(-1, 3)
                nodes.append(Line(pos=seg.astype('f4'), connect='segments', color='#172536', width=1.2))
        stats.append({'axis': sec['axis'], 'triangles': n_triangles,
                      'sand_fraction_finite_stage_columns': int(np.isfinite(q).sum()) if q is not None else None,
                      'sand_fraction_missing_stage_columns': int(np.isnan(q[:, sec['valid']]).sum()) if q is not None else None,
                      'actual_fixed_m': sec['actual_fixed_m'], 'requested_fixed_m': sec['requested_fixed_m']})
    return nodes, stats


def create_canvas(model, sections, record, display_ve=500., max_surfaces=10,
                  surface_opacity=.38, size=(1660, 900), color_limits=None,
                  curtain_field='stage'):
    import cigvis
    from vispy import app, scene
    from cigvis.vispynodes import VisCanvas
    from stratigraphy_cigvis_nodes import build_surface_nodes
    app.use_app('pyside6')
    cigvis.set_order(True)
    cigvis.set_axis_reversed(False, False, False)
    if not np.isfinite(display_ve) or display_ve <= 0:
        raise ValueError('显示垂向夸张必须大于0。')
    if not sections or any(sec['record'] != record for sec in sections):
        raise ValueError('三维层面与剖面必须属于同一个所选时刻。')
    if curtain_field not in ('stage', 'mixed', 'increment', 'filled'):
        raise ValueError('未知的三维剖面着色变量。')
    active_limits = tuple(color_limits or (0., max(float(model.morph_days[-1]), 1.)))
    surfaces, surface_meta = build_surface_nodes(model.volume_dir, record=record,
                        max_surfaces=max_surfaces, xy_stride=2, opacity=surface_opacity,
                        color_limits=active_limits)
    ranges = surface_meta['axis_ranges_km']
    sand_values = ({sec['axis']: model.sand_fraction_for_section(sec, curtain_field)[0] for sec in sections}
                   if curtain_field != 'stage' else None)
    curtains, curtain_stats = _curtain_nodes(sections, max(float(model.morph_days[-1]), 1.),
                                             active_limits, sand_values)
    # A local frame makes the selected section readable independently of the
    # full-domain surface overview. Both frames retain absolute world coordinates.
    all_x, all_y = [], []
    zlow, zhigh = np.inf, -np.inf
    for sec in sections:
        h = sec['horizontal_m'] / 1000
        if sec['axis'] == 'x':
            all_x.extend([float(h[0]), float(h[-1])]); all_y.append(sec['actual_fixed_m'] / 1000)
        else:
            all_y.extend([float(h[0]), float(h[-1])]); all_x.append(sec['actual_fixed_m'] / 1000)
        zlow = min(zlow, float(np.nanmin(sec['base_elevation_m'])) / 1000)
        zhigh = max(zhigh, float(np.nanmax(sec['top_elevation_m'])) / 1000)
    def nonzero_span(values):
        lo, hi = min(values), max(values)
        return [lo - .5, hi + .5] if hi - lo < .01 else [lo, hi]
    local = {'x': nonzero_span(all_x), 'y': nonzero_span(all_y), 'z': [zlow, zhigh]}
    if local['z'][1] <= local['z'][0]:
        local['z'][1] = local['z'][0] + .001
    groups = [surfaces + _axis_nodes(ranges), curtains + _axis_nodes(local)]
    def make_bar(cmap, limits, label):
        result = cigvis.create_colorbar(cmap, list(limits), label_str=label)
        nodes = result if isinstance(result, list) else [result]
        for node in nodes:
            node.update_params(cbar_size=(100, 640))
        return nodes
    colorbars = make_bar('turbo', active_limits, 'Saved stage / morph days')
    if curtain_field != 'stage':
        groups[0].extend(colorbars)
        colorbars = make_bar('YlOrBr', (0., 100.), 'Sand volume fraction / %')
    for node in colorbars:
        node.update_params(cbar_size=(100, 640))
    groups[1].extend(colorbars)
    class DepthClearingCanvas(VisCanvas):
        def _draw_scene(self, bgcolor=None):
            # Transparent surfaces leave depth writes disabled. OpenGL clear
            # respects that flag, so restore it before EACH frame/FBO clear.
            # Depth testing on the meshes remains enabled for correct occlusion.
            self.context.set_state(depth_mask=True)
            return super()._draw_scene(bgcolor)

    canvas = DepthClearingCanvas(visual_nodes=groups, size=size, grid=(1, 2), share=False, bgcolor='white',
                       axis_scales=(1., 1., display_ve), azimuth=-55, elevation=28,
                       fov=35, zoom_factor=.88, keys=None,
                       title=f'Delft3D strata | record {record} | display VE {display_ve:g}')
    # VisCanvas initializes later panels with the first panel's camera centre;
    # explicitly fit each to its own physical extent after construction.
    for view, limits in zip(canvas.view, (ranges, local)):
        view.camera.set_range(x=limits['x'], y=limits['y'], z=limits['z'], margin=.15)
        view.camera.azimuth = -55; view.camera.elevation = 28
        view.camera._flip_factors = [1., 1., display_ve]
        view.camera._update_transform()
    scene.Text(f'Whole-model saved surfaces ({len(surfaces)})', parent=canvas.scene,
               pos=(20, 22), color='#1e3449', font_size=12, anchor_x='left', anchor_y='top')
    right_label = {'stage': 'Saved-stage section curtains',
                   'mixed': 'Actual mixed-bed sand fraction',
                   'increment': 'Estimated net-growth sand fraction (grey = unresolved)',
                   'filled': 'Sand fraction with estimated gap filling'}[curtain_field]
    scene.Text(right_label, parent=canvas.scene,
               pos=(size[0] / 2 + 20, 22), color='#1e3449', font_size=12, anchor_x='left', anchor_y='top')
    scene.Text(f'Record {record} | hydro {model.hydro_days[record]:.2f} d | morph {model.morph_days[record]:.2f} d | DISPLAY VE {display_ve:g} | elevation UP',
               parent=canvas.scene, pos=(20, size[1] - 18), color='#344154', font_size=10,
               anchor_x='left', anchor_y='bottom')
    meta = {'record': int(record), 'hydro_days': float(model.hydro_days[record]),
            'morph_days': float(model.morph_days[record]), 'display_vertical_exaggeration': float(display_ve),
            'color_limits_morph_days': list(active_limits),
            'curtain_field': curtain_field,
            'curtain_field_interpretation': {
                'stage': 'Saved-time-inferred geometric stages; no composition assigned.',
                'mixed': 'Actual selected-record whole-bed volume fraction, vertically uniform within its one stored mixed bed.',
                'increment': 'Positive component equivalent-bed-thickness inventory differences using the configured dry bulk densities; grey marks unresolved exchanges/erosion. This is not recovered final layered lithology.',
                'filled': 'Original estimates preserved; gaps filled from same-stage spatial neighbors or contemporaneous mixed-bed fractions. Not recovered lithology.'
            }[curtain_field],
            'physical_vertical_scale': 1., 'vertices_units': 'global XYZ km; elevation positive up',
            'surface_selection': surface_meta, 'curtain_meshes': curtain_stats,
            'panel_axis_ranges_km': [ranges, local], 'renderer': 'CIGVis VisCanvas / VisPy / PySide6'}
    if getattr(model, 'source_info', None):
        meta['source_project'] = model.source_info
    if curtain_field == 'filled':
        from sand_completion import FILL_SETTINGS
        meta['sand_completion'] = FILL_SETTINGS
        for sec, stats in zip(sections, curtain_stats):
            _, source = model.completed_sand_for_section(sec)
            stats['fill_source_counts'] = {str(code): int((source == code).sum()) for code in range(4)}
    return canvas, meta


def save_canvas(canvas, metadata, output):
    from vispy import app, io
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        app.process_events()
    records = []
    for view, limits in zip(canvas.view, metadata['panel_axis_ranges_km']):
        x, y = np.mean(limits['x']), np.mean(limits['y'])
        z0, z1 = limits['z']
        points = [[x, y, z0 + .2*(z1-z0), 1], [x, y, z0 + .8*(z1-z0), 1]]
        projected = view.scene.node_transform(canvas.scene).map(points)
        pixel = projected[:, :2] / projected[:, 3, None]
        dy = float(pixel[1, 1] - pixel[0, 1])
        if dy >= 0 or not np.all(np.asarray(view.camera._flip_factors) > 0):
            raise AssertionError('正向上高程的屏幕投影校验失败。')
        records.append({'higher_minus_lower_screen_y_px': dy, 'passed': True})
    metadata['elevation_projection_checks'] = records
    pixels = canvas.render(alpha=False)
    # Projection landmarks alone do not establish that OpenGL drew geometry.
    # Check repeated frames also retain visible coloured meshes in each panel.
    first = pixels.astype(np.int16)
    coloured = first.max(axis=2) - first.min(axis=2) > 35
    half = first.shape[1] // 2
    counts = [int(coloured[60:-50, :half].sum()), int(coloured[60:-50, half:-140].sum())]
    if metadata['morph_days'] > 0 and min(counts) < 1000:
        raise AssertionError('三维几何未正确绘出，已拒绝导出空白图。')
    metadata['visible_coloured_pixels_by_panel'] = counts
    io.write_png(str(output), pixels)
    output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return output
