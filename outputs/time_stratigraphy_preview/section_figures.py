"""Scientific 2-D figures and reproducible exports for interactive sections."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import PercentFormatter
from colorbar_settings import sand_color_style

matplotlib.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                            'axes.unicode_minus': False, 'font.size': 9})


def draw_sections(model, sections, record, x_m, y_m, figure=None,
                  stage_limits=None, bed_limits=None, sand_values=None, sand_mode='increment', sand_colors=None):
    """Draw map, saved-stage sections, and matching sand-fraction sections."""
    fig = figure if figure is not None else Figure(figsize=(17.8, 8.8), layout='constrained')
    fig.clear()
    grid = fig.add_gridspec(max(2, len(sections)), 3, width_ratios=(1.05, 1.5, 1.5))
    map_ax = fig.add_subplot(grid[:, 0])
    top = np.asarray(model.raw[record], dtype=float)
    if bed_limits is None:
        bed_limits = (float(np.nanmin(top)), float(np.nanmax(top)))
    image = map_ax.pcolormesh(model.x / 1000, model.y / 1000,
                             np.ma.masked_invalid(top.T), shading='nearest', cmap='turbo',
                             vmin=bed_limits[0], vmax=bed_limits[1])
    fig.colorbar(image, ax=map_ax, label='当前床面高程 / m', shrink=.65, pad=.025)
    map_ax.plot(x_m / 1000, y_m / 1000, 'o', ms=6, mec='white', mfc='#d62f33')
    colours = {'x': '#db3b30', 'y': '#7e44ab'}
    for sec in sections:
        h = sec['horizontal_m'] / 1000
        fixed = sec['actual_fixed_m'] / 1000
        if sec['axis'] == 'x':
            map_ax.plot(h, np.full_like(h, fixed), color=colours['x'], lw=1.4, label='XZ：固定Y')
        else:
            map_ax.plot(np.full_like(h, fixed), h, color=colours['y'], lw=1.4, label='YZ：固定X')
    map_ax.set(title='平面定位图（点击选择X、Y）', xlabel='原工程 X / km', ylabel='原工程 Y / km', aspect='equal')
    map_ax.legend(loc='lower right', fontsize=8)
    norm = Normalize(*(stage_limits or (0, max(1., float(model.morph_days[-1])))), clip=True)
    cmap = matplotlib.colormaps['turbo']
    section_axes = []
    sand_axes = []
    sand_cmap, sand_norm, sand_ticks = sand_color_style(sand_colors)
    for row, sec in enumerate(sections):
        row_key = row if len(sections) > 1 else slice(None)
        ax = fig.add_subplot(grid[row_key, 1])
        h, ss, valid = sec['horizontal_m'] / 1000, sec['interfaces'], sec['valid']
        ax.fill_between(h, sec['base_elevation_m'], ss[0], where=valid,
                        color='#c6c8ca', lw=0, label='幸存初始底床')
        for stage in range(1, len(ss)):
            ax.fill_between(h, ss[stage - 1], ss[stage], where=valid,
                            facecolor=cmap(norm(sec['morph_days'][stage])), lw=0, rasterized=True)
        chosen = np.unique(np.linspace(0, record, min(record + 1, 48)).round().astype(int))
        previous = None
        for stage in chosen:
            if previous is None or np.nanmax(abs(ss[stage] - previous)) >= .001:
                ax.plot(h, ss[stage], color='#161c28', lw=.38, alpha=.6)
                previous = ss[stage]
        ax.plot(h, sec['top_elevation_m'], color='#111827', lw=1.2)
        ax.plot(h, sec['base_elevation_m'], color='#636b75', lw=.7, ls='--')
        fixed_axis = 'Y' if sec['axis'] == 'x' else 'X'
        label = sec['axis'].upper() + 'Z'
        ax.set(title=f"{label}｜固定{fixed_axis}请求 {sec['requested_fixed_m']:.3f} m → 实际 {sec['actual_fixed_m']:.3f} m",
               xlabel=f"原工程 {sec['axis'].upper()} / km", ylabel='原始高程 Z / m（向上）',
               xlim=(h[0], h[-1]))
        lo = float(np.nanmin(sec['base_elevation_m']))
        hi = float(np.nanmax(sec['top_elevation_m']))
        pad = max(.3, (hi - lo) * .07)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=.15, lw=.4)
        section_axes.append(ax)
        if sand_values is not None:
            sand_ax = fig.add_subplot(grid[row_key, 2], sharex=ax, sharey=ax)
            q = sand_values[sec['axis']]
            if q.shape != ss.shape:
                raise ValueError('含砂率数组与剖面层序形状不一致。')
            edges_h = np.empty(len(h) + 1)
            edges_h[1:-1] = (h[:-1] + h[1:]) / 2
            edges_h[0] = h[0] - (h[1] - h[0]) / 2
            edges_h[-1] = h[-1] + (h[-1] - h[-2]) / 2
            z_at_centres = np.vstack((sec['base_elevation_m'], ss))
            if not sec['valid'].all():
                for boundary in z_at_centres:
                    boundary[~sec['valid']] = np.interp(h[~sec['valid']], h[sec['valid']],
                                                         boundary[sec['valid']])
            z_at_edges = np.empty((len(z_at_centres), len(edges_h)))
            z_at_edges[:, 1:-1] = (z_at_centres[:, :-1] + z_at_centres[:, 1:]) / 2
            z_at_edges[:, 0] = z_at_centres[:, 0]
            z_at_edges[:, -1] = z_at_centres[:, -1]
            thickness = np.diff(z_at_centres, axis=0)
            visible = sec['valid'][None, :] & (thickness >= .001)
            # Missing composition stays grey; values below a manual minimum
            # saturate to the minimum colour instead of masquerading as missing.
            from matplotlib.colors import ListedColormap
            sand_ax.pcolormesh(np.broadcast_to(edges_h, z_at_edges.shape), z_at_edges,
                               np.ma.masked_where(~visible, np.zeros_like(q)), shading='flat',
                               cmap=ListedColormap(['#9ea3ab']), rasterized=True)
            display = np.ma.masked_where(~visible | ~np.isfinite(q), q)
            sand_ax.pcolormesh(np.broadcast_to(edges_h, z_at_edges.shape), z_at_edges,
                               display, shading='flat', cmap=sand_cmap, norm=sand_norm,
                               rasterized=True)
            for stage in chosen:
                sand_ax.plot(h, ss[stage], color='#172536', lw=.35, alpha=.46)
            sand_ax.plot(h, sec['top_elevation_m'], color='#111827', lw=1.1)
            sand_ax.plot(h, sec['base_elevation_m'], color='#636b75', lw=.7, ls='--')
            mode_title = {'increment': '分期净增量估算', 'mixed': '当前混合床平均',
                          'filled': '分期估算＋缺测补全'}[sand_mode]
            sand_ax.set(title=f'{label} 含砂率｜{mode_title}',
                        xlabel=f'原工程 {sec["axis"].upper()} / km', ylabel='原始高程 Z / m（向上）',
                        xlim=(h[0], h[-1]))
            sand_ax.grid(alpha=.15, lw=.4)
            sand_axes.append(sand_ax)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=section_axes,
                 label='保存沉积阶段 / 形态天', shrink=.8, pad=.025)
    if sand_axes:
        fig.colorbar(ScalarMappable(norm=sand_norm, cmap=sand_cmap), ax=sand_axes,
                     label='砂体积含量 / %', ticks=sand_ticks,
                     format=PercentFormatter(xmax=1), shrink=.8, pad=.025)
    sand_caption = {'increment': '分期砂泥库存净增量估算；灰色为无法可靠估算的组分',
                    'mixed': '所选时刻实际保存的整床平均含砂率',
                    'filled': '缺测已按同阶段邻域或当期混合床均值补全（估算）'}[sand_mode]
    fig.suptitle(f"时间层序剖面｜记录 {record} / {model.record_count - 1}｜水动力 {model.hydro_days[record]:.2f} 天"
                 f"｜形态 {model.morph_days[record]:.2f} 天\n"
                 f'截至所选时刻重建；右列含砂率：{sand_caption}', fontsize=12)
    return fig, map_ax


def export_sections(output, model, sections, record, x_m, y_m, figure, selection):
    """Export exact section arrays and long-form CSV alongside the figure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / 'sections.png', dpi=180, facecolor='white')
    descriptions = []
    for sec in sections:
        axis = sec['axis']
        arrays = {key: value for key, value in sec.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(output / f'{axis}z_section.npz', **arrays)
        with (output / f'{axis}z_interfaces.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['record_zero_based', 'hydro_days', 'morph_days', 'horizontal_m',
                             'inferred_interface_elevation_m', 'surviving_interval_thickness_m',
                             'original_bed_top_elevation_m'])
            for stage in range(record + 1):
                for column in np.flatnonzero(sec['valid']):
                    thickness = (sec['interfaces'][stage, column] - sec['interfaces'][stage - 1, column]) if stage else ''
                    writer.writerow([stage, float(sec['hydro_days'][stage]), float(sec['morph_days'][stage]),
                                     float(sec['horizontal_m'][column]), float(sec['interfaces'][stage, column]),
                                     float(thickness) if thickness != '' else '',
                                     float(sec['bed_top_history'][stage, column])])
        descriptions.append({key: value for key, value in sec.items() if not isinstance(value, np.ndarray)})
    metadata = {'selected_record_zero_based': int(record),
                'selected_hydrodynamic_days': float(model.hydro_days[record]),
                'selected_morphological_days': float(model.morph_days[record]),
                'requested_xy_m': [float(x_m), float(y_m)], 'selection': selection,
                'sections': descriptions,
                'coordinate_convention': 'Global Cartesian X/Y metres; elevation metres, positive up.',
                'method': 'For each selected column, S[t]=min(raw_bed_top[t:selected_record+1]); later records are excluded.',
                'interpretation': 'Time-series-inferred geometry; saved-stage colours do not represent recovered sand/mud composition or exact formation ages.',
                'source_cache_metadata': str(model.volume_dir / 'metadata.json')}
    if getattr(model, 'source_info', None):
        metadata['source_project'] = model.source_info
    if selection.get('sand_mode_2d') == 'filled':
        from sand_completion import FILL_SETTINGS
        metadata['sand_completion'] = FILL_SETTINGS
    (output / 'selection.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return output
