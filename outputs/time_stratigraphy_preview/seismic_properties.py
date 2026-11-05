"""Six physical-property sections, with reproducible numeric and PNG exports."""
from pathlib import Path
import json

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from colorbar_settings import sand_color_style

matplotlib.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                            'axes.unicode_minus': False, 'font.size': 9})

PROPERTY_FIELDS = (
    ('sand_fraction', '含砂率 / %', 100., 'sand'),
    ('vp_m_s', '速度 Vp / (m/s)', 1., 'turbo'),
    ('density_kg_m3', r'密度 / (kg/m$^3$)', 1., 'turbo'),
    ('impedance_kg_m2_s', r'声阻抗 / ($10^6$ kg/(m$^2$·s))', 1e-6, 'turbo'),
    ('porosity_delta', '孔隙度变化 Δφ', 1., 'seismic'),
    ('reflectivity', '反射系数 R', 1., 'seismic'),
)


def _data(cut, name):
    if name == 'impedance_kg_m2_s' and name not in cut:
        return cut['vp_m_s'] * cut['density_kg_m3']
    return cut[name]


def draw_properties(result, figure=None, sand_colors=None, fields=None):
    figure = figure or Figure(figsize=(16, 8 * len(result['sections'])), layout='constrained')
    figure.clear()
    fields = fields or PROPERTY_FIELDS
    cuts = result['sections']
    columns = min(3, len(fields))
    rows_per_cut = (len(fields) + columns - 1) // columns
    axes = figure.subplots(len(cuts) * rows_per_cut, columns, squeeze=False)
    def panel(row, column):
        return axes[row * rows_per_cut + column // columns, column % columns]
    for column, (name, title, scale, palette) in enumerate(fields):
        values = [(_data(cut, name) * scale)[np.broadcast_to(cut['valid'][:, None], _data(cut, name).shape)] for cut in cuts]
        finite = np.concatenate([a[np.isfinite(a)] for a in values])
        ticks = None
        if palette == 'sand':
            cmap, norm, ticks = sand_color_style(sand_colors, percent=True)
        else:
            cmap = matplotlib.colormaps[palette]
            if name in ('porosity_delta', 'reflectivity'):
                limit = max(float(np.max(abs(finite))) if finite.size else 0.,
                            .001 if name == 'porosity_delta' else .0001)
                norm = Normalize(-limit, limit)
            else:
                lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
                norm = Normalize(lo, hi if hi > lo else lo + 1.)
        for row, cut in enumerate(cuts):
            ax = panel(row, column)
            h, z = cut['horizontal_m'] / 1000, cut['depth_m']
            dh, dz = float(np.median(np.diff(h))), float(np.median(np.diff(z)))
            valid = np.broadcast_to(cut['valid'][:, None], _data(cut, name).shape).copy()
            if name == 'sand_fraction':
                valid &= cut['occupancy'] > 0
            data = np.ma.masked_where(~valid, _data(cut, name) * scale)
            ax.set_facecolor('#eeeeee')
            ax.imshow(data.T, origin='upper', aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                extent=[h[0] - dh/2, h[-1] + dh/2, z[-1] + dz/2, z[0] - dz/2])
            ax.set(xlim=(h[0], h[-1]), ylim=(z[-1], z[0]),
                   title=f"{cut['axis'].upper()}Z · {title}", xlabel=f"原工程 {cut['axis'].upper()} / km")
            if column % columns == 0:
                fixed_axis = 'Y' if cut['axis'] == 'x' else 'X'
                ax.set_ylabel(f"{fixed_axis}={cut['actual_fixed_m']/1000:.3f} km\n情景深度 / m")
            figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                           ticks=ticks, shrink=.8, pad=.025, label=title)
    meta = result['metadata']; params = meta['parameters']
    dimension = '二维' if params.get('model_dimension') == '2d' else '三维模型切片'
    figure.suptitle(f"{dimension}物性模型｜记录 {meta['selected_record']} · 形态 {meta['selected_morph_days']:.2f} 天\n"
                   f"物理层厚 ×{params['vertical_scale']:g} · 非均质强度 {params['porosity_strength']:g}｜Vp、密度、声阻抗、反射系数已应用当前参数", fontsize=11)
    return figure


def load_saved_sections(output):
    output = Path(output)
    metadata = json.loads((output / 'model_parameters.json').read_text(encoding='utf-8'))
    cuts = []
    for description in metadata['sections']:
        cut = dict(description)
        folder = output / (cut['axis'] + 'z')
        for file in folder.glob('*.npy'):
            cut[file.stem] = np.load(file, allow_pickle=False)
        cuts.append(cut)
    return {'sections': cuts, 'metadata': metadata}


def export_property_figures(output, result, sand_colors=None):
    """Export combined six panels and each requested field as a standalone PNG."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(16, 8 * len(result['sections'])), layout='constrained')
    FigureCanvasAgg(fig)
    draw_properties(result, fig, sand_colors)
    fig.savefig(output / 'model_properties.png', dpi=160, facecolor='white'); fig.clear()
    for cut in result['sections']:
        folder = output / (cut['axis'] + 'z'); folder.mkdir(parents=True, exist_ok=True)
        single = {'sections': [cut], 'metadata': result['metadata']}
        fig = Figure(figsize=(16, 8), layout='constrained'); FigureCanvasAgg(fig)
        draw_properties(single, fig, sand_colors)
        fig.savefig(folder / 'model_properties.png', dpi=160, facecolor='white'); fig.clear()
        for field in PROPERTY_FIELDS:
            fig = Figure(figsize=(9, 5.5), layout='constrained'); FigureCanvasAgg(fig)
            draw_properties(single, fig, sand_colors, [field])
            fig.savefig(folder / f'{field[0]}.png', dpi=160, facecolor='white'); fig.clear()
    (output / 'model_plot_settings.json').write_text(json.dumps({
        'fields': [field[0] for field in PROPERTY_FIELDS],
        'sand_color_settings': sand_colors, 'axis_order': 'horizontal,depth',
        'sand_display': 'percent; grey outside fan', 'data_changed_by_plotting': False},
        ensure_ascii=False, indent=2), encoding='utf-8')


def export_section_arrays(output, result):
    """NPY cuts from a completed forward run, including physical properties."""
    output = Path(output); descriptions = []
    for cut in result['sections']:
        folder = output / (cut['axis'] + 'z'); folder.mkdir(parents=True, exist_ok=True)
        arrays = {name: value for name, value in cut.items() if isinstance(value, np.ndarray)}
        if 'impedance_kg_m2_s' not in arrays:
            arrays['impedance_kg_m2_s'] = _data(cut, 'impedance_kg_m2_s')
        for name, array in arrays.items():
            np.save(folder / f'{name}.npy', array, allow_pickle=False)
        descriptions.append({k: v for k, v in cut.items() if not isinstance(v, np.ndarray)})
    report = dict(result['metadata'], sections=descriptions, axis_order='horizontal,depth; depth positive down',
                  files='Each cut folder contains numeric NPY arrays and horizontal_m/depth_m/valid coordinates and mask.')
    (output / 'section_model_parameters.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
