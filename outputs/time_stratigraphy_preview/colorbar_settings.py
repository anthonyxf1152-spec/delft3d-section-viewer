"""Resolve displayed colour ranges without changing physical arrays."""
import numpy as np

# Sampled from the centre of the user-supplied reference colour bar,
# normalized from its lower to upper endpoint. The quantity remains sand %.
SAND_REFERENCE_COLORS = ['#301c8a', '#4743e7', '#4269fe', '#2d8af4', '#1ca9df',
                         '#0fbebb', '#49cb85', '#9dc943', '#e6bb2d', '#fccf30', '#feff16']
DEFAULT_SAND_COLORS = {'min_percent': 0., 'mid_percent': 50., 'max_percent': 100.,
                       'palette': 'reference', 'low_color': SAND_REFERENCE_COLORS[0],
                       'mid_color': SAND_REFERENCE_COLORS[5], 'high_color': SAND_REFERENCE_COLORS[-1]}


def validate_sand_colors(config=None):
    from matplotlib.colors import to_hex
    value = dict(DEFAULT_SAND_COLORS)
    value.update(config or {})
    for key in ('min_percent', 'mid_percent', 'max_percent'):
        value[key] = float(value[key])
    low, middle, high = [value[key] for key in ('min_percent', 'mid_percent', 'max_percent')]
    if not np.isfinite([low, middle, high]).all() or not 0 <= low < middle < high <= 100:
        raise ValueError('含砂率色标须满足：0% ≤ 最小值 < 中间值 < 最大值 ≤ 100%。')
    if value['palette'] not in ('reference', 'custom'):
        raise ValueError('未知含砂率配色。')
    for key in ('low_color', 'mid_color', 'high_color'):
        value[key] = to_hex(value[key], keep_alpha=False)
    return {key: value[key] for key in DEFAULT_SAND_COLORS}


def sand_color_style(config=None, *, percent=False):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    value = validate_sand_colors(config)
    colors = (SAND_REFERENCE_COLORS if value['palette'] == 'reference' else
              [value['low_color'], value['mid_color'], value['high_color']])
    cmap = LinearSegmentedColormap.from_list('delft_sand_reference', colors, N=1025)
    cmap.set_under(colors[0]); cmap.set_over(colors[-1]); cmap.set_bad((0, 0, 0, 0))
    factor = 1 if percent else 100
    low, middle, high = [value[key] / factor for key in ('min_percent', 'mid_percent', 'max_percent')]
    norm = TwoSlopeNorm(vmin=low, vcenter=middle, vmax=high)
    ticks = np.array([low, (low + middle) / 2, middle, (middle + high) / 2, high])
    return cmap, norm, ticks


def color_limits(model, record, field, mode, manual=None):
    if field not in ('stage', 'bed') or mode not in ('global', 'current', 'manual'):
        raise ValueError('未知色标对象或范围模式。')
    if mode == 'manual':
        if manual is None or len(manual) != 2:
            raise ValueError('手动色标需要下限和上限。')
        low, high = map(float, manual)
        if not np.isfinite([low, high]).all() or high <= low or (field == 'stage' and low < 0):
            raise ValueError('色标上下限必须是有限数，且上限大于下限；沉积阶段下限不能为负。')
        return (low, high)
    if field == 'stage':
        return (0., max(1., float(model.morph_days[-1 if mode == 'global' else record])))
    if mode == 'current':
        data = np.asarray(model.raw[record])
        values = data[model.valid]
        low, high = float(values.min()), float(values.max())
    else:
        cached = getattr(model, '_global_bed_color_limits', None)
        if cached is not None:
            return cached
        low, high = np.inf, -np.inf
        for i in range(model.record_count):
            values = np.asarray(model.raw[i])[model.valid]
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
        model._global_bed_color_limits = (low, high if high > low else low + 1.)
        return model._global_bed_color_limits
    return (low, high if high > low else low + 1.)
