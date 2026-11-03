"""Resolve displayed colour ranges without changing physical arrays."""
import numpy as np


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
