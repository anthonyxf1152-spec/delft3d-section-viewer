"""Explicit spatial estimates for unresolved saved-stage sand fractions.

All calculations use the native XY grid, independent of the displayed cut.
No interpolated value is used as a donor. Original inventory-derived values
are retained; completed values are display estimates, not mass-balanced strata.
"""
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree


FILL_SETTINGS = {
    'method': 'Same-stage inverse-distance-squared interpolation on native XY',
    'max_distance_m': 2000., 'max_neighbors': 8,
    'component_constraint': 'Same four-connected surviving stage footprint',
    'fallback': 'Same-cell whole mixed-bed fraction at that stage record',
    'source_codes': {'0': 'absent/unresolved', '1': 'original inventory estimate or initial bed',
                     '2': 'spatial interpolation', '3': 'same-stage mixed-bed fallback'},
    'interpretation': 'Estimated completion only; not recovered lithology or a sediment mass balance.',
}


def fill_stage(q, visible, mixed, x, y):
    """Complete one stage; preserve every finite q exactly, never bridge components."""
    values = np.asarray(q, dtype='f4').copy()
    source = np.where(np.isfinite(q), 1, 0).astype('u1')
    missing = visible & ~np.isfinite(q)
    donors = visible & np.isfinite(q)
    if missing.any() and donors.any():
        components, _ = label(visible)
        donor_i, donor_j = np.nonzero(donors)
        target_i, target_j = np.nonzero(missing)
        tree = cKDTree(np.column_stack((x[donor_i], y[donor_j])))
        distance, index = tree.query(np.column_stack((x[target_i], y[target_j])),
                                     k=list(range(1, FILL_SETTINGS['max_neighbors'] + 1)),
                                     distance_upper_bound=FILL_SETTINGS['max_distance_m'])
        available = np.isfinite(distance)
        safe_index = np.minimum(index, len(donor_i) - 1)
        available &= (components[donor_i[safe_index], donor_j[safe_index]] ==
                      components[target_i, target_j, None])
        weights = np.where(available, 1. / np.maximum(distance, 1e-6)**2, 0.)
        total = weights.sum(axis=1)
        usable = total > 0
        donor_values = q[donor_i[safe_index], donor_j[safe_index]]
        estimated = np.divide((weights * donor_values).sum(axis=1), total,
                              out=np.zeros_like(total), where=usable)
        values[target_i[usable], target_j[usable]] = estimated[usable]
        source[target_i[usable], target_j[usable]] = 2
    fallback = missing & ~np.isfinite(values) & np.isfinite(mixed)
    values[fallback] = mixed[fallback]
    source[fallback] = 3
    return values, source


def build_completed_volume(model, record):
    """Build only through the chosen time, keeping provenance for each stage/cell."""
    model._load_sediment_inventory()
    surfaces = model.interfaces_until(record)
    values = np.full(surfaces.shape, np.nan, dtype='f4')
    sources = np.zeros(surfaces.shape, dtype='u1')
    previous_sand = previous_mud = None
    for stage in range(record + 1):
        sand = np.asarray(model._sand_equivalent[stage], dtype='f8')
        mud = np.asarray(model._mud_equivalent[stage], dtype='f8')
        total = sand + mud
        mixed = np.divide(sand, total, out=np.full_like(total, np.nan),
                          where=model.valid & (total > 1e-7))
        if stage == 0:
            values[stage] = mixed
            sources[stage] = np.isfinite(mixed)
        else:
            ds, dm = sand - previous_sand, mud - previous_mud
            dh = (surfaces[stage] - surfaces[stage - 1]).astype('f8')
            net = ds + dm
            supported = (model.valid & (dh >= .001) & (ds >= 0) & (dm >= 0) &
                         (abs(net - dh) <= 1e-4))
            q = np.divide(ds, net, out=np.full_like(net, np.nan), where=supported).astype('f4')
            values[stage], sources[stage] = fill_stage(q, model.valid & (dh > 1e-7),
                                                       mixed, model.x, model.y)
        previous_sand, previous_mud = sand, mud
    values.flags.writeable = False
    sources.flags.writeable = False
    return values, sources
