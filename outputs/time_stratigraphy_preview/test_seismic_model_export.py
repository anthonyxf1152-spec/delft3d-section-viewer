from dataclasses import replace
from pathlib import Path
from unittest import mock
import json
import tempfile
import unittest

import numpy as np

from seismic_forward import (
    MAX_GRID_VOXELS,
    MODEL_VOLUME_NAMES,
    SeismicParameters,
    _line,
    build_current_model,
    estimate_current_grid,
    export_current_model,
    model_sections,
    reflection,
)
from stratigraphy_model import StratigraphyModel
from series_loader import close_model


class EstimationOnlyModel:
    """Minimum model proving grid estimation does not touch physical arrays."""

    def __init__(self, x, y, record_count=3):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.record_count = record_count

    def _record(self, record):
        record = int(record)
        if not 0 <= record < self.record_count:
            raise IndexError(record)
        return record

    @property
    def raw(self):
        raise AssertionError('grid estimation/build rejection must not read bed history')

    @property
    def valid(self):
        raise AssertionError('grid estimation/build rejection must not read validity arrays')


def simple_sections(x, y, record=1):
    return [
        {'axis': 'x', 'record': record, 'horizontal_m': np.asarray(x, dtype=float),
         'actual_fixed_m': float(y[len(y) // 2]), 'requested_fixed_m': float(y[len(y) // 2])},
        {'axis': 'y', 'record': record, 'horizontal_m': np.asarray(y, dtype=float),
         'actual_fixed_m': float(x[len(x) // 2]), 'requested_fixed_m': float(x[len(x) // 2])},
    ]


class SeismicModelExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='delft-model-export-test-')
        self.root = Path(self.temp.name)
        self.x = np.arange(5.) * 250
        self.y = np.arange(5.) * 250
        xx, yy = np.meshgrid(self.x, self.y, indexing='ij')
        initial = .001 * yy - .0002 * xx
        raw = np.array([initial, initial + 10, initial + 20], dtype='f4')
        raw[-1, 2, 2] -= 5
        self.raw = raw
        self.valid = np.ones((5, 5), bool)
        np.save(self.root / 'bed_top_elevation_m.npy', raw)
        np.savez(self.root / 'native_geometry.npz', x_m=self.x, y_m=self.y,
                 valid=self.valid, native_m_indices=np.arange(5),
                 native_n_indices=np.arange(5), morph_days=[0., 10., 20.],
                 hydro_days=[0., 1., 2.], initial_base_elevation_m=initial - 4.5)
        self.model = StratigraphyModel(self.root)
        self.sections = self.model.extract_sections(2, 500, 500, 'both', 500)
        self.params = SeismicParameters(spacing_xy_m=250., spacing_z_m=10., depth_max_m=1500.)
        self.hint = (np.full(raw.shape, .8, dtype='f4'),
                     np.ones(raw.shape, dtype='u1'))

    def tearDown(self):
        close_model(self.model)
        self.temp.cleanup()

    def build(self):
        return build_current_model(self.model, self.sections, self.params,
                                   filled_hint=self.hint)

    def test_estimate_is_exact_for_nondivisible_spacing_without_reading_model_arrays(self):
        model = EstimationOnlyModel(self.x, self.y)
        sections = simple_sections(self.x, self.y)
        params = SeismicParameters(spacing_xy_m=300., spacing_z_m=7., depth_max_m=1001.)

        estimate = estimate_current_grid(model, sections, params)

        # Global XY lattice ends at 900 m because 1000 is not divisible by 300;
        # the Z construction deliberately includes the nearest 7 m sample at 1001.
        self.assertEqual(estimate['grid_shape_xyz'], [4, 4, 144])
        self.assertEqual(estimate['grid_bounds_m'],
                         {'x': [0., 900.], 'y': [0., 900.], 'depth': [0., 1001.]})
        self.assertEqual(estimate['voxel_count'], 4 * 4 * 144)
        self.assertEqual(estimate['single_float32_array_bytes'], 4 * 4 * 144 * 4)
        self.assertEqual([cut['shape_horizontal_depth'] for cut in estimate['sections']],
                         [[4, 144], [4, 144]])
        self.assertTrue(estimate['within_grid_limit'])
        self.assertGreater(estimate['estimated_forward_peak_bytes'],
                           estimate['estimated_model_peak_bytes'])
        self.assertGreater(estimate['estimated_npy_payload_bytes'],
                           8 * estimate['single_float32_array_bytes'])
        with self.assertRaises(ValueError):
            estimate_current_grid(model, sections, replace(params, spacing_z_m=20.))

    def test_limit_flag_and_build_rejection_happen_before_physical_arrays_are_read(self):
        axis = np.arange(0., 10001., 100.)
        model = EstimationOnlyModel(axis, axis)
        section = simple_sections(axis, axis)[:1]
        params = SeismicParameters(spacing_xy_m=25., spacing_z_m=2., depth_max_m=1000.)

        estimate = estimate_current_grid(model, section, params)
        self.assertGreater(estimate['voxel_count'], MAX_GRID_VOXELS)
        self.assertFalse(estimate['within_grid_limit'])
        with self.assertRaisesRegex(ValueError, '网格过大'):
            build_current_model(model, section, params)

    def test_build_and_pickle_free_export_do_not_call_psf_and_preserve_source(self):
        source_before = np.load(self.root / 'bed_top_elevation_m.npy', allow_pickle=False).copy()
        with mock.patch('seismic_forward.psf_image',
                        side_effect=AssertionError('model construction/export must not convolve')) as psf:
            prepared = self.build()
            output = export_current_model(self.root / 'saved_model', prepared, self.sections)
        psf.assert_not_called()

        shape = tuple(prepared['metadata']['grid_shape_xyz'])
        self.assertEqual(tuple(prepared['coordinates']), ('x_m', 'y_m', 'depth_m'))
        self.assertEqual(tuple(prepared['volumes']), MODEL_VOLUME_NAMES)
        self.assertEqual(set(prepared['surfaces']),
                         {'valid_xy', 'fan_top_depth_m', 'fan_base_depth_m'})
        self.assertFalse(prepared['metadata']['convolution_performed'])
        self.assertEqual(prepared['metadata']['grid_estimate']['grid_shape_xyz'], list(shape))
        for volume in prepared['volumes'].values():
            self.assertEqual(volume.shape, shape)
            self.assertEqual(volume.dtype, np.dtype('f4'))

        phi = prepared['volumes']['porosity_delta']
        vp = prepared['volumes']['vp_m_s']
        rho = prepared['volumes']['density_kg_m3']
        np.testing.assert_allclose(prepared['volumes']['reflectivity'], reflection(vp, rho),
                                   rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            prepared['volumes']['baseline_reflectivity'],
            reflection(vp + 6000 * phi, rho + 1650 * phi), rtol=0, atol=2e-7)

        for name, values in prepared['coordinates'].items():
            np.testing.assert_array_equal(np.load(output / f'{name}.npy', allow_pickle=False), values)
        for name, values in prepared['volumes'].items():
            loaded = np.load(output / f'{name}.npy', allow_pickle=False)
            np.testing.assert_array_equal(loaded, values)
            self.assertNotEqual(loaded.dtype, np.dtype('O'))
        for name, values in prepared['surfaces'].items():
            np.testing.assert_array_equal(np.load(output / f'{name}.npy', allow_pickle=False), values)

        impedance = np.load(output / 'impedance_kg_m2_s.npy', mmap_mode='r', allow_pickle=False)
        try:
            np.testing.assert_array_equal(impedance, vp * rho)
            cuts = model_sections(prepared, self.sections)
            for section, cut in zip(self.sections, cuts):
                prefix = output / f"{section['axis']}z"
                self.assertTrue((prefix / 'reflectivity.npy').is_file())
                self.assertTrue((prefix / 'impedance_kg_m2_s.npy').is_file())
                expected = _line(impedance, prepared['coordinates']['x_m'],
                                 prepared['coordinates']['y_m'], section['axis'],
                                 section['actual_fixed_m'], cut['horizontal_m'])
                np.testing.assert_array_equal(
                    np.load(prefix / 'impedance_kg_m2_s.npy', allow_pickle=False), expected)
        finally:
            impedance._mmap.close()

        report = json.loads((output / 'model_parameters.json').read_text(encoding='utf-8'))
        self.assertEqual(report['parameters']['spacing_xy_m'], self.params.spacing_xy_m)
        self.assertEqual(report['parameters']['spacing_z_m'], self.params.spacing_z_m)
        self.assertEqual(report['arrays']['vp_m_s.npy']['axis_order'], 'X,Y,depth')
        self.assertEqual(report['arrays']['xz/vp_m_s.npy']['axis_order'], 'horizontal,depth')
        self.assertEqual(report['array_units']['vp_m_s'], 'm/s')
        self.assertEqual(report['array_units']['density_kg_m3'], 'kg/m3')
        self.assertEqual(report['array_units']['impedance_kg_m2_s'], 'kg/(m2*s)')
        self.assertIn('Positive down', report['depth_convention'])
        np.testing.assert_array_equal(
            np.load(self.root / 'bed_top_elevation_m.npy', allow_pickle=False), source_before)

    def test_export_never_overwrites_and_removes_failed_staging_directory(self):
        prepared = self.build()
        output = export_current_model(self.root / 'complete_bundle', prepared, self.sections)
        report_before = (output / 'model_parameters.json').read_bytes()
        with self.assertRaises(FileExistsError):
            export_current_model(output, prepared, self.sections)
        self.assertEqual((output / 'model_parameters.json').read_bytes(), report_before)

        original_save = np.save
        calls = 0

        def fail_during_save(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('simulated disk write failure')
            return original_save(*args, **kwargs)

        failed = self.root / 'failed_bundle'
        with mock.patch('seismic_forward.np.save', side_effect=fail_during_save):
            with self.assertRaisesRegex(OSError, 'simulated'):
                export_current_model(failed, prepared, self.sections)
        self.assertFalse(failed.exists())
        self.assertEqual(list(self.root.glob('.model-writing-*')), [])


if __name__ == '__main__':
    unittest.main()
