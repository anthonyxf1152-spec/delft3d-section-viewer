from dataclasses import replace
from pathlib import Path
from unittest import mock
import json
import tempfile
import unittest

import numpy as np
from scipy.fft import rfft, rfftfreq

from seismic_forward import (SeismicParameters, estimate_current_grid, build_current_model,
                             forward_current, export_current_model, reflection)
from seismic_2d import psf_image_2d, FIELDS
from stratigraphy_model import StratigraphyModel
from series_loader import close_model
from test_seismic_model_export import EstimationOnlyModel, simple_sections


class Seismic2DTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='delft-2d-psf-test-')
        self.root = Path(self.temp.name)
        x = np.arange(11.) * 250; y = np.arange(9.) * 250
        xx, yy = np.meshgrid(x, y, indexing='ij')
        initial = .001 * yy - .0002 * xx
        raw = np.array([initial, initial + 10, initial + 20], dtype='f4')
        raw[-1, 5, 4] -= 5
        np.save(self.root / 'bed_top_elevation_m.npy', raw)
        np.savez(self.root / 'native_geometry.npz', x_m=x, y_m=y,
            valid=np.ones((len(x), len(y)), bool), native_m_indices=np.arange(len(x)),
            native_n_indices=np.arange(len(y)), morph_days=[0., 10., 20.], hydro_days=[0., 1., 2.],
            initial_base_elevation_m=initial - 4.5)
        self.model = StratigraphyModel(self.root)
        self.sections = self.model.extract_sections(2, 1250, 1000, 'both', 500)
        self.params = SeismicParameters(model_dimension='2d', spacing_xy_m=125.)
        self.hint = (np.full(raw.shape, .8, dtype='f4'), np.ones(raw.shape, dtype='u1'))

    def tearDown(self):
        close_model(self.model); self.temp.cleanup()

    def build(self, params=None):
        return build_current_model(self.model, self.sections, params or self.params, filled_hint=self.hint)

    def test_true_2d_psf_flat_reflector_phase_frequency_linearity_and_horizontal_response(self):
        r = np.zeros((96, 241), dtype='f4'); r[:, 120] = .1
        image = psf_image_2d(r, [125., 5.], 2800., 30.)
        trace = image[48]
        peak_hz = rfftfreq(len(trace), 2 * 5 / 2800)[np.argmax(abs(rfft(trace)))]
        self.assertLess(abs(peak_hz - 30), 3.)
        self.assertEqual(int(np.argmax(trace)), 120)
        np.testing.assert_allclose(trace[115:120], trace[121:126][::-1], atol=.001)
        point = np.zeros_like(r); point[48, 120] = .1
        # Resolve the lateral PSF with fine horizontal sampling; at 125 m
        # neighbouring traces are already far outside the central lobe.
        spread = psf_image_2d(point, [25., 5.], 2800., 30.)
        self.assertGreater(float(np.max(abs(spread[47]))), .01 * float(np.max(abs(spread[48]))))
        np.testing.assert_allclose(psf_image_2d(point * 2, [25., 5.], 2800., 30.), spread * 2, atol=1e-7)
        np.testing.assert_array_equal(psf_image_2d(np.zeros_like(r), [125., 5.], 2800., 30.), 0)

    def test_exact_grid_estimate_guards_crops_nondivisible_depth_and_no_physical_array_reads(self):
        model = EstimationOnlyModel(self.model.x, self.model.y)
        params = replace(self.params, spacing_xy_m=160., spacing_z_m=7., depth_max_m=1001.)
        estimate = estimate_current_grid(model, self.sections, params)
        self.assertEqual(estimate['model_dimension'], '2d')
        self.assertEqual(estimate['sections'][0]['computational_shape_horizontal_depth'], [16, 144])
        self.assertEqual(estimate['sections'][1]['computational_shape_horizontal_depth'], [13, 144])
        self.assertEqual(estimate['voxel_count'], (16 + 13) * 144)
        self.assertEqual(estimate['sections'][0]['depth_bounds_m'], [0., 1001.])
        self.assertGreater(estimate['estimated_forward_peak_bytes'], estimate['estimated_model_peak_bytes'])
        for sec, grid in zip(self.sections, estimate['sections']):
            h = np.arange(grid['computational_shape_horizontal_depth'][0]) * 160
            expected = np.sum((h >= sec['horizontal_m'][0]) & (h <= sec['horizontal_m'][-1]))
            self.assertEqual(grid['shape_horizontal_depth'], [expected, 144])

    def test_2d_builder_does_not_use_3d_grid_or_psf_and_properties_obey_current_parameters(self):
        with (mock.patch('seismic_forward._current_grid', side_effect=AssertionError('3-D grid forbidden')),
              mock.patch('seismic_2d.psf_image_2d', side_effect=AssertionError('no convolution during build'))):
            prepared = self.build()
        self.assertFalse(prepared['metadata']['convolution_performed'])
        for full, display, estimate in zip(prepared['computational_sections'], prepared['sections'], prepared['metadata']['grid_estimate']['sections']):
            self.assertEqual(full['vp_m_s'].shape, tuple(estimate['computational_shape_horizontal_depth']))
            self.assertEqual(display['vp_m_s'].shape, tuple(estimate['shape_horizontal_depth']))
            for name in FIELDS:
                self.assertEqual(full[name].ndim, 2)
                self.assertEqual(full[name].dtype, np.dtype('f4'))
                self.assertTrue(np.isfinite(full[name]).all())
            np.testing.assert_array_equal(display['impedance_kg_m2_s'], display['vp_m_s'] * display['density_kg_m3'])
            np.testing.assert_array_equal(display['reflectivity'], reflection(display['vp_m_s'], display['density_kg_m3']))
            self.assertLessEqual(float(np.max(abs(display['porosity_delta']))), .025001)
        zero = self.build(replace(self.params, porosity_strength=0.))
        again = self.build()
        for cut, base, same in zip(prepared['sections'], zero['sections'], again['sections']):
            np.testing.assert_array_equal(cut['sand_fraction'], base['sand_fraction'])
            np.testing.assert_array_equal(cut['vp_m_s'], same['vp_m_s'])
            np.testing.assert_allclose(base['vp_m_s'] - cut['vp_m_s'], 6000 * cut['porosity_delta'], atol=.001)

    def test_model_npy_save_before_convolution_and_saved_grid_reproduces_forward(self):
        prepared = self.build()
        with mock.patch('seismic_2d.psf_image_2d', side_effect=AssertionError('model saving must not convolve')):
            output = export_current_model(self.root / 'model', prepared, self.sections)
        report = json.loads((output / 'model_parameters.json').read_text(encoding='utf-8'))
        self.assertEqual(report['parameters']['model_dimension'], '2d')
        self.assertIn('source_project', report)
        for name, description in report['arrays'].items():
            array = np.load(output / name, allow_pickle=False)
            self.assertEqual(list(array.shape), description['shape'])
            self.assertNotEqual(array.dtype, np.dtype('O'))
        with mock.patch('seismic_reference.psf_image', side_effect=AssertionError('3-D convolution forbidden')):
            result = forward_current(self.model, self.sections, self.params, filled_hint=self.hint)
        for cut in result['sections']:
            folder = output / (cut['axis'] + 'z')
            for name in FIELDS:
                np.testing.assert_array_equal(np.load(folder / f'{name}.npy'), cut[name])
            full_r = np.load(folder / 'computational_grid/reflectivity.npy')
            full_h = np.load(folder / 'computational_grid/horizontal_m.npy')
            image = psf_image_2d(full_r, [125., 5.], 2800., 30.)
            select = (full_h >= cut['horizontal_m'][0]) & (full_h <= cut['horizontal_m'][-1])
            np.testing.assert_array_equal(image[select], cut['seismic'])
        with self.assertRaises(FileExistsError):
            export_current_model(output, prepared, self.sections)

    def test_limit_invalid_sampling_depth_and_missing_composition_fail_before_compute(self):
        axis = np.linspace(0, 100000., 101)
        model = EstimationOnlyModel(axis, axis)
        params = replace(self.params, spacing_xy_m=25., spacing_z_m=1., depth_max_m=4000.)
        sections = simple_sections(axis, axis)
        estimate = estimate_current_grid(model, sections, params)
        self.assertFalse(estimate['within_grid_limit'])
        with self.assertRaisesRegex(ValueError, '网格过大'):
            build_current_model(model, sections, params)
        with self.assertRaises(ValueError):
            estimate_current_grid(self.model, self.sections, replace(self.params, spacing_z_m=50.))
        with self.assertRaisesRegex(ValueError, '深度范围'):
            self.build(replace(self.params, vertical_scale=64., burial_depth_m=1000.))
        fractions = self.hint[0].copy(); fractions[1, 5, 4] = np.nan
        with self.assertRaisesRegex(ValueError, '未知'):
            build_current_model(self.model, self.sections, self.params, filled_hint=(fractions, self.hint[1]))

    def test_zero_strength_identical_seismic_and_invalid_support_mask(self):
        valid = self.model.valid.copy(); valid[0] = False; self.model.valid = valid
        result = forward_current(self.model, self.sections, replace(self.params, porosity_strength=0.), filled_hint=self.hint)
        for cut in result['sections']:
            np.testing.assert_array_equal(cut['seismic'], cut['seismic_baseline'])
            np.testing.assert_array_equal(cut['porosity_delta'], 0)
        full = self.build()['computational_sections'][0]
        self.assertFalse(full['valid'][0])

    def test_earlier_time_excludes_future_and_failed_npy_write_cleans_staging(self):
        sections = self.model.extract_sections(1, 1250, 1000, 'both', 500)
        hint = tuple(a[:2] for a in self.hint)
        before = forward_current(self.model, sections, self.params, filled_hint=hint)
        values = np.array(self.model.raw); self.model.raw._mmap.close()
        values[2] = -10000.; self.model.raw = values
        after = forward_current(self.model, sections, self.params, filled_hint=hint)
        for a, b in zip(before['sections'], after['sections']):
            np.testing.assert_array_equal(a['seismic'], b['seismic'])
        prepared = build_current_model(self.model, sections, self.params, filled_hint=hint)
        output = self.root / 'failed_output'
        with mock.patch('seismic_2d.np.save', side_effect=OSError('simulated disk failure')):
            with self.assertRaisesRegex(OSError, 'simulated disk'):
                export_current_model(output, prepared, sections)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob('.model-writing-*')), [])


if __name__ == '__main__':
    unittest.main()
