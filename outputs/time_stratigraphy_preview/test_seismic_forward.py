from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
import json

import numpy as np

from seismic_forward import SeismicParameters, forward_current, reflection, _line, _normal_field
from stratigraphy_model import StratigraphyModel
from series_loader import close_model


class SeismicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='delft-seismic-test-')
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
        np.savez(self.root / 'native_geometry.npz', x_m=self.x, y_m=self.y, valid=self.valid,
            native_m_indices=np.arange(5), native_n_indices=np.arange(5),
            morph_days=[0., 10., 20.], hydro_days=[0., 1., 2.], initial_base_elevation_m=initial - 4.5)
        self.model = StratigraphyModel(self.root)
        self.sections = self.model.extract_sections(2, 500, 500, 'both', 500)
        self.params = SeismicParameters(spacing_xy_m=250.)
        self.hint = (np.full(raw.shape, .8, dtype='f4'), np.ones(raw.shape, dtype='u1'))

    def tearDown(self):
        close_model(self.model)
        self.temp.cleanup()

    def run_forward(self, **kwargs):
        return forward_current(self.model, self.sections, kwargs.pop('params', self.params),
                               filled_hint=kwargs.pop('hint', self.hint), **kwargs)

    def test_impedance_boundary_reflection_sign_and_magnitude(self):
        vp = np.ones((2, 2, 4), dtype='f4') * 2000
        rho = np.ones_like(vp) * 2000
        np.testing.assert_array_equal(reflection(vp, rho), 0)
        vp[:, :, 2:] *= 1.5
        r = reflection(vp, rho)
        np.testing.assert_allclose(r[:, :, 1], .2)
        np.testing.assert_array_equal(r[:, :, [0, 2, 3]], 0)

    def test_zero_heterogeneity_equals_baseline_and_real_heterogeneity_changes_impedance(self):
        zero = self.run_forward(params=replace(self.params, porosity_strength=0.))
        for section in zero['sections']:
            np.testing.assert_array_equal(section['seismic'], section['seismic_baseline'])
            np.testing.assert_array_equal(section['porosity_delta'], 0)
        result = self.run_forward()
        for base, new in zip(zero['sections'], result['sections']):
            np.testing.assert_array_equal(base['sand_fraction'], new['sand_fraction'])
            self.assertGreater(float(np.max(abs(new['seismic'] - new['seismic_baseline']))), 1e-5)
            self.assertLessEqual(float(np.max(abs(new['porosity_delta']))), .025001)
            np.testing.assert_allclose(base['vp_m_s'] - new['vp_m_s'], 6000 * new['porosity_delta'], atol=.001)

    def test_shared_xyz_crossing_coordinates_and_depth_geometry(self):
        result = self.run_forward()
        a, b = result['sections']
        self.assertEqual(a['actual_fixed_m'], 500.)
        for key in ('seismic', 'vp_m_s', 'sand_fraction', 'reflectivity'):
            np.testing.assert_allclose(a[key][2], b[key][2], atol=1e-7)
        # The selected incision remains deeper, rather than being flattened.
        self.assertGreater(float(a['fan_top_depth_m'][2]), float(a['fan_top_depth_m'][1]))
        np.testing.assert_allclose(a['fan_top_depth_m'][[1, 2]], [1080, 1160], atol=.001)
        np.testing.assert_allclose(a['fan_base_depth_m'], 1400, atol=.001)
        self.assertTrue(np.all(a['depth_m'][1:] > a['depth_m'][:-1]))
        np.testing.assert_array_equal(np.load(self.root / 'bed_top_elevation_m.npy'), self.raw)

    def test_reproducible_seed_and_roi_invariant_spatial_random_field(self):
        x = np.arange(9.) * 125; y = np.arange(7.) * 125; z = np.arange(11.) * 5
        whole = _normal_field(x, y, z, 125, (0, 0), 42)
        crop = _normal_field(x[2:7], y[1:6], z, 125, (0, 0), 42)
        np.testing.assert_array_equal(whole[2:7, 1:6], crop)
        a = self.run_forward(); b = self.run_forward()
        for aa, bb in zip(a['sections'], b['sections']):
            np.testing.assert_array_equal(aa['seismic'], bb['seismic'])

    def test_missing_composition_and_initial_record_are_not_invented(self):
        missing = self.hint[0].copy(); missing[1, 1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, '未知'):
            self.run_forward(hint=(missing, self.hint[1]))
        initial = self.model.extract_sections(0, 500, 500, 'both', 500)
        with self.assertRaisesRegex(ValueError, '初始'):
            forward_current(self.model, initial, self.params, filled_hint=self.hint)

    def test_scenario_outside_depth_range_and_invalid_sampling_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '深度范围'):
            self.run_forward(params=replace(self.params, vertical_scale=64, burial_depth_m=1000))
        for config in ({'frequency_hz': float('nan')}, {'spacing_z_m': 20.},
                       {'vertical_scale': 0.}, {'porosity_strength': -1.}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                replace(self.params, **config).validate()

    def test_cut_uses_exact_fixed_coordinate_without_snapping_to_compute_lattice(self):
        xx, yy = np.meshgrid(self.x, self.y, indexing='ij')
        field = (2 * xx + yy)[:, :, None] * np.array([1., 2.])
        actual = _line(field, self.x, self.y, 'x', 375., self.x)
        np.testing.assert_allclose(actual[:, 0], self.x * 2 + 375)

    def test_flat_reflector_has_requested_dominant_frequency_and_zero_phase(self):
        from scipy.fft import rfft, rfftfreq
        from seismic_reference import psf_image
        r = np.zeros((48, 48, 241), dtype='f4'); r[:, :, 120] = .1
        image, _, _ = psf_image(r, [125., 125., 5.], 2800., 30.)
        trace = image[24, 24]
        peak_hz = rfftfreq(len(trace), 2 * 5 / 2800)[np.argmax(abs(rfft(trace)))]
        self.assertLess(abs(peak_hz - 30), 3.)
        self.assertEqual(int(np.argmax(trace)), 120)
        np.testing.assert_allclose(trace[115:120], trace[121:126][::-1], atol=.001)

    def test_earlier_selected_record_excludes_all_later_geometry(self):
        sections = self.model.extract_sections(1, 500, 500, 'x', 500)
        hint = tuple(a[:2] for a in self.hint)
        a = forward_current(self.model, sections, self.params, filled_hint=hint)
        values = np.array(self.model.raw)
        self.model.raw._mmap.close()
        values[2] = -10000.  # Extreme later erosion must not erase this view.
        self.model.raw = values
        b = forward_current(self.model, sections, self.params, filled_hint=hint)
        np.testing.assert_array_equal(a['sections'][0]['seismic'], b['sections'][0]['seismic'])
        np.testing.assert_array_equal(a['sections'][0]['fan_base_depth_m'], b['sections'][0]['fan_base_depth_m'])
        self.assertEqual(b['metadata']['selected_record'], 1)

    def test_inactive_native_columns_are_masked_not_zero_sand_observations(self):
        valid = self.model.valid.copy(); valid[0] = False; self.model.valid = valid
        result = self.run_forward()
        self.assertFalse(result['sections'][0]['valid'][0])
        self.assertTrue(np.isfinite(result['sections'][0]['seismic']).all())


if __name__ == '__main__':
    unittest.main()
