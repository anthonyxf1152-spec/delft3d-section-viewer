from pathlib import Path
from unittest import mock
import json
import tempfile
import unittest

import numpy as np

from seismic_forward import (SeismicParameters, build_current_model, save_current_model,
    forward_current, estimate_current_grid, MODEL_VOLUME_NAMES, model_sections)
from seismic_streaming import save_3d_model_streaming
from stratigraphy_model import StratigraphyModel
from series_loader import close_model


class StreamingModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='delft-stream-test-')
        self.root = Path(self.temp.name)
        x, y = np.arange(17.) * 500, np.arange(19.) * 500
        xx, yy = np.meshgrid(x, y, indexing='ij')
        initial = .001 * yy - .0002 * xx
        raw = np.array([initial, initial + 10, initial + 20], dtype='f4')
        raw[-1, 8, 9] -= 5
        np.save(self.root / 'bed_top_elevation_m.npy', raw)
        np.savez(self.root / 'native_geometry.npz', x_m=x, y_m=y, valid=np.ones(xx.shape, bool),
            native_m_indices=np.arange(len(x)), native_n_indices=np.arange(len(y)),
            morph_days=[0., 10., 20.], hydro_days=[0., 1., 2.], initial_base_elevation_m=initial - 4.5)
        self.model = StratigraphyModel(self.root)
        self.params = SeismicParameters(spacing_xy_m=500., spacing_z_m=10., depth_max_m=1500.)
        self.sections = self.model.extract_sections(2, 4000, 4500, 'both', 2500.)
        self.hint = (np.full(raw.shape, .8, dtype='f4'), np.ones(raw.shape, dtype='u1'))

    def tearDown(self):
        close_model(self.model); self.temp.cleanup()

    def test_tiles_match_whole_model_including_noise_halos_and_invalid_source_support(self):
        valid = self.model.valid.copy()
        valid[:2] = False; valid[5:12, 6:13] = False
        self.model.valid = valid
        whole = build_current_model(self.model, self.sections, self.params, filled_hint=self.hint)
        with mock.patch('seismic_forward.psf_image', side_effect=AssertionError('saving must not convolve')):
            output = save_3d_model_streaming(self.root / 'streamed', self.model, self.sections,
                self.params, filled_hint=self.hint, tile_shape_xy=(3, 4))
        for name in MODEL_VOLUME_NAMES:
            np.testing.assert_array_equal(np.load(output / f'{name}.npy', allow_pickle=False), whole['volumes'][name])
        for name in whole['surfaces']:
            np.testing.assert_array_equal(np.load(output / f'{name}.npy'), whole['surfaces'][name])
        np.testing.assert_array_equal(np.load(output / 'impedance_kg_m2_s.npy'),
            whole['volumes']['vp_m_s'] * whole['volumes']['density_kg_m3'])
        for cut in model_sections(whole, self.sections):
            for name in MODEL_VOLUME_NAMES:
                np.testing.assert_array_equal(np.load(output / (cut['axis'] + 'z') / f'{name}.npy'), cut[name])
        report = json.loads((output / 'model_parameters.json').read_text(encoding='utf-8'))
        self.assertEqual(report['grid_shape_xyz'], whole['metadata']['grid_shape_xyz'])
        self.assertGreater(report['streaming_save']['tile_count'], 1)
        self.assertFalse(report['convolution_performed'])
        self.assertEqual(report['arrays']['vp_m_s.npy']['axis_order'], 'X,Y,depth')

    def test_forward_limit_does_not_block_complete_numeric_3d_npy_save(self):
        with mock.patch('seismic_forward.MAX_GRID_VOXELS', 10):
            estimate = estimate_current_grid(self.model, self.sections, self.params)
            self.assertFalse(estimate['within_grid_limit'])
            with self.assertRaisesRegex(ValueError, '网格过大'):
                forward_current(self.model, self.sections, self.params, filled_hint=self.hint)
            with mock.patch('seismic_forward.psf_image', side_effect=AssertionError('no PSF')):
                output = save_current_model(self.root / 'beyond_forward_limit', self.model,
                    self.sections, self.params, filled_hint=self.hint)
        report = json.loads((output / 'model_parameters.json').read_text(encoding='utf-8'))
        self.assertTrue(report['model_save_ignores_forward_voxel_limit'])
        self.assertFalse(report['grid_estimate']['within_grid_limit'])
        for name, info in report['arrays'].items():
            array = np.load(output / name, mmap_mode='r', allow_pickle=False)
            self.assertEqual(list(array.shape), info['shape'])
            self.assertEqual(str(array.dtype), info['dtype'])
            array._mmap.close()

    def test_failed_tile_cleans_files_and_existing_output_is_not_overwritten(self):
        failed = self.root / 'failed'
        from seismic_streaming import _mapped_write
        calls = 0
        def fail_write(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 10:
                raise OSError('simulated tile write failure')
            return _mapped_write(*args, **kwargs)
        with mock.patch('seismic_streaming._mapped_write', side_effect=fail_write):
            with self.assertRaisesRegex(OSError, 'simulated tile'):
                save_3d_model_streaming(failed, self.model, self.sections, self.params,
                    filled_hint=self.hint, tile_shape_xy=(3, 4))
        self.assertFalse(failed.exists())
        self.assertEqual(list(self.root.glob('.model-writing-*')), [])
        existing = self.root / 'existing'; existing.mkdir()
        sentinel = existing / 'keep.txt'; sentinel.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            save_current_model(existing, self.model, self.sections, self.params, filled_hint=self.hint)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
