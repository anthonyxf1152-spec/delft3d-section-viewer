"""Continuation branches, gaps, erosion, cache isolation and failure rollback."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import series_loader as series
from stratigraphy_model import StratigraphyModel


class SeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='delft-series-test-')
        self.root = Path(self.temp.name)
        self.specs = {}
        self.a = self.make('a', [0., 1., 2.], [5., 6., 7.])
        self.b = self.make('b', [1., 2., 3.], [6., 6.5, 5.5])
        self.models = []
        self.patches = [patch.object(series.single, 'inspect_source', side_effect=self.inspect),
                        patch.object(series.single, 'load_project', side_effect=self.single_load)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for model in self.models:
            series.close_model(model)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def make(self, name, times, heights, *, base=0., sand_ratio=.6, x=(0., 100.)):
        path = self.root / name
        path.mkdir()
        dat = path / (name + '.dat'); dat.write_bytes(name.encode())
        definition = dat.with_suffix('.def'); definition.write_bytes(b'definition')
        raw = np.broadcast_to(np.array(heights, dtype='f4')[:, None, None], (len(times), 2, 2)).copy()
        thickness = raw - base
        for filename, values in zip(series.ARRAYS, (raw, thickness * sand_ratio, thickness * (1 - sand_ratio))):
            np.save(path / filename, values.astype('f4'))
        np.savez(path / 'native_geometry.npz', x_m=x, y_m=[0., 100.], valid=np.ones((2, 2), bool),
                 native_m_indices=[0, 1], native_n_indices=[0, 1], hydro_days=times, morph_days=np.array(times) * 10,
                 initial_base_elevation_m=np.full((2, 2), base), final_base_elevation_m=np.full((2, 2), base))
        (path / 'metadata.json').write_text(json.dumps({'build_complete': True, 'shape_time_x_y': list(raw.shape)}))
        (path / 'sand_cache_metadata.json').write_text(json.dumps({'complete': True, 'shape_time_xy': list(raw.shape),
            'files': list(series.ARRAYS[1:]), 'fraction_vs_LYRFRAC_max_error': 0., 'height_vs_DP_BEDLYR_max_error_m': 0.}))
        source = {'dat_file': str(dat), 'def_file': str(definition)}
        self.specs[str(dat)] = (path, len(times))
        return source

    def inspect(self, dat_file, def_file=None):
        return dict(dat_file=str(dat_file), def_file=str(def_file), records=self.specs[str(dat_file)][1],
                    sediment_names=['sand', 'mud'], source_signature=series.single.source_signature(dat_file, def_file))

    def single_load(self, dat_file, def_file=None, **kwargs):
        model = StratigraphyModel(self.specs[str(dat_file)][0])
        model.source_info = self.inspect(dat_file, def_file)
        self.models.append(model)
        return model

    def load(self, sources=None, **kwargs):
        model = series.load_series(sources or [self.a, self.b], cache_root=self.root / 'cache',
                                   progress=lambda _: None, **kwargs)
        self.models.append(model)
        return model

    def test_later_branch_replaces_tail_with_verified_anchor_and_cross_period_erosion(self):
        model = self.load()
        np.testing.assert_array_equal(model.hydro_days, [0, 1, 2, 3])
        np.testing.assert_array_equal(model.raw[:, 0, 0], [5, 6, 6.5, 5.5])
        np.testing.assert_array_equal(model.interfaces_until(3)[:, 0, 0], [5, 5.5, 5.5, 5.5])
        seam = model.source_info['seams'][0]
        self.assertEqual(seam['discarded_previous_tail_records'], 1)
        self.assertTrue(seam['duplicate_anchor_removed'])
        self.assertEqual(model.source_info['record_sources'][1], {'source_index': 1, 'source_record': 0})
        self.assertTrue(self.load().cache_reused)

    def test_real_gap_keeps_time_without_interpolation(self):
        c = self.make('c', [5., 6.], [7.5, 8.])
        model = self.load([self.a, c])
        np.testing.assert_array_equal(model.hydro_days, [0, 1, 2, 5, 6])
        self.assertEqual(model.source_info['seams'][0]['unobserved_interval']['duration_hydro_days'], 3.)

    def test_reset_offsets_both_clocks_and_deduplicates(self):
        c = self.make('c', [0., 1., 2.], [7., 8., 9.])
        model = self.load([self.a, c], time_mode='reset')
        np.testing.assert_array_equal(model.hydro_days, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(model.morph_days, [0, 10, 20, 30, 40])
        self.assertEqual(model.source_info['seams'][0]['hydro_shift_days'], 2.)

    def test_geometry_grid_base_and_inventory_mismatch_rejected(self):
        for name, config in [('bed', {'heights': [6.1, 8.]}),
                             ('grid', {'x': (0., 101.)}), ('base', {'base': -1.}),
                             ('inventory', {'sand_ratio': .5})]:
            with self.subTest(name=name):
                heights = config.pop('heights', [6., 8.])
                c = self.make(name, [1., 3.], heights, **config)
                with self.assertRaises(ValueError):
                    self.load([self.a, c])
                self.assertEqual(list((self.root / 'cache').glob('build-*')), [])

    def test_wrong_order_and_duplicate_file_rejected(self):
        for sources in ([self.b, self.a], [self.a, self.a]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                self.load(sources)

    def test_corrupt_merged_cache_rebuild_and_failed_merge_preserves_old_result(self):
        model = self.load()
        series.close_model(model)
        (model.volume_dir / 'native_geometry.npz').write_bytes(b'corrupt')
        rebuilt = self.load()
        self.assertFalse(rebuilt.cache_reused)
        c = self.make('bad', [2., 3.], [99., 100.])
        with self.assertRaises(ValueError):
            self.load([self.a, c])
        np.testing.assert_array_equal(rebuilt.raw[:, 0, 0], [5, 6, 6.5, 5.5])

    def test_source_change_before_commit_rolls_back_combined_cache(self):
        original = series._build
        def changing(*args):
            original(*args)
            Path(self.b['dat_file']).write_bytes(b'changed source')
        with patch.object(series, '_build', side_effect=changing), self.assertRaisesRegex(ValueError, '变化'):
            self.load()
        self.assertEqual(list((self.root / 'cache').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
