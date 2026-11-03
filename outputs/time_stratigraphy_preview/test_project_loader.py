"""Portable transactional loader checks; no real Delft3D result files required."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import project_loader as loader


def fake_inspect(dat_file, def_file=None):
    dat, defn = loader.resolve_pair(dat_file, def_file)
    return {'dat_file': str(dat), 'def_file': str(defn), 'sediment_names': ['sand', 'mud'],
            'records': 2, 'source_signature': loader.source_signature(dat, defn)}


def fake_bed(dat_file, output, **kwargs):
    output = Path(output)
    value = Path(dat_file).read_bytes()[0]
    raw = np.full((2, 2, 2), value, dtype='f4')
    for name in ('bed_top_elevation_m.npy', 'surviving_interfaces_elevation_m.npy'):
        np.save(output / name, raw)
    np.savez(output / 'native_geometry.npz', x_m=[0., 100.], y_m=[0., 100.],
             valid=np.ones((2, 2), bool), morph_days=[0., 1.], hydro_days=[0., 1.],
             initial_base_elevation_m=np.zeros((2, 2)), native_m_indices=[0, 1], native_n_indices=[0, 1])
    (output / 'metadata.json').write_text(json.dumps({'build_complete': True, 'shape_time_x_y': [2, 2, 2]}))


def fake_sand(output, **kwargs):
    output = Path(output)
    for name in ('sand_equivalent_thickness_m.npy', 'mud_equivalent_thickness_m.npy'):
        np.save(output / name, np.ones((2, 2, 2), dtype='f4'))
    (output / 'sand_cache_metadata.json').write_text(json.dumps({'complete': True, 'shape_time_xy': [2, 2, 2]}))


class ProjectLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='delft3d-loader-test-')
        self.root = Path(self.temp.name)
        self.dat = self.root / 'result.dat'; self.dat.write_bytes(b'A first dataset')
        self.defn = self.root / 'custom-definition.def'; self.defn.write_bytes(b'definition A')
        self.cache = self.root / 'cache'
        self.models = []
        self.patches = [patch.object(loader, 'inspect_source', side_effect=fake_inspect),
                        patch.object(loader, 'build_cache', side_effect=fake_bed),
                        patch.object(loader, 'build_sand_cache', side_effect=fake_sand)]
        self.mocks = [item.start() for item in self.patches]

    def tearDown(self):
        for model in self.models:
            if not model.raw._mmap.closed:
                model.raw._mmap.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def load(self, dat=None, **kwargs):
        model = loader.load_project(dat or self.dat, self.defn,
                                    cache_root=self.cache, progress=lambda message: None, **kwargs)
        self.models.append(model)
        return model

    def test_explicit_def_reuse_and_configuration_isolation(self):
        first = self.load()
        again = self.load()
        self.assertFalse(first.cache_reused)
        self.assertTrue(again.cache_reused)
        self.assertEqual(self.mocks[1].call_count, 1)
        self.assertEqual(first.source_info['def_file'], str(self.defn.resolve()))
        changed = self.load(sand_density=1500.)
        self.assertNotEqual(first.volume_dir, changed.volume_dir)
        self.assertFalse(changed.cache_reused)

    def test_different_files_have_independent_values(self):
        first = self.load()
        second_dat = self.root / 'another.dat'; second_dat.write_bytes(b'B different dataset')
        second = self.load(second_dat)
        self.assertNotEqual(first.volume_dir, second.volume_dir)
        self.assertEqual(first.raw[0, 0, 0], ord('A'))
        self.assertEqual(second.raw[0, 0, 0], ord('B'))

    def test_failure_leaves_existing_project_usable_and_removes_staging(self):
        first = self.load()
        old_path = first.volume_dir
        self.mocks[2].side_effect = ValueError('bad density closure')
        with self.assertRaisesRegex(ValueError, 'bad density closure'):
            self.load(sand_density=1234.)
        self.assertTrue(old_path.is_dir())
        self.assertEqual(first.raw[0, 0, 0], ord('A'))
        self.assertEqual(list(self.cache.glob('build-*')), [])
        self.assertTrue(self.load().cache_reused)

    def test_source_change_during_build_aborts_commit(self):
        def changing_inventory(output, **kwargs):
            fake_sand(output, **kwargs)
            self.defn.write_bytes(b'changed definition during read')
        self.mocks[2].side_effect = changing_inventory
        with self.assertRaisesRegex(ValueError, '发生变化'):
            self.load()
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_corrupt_cache_is_rebuilt(self):
        first = self.load()
        first.raw._mmap.close()
        (first.volume_dir / 'native_geometry.npz').write_bytes(b'broken compressed file')
        rebuilt = self.load()
        self.assertFalse(rebuilt.cache_reused)
        self.assertEqual(self.mocks[1].call_count, 2)
        self.assertTrue(rebuilt.volume_dir.is_dir())

    def test_invalid_density_and_components_rejected(self):
        for config in ({'sand_density': 0}, {'mud_density': float('nan')},
                       {'sand_index': 1, 'mud_index': 1}, {'sand_index': .5}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.load(**config)
        self.assertFalse(self.cache.exists())


if __name__ == '__main__':
    unittest.main()
