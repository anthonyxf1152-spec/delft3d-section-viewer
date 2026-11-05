from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from saved_sand_data import SavedSandModel, sampled_indices, _index_to_world
from saved_sand_scene import plane_geometry


class SavedSandDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='saved-sand-test-')
        self.root = Path(self.temp.name)
        self.shape = (9,11,13)
        self.axes = [30000 + np.arange(9)*25., 40000 + np.arange(11)*50., np.arange(13)*5.]
        i,j,k = np.indices(self.shape)
        self.q = ((i + j + k) / 30).astype('f4')
        occ = np.zeros(self.shape,dtype='f4'); occ[:,:,3:10] = 1
        valid = np.ones(self.shape[:2],bool); valid[0] = False
        for name,array in [('sand_fraction',self.q),('occupancy',occ),('valid_xy',valid),
                           *zip(('x_m','y_m','depth_m'),self.axes)]:
            np.save(self.root / f'{name}.npy',array,allow_pickle=False)
        metadata = {'complete':True,'grid_shape_xyz':list(self.shape),
            'sections':[{'axis':'x','actual_fixed_m':40100.},{'axis':'y','actual_fixed_m':30150.}]}
        (self.root/'model_parameters.json').write_text(json.dumps(metadata),encoding='utf-8')
        self.model = None

    def tearDown(self):
        if self.model is not None:
            self.model.close()
        self.temp.cleanup()

    def load(self,path=None):
        self.model = SavedSandModel(path or self.root)
        return self.model

    def test_readonly_memory_map_coordinates_positions_and_independence_of_native_files(self):
        model = self.load(self.root/'sand_fraction.npy')
        self.assertIsInstance(model.q,np.memmap)
        self.assertFalse(model.q.flags.writeable)
        self.assertEqual(model.shape,self.shape)
        self.assertEqual(model.initial_indices(),[6,2,6])
        with self.assertRaises(ValueError):
            model.q[1,1,1] = .5
        model.close()
        with self.assertRaisesRegex(ValueError,'关闭'):
            model.plane(0,2)

    def test_all_three_plane_orientations_and_depth_down_physical_coordinates(self):
        model = self.load()
        for axis,index in [(0,4),(1,5),(2,6)]:
            plane = model.plane(axis,index)
            expected = np.take(self.q,index,axis=axis)
            np.testing.assert_array_equal(plane['values'].reshape(plane['shape']),expected)
            coords = plane['vertices'].reshape(tuple(plane['shape'])+(3,))
            constant = self.axes[axis][index] * (-.001 if axis==2 else .001)
            np.testing.assert_allclose(coords[...,axis],constant,atol=1e-5)
            if axis != 2:
                self.assertLess(float(coords[0,-1,2]),float(coords[0,0,2]))
            vertices,faces,values = plane_geometry(plane)
            self.assertEqual(len(vertices),expected.size)
            self.assertEqual(len(values),expected.size)
            self.assertGreater(len(faces),0)
            self.assertTrue(np.all(plane['valid'].ravel()[faces]))
        self.assertFalse(model.plane(0,0)['valid'].any())

    def test_display_sampling_is_bounded_keeps_endpoints_and_does_not_change_source(self):
        model = self.load()
        for shape,budget in [((711,801,51),2_000_000),((1201,1251,361),500_000)]:
            picks = sampled_indices(shape,budget,max_axis=320,preserve_depth=True)
            self.assertLessEqual(np.prod([len(a) for a in picks]),budget)
            for n,indices in zip(shape,picks):
                self.assertEqual(indices[0],0); self.assertEqual(indices[-1],n-1)
                self.assertTrue(np.all(np.diff(indices)>0))
        before = (self.root/'sand_fraction.npy').read_bytes()
        sampled,inside,picks = model.surface_grid(budget=128,threshold_percent=60)
        self.assertLessEqual(sampled.size,128)
        self.assertFalse(inside[0].any())
        self.assertTrue(np.all(sampled[inside]>=.6))
        plane = model.plane(2,6,budget=40)
        self.assertLessEqual(len(plane['values']),40)
        self.assertEqual((self.root/'sand_fraction.npy').read_bytes(),before)

    def test_mismatched_axes_mask_percent_values_and_incomplete_metadata_are_rejected(self):
        cases = [
            ('x_m',np.arange(8.)),
            ('valid_xy',np.ones((2,3),bool)),
            ('sand_fraction',self.q*100),
            ('occupancy',np.ones((2,3,4),dtype='f4')),
            ('depth_m',-np.arange(13.)),
        ]
        for name,broken in cases:
            path = self.root/f'{name}.npy'; original = path.read_bytes()
            with self.subTest(name=name):
                np.save(path,broken,allow_pickle=False)
                with self.assertRaises(ValueError):
                    SavedSandModel(self.root)
                path.write_bytes(original)
        (self.root/'model_parameters.json').write_text('{"complete":false}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'未保存完整'):
            SavedSandModel(self.root)

    def test_two_dimensional_cut_file_is_rejected_and_index_to_world_preserves_depth_sign(self):
        np.save(self.root/'xz.npy',self.q[3],allow_pickle=False)
        with self.assertRaisesRegex(ValueError,'三维数组'):
            SavedSandModel(self.root/'xz.npy')
        coordinates = _index_to_world(np.array([[0,0,0],[1,1,1],[-.5,2,2]],dtype='f4'),
            [np.array([30000.,30025.,30050.]),np.array([40000.,40050.,40100.]),np.array([0.,5.,10.])])
        np.testing.assert_allclose(coordinates,[[30,40,0],[30.025,40.05,-.005],[29.9875,40.1,-.01]],atol=1e-5)


if __name__ == '__main__':
    unittest.main()
