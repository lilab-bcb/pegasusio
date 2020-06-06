import unittest

import pegasusio as io

class TestIO(unittest.TestCase):

    def test_read_h5ad(self):
        data = io.read_input("tests/pegasusio-test-data/case1/pbmc3k.h5ad", genome = 'hg19')
        self.assertEqual(data.shape, (2638, 1838))

    def test_mixture_data(self):
        data = io.read_input("tests/pegasusio-test-data/case2/1k_hgmm_v3_filtered_feature_bc_matrix.h5")
        data.select_data('mm10-rna')
        self.assertEqual(data.shape, (1063, 54232))
        data.select_data('hg19-rna')
        self.assertEqual(data.shape, (1063, 57905))

    def test_read_10x_mtx(self):
        data = io.read_input("tests/pegasusio-test-data/case3/42468c97-1c5a-4c9f-86ea-9eaa1239445a.mtx", genome = 'hg19')
        self.assertEqual(data.shape, (2544, 58347))

    def test_read_loom(self):
        data = io.read_input("tests/pegasusio-test-data/case3/pancreas.loom", genome = 'hg19')
        self.assertEqual(data.shape, (2544, 58347))

    def test_read_zarr(self):
        data = io.read_input("tests/pegasusio-test-data/case4/MantonBM1_1_dbls.zarr")
        self.assertEqual(data.shape, (4274, 19360))

if __name__ == '__main__':
    unittest.main()
