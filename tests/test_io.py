import unittest

import pegasusio as io

class TestIO(unittest.TestCase):

    def test_h5ad(self):
        data = io.read_input("pegasusio-test-data/case1/pbmc3k.h5ad", genome = 'hg19')
        io.write_output(data, "pegasusio-test-data/case1/pbmc3k_out.h5ad")
        data = io.read_input("pegasusio-test-data/case1/pbmc3k_out.h5ad")

        self.assertEqual(data.shape, (2638, 1838), "Count matrix shape differs!")
        self.assertEqual(data.get_genome(), "hg19", "Genome differs!")
        self.assertEqual(data.get_modality(), "rna", "Modality differs!")

    def test_mixture_data(self):
        data = io.read_input("pegasusio-test-data/case2/1k_hgmm_v3_filtered_feature_bc_matrix.h5")
        data.select_data('mm10-rna')
        self.assertEqual(data.shape, (1063, 54232), "Mouse data shape differs!")
        self.assertEqual(data.get_genome(), "mm10", "Mouse data genome differs!")
        self.assertEqual(data.get_modality(), "rna", "Mouse data modality differs!")
        data.select_data('hg19-rna')
        self.assertEqual(data.shape, (1063, 57905), "Human data shape differs!")
        self.assertEqual(data.get_genome(), "hg19", "Human data genome differs!")
        self.assertEqual(data.get_modality(), "rna", "Human data modality differs!")

    def test_10x_mtx(self):
        data = io.read_input("pegasusio-test-data/case3/42468c97-1c5a-4c9f-86ea-9eaa1239445a.mtx", genome = 'hg19')
        io.write_output(data, "pegasusio-test-data/case3/test.mtx")
        data = io.read_input("pegasusio-test-data/case3/test.mtx")

        self.assertEqual(data.shape, (2544, 58347), "Count matrix shape differs!")
        self.assertEqual(data.get_genome(), "hg19", "Genome differs!")
        self.assertEqual(data.get_modality(), "rna", "Modality differs!")

    def test_loom(self):
        data = io.read_input("pegasusio-test-data/case3/pancreas.loom", genome = 'hg19')
        io.write_output(data, "pegasusio-test-data/case3/pancreas_out.loom")
        data = io.read_input("pegasusio-test-data/case3/pancreas_out.loom")

        self.assertEqual(data.shape, (2544, 58347), "Count matrix shape differs!")
        self.assertEqual(data.get_genome(), "hg19", "Genome differs!")
        self.assertEqual(data.get_modality(), "rna", "Modality differs!")

    def test_zarr(self):
        data = io.read_input("pegasusio-test-data/case4/MantonBM1_1_dbls.zarr")
        io.write_output(data, "pegasusio-test-data/case4/MantonBM_out.zarr")
        data = io.read_input("pegasusio-test-data/case4/MantonBM_out.zarr")

        self.assertEqual(data.shape, (4274, 19360), "Count matrix shape differs!")
        self.assertEqual(data.get_genome(), "GRCh38", "Genome differs!")
        self.assertEqual(data.get_modality(), "rna", "Modality differs!")

if __name__ == '__main__':
    unittest.main()
