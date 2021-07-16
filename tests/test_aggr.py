import unittest

import pegasusio as io

class TestAggr(unittest.TestCase):

    def test_aggregate_matrices(self):
        data = io.read_input("pegasusio-test-data/aggr_result.zarr.zip")

        self.assertEqual(data.shape, (16303, 36601), "Count matrix shape differs!")
        self.assertTrue('Donor' in data.obs, "Attribute 'Donor' is not included in the resulting count matrix!")
        self.assertEqual(data.obs['Channel'].cat.categories.size, 3, "Some sample is not included in the result count matrix!")
        self.assertLessEqual(data.obs['percent_mito'].max(), 10, "Filtration based on '--percent-mito' fails!")
        self.assertGreaterEqual(data.obs['n_genes'].min(), 100, "Filtration based on '--min-genes' fails!")


if __name__ == '__main__':
    unittest.main()
