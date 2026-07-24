import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import pegasusio as io


class TestAggr(unittest.TestCase):

    def test_aggregate_count_matrix(self):
        def make_data(barcodes, features, counts):
            obs = pd.DataFrame(index=pd.Index(barcodes, name="barcodekey"))
            var = pd.DataFrame(index=pd.Index(features, name="featurekey"))
            unidata = io.UnimodalData(
                obs,
                var,
                {"X": csr_matrix(counts, dtype=np.int32)},
                {"genome": "GRCh38", "modality": "rna"},
            )
            return io.MultimodalData(unidata)

        sample1 = make_data(
            ["cell1", "cell2"],
            ["gene1", "gene2"],
            [[1, 2], [0, 3]],
        )
        sample2 = make_data(
            ["cell3", "cell4"],
            ["gene2", "gene3"],
            [[4, 5], [6, 0]],
        )
        sample3 = make_data(
            ["cell5", "cell6"],
            ["gene3", "gene4", "gene1"],
            [[7, 8, 9], [10, 0, 11]],
        )

        result = io.aggregate_matrices(
            {
                "Sample": ["sample1", "sample2", "sample3"],
                "Object": [sample1, sample2, sample3],
            }
        )
        actual = pd.DataFrame(
            result.X.toarray(),
            index=result.obs_names,
            columns=result.var_names,
        )
        expected = pd.DataFrame(
            [
                [1, 2, 0, 0],
                [0, 3, 0, 0],
                [0, 4, 5, 0],
                [0, 6, 0, 0],
                [9, 0, 7, 8],
                [11, 0, 10, 0],
            ],
            index=pd.Index(
                [
                    "sample1-cell1",
                    "sample1-cell2",
                    "sample2-cell3",
                    "sample2-cell4",
                    "sample3-cell5",
                    "sample3-cell6",
                ],
                name="barcodekey",
            ),
            columns=pd.Index(
                ["gene1", "gene2", "gene3", "gene4"],
                name="featurekey",
            ),
            dtype=np.int32,
        )

        pd.testing.assert_frame_equal(
            actual.loc[expected.index, expected.columns],
            expected,
        )

    def test_aggregate_matrices(self):
        data = io.read_input("pegasusio-test-data/aggr_result.zarr.zip")

        self.assertEqual(data.shape, (16303, 36601), "Count matrix shape differs!")
        self.assertTrue('Donor' in data.obs, "Attribute 'Donor' is not included in the resulting count matrix!")
        self.assertEqual(data.obs['Channel'].cat.categories.size, 3, "Some sample is not included in the result count matrix!")
        self.assertLessEqual(data.obs['percent_mito'].max(), 10, "Filtration based on '--percent-mito' fails!")
        self.assertGreaterEqual(data.obs['n_genes'].min(), 100, "Filtration based on '--min-genes' fails!")


if __name__ == '__main__':
    unittest.main()
