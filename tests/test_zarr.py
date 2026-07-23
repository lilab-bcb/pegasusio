import os
import tempfile
import unittest
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import pegasusio as io


def _make_unimodal_data():
    obs = pd.DataFrame(
        {
            "Channel": pd.Categorical(["channel1", "channel2", "channel1"]),
            "n_counts": np.array([3, 4, 9], dtype=np.int64),
        },
        index=pd.Index(["cell1", "cell2", "cell3"], dtype=object, name="barcodekey"),
    )
    var = pd.DataFrame(
        {
            "featureid": ["ENSG1", "ENSG2", "ENSG3"],
            "featuretype": pd.Categorical(
                ["Gene Expression", "Gene Expression", "Gene Expression"]
            ),
        },
        index=pd.Index(["GeneA", "GeneB", "GeneC"], dtype=object, name="featurekey"),
    )
    X = csr_matrix(
        np.array(
            [
                [1, 0, 2],
                [0, 3, 1],
                [4, 0, 5],
            ],
            dtype=np.float32,
        )
    )
    counts = csr_matrix(
        np.array(
            [
                [2, 0, 3],
                [0, 4, 1],
                [5, 0, 6],
            ],
            dtype=np.float32,
        )
    )
    umap = np.array([[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]], dtype=np.float32)
    loadings = np.array(
        [[0.4, 1.4], [0.5, 1.5], [0.6, 1.6]], dtype=np.float32
    )
    connectivities = csr_matrix(
        np.array(
            [
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
    )

    return io.UnimodalData(
        obs,
        var,
        {"X": X, "counts": counts},
        {
            "genome": "GRCh38",
            "modality": "rna",
            "uid": "GRCh38-rna",
            "source": "zarr-zip-test",
        },
        barcode_multiarrays={"X_umap": umap},
        feature_multiarrays={"PCs": loadings},
        barcode_multigraphs={"connectivities": connectivities},
        cur_matrix="X",
    )


class TestZarrZip(unittest.TestCase):

    def _assert_roundtrip_equal(self, data, expected):
        unidata = data.current_data()

        self.assertEqual(data.current_key(), "GRCh38-rna")
        self.assertEqual(unidata.shape, expected.shape)
        self.assertEqual(unidata.get_genome(), "GRCh38")
        self.assertEqual(unidata.get_modality(), "rna")
        self.assertEqual(unidata.get_uid(), "GRCh38-rna")
        self.assertEqual(unidata.metadata["source"], "zarr-zip-test")
        pd.testing.assert_frame_equal(unidata.obs, expected.obs, check_dtype=False)
        pd.testing.assert_frame_equal(unidata.var, expected.var, check_dtype=False)
        np.testing.assert_array_equal(unidata.X.toarray(), expected.X.toarray())
        np.testing.assert_array_equal(
            unidata.matrices["counts"].toarray(),
            expected.matrices["counts"].toarray(),
        )
        np.testing.assert_array_equal(unidata.obsm["X_umap"], expected.obsm["X_umap"])
        np.testing.assert_array_equal(unidata.varm["PCs"], expected.varm["PCs"])
        np.testing.assert_array_equal(
            unidata.obsp["connectivities"].toarray(),
            expected.obsp["connectivities"].toarray(),
        )

    def test_read_zarr_zip(self):
        expected = _make_unimodal_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.zarr.zip")
            io.write_output(expected, path)

            data = io.read_input(path)

        self._assert_roundtrip_equal(data, expected)

    def test_write_zarr_zip(self):
        expected = _make_unimodal_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.zarr.zip")
            io.write_output(expected, path)

            self.assertTrue(zipfile.is_zipfile(path))
            with zipfile.ZipFile(path, mode="r") as zf:
                names = zf.namelist()
            duplicated = [name for name, count in Counter(names).items() if count > 1]
            self.assertEqual(duplicated, [])
            self.assertTrue(
                any(name.endswith(".zgroup") or name.endswith("zarr.json") for name in names)
            )

            data = io.read_input(path)

        self._assert_roundtrip_equal(data, expected)


if __name__ == '__main__':
    unittest.main()
