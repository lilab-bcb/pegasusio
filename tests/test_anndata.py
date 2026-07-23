import unittest

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import pegasusio as io


class TestAnnData(unittest.TestCase):

    def test_calc_qc_filters_with_anndata_input(self):
        adata = anndata.AnnData(
            X=csr_matrix(
                np.array(
                    [
                        [1, 0, 2],
                        [0, 0, 0],
                        [5, 1, 0],
                        [2, 2, 1],
                    ],
                    dtype=np.float32,
                )
            ),
            obs=pd.DataFrame(
                {
                    "demux_type": pd.Categorical(["singlet", "doublet", "singlet", "singlet"]),
                    "assignment": pd.Categorical(["sample1", "sample2", "sample1", "sample3"]),
                },
                index=pd.Index(["cell1", "cell2", "cell3", "cell4"], name="barcodekey"),
            ),
            var=pd.DataFrame(
                index=pd.Index(["MT-ND1", "GeneA", "GeneB"], name="featurekey")
            ),
            uns={"modality": "rna"},
        )

        io.calc_qc_filters(
            adata,
            select_singlets=True,
            min_genes=2,
            min_umis=3,
            mito_prefix="MT-",
            percent_mito=60.0,
        )

        np.testing.assert_array_equal(adata.obs["n_genes"].values, [2, 0, 2, 3])
        np.testing.assert_array_equal(adata.obs["n_counts"].values, [3, 0, 6, 5])
        np.testing.assert_allclose(
            adata.obs.loc[["cell1", "cell3"], "percent_mito"].values,
            np.array([100.0 / 3.0, 500.0 / 6.0]),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            adata.obs["passed_qc"].values,
            np.array([True, False, False, True]),
        )

    def test_unimodal_data_anndata_roundtrip(self):
        obs = pd.DataFrame(
            {
                "sample": pd.Categorical(["sample1", "sample2", "sample1"]),
                "n_counts": [3, 4, 7],
            },
            index=pd.Index(["cell1", "cell2", "cell3"], name="barcodekey"),
        )
        var = pd.DataFrame(
            {
                "featureid": ["ENSG1", "ENSG2"],
                "featuretype": pd.Categorical(["Gene Expression", "Gene Expression"]),
            },
            index=pd.Index(["GeneA", "GeneB"], name="featurekey"),
        )
        X = csr_matrix(np.array([[1, 0], [0, 2], [3, 4]], dtype=np.float32))
        counts = csr_matrix(np.array([[2, 0], [0, 3], [4, 5]], dtype=np.float32))
        pca = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        loadings = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
        neighbors = csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32))

        unidata = io.UnimodalData(
            obs.copy(),
            var.copy(),
            {"X": X, "counts": counts},
            {"genome": "GRCh38", "modality": "rna", "uid": "GRCh38-rna", "pipeline": "test"},
            barcode_multiarrays={"X_pca": pca},
            feature_multiarrays={"PCs": loadings},
            barcode_multigraphs={"connectivities": neighbors},
            cur_matrix="X",
        )

        adata = unidata.to_anndata()
        roundtrip = io.UnimodalData(adata)

        self.assertEqual(adata.shape, unidata.shape)
        np.testing.assert_array_equal(adata.X.toarray(), X.toarray())
        np.testing.assert_array_equal(adata.layers["counts"].toarray(), counts.toarray())
        np.testing.assert_array_equal(adata.obsm["X_pca"], pca)
        np.testing.assert_array_equal(adata.varm["PCs"], loadings)
        np.testing.assert_array_equal(adata.obsp["connectivities"].toarray(), neighbors.toarray())

        self.assertEqual(roundtrip.shape, unidata.shape)
        self.assertEqual(roundtrip.get_genome(), "GRCh38")
        self.assertEqual(roundtrip.get_modality(), "rna")
        self.assertEqual(roundtrip.get_uid(), "GRCh38-rna")
        self.assertEqual(roundtrip.metadata["pipeline"], "test")
        pd.testing.assert_frame_equal(roundtrip.obs, obs)
        pd.testing.assert_frame_equal(roundtrip.var, var)
        np.testing.assert_array_equal(roundtrip.X.toarray(), X.toarray())
        np.testing.assert_array_equal(roundtrip.matrices["counts"].toarray(), counts.toarray())
        np.testing.assert_array_equal(roundtrip.obsm["X_pca"], pca)
        np.testing.assert_array_equal(roundtrip.varm["PCs"], loadings)
        np.testing.assert_array_equal(roundtrip.obsp["connectivities"].toarray(), neighbors.toarray())


if __name__ == '__main__':
    unittest.main()
