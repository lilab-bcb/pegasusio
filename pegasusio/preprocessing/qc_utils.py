import numpy as np

from pegasusio import UnimodalData

import logging
logger = logging.getLogger(__name__)

def qc_metrics(
    data: UnimodalData,
    mito_prefix: str = "MT-",
    min_genes: int = 500,
    max_genes: int = 6000,
    min_umis: int = 100,
    max_umis: int = 600000,
    percent_mito: float = 10.0,
    percent_cells: float = 0.05,
) -> None:
    """Generate Quality Control (QC) metrics on the dataset.

    Parameters
    ----------
    data: ``anndata.AnnData``
       Annotated data matrix with rows for cells and columns for genes.
    mito_prefix: ``str``, optional, default: ``"MT-"``
       Prefix for mitochondrial genes.
    min_genes: ``int``, optional, default: ``500``
       Only keep cells with at least ``min_genes`` genes.
    max_genes: ``int``, optional, default: ``6000``
       Only keep cells with less than ``max_genes`` genes.
    min_umis: ``int``, optional, default: ``100``
       Only keep cells with at least ``min_umis`` UMIs.
    max_umis: ``int``, optional, default: ``600,000``
       Only keep cells with less than ``max_umis`` UMIs.
    percent_mito: ``float``, optional, default: ``10.0``
       Only keep cells with percent mitochondrial genes less than ``percent_mito`` % of total counts.
    percent_cells: ``float``, optional, default: ``0.05``
       Only assign genes to be ``robust`` that are expressed in at least ``percent_cells`` % of cells.

    Returns
    -------
    ``None``

    Update ``data.obs``:

        * ``n_genes``: Total number of genes for each cell.
        * ``n_counts``: Total number of counts for each cell.
        * ``percent_mito``: Percent of mitochondrial genes for each cell.
        * ``passed_qc``: Boolean type indicating if a cell passes the QC process based on the QC metrics.

    Update ``data.var``:

        * ``n_cells``: Total number of cells in which each gene is measured.
        * ``percent_cells``: Percent of cells in which each gene is measured.
        * ``robust``: Boolean type indicating if a gene is robust based on the QC metrics.
        * ``highly_variable_features``: Boolean type indicating if a gene is a highly variable feature. By default, set all robust genes as highly variable features.

    Examples
    --------
    >>> pg.qcmetrics(adata)
    """

    data.obs["passed_qc"] = False

    data.obs["n_genes"] = data.X.getnnz(axis=1)
    data.obs["n_counts"] = data.X.sum(axis=1).A1

    mito_prefixes = mito_prefix.split(",")

    def startswith(name):
        for prefix in mito_prefixes:
            if name.startswith(prefix):
                return True
        return False

    mito_genes = data.var_names.map(startswith).values.nonzero()[0]
    data.obs["percent_mito"] = (
        data.X[:, mito_genes].sum(axis=1).A1
        / np.maximum(data.obs["n_counts"].values, 1.0)
    ) * 100

    # Assign passed_qc
    filters = [
        data.obs["n_genes"] >= min_genes,
        data.obs["n_genes"] < max_genes,
        data.obs["n_counts"] >= min_umis,
        data.obs["n_counts"] < max_umis,
        data.obs["percent_mito"] < percent_mito,
    ]

    data.obs.loc[np.logical_and.reduce(filters), "passed_qc"] = True

    var = data.var
    data = data[
        data.obs["passed_qc"]
    ]  # compute gene stats in space of filtered cells only

    var["n_cells"] = data.X.getnnz(axis=0)
    var["percent_cells"] = (var["n_cells"] / data.shape[0]) * 100
    var["robust"] = var["percent_cells"] >= percent_cells
    var["highly_variable_features"] = var[
        "robust"
    ]  # default all robust genes are "highly" variable

def filter_data(data: UnimodalData) -> None:
    """ Filter data based on qc_metrics calculated in ``pg.qc_metrics``.

    Parameters
    ----------
    data: ``anndata.AnnData``
        Annotated data matrix with rows for cells and columns for genes.

    Returns
    -------
    ``None``

    Update ``data`` with cells and genes after filtration.

    Examples
    --------
    >>> pg.filter_data(adata)
    """

    assert "passed_qc" in data.obs
    prior_shape = data.shape
    data._inplace_subset_obs(data.obs["passed_qc"].values)
    data._inplace_subset_var((data.var["n_cells"] > 0).values)
    logger.info(
        "After filtration, {nc}/{ncp} cells and {ng}/{ngp} genes are kept. Among {ng} genes, {nrb} genes are robust.".format(
            nc=data.shape[0],
            ng=data.shape[1],
            ncp=prior_shape[0],
            ngp=prior_shape[1],
            nrb=data.var["robust"].sum(),
        )
    )
