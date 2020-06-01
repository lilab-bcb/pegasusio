import numpy as np

from pegasusio import UnimodalData

import logging
logger = logging.getLogger(__name__)



def apply_qc_filters(
    unidata: UnimodalData,
    select_singlets: bool = False,
    min_genes: int = None,
    max_genes: int = None,
    min_umis: int = None,
    max_umis: int = None,
    mito_prefix: str = None,
    percent_mito: float = None
) -> None:
    """Generate Quality Control (QC) metrics and filter dataset based on the QCs.

    Parameters
    ----------
    data: ``UnimodalData``
       Unimodal data matrix with rows for cells and columns for genes.
    select_singlets: ``bool``, optional, default ``False``
        If select only singlets.
    min_genes: ``int``, optional, default: None
       Only keep cells with at least ``min_genes`` genes.
    max_genes: ``int``, optional, default: None
       Only keep cells with less than ``max_genes`` genes.
    min_umis: ``int``, optional, default: None
       Only keep cells with at least ``min_umis`` UMIs.
    max_umis: ``int``, optional, default: None
       Only keep cells with less than ``max_umis`` UMIs.
    mito_prefix: ``str``, optional, default: None
       Prefix for mitochondrial genes.
    percent_mito: ``float``, optional, default: None
       Only keep cells with percent mitochondrial genes less than ``percent_mito`` % of total counts. Only when both mito_prefix and percent_mito set, the mitochondrial filter will be triggered.

    Returns
    -------
    ``None``

    Update ``unidata.obs``:

        * ``n_genes``: Total number of genes for each cell.
        * ``n_counts``: Total number of counts for each cell.
        * ``percent_mito``: Percent of mitochondrial genes for each cell.
        * ``demux_type``: this column might be deleted if select_singlets is on.

    Examples
    --------
    >>> apply_qc_filters(unidata, min_umis = 500, select_singlets = True)
    """
    assert unidata.uns["modality"] == "rna"

    filters = []

    if select_singlets and ("demux_type" in unidata.obs):
        filters.append(unidata.obs["demux_type"] == "singlet")
        unidata.obs.drop(columns="demux_type", inplace=True)

    min_cond = min_genes is not None
    max_cond = max_genes is not None
    if min_cond or max_cond:
        unidata.obs["n_genes"] = unidata.X.getnnz(axis=1)
        if min_cond:
            filters.append(unidata.obs["n_genes"] >= min_genes)
        if max_cond:
            filters.append(unidata.obs["n_genes"] < max_genes)

    min_cond = min_umis is not None
    max_cond = max_umis is not None
    calc_mito = (mito_prefix is not None) and (percent_mito is not None)
    if min_cond or max_cond or calc_mito:
        unidata.obs["n_counts"] = unidata.X.sum(axis=1).A1
        if min_cond:
            filters.append(unidata.obs["n_counts"] >= min_umis)
        if max_cond:
            filters.append(unidata.obs["n_counts"] < max_umis)
        if calc_mito:
            mito_prefixes = mito_prefix.split(",")

            def _startswith(name):
                for prefix in mito_prefixes:
                    if name.startswith(prefix):
                        return True
                return False

            mito_genes = unidata.var_names.map(_startswith).values.nonzero()[0]

            unidata.obs["percent_mito"] = (
                unidata.X[:, mito_genes].sum(axis=1).A1
                / np.maximum(unidata.obs["n_counts"].values, 1.0)
            ) * 100

            filters.append(unidata.obs["percent_mito"] < percent_mito)

    if len(filters) > 0:
        selected = np.logical_and.reduce(filters)
        prior_n = unidata.shape[0]
        unidata._inplace_subset_obs(selected)
        logger.info(f"After filtration, {unidata.shape[0]} out of {prior_n} cell barcodes are kept in UnimodalData object {unidata.get_uid()}.")
