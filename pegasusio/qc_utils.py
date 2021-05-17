import numpy as np
import pandas as pd

from pegasusio import UnimodalData

import logging
logger = logging.getLogger(__name__)



class DictWithDefault:
    ### Used for parsing mito prefix
    def __init__(self, string: str):
        self.mapping = {}
        self.default = None

        if string is not None:
            fields = string.split(',')
            for field in fields:
                if field.find(':') >= 0:
                    key, value = field.split(':')
                    self.mapping[key] = value
                else:
                    self.default = field

    def get(self, key: str) -> str:
        return self.mapping.get(key, self.default)



def calc_qc_filters(
    unidata: UnimodalData,
    select_singlets: bool = False,
    remap_string: str = None,
    subset_string: str = None,
    min_genes: int = None,
    max_genes: int = None,
    min_umis: int = None,
    max_umis: int = None,
    mito_prefix: str = None,
    percent_mito: float = None
) -> None:
    """Calculate Quality Control (QC) metrics and mark barcodes based on the combination of QC metrics.

    Parameters
    ----------
    unidata: ``UnimodalData``
       Unimodal data matrix with rows for cells and columns for genes.
    select_singlets: ``bool``, optional, default ``False``
        If select only singlets.
    remap_string: ``str``, optional, default ``None``
        Remap singlet names using <remap_string>, where <remap_string> takes the format "new_name_i:old_name_1,old_name_2;new_name_ii:old_name_3;...". For example, if we hashed 5 libraries from 3 samples sample1_lib1, sample1_lib2, sample2_lib1, sample2_lib2 and sample3, we can remap them to 3 samples using this string: "sample1:sample1_lib1,sample1_lib2;sample2:sample2_lib1,sample2_lib2". In this way, the new singlet names will be in metadata field with key 'assignment', while the old names will be kept in metadata field with key 'assignment.orig'.
    subset_string: ``str``, optional, default ``None``
        If select singlets, only select singlets in the <subset_string>, which takes the format "name1,name2,...". Note that if --remap-singlets is specified, subsetting happens after remapping. For example, we can only select singlets from sampe 1 and 3 using "sample1,sample3".
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
        * ``passed_qc``: Boolean type indicating if a cell passes the QC process based on the QC metrics.
        * ``demux_type``: this column might be deleted if select_singlets is on.

    Examples
    --------
    >>> calc_qc_filters(unidata, min_umis = 500, select_singlets = True)
    """
    assert unidata.uns["modality"] == "rna"

    filters = []

    if select_singlets and ("demux_type" in unidata.obs):
        if remap_string is not None:
            if "assignment" not in unidata.obs:
                raise ValueError("No assignment field detected!")
            unidata.obs["assignment.orig"] = unidata.obs["assignment"]

            remap = {}
            tokens = remap_string.split(";")
            for token in tokens:
                new_key, old_str = token.split(":")
                old_keys = old_str.split(",")
                for key in old_keys:
                    remap[key] = new_key

            unidata.obs["assignment"] = pd.Categorical(unidata.obs["assignment"].apply(lambda x: remap[x] if x in remap else x))
            logger.info("Singlets are remapped.")


        if subset_string is None:
            filters.append(unidata.obs["demux_type"] == "singlet")
        else:
            if "assignment" not in unidata.obs:
                raise ValueError("No assignment field detected!")

            subset = np.array(subset_string.split(","))
            filters.append(np.isin(unidata.obs["assignment"], subset))

        unidata.uns["__del_demux_type"] = True

    if "n_genes" not in unidata.obs:
        unidata.obs["n_genes"] = unidata.X.getnnz(axis=1)

    if "n_counts" not in unidata.obs:
        unidata.obs["n_counts"] = unidata.X.sum(axis=1).A1

    if min_genes is not None:
        filters.append(unidata.obs["n_genes"] >= min_genes)
    if max_genes is not None:
        filters.append(unidata.obs["n_genes"] < max_genes)
    if min_umis is not None:
        filters.append(unidata.obs["n_counts"] >= min_umis)
    if max_umis is not None:
        filters.append(unidata.obs["n_counts"] < max_umis)

    if (mito_prefix is not None) and (percent_mito is not None):
        mito_genes = unidata.var_names.map(lambda x: x.startswith(mito_prefix)).values.nonzero()[0]
        unidata.obs["percent_mito"] = (
            unidata.X[:, mito_genes].sum(axis=1).A1
            / np.maximum(unidata.obs["n_counts"].values, 1.0)
        ) * 100
        filters.append(unidata.obs["percent_mito"] < percent_mito)

    if len(filters) > 0:
        selected = np.logical_and.reduce(filters)
        unidata.obs["passed_qc"] = selected
    else:
        unidata.obs["passed_qc"] = True


def apply_qc_filters(unidata: UnimodalData):
    """ Apply QC filters to filter out low quality cells """
    if "passed_qc" in unidata.obs:
        prior_n = unidata.shape[0]
        unidata._inplace_subset_obs(unidata.obs["passed_qc"])

        cols = ["passed_qc"]
        if unidata.uns.get("__del_demux_type", False):
            cols.append("demux_type")
            if "assignment" in unidata.obs:
                # remove categories that contain no elements
                series = unidata.obs["assignment"].value_counts(sort = False)
                unidata.obs["assignment"] = pd.Categorical(unidata.obs["assignment"], categories = series[series > 0].index.astype(str))
            # del unidata.uns["__del_demux_type"]

        unidata.obs.drop(columns=cols, inplace=True)
        if len(unidata.obsm) > 0:
            unidata.obsm.clear()
        if len(unidata.varm) > 0:
            unidata.varm.clear()
        for key in list(unidata.uns):
            if key not in {'genome', 'modality', 'norm_count', 'df_qcplot'}:
                del unidata.uns[key]
        logger.info(f"After filtration, {unidata.shape[0]} out of {prior_n} cell barcodes are kept in UnimodalData object {unidata.get_uid()}.")
