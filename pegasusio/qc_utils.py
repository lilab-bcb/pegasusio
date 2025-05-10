import numpy as np
import pandas as pd

from pegasusio import UnimodalData
from typing import Optional, List, Union

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
    remap_string: Optional[str] = None,
    subset_string: Optional[str] = None,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    min_umis: Optional[int] = None,
    max_umis: Optional[int] = None,
    mito_prefix: Optional[str] = None,
    percent_mito: Optional[float] = None,
    ribo_prefix: Optional[Union[str, List[str]]] = None,
    percent_ribo: Optional[float] = None,
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
       Only keep cells with percent of mitochondrial genes less than ``percent_mito`` % of total counts. The mitochondrial filtering is triggered only when both ``mito_prefix`` and ``percent_mito`` are both set.
    ribo_prefix: ``str`` or ``List[str]``, optional, default: None
        Prefix(es) for ribosomal genes.
    percent_ribo: ``float``, optional, default: None
        Only keep cells with percent of ribosomal genes less than ``percent_ribo`` % of total counts. The ribosomal filtering is triggered only when both ``ribo_prefix`` and ``percent_ribo`` are both set.

    Returns
    -------
    ``None``

    Update ``unidata.obs``:

        * ``n_genes``: Total number of genes for each cell.
        * ``n_counts``: Total number of counts for each cell.
        * ``percent_mito``: Percent of mitochondrial genes for each cell.
        * ``percent_ribo``: Percent of ribosomal genes for each cell.
        * ``passed_qc``: Boolean type indicating if a cell passes the QC process based on the QC metrics.
        * ``demux_type``: this column might be deleted if select_singlets is on.

    Examples
    --------
    >>> calc_qc_filters(unidata, min_umis = 500, select_singlets = True)
    """
    assert unidata.uns["modality"] in {"rna", "visium"}

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

    if mito_prefix is not None:
        mito_genes = unidata.var_names.map(lambda x: x.startswith(mito_prefix)).values.nonzero()[0]
        unidata.obs["percent_mito"] = (
            unidata.X[:, mito_genes].sum(axis=1).A1
            / np.maximum(unidata.obs["n_counts"].values, 1.0)
        ) * 100
        if percent_mito is not None:
            filters.append(unidata.obs["percent_mito"] < percent_mito)

    if ribo_prefix is not None:
        if not isinstance(ribo_prefix, list):
            ribo_prefix = [ribo_prefix]
        ribo_prefix = tuple(ribo_prefix)
        ribo_genes = unidata.var_names.map(lambda x: x.startswith(ribo_prefix)).values.nonzero()[0]
        unidata.obs["percent_ribo"] = (
            unidata.X[:, ribo_genes].sum(axis=1).A1
            / np.maximum(unidata.obs["n_counts"].values, 1.0)
        ) * 100
        if percent_ribo is not None:
            filters.append(unidata.obs["percent_ribo"] < percent_ribo)

    if len(filters) > 0:
        selected = np.logical_and.reduce(filters)
        unidata.obs["passed_qc"] = selected
    else:
        unidata.obs["passed_qc"] = True


def apply_qc_filters(unidata: UnimodalData, uns_white_list: str = None):
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
            del unidata.uns["__del_demux_type"]

        unidata.obs.drop(columns=cols, inplace=True)
        if len(unidata.obsm) > 0:
            for key in list(unidata.obsm):
                if key not in ['X_spatial']:
                    del unidata.obsm[key]
        if len(unidata.varm) > 0:
            unidata.varm.clear()
        if uns_white_list is not None:
            white_list = set(uns_white_list.split(',') + ['genome', 'modality', 'uid'])
            for key in list(unidata.uns):
                if key not in white_list:
                    del unidata.uns[key]
        logger.info(f"After filtration, {unidata.shape[0]} out of {prior_n} cell barcodes are kept in UnimodalData object {unidata.get_uid()}.")
