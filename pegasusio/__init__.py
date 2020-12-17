import sys
import logging
import warnings


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

warnings.filterwarnings("ignore", category=FutureWarning, module='anndata')


modalities = ['rna', 'citeseq', 'hashing', 'tcr', 'bcr', 'crispr', 'atac', 'cyto', 'nanostring']

from .decorators import timer, run_gc

from .unimodal_data import UnimodalData
from .vdj_data import VDJData
from .citeseq_data import CITESeqData
from .cyto_data import CytoData
from .nanostring_data import NanostringData
from .qc_utils import calc_qc_filters, apply_qc_filters, DictWithDefault
from .multimodal_data import MultimodalData
from .aggr_data import AggrData, _get_fillna_dict
from .readwrite import infer_file_type, read_input, write_output, write_scp_file
from .data_aggregation import aggregate_matrices

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # < Python 3.8: Use backport module
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
    del version
except PackageNotFoundError:
    pass
