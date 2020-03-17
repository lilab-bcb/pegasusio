import sys
import logging


logger = logging.getLogger("pegasusio")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

import fast_funcs, io_funcs

from .unimodal_data import UnimodalData
from .multimodal_data import MultimodalData
from .zarr_utils import ZarrFile
from .hdf5_utils import load_10x_h5_file, load_pegasus_h5_file
from .mtx_utils import load_mtx_file, write_mtx_file

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
