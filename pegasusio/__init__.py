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


from .unimodal_data import UnimodalData
from .multimodal_data import MultimodalData
from .readwrite import infer_file_type, read_input, write_output

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
