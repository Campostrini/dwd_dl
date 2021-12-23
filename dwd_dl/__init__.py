import logging
import sys

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from .logger import INFOFORMATTER, DEBUGFORMATTER

_ch = logging.StreamHandler(stream=sys.stdout)
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter(INFOFORMATTER))

log.addHandler(_ch)

from . import cfg
from . import img
from . import dataset
from . import train
from . import utils
from . import yaml_utils
from . import cli
from . import unet
from . import stats
