# /Users/linliu/NewSpaGRN/src/spagrn/__init__.py
__package__ = 'spagrn'

__all__ = [
    'regulatory_network',
    'hotspot',
    'knn',
    'local_stats',
    'autocor',
    'corexp',
    'c_autocor',
    'm_autocor',
    'g_autocor',
    'network',
    'danb_model',
    'bernoulli_model',
    'normal_model',
    'none_model',
    'utils',
    'plot'
]

from . import regulatory_network
from . import hotspot
from . import knn
from . import local_stats
from . import autocor
from . import corexp
from . import c_autocor
from . import m_autocor
from . import g_autocor
from . import network
from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model
from . import utils
from . import plot

# Deprecated imports to maintain compatibility with older scripts
from .regulatory_network import InferNetwork
from .network import Network

COLORS = [
    '#d60000', '#e2afaf', '#018700', '#a17569', '#e6a500', '#004b00',
    '#6b004f', '#573b00', '#005659', '#5e7b87', '#0000dd', '#00acc6',
    '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c',
    '#8287ff', '#ff7ed1', '#8e7b01', '#9e4b00', '#8eba00', '#a57bb8',
    '#5901a3', '#8c3bff', '#a03a52', '#a1c8c8', '#f2007b', '#ff7752',
    '#bac389', '#15e18c', '#60383b', '#546744', '#380000', '#e252ff',
]