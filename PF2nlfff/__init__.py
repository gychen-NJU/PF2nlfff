# PF2nlfff

# version
__version__ = "2.0.0"

# author
__author__ = "Guoyin Chen"
__email__ = "gychen@smail.nju.edu.cn"

# Lazy import to avoid dependency issues
__all__ = ['PROGAN']

# Only import PROGAN when accessed
from importlib import import_module

def __getattr__(name):
    if name == 'PROGAN':
        from .progan import PROGAN
        return PROGAN
    raise AttributeError(f"module 'PF2nlfff' has no attribute '{name}'")
