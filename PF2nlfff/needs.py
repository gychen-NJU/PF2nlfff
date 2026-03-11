import os
import json
import pandas
import numpy as np
from numpy.linalg import norm
import re
import glob
import pandas as pd
import argparse
import json
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import clear_output
from collections import OrderedDict

import torch
import torchvision.transforms.functional
from torch import nn
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
from torch.optim.lr_scheduler import StepLR

import time
from multiprocessing import Pool, cpu_count

# Make pyevtk optional
try:
    from pyevtk.vtk import VtkFile, VtkRectilinearGrid
except ImportError:
    VtkFile = None
    VtkRectilinearGrid = None
    print("Warning: pyevtk not installed. Visualization functionality will be limited.")

import sys
from tqdm import tqdm
from IPython.display import clear_output
from scipy import stats