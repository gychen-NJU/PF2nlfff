from ..needs import *
from .math import *
import importlib.resources

# ===================================================== #
#                       evaluation                      #
# ===================================================== #    
def ShapeClip(data, is_print_info = False):
    n1, n2, n3 = data.shape[1:]
    ret = data[:,:,:,:n3]
    if is_print_info:
        print('original size: ', [n1, n2, n3])

    if n1%2 !=0:
        n1 = n1-1
    if n2%2 != 0:
        n2 = n2-1
    if n3%2 !=0:
        n3 = n3-1
    ret = ret[:,:n1,:n2,:n3]

    if n1%16!=0:
        edge = n1%16//2
        ret = ret[:,edge:-edge,:,:]
    if n2%16!=0:
        edge = n2%16//2
        ret = ret[:,:,edge:-edge,:]
    if n3%16!=0:
        edge = n3%16//2
        ret = ret[:,:,:,:-2*edge]

    n1, n2, n3 = ret.shape[1:]
    if is_print_info:
        print('modified size: ', [n1, n2, n3])
    
    return ret

def evaluate(b_cube, is_eval=False):
    is_tensor = isinstance(b_cube, torch.Tensor)
    epsilon = 1.e-32
    divB = div(b_cube)
    rotB = rot(b_cube)
    if is_eval:
        divB = divB[:,1:-1,1:-1,1:-1]
        rotB = rotB[:,1:-1,1:-1,1:-1]
        b_cube = b_cube[:,1:-1,1:-1,1:-1]
    lorentz = cube_cross(rotB, b_cube)
    force = (torch.norm(lorentz, dim=0) if is_tensor else norm(lorentz, axis=0))
    B = (torch.norm(b_cube,dim=0) if is_tensor else norm(b_cube, axis=0))
    J = (torch.norm(rotB, dim=0) if is_tensor else norm(rotB, axis=0))
    sine = force/(J*B+epsilon)
    sigma_J = ((sine*J).sum()/J.sum())
    f_i = divB/(B+epsilon)/6.
    f_i = (torch.abs(f_i).mean() if is_tensor else np.abs(f_i).mean())
    L = (B**(-2.)*force**2+divB**2).sum()
    
    return f_i, sigma_J, L

# verctor correlation metric
def vec_corr_metric(b,B):
    '''
    b: the prediction magnetic field
    B: the reference magnetic field
    '''
    ret = cube_dot(B,b).sum()/((cube_dot(B,B).sum())*(cube_dot(b,b).sum()))**0.5
    return ret

#  Cauchy–Schwarz metric
def CS_metric(b,B):
    Bdotb = cube_dot(B,b)
    Bi = cube_dot(B,B)**0.5
    bi = cube_dot(b,b)**0.5
    ret = (Bdotb/(Bi*bi)).mean()
    return ret

# normalized vector error
def norm_vec_err(b,B):
    ret = norm(b-B, axis=0).sum()/norm(B,axis=0).sum()
    return ret

# mean verctor error
def mean_vec_err(b,B):
    ret  = (norm(b-B, axis=0)/norm(B,axis=0)).mean()
    return ret

# magnetic energy fraction
def mag_e_frac(b,B):
    ret = cube_dot(b,b).sum()/cube_dot(B,B).sum()
    return ret

class samples():
    def __init__(self):
        self._LL = None
        self._TD = None
    
    @property
    def LL(self):
        if self._LL is None:
            with importlib.resources.path(
                'PF2nlfff.data', 'LL_data.npy'
            ) as path:
                self._LL = np.load(path)
                # Convert to native byte order to avoid PyTorch error
                if not self._LL.dtype.isnative:
                    self._LL = self._LL.astype(self._LL.dtype.newbyteorder('='))
        return self._LL
    
    @property
    def TD(self):
        if self._TD is None:
            with importlib.resources.path(
                'PF2nlfff.data', 'TD_data.npy'
            ) as path:
                self._TD = np.load(path)
                # Convert to native byte order to avoid PyTorch error
                if not self._TD.dtype.isnative:
                    self._TD = self._TD.astype(self._TD.dtype.newbyteorder('='))
        return self._TD
