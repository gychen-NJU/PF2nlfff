# PF2nlfff
Code for the PF2nlfff projection

***
## Install
```
git clone https://github.com/gychen-NJU/PF2nlfff

cd PF2nlfff/codes
```

Install with **pip**

`pip install -e .`

Install with **conda**
```
conda activate yourenvs
conda install conda-build
conda develop .
```
***
## usage
You can easily build the NonLinear Force-Free Field (NLFFF) with the traditional extrapolation method or the Neural Network Method.

### Extrapolation:
You can do Potential extrapolation with Green function method and NLFFF extrapolation with Optimization method with the script [extrapolation.py](utils/extrapolation.py).

And The extrapolation parameters is in the configure setting file [config.jons](config.jons).

When using it, opon your terminal and use the following commond:
```
cd utils
conda activate yourenvs
python extrapolation.py --config '../config.jons'
```
Demo's output is like follows:
```
Potential Field:  fi=8.982e-03, sigma_J = 0.792
save PF data in: PF_data.npy
save VTR data in: PF_data: 
your device is:  cuda
Progress: 10000step [00:12, 806.58step/s, fi=0.0101, sigma_J=0.308]                                                                                                                           
save opt pictures:  nlfff_opt_pic.png
```
