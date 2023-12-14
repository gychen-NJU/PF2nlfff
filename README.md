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

### PICNN
If you want build a NLFFF with neural network.
you need to unzip the files in [Unet3D](UNet3D) and get a file named 'UNet3D.pkl', then add it to the [config.jons](config.jons).
Then use the script [PICNN.py](utils/PICNN.py), one can training a satisfactory NLFFF model.

The commond is as follows:
```
cd utils
codna activate yourenvs
python PICNN.py --config '../config.json'
```
If it runs properly, the output is as follows:
```
original size:  [101, 101, 50]
modified size:  [96, 96, 48]
 ### Begin Network Training ###
Iter 00000, Loss: 3.695e+07, Loss_div: 2.756e+07,Loss_ff: 9.240e+06, Loss_bc: 1.410e+05, wall_time: 4.921e-01 sec, lr: 1.000e-03
create the DIR. : ./trainer/3D_models/
Iter 00010, Loss: 1.504e+06, Loss_div: 6.011e+05,Loss_ff: 3.083e+05, Loss_bc: 5.943e+05, wall_time: 1.941e+01 sec, lr: 1.000e-03
Iter 00020, Loss: 1.278e+06, Loss_div: 7.307e+05,Loss_ff: 2.563e+05, Loss_bc: 2.907e+05, wall_time: 3.385e+01 sec, lr: 9.900e-04
Iter 00030, Loss: 6.432e+05, Loss_div: 2.598e+05,Loss_ff: 2.228e+05, Loss_bc: 1.601e+05, wall_time: 4.865e+01 sec, lr: 9.900e-04
Iter 00040, Loss: 5.300e+05, Loss_div: 1.912e+05,Loss_ff: 2.072e+05, Loss_bc: 1.312e+05, wall_time: 6.081e+01 sec, lr: 9.801e-04
Iter 00050, Loss: 4.995e+05, Loss_div: 2.397e+05,Loss_ff: 1.739e+05, Loss_bc: 8.557e+04, wall_time: 7.011e+01 sec, lr: 9.801e-04
Iter 00060, Loss: 4.475e+05, Loss_div: 2.547e+05,Loss_ff: 1.367e+05, Loss_bc: 5.572e+04, wall_time: 7.932e+01 sec, lr: 9.703e-04
Iter 00070, Loss: 3.347e+05, Loss_div: 1.634e+05,Loss_ff: 1.300e+05, Loss_bc: 4.098e+04, wall_time: 8.851e+01 sec, lr: 9.703e-04
Iter 00080, Loss: 3.408e+05, Loss_div: 1.841e+05,Loss_ff: 1.246e+05, Loss_bc: 3.182e+04, wall_time: 9.590e+01 sec, lr: 9.606e-04
```
After a while, you can get a force-free 3D magnetic field model.
