import argparse
import json
from tqdm import tqdm

from PF2nlfff.mf_module import *
from PF2nlfff.tools import *

parser = argparse.ArgumentParser(description='Extrapolate the potential field from a boundary file with Green function method')
parser.add_argument('--config', type=str, help='file path for the configration file', required=True, default='config.json')

args = parser.parse_args()
with open(args.config, 'r') as json_file:
    config_data = json.load(json_file)

boundary_file = config_data["extrapolation"]["potential_field"]["boundary_file"]
PF_exist = config_data["extrapolation"]["potential_field"]["PF_exist"]
PF_file = config_data["extrapolation"]["potential_field"]["PF_file"]
n3 = config_data["extrapolation"]["potential_field"]["nz"]
PF_savepath = config_data["files"]["PF_files"]["data_name"]
is_save_vtr = config_data["extrapolation"]["potential_field"]["is_save_vtr"]
nlfff_extrapolation = config_data["extrapolation"]["nlfff"]["nlfff_extrapolation"]

boundary = np.load(boundary_file)
if not PF_exist:
    PF = extrapolate_potential(boundary, n3=n3)
else:
    PF = np.load(PF_file)
fi,sJ,_=evaluate(PF,is_eval=False)
print('Potential Field:  fi=%.3e, sigma_J = %.3f' % (fi, sJ))

np.save(PF_savepath, PF)
print('save PF data in: %s' % PF_savepath)
                                           
if is_save_vtr:
    vtr_name = config_data["files"]["PF_files"]["vtr_name"]
    lb_ub = config_data["files"]["PF_files"]["lb_ub"]
    Bcube2vtr(PF, lb_ub = lb_ub, savepath=vtr_name)
    print('save VTR data in: %s: ' % vtr_name)
    
    if nlfff_extrapolation:
        PF_data = PF
        dt = config_data["extrapolation"]["nlfff"]["dt"]
        mu = config_data["extrapolation"]["nlfff"]["mu"]
        mf_module = use_mf(PF_data, dt=dt, mu=mu)
        device = config_data["extrapolation"]["nlfff"]["device"]
        if device != 'auto':
            mf_device = torch.device(device)
            mf_module.device = mf_device
        else:
            mf_device = mf_module.device
        print('your device is: ', mf_module.device)

        max_step = config_data["extrapolation"]["nlfff"]["max_step"]
        progress_bar = tqdm(total=max_step, desc="Progress", unit="step")
        B_mf = torch.from_numpy(PF_data).to(mf_device)
        fi, sigma_J, L0 = evaluate(PF_data, is_eval=True)
        fi_list = [fi]
        sigmaJ_list = [sigma_J]
        L_list = [1.]

        for i in range(max_step):
            B_mf = use_mf(B_mf, dt=dt)()
            fi, sigma_J, L = evaluate(B_mf, is_eval=False)
            fi_list.append(fi)
            sigmaJ_list.append(sigma_J)
            L_list.append(L/L0)

            progress_bar.set_postfix({"fi": fi, "sigma_J": sigma_J})
            progress_bar.update(10)
        
        progress_bar.close()
    nlfff_output = config_data["files"]["nlfff"]["data_name"]
    np.save(nlfff_output,B_mf.detach().cpu().numpy())
        
        
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(20,6.18*2))

    plt.subplot(221)
    plt.plot(L_list)
    plt.yscale('log')
    plt.ylabel('L')

    plt.subplot(222)
    plt.plot(sigmaJ_list)
    plt.ylabel('$\sigma_J$')

    plt.subplot(223)
    plt.plot(fi_list)
    plt.yscale('log')
    plt.ylabel(r'$\left<\|f_i\|\right>$')
    
    fig_name = config_data["files"]["nlfff"]["pic"]
    if fig_name != None:
        plt.savefig(fig_name, bbox_inches='tight')
        print('save opt pictures: ', fig_name)
    plt.close()