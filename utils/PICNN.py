import argparse
import json

from PF2nlfff.mf_module import *
from PF2nlfff.tools import *
from PF2nlfff.model_3D import *

parser = argparse.ArgumentParser(description='Extrapolate the potential field from a boundary file with Green function method')
parser.add_argument('--config', type=str, help='file path for the configration file', required=True, default='config.json')

args = parser.parse_args()
with open(args.config, 'r') as json_file:
    config_data = json.load(json_file)
    
net_pkl = config_data["picnn"]["existed_net"]
if net_pkl != None:
    net = torch.load(net_pkl)
for name, param in net.named_parameters():
    param.data = param.data.to('cuda:0')
    # print(param.device)
net.decoder_device = torch.device('cuda:0')
net.encoder_device = torch.device('cuda:0')
PF_file = config_data["picnn"]["PF_file"]
if PF_file == None:
    bounary_file = config_data["picnn"]["boundary_file"]
    boundary = np.load(boundary_file)
    PF_data = extrapolate_potential(boundary, n3='auto')
else:
    PF_data = np.load(PF_file)
    PF_data = adc_shape(PF_data)
    
if net == None:
    model = PICNN(PF_data)
else:
    model = PICNN(PF_data, net=net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu"')))

print(" ### Begin Network Training ###")
model.L_bc = config_data["picnn"]["L_bc"]
model.L_div = config_data["picnn"]["L_div"]
model.L_ff = config_data["picnn"]["L_ff"]
model.print_epoch=config_data["picnn"]["print_epoch"]
model.save_epoch=config_data["picnn"]["save_epoch"]
model._use_mf = config_data["picnn"]["use_mf"]
LR = config_data["picnn"]["lr"]
step_size = config_data["picnn"]["step_size"]
gamma=config_data["picnn"]["gamma"]
nepoch = config_data["picnn"]["nepoch"]
model.train(nepoch, lr=LR, Delta=step_size, gamma=gamma)
print("!!! Finish Training !!!")

is_plot_loss = config_data["picnn"]["is_plot_loss"]
if is_plot_loss:
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(20,6.18*2))
    plt.subplot(221)
    plt.plot(model.loss_list)
    plt.yscale('log')
    plt.ylabel('Loss')

    plt.subplot(222)
    plt.plot(model.L_list)
    plt.yscale('log')
    plt.ylabel('L')

    plt.subplot(223)
    plt.plot(model.sJ_list)
    plt.ylabel('$\sigma_J$')

    plt.subplot(224)
    plt.plot(model.fi_list)
    plt.yscale('log')
    plt.ylabel(r'$\left<\|f_i\|\right>$')
    
    loss_pic_name = config_data["picnn"]["loss_pic_name"]
    plt.savefig(loss_pic_name, bbox_inches='tight')
    print('Save the loss figure: %s' % loss_pic_name)
    plt.close()

model.net.eval()
with torch.no_grad():
    pred = model.net(torch.from_numpy(PF_data).unsqueeze(0))[0].detach().cpu().numpy()
pred_file = config_data["picnn"]["nlfff_pred_data"]
np.save(pred_file, pred)
print("Save the Predicted field: %s" % pred_file)
is_save_vtr = config_data["picnn"]["is_plot_loss"]
if is_save_vtr:
    vtr_name = config_data["picnn"]["vtr_file"]
    lb_ub = config_data["picnn"]["lb_ub"]
    Bcube2vtr(pred, lb_ub=lb_ub, savepath=vtr_name)
    print("Save the Predicted VTR data: %s" % (vtr_name+'.vtr'))
