from .needs import *
from .networks.model_3D import PICNN, UNet3D
from .extrapolation.PF_module import potential_field_extrapolation as PF
from .extrapolation.fast_module import fast_relax
from .utils.data import evaluate, ShapeClip
import importlib.resources
import time

class PROGAN():
    def __init__(self, boundary, **kwargs):
        """
        boundary: Tensor (3,nx,ny)
            bottom boundary for extrapolation
        kwargs:
            device: device to do extrapolation
        """
        self.boundary = boundary.clone()
        self.nx,self.ny = boundary.shape[1:]
        self.device = kwargs.get('device',boundary.device)

    def __call__(
        self,
        nz = None,
        PF_config = dict(),
        fast_config = dict(),
        nn_config = dict(),
        **kwargs
    ):
        """
        Extrapolation via boundary with PRO-GAN

        ==========
        Parameter:
            nz: int
                number of z-direction grid points
            --------
            PF_config: dict
                configuration for Potential field extrapolation
                device: 
                max_memory(GB): 48
            --------
            fast_config: dict
                configuration for fast extrapolation
                lr: 1e-1
                device:
                is_print: False
                print_interval: 1
            --------
            nn_config: dict
                configuration for neural network relaxation
                lr: 1e-3
                epoch: 1000
                Delta: 1
                gamma: 0.994
                print_epoch: 50
                save_epoch: 10000
                save_best: False
                save_first: False
                opt_mt: 1e-5
                save_models_dir: './trainer/3D_models/'
                save_model_name: netG
        """
        time0 = time.time()
        nx = self.nx
        ny = self.ny
        nz = nz if nz else min(self.nx,self.ny)
        setattr(self, 'nz', nz)
        print(f"{' Extrapolating Potential Field ':=^100}")
        print(f" nx:{nx} | ny:{ny} | nz:{nz}")
        print(f" device: {self.device}")
        device = PF_config.pop('device',self.device)
        boundary = self.boundary.clone().to(device)
        PF_data = PF(boundary[2],nz,device=device,**PF_config)
        print(f" Wall Time: {(time.time()-time0)/60:9.3f} [min]")
        print('\n')
        print(f"{' Fast NLFFF Extrapolation ':=^100}")
        PF_tensor = PF_data.clone()
        PF_tensor[...,0] = boundary.clone()
        fi,sj,L = evaluate(PF_tensor)
        print(f" Initial: fi={fi:.5e} | sj={sj:.5e} | L={L:.5e}")
        device = fast_config.pop('device',self.device)
        fast_data = fast_relax(PF_tensor.clone(),boundary,device=device,**fast_config)
        fi,sj,L = evaluate(fast_data)
        print(f" Final  : fi={fi:.5e} | sj={sj:.5e} | L={L:.5e}\n")
        print(f" Wall Time: {(time.time()-time0)/60:9.3f} [min]")
        if nn_config:
            print(f"{' Physics-Reinforced Retraining ':=^100}")
            size0 = PF_tensor.shape
            PF_tensor = ShapeClip(PF_tensor)
            bound_clip = PF_tensor[...,0].clone()
            if PF_tensor.shape!=size0:
                print(f"Clip data from {size0} to {PF_tensor.shape}")
            with importlib.resources.path('PF2nlfff.data','netG.pth') as p:
                netG = UNet3D()
                netG.load_state_dict(torch.load(p,weights_only=False,map_location=device))
                netG = netG.to(device)
                netG.encoder_device = device
                netG.decoder_device = device
            device = nn_config.pop('device',self.device)
            picnn = PICNN(PF_tensor,net=netG,device=device)
            picnn.train(**nn_config)
            picnn.net.eval()
            with torch.no_grad():
                nn_data = picnn.net(PF_tensor.unsqueeze(0)).squeeze(0).detach().clone()
                nn_data[...,0] = bound_clip.clone()
            hold = nn_data.detach().clone()
            nn_data = fast_relax(nn_data,bound_clip,device=device)
            fi,sj,L = evaluate(nn_data)
            _,_,L0  = evaluate(hold)
            if L0<L:
                nn_data = hold.clone()
                fi,sj,L = evaluate(nn_data)
            del hold
            print(f" Network: fi={fi:.5e} | sj={sj:.5e} | L={L:.5e}")
            print(f" Wall Time: {(time.time()-time0)/60:9.3f} [min]")
            print(f"{' Physics-Reinforced Retraining End ':=^100}")
            training_hist = dict(
                fi = picnn.fi_list,
                sj = picnn.sJ_list,
                L = picnn.L_list,
                loss = picnn.loss_list,
                netG_path = os.path.join(picnn.save_models_dir,f"{picnn.save_model_name}_{picnn.iter:05d}.pkl")
            )
        else:
            nn_data = None
            training_hist = None
        res = DotDict(
            dict(
                PF = PF_data.detach().cpu().numpy(),
                fast = fast_data.detach().cpu().numpy(),
                nn = nn_data.detach().cpu().numpy(),
                training_hist = training_hist,
            )
        )
        return res
        

class DotDict(dict):
    """
    Dict subclass that support accessing dict keys through attribute
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # recursively transform nested dicts to DotDict
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, DotDict):
                self[k] = DotDict(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        # recursively transform nested dicts to DotDict
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __repr__(self):
        items = []
        for k, v in self.items():
            items.append(f"{k}={v}")
        return f"DotDict({{{', '.join(items)}}})"

    def copy(self):
        return DotDict({k: (DotDict(v) if isinstance(v, dict) else v) for k, v in self.items()})
