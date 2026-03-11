from ..needs import *
from ..utils.math import *
from ..utils.data import evaluate


# ===================================================== #
#          NLFFF with the optimizing method             #
# ===================================================== #
class use_opt():
    def __init__(self, PF, boundary, dt=1.e-5, mu=1.0,**kwargs):
        self.dt = dt
        self.mu = mu
        self.iter = 0
        self.print_epoch = kwargs.get('print_epoch', 100)
        self.wall_time = 0

        b_cube = PF
        b_cube[:,:,:,0] = boundary
        self.is_tensor = isinstance(b_cube, torch.Tensor)
        self.device = kwargs.get('device',PF.device)
        self.b_cube = b_cube.to(self.device)
        self.boundary = boundary.to(self.device)
        
        self.L_list = []
        self.fi_list = []
        self.sigmaJ_list = []
        fi, sigma_J, self.L0 = evaluate(self.b_cube)
        self.fi_list.append(fi)
        self.sigmaJ_list.append(sigma_J)
        self.L_list.append(1.)
        
    def return_F(self,b_cube):
        rotB = rot(b_cube)
        divB = div(b_cube)
        Omega = torch.norm(b_cube, dim=0)**(-2.)*(cube_cross(rotB, b_cube)-divB*b_cube)  
        
        F = rot(cube_cross(Omega,b_cube))-cube_cross(Omega, rotB)-\
            grad(cube_dot(Omega,b_cube))+Omega*divB+cube_dot(Omega,Omega)*b_cube    
        return F
    
    def RK45_opt(self,b_cube, dt, mu):
        x = b_cube
        F1 = self.return_F(x)
        F2 = self.return_F(x+0.5*F1*dt*mu)
        F3 = self.return_F(x+0.5*F2*dt*mu)
        F4 = self.return_F(x+1.0*F3*dt*mu)
        F = 1/6*(F1+2*F2+2*F3+F4)
        dB = mu*dt*F
        return x+dB
        
    def opt_iteration(self, max_step = 1e4, **kwargs):
        dt = kwargs.get('dt', self.dt)
        mu = kwargs.get('mu', self.mu)
        NP = kwargs.get('print_epoch', self.print_epoch)
        IP = kwargs.get('is_print',True)
        first = True
        start_iter = self.iter
        if start_iter == 0:
            self.wall_time = 0
        start_time = time.time()
        if not self.is_tensor:
            B = torch.from_numpy(self.b_cube).to(self.device)
            boundary = torch.from_numpy(self.boundary).to(self.device)
        else:
            B = self.b_cube.to(self.device)
            boundary = self.boundary.to(self.device)
        
        for i in range(max_step):
            self.iter+=1
            B = self.RK45_opt(B, dt, mu)
            fi, sigma_J, L = evaluate(B)
            self.fi_list.append(fi)
            self.sigmaJ_list.append(sigma_J)
            self.L_list.append(L/self.L0)
            B[:,:,:,0] = boundary
            if IP and (self.iter % NP==0 or first or i==max_step-1):
                self.wall_time=time.time()-start_time
                print('iter: %05d/%05d, fi: %.4e, sigma_J: %.4e, L:%.4e, wall_time: %8.3f min' % 
                      (self.iter, max_step+start_iter, fi, sigma_J, L, self.wall_time/50))
                first = False
        
        setattr(self,'bcube',B)
        B = B if self.is_tensor else B.detach().cpu().numpy()
        return B

    def plot_optmization(self):
        plt.plot(self.fi_list, label='r$\langle |f_i|\rangle$',c='C0')
        plt.ylabel(r'$\langle |f_i|\rangle$', color='C0')
        plt.yscale('log')
        plt.gca().tick_params('y', colors='C0')
        ax1 = plt.gca()
        ax1.spines['left'].set_color('C0')
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_color('C0')
        for tick in ax1.yaxis.get_minor_ticks():
            tick.label1.set_color('C0')
        
        ax2 = ax1.twinx()
        plt.plot(self.sigmaJ_list, label=r'$\sigma_J$', c='C1')
        plt.ylabel(r'$\sigma_J$', color='C1')
        plt.gca().tick_params('y', colors='C1')
        for tick in ax2.yaxis.get_major_ticks():
            tick.label1.set_color('C1')
        for tick in ax2.yaxis.get_minor_ticks():
            tick.label1.set_color('C1')
        
    def __call__(self,**kwargs):
        ret = self.opt_iteration(max_step=1,**kwargs)
        return ret
    

