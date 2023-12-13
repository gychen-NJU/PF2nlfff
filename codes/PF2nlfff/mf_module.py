from packages import *

# ===================================================== #
#            Green function extrapolation               #
# ===================================================== #
def calculate_phi_gpu(args, rr_cube, rp_cube, bn):
    i, j, k = args
    nx, ny = bn.shape
    rr_temp = rr_cube[:,i,j,k][:,None,None].expand(3, nx, ny)
    green_kenel = 1/(2*torch.tensor(np.pi))/torch.norm((rr_temp-rp_cube), dim=0)
    phi_val = torch.sum(bn.clone().detach()*green_kenel)

    return phi_val

def calculate_phi_cpu(args):
    i, j, k, rr_cube, rp_cube, bn = args
    nx, ny = bn.shape
    rr_temp = np.broadcast_to(rr_cube[:,i,j,k][:,np.newaxis,np.newaxis], (3, nx, ny))
    green_kenel = 1/(2*np.pi)/np.linalg.norm((rr_temp-rp_cube), axis=0)
    phi_val = np.sum(bn*green_kenel)

    return phi_val

def extrapolate_potential(bottom_boundary, n3='auto', device = 'auto', zscale=1.0):
    start_time = time.time()
    print('\rboundary size: ', bottom_boundary.shape)
    print('\rextrapolating the potential field with the Green function method')
    if device =='auto':
        device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        device = torch.device(device)
    print('\ryour device is: %s' % device.type)
    use_gpu = (True if device.type=='cuda' else False)
    if use_gpu:
        bn = torch.tensor(bottom_boundary[2,:,:]).float().to(device)
        nx, ny = bn.shape
        if n3 == 'auto':
            nz = (np.ceil(3/8*(nx+ny)/4)*4).long()
        else:
            nz = torch.tensor(n3).long()
        i, j, k = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz))
        rr_cube = torch.stack((i, j, zscale*k), dim=0).float().to(device)
        i, j = torch.meshgrid(torch.arange(nx), torch.arange(ny))
        r_ij = torch.stack((i,j, -1./torch.sqrt(2*torch.tensor(np.pi))*torch.ones_like(i)), dim=0)
        rp_cube = r_ij.expand(3, nx, ny).float().to(device)

        phi_cube = torch.zeros(nx, ny, nz)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    args = (i, j, k)
                    phi_cube[i, j, k] = calculate_phi_gpu(args, rr_cube, rp_cube, bn)

        bx = torch.zeros(nx, ny, nz).float().to(device)
        by = torch.zeros(nx, ny, nz).float().to(device)
        bz = torch.zeros(nx, ny, nz).float().to(device)
        bx[1:-1,:,:] = (phi_cube[2:,:,:]-phi_cube[:-2,:,:])/2
        by[:,1:-1,:] = (phi_cube[:,2:,:]-phi_cube[:,:-2,:])/2
        bz[:,:,1:-1] = (phi_cube[:,:,2:]-phi_cube[:,:,:-2])/2
        bx[0,:,:] = phi_cube[1,:,:]-phi_cube[0,:,:]
        by[:,0,:] = phi_cube[:,1,:]-phi_cube[:,0,:]
        bz[:,:,0] = phi_cube[:,:,1]-phi_cube[:,:,0]
        bx[-1,:,:] = phi_cube[-1,:,:]-phi_cube[-2,:,:]
        by[:,-1,:] = phi_cube[:,-1,:]-phi_cube[:,-2,:]
        bz[:,:,-1] = phi_cube[:,:,-1]-phi_cube[:,:,-2]

        b_cube = -torch.stack((bx,by,bz), dim=0)
        b_cube[:,:,:,0] = torch.tensor(bottom_boundary)
        
    else:
        bn = bottom_boundary[2,:,:]
        nx, ny = bn.shape
        if n3=='auto':
            nz = (np.ceil(3/8*(nx+ny)/4)*4).astype(int)
        else:
            nz = n3
        i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        rr_cube = np.stack((i, j, zscale*k), axis=0)
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        r_ij = np.stack((i,j, -1./np.sqrt(2*np.pi)*np.ones_like(i)), axis=0)
        rp_cube = np.broadcast_to(r_ij[...], (3, nx, ny))

        num_cores = cpu_count()
        print("\rCPU kernel counts：", num_cores)

        phi_cube = np.zeros((nx, ny, nz))
        # build the args tabel
        args_list = [(i, j, k, rr_cube, rp_cube, bn) for i in range(nx)\
                                      for j in range(ny)\
                                      for k in range(nz)]

        # create the process pooling
        with Pool() as pool:
            result_list = pool.map(calculate_phi_cpu, args_list)

        # fill the results to 'phi_cube'
        for idx, (i, j, k) in enumerate([(i, j, k) for i in range(nx)\
                                      for j in range(ny)\
                                      for k in range(nz)]):
               phi_cube[i, j, k] = result_list[idx]

        bx = np.zeros((nx,ny,nz))
        by = np.zeros((nx,ny,nz))
        bz = np.zeros((nx,ny,nz))
        bx[1:-1,:,:] = (phi_cube[2:,:,:]-phi_cube[:-2,:,:])/2
        by[:,1:-1,:] = (phi_cube[:,2:,:]-phi_cube[:,:-2,:])/2
        bz[:,:,1:-1] = (phi_cube[:,:,2:]-phi_cube[:,:,:-2])/2
        bx[0,:,:] = phi_cube[1,:,:]-phi_cube[0,:,:]
        by[:,0,:] = phi_cube[:,1,:]-phi_cube[:,0,:]
        bz[:,:,0] = phi_cube[:,:,1]-phi_cube[:,:,0]
        bx[-1,:,:] = phi_cube[-1,:,:]-phi_cube[-2,:,:]
        by[:,-1,:] = phi_cube[:,-1,:]-phi_cube[:,-2,:]
        bz[:,:,-1] = phi_cube[:,:,-1]-phi_cube[:,:,-2]
        b_cube = -np.array([bx,by,bz])
        b_cube[:,:,:,0] = bottom_boundary

    end_time = time.time()
    execution_time = end_time - start_time
    print('\rexecution time: %.1f sec' % execution_time)
    print('\r!!! Finish !!!')
    return (b_cube.detach().cpu().numpy() if isinstance(b_cube, torch.Tensor) else b_cube)


# ===================================================== #
#          NLFFF with the optimizing method             #
# ===================================================== #
class use_mf():
    def __init__(self, b_cube, dt=1.e-5, mu=1.0):
        self.dt = dt
        self.mu = mu
        self.iter = 0
        self.print_epoch = 100
        self.wall_time = 0
        
        self.is_tensor = isinstance(b_cube, torch.Tensor)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.b_cube = (b_cube if self.is_tensor else torch.from_numpy(b_cube)).to(self.device)      
        self.boundary = (b_cube[:,:,:,0] if self.is_tensor else torch.from_numpy(b_cube[:,:,:,0]))
        
        self.L_list = []
        self.fi_list = []
        self.sigmaJ_list = []
        fi, sigma_J, self.L0 = evaluate(b_cube)
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
    
    def RK45_mf(self,b_cube, dt, mu):
        x = b_cube
        F1 = self.return_F(x)
        F2 = self.return_F(x+0.5*F1*dt*mu)
        F3 = self.return_F(x+0.5*F2*dt*mu)
        F4 = self.return_F(x+1.0*F3*dt*mu)
        F = 1/6*(F1+2*F2+2*F3+F4)
        dB = mu*dt*F
        return x+dB
        
    def mf_module(self, max_step = 1e4):
        first = True
        start_iter = self.iter
        if start_iter == 0:
            self.wall_time = 0
        start_time = time.time()
        if not self.is_tensor:
            self.b_cube = torch.tensor(self.b_cube).clone().detach().to(self.device)
        dt = self.dt
        mu = self.mu
        B = self.b_cube
        
        for i in range(max_step):
            B = self.RK45_mf(B, dt, mu)
            fi, sigma_J, L = evaluate(B)
            self.fi_list.append(fi)
            self.sigmaJ_list.append(sigma_J)
            self.L_list.append(L/self.L0)
            self.wall_time+=time.time()-start_time
            if self.iter % self.print_epoch ==0 or first:
                print('iter: %05d/%05d, fi: %.4e, sigma_J: %.4e, L:%.4e, wall_time: %.4e' % 
                      (self.iter, max_step+start_iter, fi, sigma_J, L, self.wall_time))
                first = False
            self.iter+=1

            
        progress_bar.close()
        
    def __call__(self):
        ret = self.RK45_mf(self.b_cube, self.dt, self.mu)
        ret[:,:,:,0] = self.boundary
        ret = (ret if self.is_tensor else ret.detach().cpu().numpy())
        return ret
    
    
# ===================================================== #
#                    vector analysis                    #
# ===================================================== #
# differential oprator
def jacobi_matrix(b_cube):
    is_array = False
    if isinstance(b_cube, np.ndarray):
        b_cube = torch.from_numpy(b_cube)
        is_array=True
    elif not isinstance(b_cube, torch.Tensor):
        raise ValueError("Input must be a NumPy array or a PyTorch tensor.")
    
    b_dx = torch.cat([(b_cube[:,-1:,:,:]-b_cube[:,-2:-1,:,:]),
                      (b_cube[:,2:,:,:]-b_cube[:,:-2,:,:])/2,
                      (b_cube[:,1:2,:,:]-b_cube[:,0:1,:,:])], dim=1)
    
    b_dy = torch.cat([(b_cube[:,:,-1:,:]-b_cube[:,:,-2:-1,:]),
                      ((b_cube[:,:,2:,:]-b_cube[:,:,:-2,:])/2),
                      (b_cube[:,:,1:2,:]-b_cube[:,:,0:1,:])], dim=2)
    
    b_dz = torch.cat([(b_cube[:,:,:,-1:]-b_cube[:,:,:,-2:-1]),
                      ((b_cube[:,:,:,2:]-b_cube[:,:,:,:-2])/2),
                      (b_cube[:,:,:,1:2]-b_cube[:,:,:,0:1])], dim=3)
    
    jacobi = torch.stack([b_dx, b_dy, b_dz], dim=1)
    return jacobi

def rot(vec):
    jacobi = jacobi_matrix(vec)
    # rotB: (dBy/dz-dBz/dy, dBz/dx-dBx/dz, dBy/dx-dBx/dy)
    rot_vec = torch.stack([jacobi[2,1]-jacobi[1,2],
                          jacobi[0,2]-jacobi[2,0],
                          jacobi[1,0]-jacobi[0,1]], dim=0)
    
    if isinstance(vec, torch.Tensor):
        return rot_vec
    else:
        return rot_vec.cpu().numpy()

def div(vec):
    jacobi = jacobi_matrix(vec)
    div_vec = (jacobi[0,0] + jacobi[1,1] + jacobi[2,2]).unsqueeze(0)
    
    if isinstance(vec, torch.Tensor):
        return div_vec
    else:
        return div_vec.cpu().numpy()

def grad(vec):
    jacobi = jacobi_matrix(vec)
    grad_vec = torch.stack([jacobi[0,0], jacobi[0,1], jacobi[0,2]], dim=0)
    
    if isinstance(vec, torch.Tensor):
        return grad_vec
    else:
        return grad_vec.cpu().numpy()

def cube_cross(a, b):
    is_array=False
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
        is_array=True
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    size = a.permute(1,2,3,0).shape
        
    a = a.permute(1,2,3,0).reshape(-1,3)
    b = b.permute(1,2,3,0).reshape(-1,3)
    axb = torch.cross(a, b).reshape(size).permute(3,0,1,2)
    
    if is_array:
        return axb.cpu().numpy()
    else:
        return axb

def cube_dot(a, b):
    is_array = False
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
        is_array = True
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    
    a_dot_b = torch.sum(a*b, dim=0).unsqueeze(0)
    if is_array:
        a_dot_b = a_dot_b.detach().numpy()
    
    return a_dot_b


# ===================================================== #
#                       evaluation                      #
# ===================================================== #    
def adc_shape(data, is_print_info = True):
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
    sigma_J = ((sine*J).sum()/J.sum()).item()
    f_i = divB/(B+epsilon)/6.
    f_i = (torch.abs(f_i).mean().item() if is_tensor else np.abs(f_i).mean())
    L = (B**(-2.)*force**2+divB**2).sum().item()
    
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