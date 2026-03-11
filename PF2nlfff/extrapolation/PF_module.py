from ..needs import *

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

def potential_field_extrapolation(
        Bn, # Normal field in the boundary
        nz = None, # the z-grid numbers
        **kwargs
        ):
    '''
    Extrapolating a potential field based on the normal boundary field via Green function's method

    =======
    Parameters:
        Bn: array-like
            Normal field in the bounary with shape (nx,ny)
        nz: int
            the grid numbers in z-directions, default is min(nx,ny)
        ---------
        **kwargs: optional parameters
            device: device use to compute
            max_memory: the memory limit for the calculation, default is 48 (GB)
            return_phi: whether to return the scalar potential, default is False

    ======
    Return:
        PF_data: array-like
            potential field array with shape (3,nx,ny,nz)
    '''
    is_array = isinstance(Bn, np.ndarray)
    device = torch.device(kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    max_memory = kwargs.get('max_memory', 48)
    return_phi = kwargs.get('return_phi', False)
    nx,ny = Bn.shape
    nz = min(nx,ny) if nz is None else nz
    if is_array:
        Bn_cube = torch.from_numpy(Bn[None,None,:,:]).to(device)
    elif isinstance(Bn, torch.Tensor):
        Bn_cube = Bn[None,None,:,:].to(device)
    else:
        raise ValueError('Bn must be a numpy array or a torch tensor')
    rr_cube = torch.stack(
        torch.meshgrid(
            torch.arange(nx),
            torch.arange(ny),
            torch.arange(nz),
            indexing='ij'
        ),
        dim=0
    )
    rp_cube = rr_cube.clone()[:,:,:,0]
    rr_cube = rr_cube.reshape(3,-1)[:,:,None,None].to(device)
    rp_cube = rp_cube[:,None,:,:].to(device)
    norm_vec = torch.zeros(3,1,1,1).to(device)
    norm_vec[2] = 1.
    phi_cube = torch.zeros(nx,ny,nz).to(device)
    phi_list = phi_cube.flatten()
    batch_size = max_memory*1024**3//(8*3*nx*ny)//2
    total_size = nx*ny*nz
    print(f'Batch Size/Total Size: {batch_size}/{total_size}')
    head_idx = 0
    tail_idx = head_idx+batch_size
    while head_idx<total_size:
        irr = rr_cube[:,head_idx:tail_idx]
        # print(irr.shape,rp_cube.shape, norm_vec.shape)
        Gkernel_k = 1/(2*torch.pi*torch.norm(irr-rp_cube+1/np.sqrt(2*torch.pi)*norm_vec,dim=0))
        # print(f'{torch.sum(Bn_cube*Gkernel_k,dim=(1,2)).shape, Gkernel_k.shape, Bn_cube.shape}')
        phi_list[head_idx:tail_idx] = torch.sum(Bn_cube[0]*Gkernel_k,dim=(1,2))
        head_idx += batch_size
        tail_idx += batch_size
    phi_cube = phi_list.reshape(nx,ny,nz)
    phi_array = phi_cube.detach().cpu().numpy()
    Bxyz = np.stack(np.gradient(phi_array,axis=(0,1,2)),axis=0)
    if not is_array:
        phi_array = torch.from_numpy(phi_array).to(device)
        Bxyz = torch.from_numpy(Bxyz).to(device)
    if return_phi:
        return Bxyz, phi_array
    else:
        return Bxyz

def extrapolate_potential_old(bottom_boundary, n3='auto', device = 'auto', zscale=1.0):
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
            nz = (np.ceil(3/8*(nx+ny)/4)*4).astype(np.int64) 
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
        b_cube = -grad(phi_cube.unsqueeze(0))
        b_cube[:,:,:,0] = torch.from_numpy(bottom_boundary).to(device)
        # bx[1:-1,:,:] = (phi_cube[2:,:,:]-phi_cube[:-2,:,:])/2
        # by[:,1:-1,:] = (phi_cube[:,2:,:]-phi_cube[:,:-2,:])/2
        # bz[:,:,1:-1] = (phi_cube[:,:,2:]-phi_cube[:,:,:-2])/2
        # bx[0,:,:] = phi_cube[1,:,:]-phi_cube[0,:,:]
        # by[:,0,:] = phi_cube[:,1,:]-phi_cube[:,0,:]
        # bz[:,:,0] = phi_cube[:,:,1]-phi_cube[:,:,0]
        # bx[-1,:,:] = phi_cube[-1,:,:]-phi_cube[-2,:,:]
        # by[:,-1,:] = phi_cube[:,-1,:]-phi_cube[:,-2,:]
        # bz[:,:,-1] = phi_cube[:,:,-1]-phi_cube[:,:,-2]
        # b_cube = -torch.stack((bx,by,bz), dim=0)
        # b_cube[:,:,:,0] = torch.tensor(bottom_boundary)
        
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