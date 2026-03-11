import torch
from ..needs import *
from ..utils.data import evaluate

def fast_relax(bcube, boundary=None, maxiters=100, **kwargs):
    """
    bcube: Tensor # (3,nx,ny,nz)
        initial magnetic field to relax
    boundary: Tensor # (3,nx,ny)
        bottom boundary
    """
    device = kwargs.get('device', bcube.device)
    lr = kwargs.get('lr',1e-1)
    IP = kwargs.get('is_print',False)
    PI = kwargs.get('print_intervals',1)
    if boundary is None:
        boundary  = bcube[:,:,:,0].clone()
    bcube = bcube.to(device)
    boundary = boundary.to(device)

    shape = bcube.shape
    free_mask = torch.ones(shape, dtype=torch.bool, device=device)
    free_mask[:,:,:,0] = False
    free_params = bcube[free_mask].clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [free_params],
        max_iter=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        lr=lr,
    )

    def closure():
        optimizer.zero_grad()
        # rebuild tensor
        x = bcube.clone()
        x[free_mask] = free_params
        # keep the bottom boundary
        x[:, :, :, 0] = boundary
        # calculate the loss
        _, _, L = evaluate(x)
        L.backward()
        return L

    loss_list = []
    # Initialize data_rcd with the initial bcube
    data_rcd = bcube.clone().detach().cpu()
    min_loss = float('inf')
    
    for step in range(maxiters): 
        loss = optimizer.step(closure)
        if IP and (((step+1)%PI==0) or (step==0) or (step==maxiters-1)):
            print(f"Step {step:3d}, Loss: {loss.item():.8e}")
        
        # Update bcube with the optimized parameters
        with torch.no_grad():
            bcube[free_mask] = free_params
            bcube[:, :, :, 0] = boundary
        
        # Update data_rcd if current loss is lower than minimum loss
        if loss.item() < min_loss:
            min_loss = loss.item()
            data_rcd = bcube.clone().detach().cpu()
        
        loss_list.append(loss.item())
    
    if kwargs.get('return_loss',False):
        return data_rcd.to(device), loss_list
    else:
        return data_rcd.to(device)