from packages import *
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# ===================================================== #
#               Integrate the field line                #
# ===================================================== #
def sphere_sample(point, r=0.5, nsample=10):
    x0, y0, z0 = point
    rlist = np.random.uniform(0, r, nsample)
    theta = np.random.uniform(0,np.pi, nsample)
    phi = np.random.uniform(0, 2*np.pi, nsample)
    xlist = rlist*np.sin(theta)*np.cos(phi)+x0
    ylist = rlist*np.sin(theta)*np.sin(phi)+y0
    zlist = rlist*np.cos(theta)+z0
    xlist = np.append(xlist, x0)
    ylist = np.append(ylist, y0)
    zlist = np.append(zlist, z0)
    ret = np.vstack([xlist,ylist,zlist]).T
    return ret

def field_line(field, star_point,ds=1.0, max_step = 10000):
    first = True
    flag = True
    fieldline1 = [star_point]
    fieldline2 = []
    x0, y0, z0 = star_point
    iters = 0
    lb = np.array([0,0,0])
    ub = np.array(field.shape[-3:])-1
    
    while True:
        xi, yi, zi = np.around(np.array([x0, y0, z0])).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        k1 = bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k1*ds/2)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        k2 = bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k2*ds/2)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        k3 = bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k3*ds)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        k4 = bvec/bb
        x0, y0, z0 = np.array([x0,y0,z0])+(k1+2*k2+2*k3+k4)*ds/6.
        iters += 1
        if x0<lb[0] or x0>ub[0] or y0<lb[1] or y0>ub[1] or z0<lb[2] or z0>ub[2] or iters>=max_step:
            if iters >= max_step:
                print('over the max step: %05d during the forward integrating' % iters)
            break
        fieldline1.append(np.array([x0, y0, z0]))
        
# back ward stream line
    x0, y0, z0 = star_point
    iters = 0
    while True:
        xi, yi, zi = np.around(np.array([x0, y0, z0])).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        k1 = -bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k1*ds/2)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        k2 = -bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k2*ds/2)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        k3 = -bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k3*ds)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        k4 = -bvec/bb
        x0, y0, z0 = np.array([x0,y0,z0])+(k1+2*k2+2*k3+k4)*ds/6.
        iters += 1
        if x0<lb[0] or x0>ub[0] or y0<lb[1] or y0>ub[1] or z0<lb[2] or z0>ub[2] or iters>=max_step:
            if iters >= max_step:
                print('over the max step: %05d during the backward integrating' % iters)
            break
        fieldline2.append(np.array([x0, y0, z0]))
        
    filedline = fieldline2[::-1]+fieldline1
    filedline = np.array(filedline)
    return filedline

# ===================================================== #
#          read boundary data from Guo's code           #
# ===================================================== #
def read_nlfff_boundary(filename):
    with open(filename, 'rb') as file:
        nx1 = np.fromfile(file, dtype='int32', count=1)[0]  
        nx2 = np.fromfile(file, dtype='int32', count=1)[0]  

        xc = np.fromfile(file, dtype='float64', count=1)[0]  
        yc = np.fromfile(file, dtype='float64', count=1)[0]  
        dx = np.fromfile(file, dtype='float64', count=1)[0]  
        dy = np.fromfile(file, dtype='float64', count=1)[0]  

        Bx = np.fromfile(file, dtype='float64', count=nx1 * nx2)  
        By = np.fromfile(file, dtype='float64', count=nx1 * nx2)  
        Bz = np.fromfile(file, dtype='float64', count=nx1 * nx2)  

    Bx = Bx.reshape(nx1, nx2)  
    By = By.reshape(nx1, nx2)  
    Bz = Bz.reshape(nx1, nx2)  
    
    return np.stack([Bx,By,Bz])


# ===================================================== #
#            import the field to VTR file               #
# ===================================================== #
def Bcube2vtr(b_cube, lb_ub = 'auto', savepath = './Bcube'):
    b_cube = b_cube.transpose(0,2,1,3)
    nx, ny, nz = np.array(b_cube.shape[-3:])-1
    if lb_ub == 'auto':
        lb = np.array([0,0,0])
        ub = np.array(b_cube.shape[1:])-1
        lb_ub = np.stack([lb,ub])
    else:
        lb_ub[:,0]=lb_ub[:,1]+lb_ub[:,0]
        lb_ub[:,1]=lb_ub[:,0]-lb_ub[:,1]
        lb_ub[:,0]=lb_ub[:,0]-lb_ub[:,1]
    lx, ly, lz = lb_ub[1]-lb_ub[0]
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    npoints = (nx + 1) * (ny + 1) * (nz + 1)
    x = np.arange(lb_ub[0,0], lx + 0.1*dx, dx, dtype='float64')
    y = np.arange(lb_ub[0,1], ly + 0.1*dy, dy, dtype='float64')
    z = np.arange(lb_ub[0,2], lz + 0.1*dz, dz, dtype='float64')
    lb, ub = lb_ub

    # savepath = "./TDm2vtk"
    w = VtkFile(savepath, VtkRectilinearGrid)
    w.openGrid(start = tuple(lb), end = tuple(ub))
    w.openPiece( start = tuple(lb), end = tuple(ub))

    # Point data
    b1 = b_cube[0,:,:,:]
    b2 = b_cube[1,:,:,:]
    b3 = b_cube[2,:,:,:]
    b1 = np.ascontiguousarray(b1)
    b2 = np.ascontiguousarray(b2)
    b3 = np.ascontiguousarray(b3)
    w.openData("Point",vectors = 'bvec', scalars={'b1', 'b2', 'b3'})
    w.addData('bvec', (b1,b2,b3))
    w.addData('b1', b1)
    w.addData('b2', b2)
    w.addData('b3', b3)
    w.closeData("Point")

    # Cell data

    # Coordinates of cell vertices
    w.openElement("Coordinates")
    w.addData("x_coordinates", x);
    w.addData("y_coordinates", y);
    w.addData("z_coordinates", z);
    w.closeElement("Coordinates");

    w.closePiece()
    w.closeGrid()

    w.appendData(data=(b1,b2,b3))
    w.appendData(data = b1)
    w.appendData(data = b2)
    w.appendData(data = b3)
    w.appendData(x).appendData(y).appendData(z)
    w.save()

# # ===================================================== #
# #            import the training data               #
# # ===================================================== #
# product_info_path = os.path.join(script_dir,'archive-202203-info.csv')
# product_info = pandas.read_csv(product_info_path)
# def read_bcube(file):
#     sample_harpnum_trec = re.search(r'\d+\.\d{8}_\d{6}_\w+',file).group()
#     sample_product_info=product_info[ product_info["harpnum_trec"]==sample_harpnum_trec]
#     sample_bout_maxlevel=int(sample_product_info["bout_maxlevel"])

#     sample_nx=int(sample_product_info["grid_x"])
#     sample_ny=int(sample_product_info["grid_y"])
#     sample_nz=int(sample_product_info["grid_z"])

#     sample_identifiers=int(sample_product_info["identifiers"])
    
#     nx=sample_nx
#     ny=sample_ny
#     nz=sample_nz

#     np_dtype_str=r"<d"
#     bin_path=file

#     # https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
#     nlfff_data = np.memmap(bin_path,
#                     dtype=np.dtype(np_dtype_str),
#                     offset=0,
#                     shape=(3, nx, ny, nz),
#                     order='C') 
#     nlfff_data = np.array(nlfff_data)
    
#     return nlfff_data

# def get_files_with_path(directory):
#     file_list = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file == "Bout.bin":
#                 file_path = os.path.join(root, file)
#                 file_list.append(file_path)
#     return file_list