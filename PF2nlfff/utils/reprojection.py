import os
import sunpy, sunpy.map

import numpy as np
import matplotlib.pyplot as plt
import sunpy.coordinates as coord

from pyevtk.hl import gridToVTK
from astropy.time import Time
from sunpy.net import Fido, attrs as a
from scipy.spatial.transform import Rotation
from sunpy.map import Map
from astropy import units as u
from sunpy.time import parse_time

def get_aia_map(observation_time=None,waveband=171):
    
    if observation_time:
        obs_time = parse_time(observation_time)
        time_range = a.Time(obs_time - 12*u.second, obs_time)
        print(f"Time range: {time_range}")
    else:
        # latest date
        time_range = a.Time(parse_time('now') - 24*u.hour, parse_time('now'))
    
    # acquire AIA 171Å data
    query = Fido.search(
        time_range,
        a.Instrument('AIA'),
        a.Wavelength(waveband*u.angstrom),
        a.Sample(12*u.second)
    )
    
    # check if find data
    if len(query[0]) == 0:
        print(f"Cannot find data in the target time range, please try another date")
        return None
    
    # download latest data
    files = Fido.fetch(query[0][-1])
    
    # check if download successful
    if len(files) == 0:
        print(f"Failed to download data, please check your network connection or try another date")
        return None
    
    print(f"Data download finished: {files[0]}")
    
    # read data
    aia_map = Map(files[0])
    return aia_map

def get_hmi_B_720s_map(datetime_str, email='hq_turtle@163.com'):
    """
    获取指定时刻的 HMI hmi.B_720s 全日面矢量磁场数据

    Parameters
    ----------
    datetime_str : str
        时间字符串，例如 '2023-06-01T12:00:00'

    Returns
    -------
    maps : dict
        {'Br': sunpy.map.Map,
         'Bt': sunpy.map.Map,
         'Bp': sunpy.map.Map}
    """

    t = Time(datetime_str)

    # JSOC 数据集
    series = a.jsoc.Series('hmi.B_720s')

    # 查询三个矢量分量
    segments = a.jsoc.Segment('inclination') | a.jsoc.Segment('azimuth') | a.jsoc.Segment('disambig') | a.jsoc.Segment('field')

    # 精确到最近的 720s 数据
    result = Fido.search(
        a.Time(t - 60*u.s, t + 60*u.s),
        series,
        segments,
        a.jsoc.Notify(email)
    )

    if len(result) == 0:
        raise RuntimeError("未找到对应时间的 hmi.B_720s 数据")

    # 下载数据
    files = Fido.fetch(result)

    # 构建 Map
    maps = {}
    for f in files:
        m = sunpy.map.Map(f)
        # segment 名在 meta 里
        seg = m.meta.get('drms_id', '').strip()
        maps[seg] = m

    return maps

def get_hmi_sharp_720s_map(datetime_str, harpnum, email='hq_turtle@163.com'):
    """
    获取指定时刻、指定 HARPNUM 的 HMI SHARP 矢量磁场数据（Br, Bt, Bp）
    """

    t = Time(datetime_str)

    series = a.jsoc.Series('hmi.sharp_cea_720s')

    segments = (
        a.jsoc.Segment('Br') |
        a.jsoc.Segment('Bt') |
        a.jsoc.Segment('Bp')
    )

    result = Fido.search(
        a.Time(t - 60*u.s, t + 60*u.s),
        series,
        a.jsoc.PrimeKey('HARPNUM', harpnum),
        segments,
        a.jsoc.Notify(email)
    )

    if len(result) == 0:
        raise RuntimeError(
            f"未找到 hmi.sharp_720s 数据: time={datetime_str}, HARPNUM={harpnum}"
        )

    files = Fido.fetch(result)

    maps = {}
    for f in files:
        m = sunpy.map.Map(f)
        seg = m.meta.get('drms_id', '').strip()
        maps[seg] = m

    return maps

def rotate(bcube,lon,lat):
    bx,by,bz = bcube
    brot = np.stack([
        +bz*np.cos(lat)*np.sin(lon)-by*np.sin(lon)*np.sin(lat)+bx*np.cos(lon),
        +bz*np.sin(lat)            +by*np.cos(lat),
        +bz*np.cos(lat)*np.cos(lon)-by*np.cos(lon)*np.sin(lat)-bx*np.sin(lon)
    ])
    return brot

class Reproject:
    """
    Handles reprojection of a 3-D magnetic-field cube onto the observer’s
    perspective defined by an HMI SHARP region.
    """

    def __init__(self, datetime_str, harpnum):
        # store user inputs
        self.datetime_str = datetime_str
        self.harpnum = harpnum

        # fetch SHARP magnetogram maps (Br, Bt, Bp)
        self.sharp_maps = get_hmi_sharp_720s_map(
            datetime_str=datetime_str,
            harpnum=harpnum,
        )

        # reorder axes of the input cube to match internal convention
        self.bcube = None
        # record spatial resolution
        self.dimensions = None

        # keep keys for later unpacking
        self.keys = self.sharp_maps.keys()
        # unpack Br, Bt, Bp maps in the same order
        self.Br, self.Bt, self.Bp = [self.sharp_maps[key] for key in self.keys]

    @staticmethod
    def _reorder(bcube):
        """Swap and permute axes so that the cube follows (comp, x, y, z)."""
        bcube_new = bcube.transpose(0, 2, 1, 3)[(1, 0, 2), :, :, :]
        return bcube_new

    def _init_params(self,**kwargs):
        """Derive pixel scales and FOV sizes in solar-radii units."""
        Br = self.Br
        # lon = Br.center.lon.value
        # lat = Br.center.lat.value
        # hgs_center = Br.center.transform_to(coord.HeliographicStonyhurst)
        # lon = hgs_center.lon.value
        # lat = hgs_center.lat.value
        hpc_center = Br.center.transform_to(coord.Helioprojective)
        Tx = hpc_center.Tx.value
        Ty = hpc_center.Ty.value
        L_unit = Br.rsun_obs.value
        lat = np.arcsin(Ty/L_unit)
        lon = np.arcsin(Tx/L_unit/np.cos(lat))
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
        scale1_cea = Br.scale.axis1.value
        scale2_cea = Br.scale.axis2.value
        rsun_arcsec = (Br.rsun_meters/Br.dsun).value*180/np.pi*3600
        lon_usr = kwargs.get('lon',None)
        lat_usr = kwargs.get('lat',None)
        if lon_usr:
            lon = lon_usr
        if lat_usr:
            lat = lat_usr

        nx, ny, nz = self.dimensions

        # convert CEA scales to arcsec and then to solar-radii
        scale1 = np.rad2deg((np.deg2rad(scale1_cea) * Br.rsun_meters / Br.dsun).value) * 3600
        scale2 = np.rad2deg((np.deg2rad(scale2_cea) * Br.rsun_meters / Br.dsun).value) * 3600

        sizex_arcsec = nx * scale1
        sizey_arcsec = ny * scale2
        sizez_arcsec = nz * (scale1 + scale2) / 2

        # normalize by solar radius
        sizex_unit = sizex_arcsec / rsun_arcsec
        sizey_unit = sizey_arcsec / rsun_arcsec
        sizez_unit = sizez_arcsec / rsun_arcsec

        self.params = {
            'lon': lon,
            'lat': lat,
            'scale1': scale1,
            'scale2': scale2,
            'sizex_unit': sizex_unit,
            'sizey_unit': sizey_unit,
            'sizez_unit': sizez_unit,
        }

    def save_aia_vts(self, aia_base_name='', waveband=171, dir_name='./'):
        aia_map = get_aia_map(self.datetime_str, waveband=waveband)
        
        # check if get_aia_map successful
        if aia_map is None:
            print(f"Failed to get AIA {waveband} map, skipping export")
            return
        
        fovx_arcsec = (aia_map.scale.axis1*aia_map.dimensions.x).value
        fovy_arcsec = (aia_map.scale.axis2*aia_map.dimensions.y).value
        rsun_arcsec = (aia_map.rsun_meters/aia_map.dsun).value*180/np.pi*3600
        fovx_unit = fovx_arcsec/rsun_arcsec
        fovy_unit = fovy_arcsec/rsun_arcsec
        N1 = int(aia_map.dimensions.x.value)
        N2 = int(aia_map.dimensions.y.value)
        x_POS = np.linspace(-fovx_unit/2, fovx_unit/2, N1)
        y_POS = np.linspace(-fovy_unit/2, fovy_unit/2, N2)
        X,Y,Z = np.meshgrid(x_POS,y_POS,np.array([0.]),indexing='ij')
        save_name = os.path.join(dir_name,'aiasdo'+str(waveband)+aia_base_name)
        gridToVTK(
            save_name,
            X,Y,Z,
            pointData={f'sdoaia{waveband}': aia_map.data.T[:,:,None]}
        )
        print(f'Exported AIA {waveband} map to {save_name}.vts')

    def __call__(self, bcube, save_vts=True, origin_name='origin', reproj_name='reproj',**kwargs):
        """
        Rotate the cube to the observer’s view and optionally export VTK files.
        """
        # store input cube
        self.bcube = self._reorder(bcube)
        # record spatial resolution
        self.dimensions = self.bcube.shape[1:]
        # compute auxiliary parameters
        self._init_params(**kwargs)

        # retrieve geometric parameters
        lon = self.params['lon']
        lat = self.params['lat']
        sizex_unit = self.params['sizex_unit']
        sizey_unit = self.params['sizey_unit']
        sizez_unit = self.params['sizez_unit']

        nx, ny, nz = self.dimensions

        # build regular grid in local Cartesian coordinates (in Rs units)
        box_x = np.linspace(-sizex_unit / 2, sizex_unit / 2, nx)
        box_y = np.linspace(-sizey_unit / 2, sizey_unit / 2, ny)
        box_z = np.linspace(1, 1 + sizez_unit, nz)
        X, Y, Z = np.meshgrid(box_x, box_y, box_z, indexing='ij')

        # compose rotation: Z(lon) then Y(-lat) to align observer
        rot = Rotation.from_euler('ZYX', [0, lon, -lat], degrees=True)
        xyz = np.stack([X, Y, Z], axis=-1)
        xyz_flat = xyz.reshape(-1, 3)
        xyz_rot = rot.apply(xyz_flat).reshape(xyz.shape)
        Xn, Yn, Zn = xyz_rot.transpose(3, 0, 1, 2)

        # ensure C-contiguous for VTK export
        Xn = np.ascontiguousarray(Xn)
        Yn = np.ascontiguousarray(Yn)
        Zn = np.ascontiguousarray(Zn)

        # rotate magnetic-field cube accordingly
        bcube_rot = rotate(self.bcube, np.deg2rad(lon), np.deg2rad(lat))

        # export original and rotated cubes as VTK structured grids
        if save_vts:
            gridToVTK(
                origin_name,
                X, Y, Z,
                pointData=dict(
                    Bx=np.ascontiguousarray(self.bcube[0]),
                    By=np.ascontiguousarray(self.bcube[1]),
                    Bz=np.ascontiguousarray(self.bcube[2]),
                )
            )
            print(f'Save original view data to {origin_name}.vts')

            gridToVTK(
                reproj_name,
                Xn, Yn, Zn,
                pointData=dict(
                    Bx=np.ascontiguousarray(bcube_rot[0]),
                    By=np.ascontiguousarray(bcube_rot[1]),
                    Bz=np.ascontiguousarray(bcube_rot[2]),
                    Bn=np.ascontiguousarray(self.bcube[2]),
                )
            )
            print(f'Save reprojected view data to {reproj_name}.vts')

        # unit vector pointing from Sun center to observer
        unit_normal_vector = np.array([
            np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat)),
            np.sin(np.deg2rad(lat)),
            np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
        ])

        # place camera slightly outside the sphere for visualization
        camera_dist = 0.2
        camera_pos = (1 + camera_dist) * unit_normal_vector
        slice_pos = 1.001 * unit_normal_vector

        self.render_params = {
            'camera_pos': camera_pos,
            'slice_pos': slice_pos,
            'radial_dir': unit_normal_vector,
        }
        print('Render params: ', self.render_params)
        ret = dict(
            reproject=dict(
                xyz = xyz_rot,
                bxyz = bcube_rot,
            ),
            original = dict(
                xyz = xyz,
                bxyz = self.bcube,
            )
        )
        return ret


if __name__ == '__main__':
    hmi_map = get_hmi_B_720s_map('2013-07-02 08:00:00')
    aia_map = get_aia_map('2013-07-02 08:00:00')
    aia_193_map = get_aia_map('2013-07-02 08:00:00', waveband=193)
    sharp_maps = get_hmi_sharp_720s_map(
        datetime_str='2013-07-02 08:00:00',
        harpnum=2923,
    )