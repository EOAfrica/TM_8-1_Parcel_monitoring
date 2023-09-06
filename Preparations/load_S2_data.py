
import xarray as xr


if __name__ == '__main__':
    infile = '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/s2_data/36LWL/30/S2-L2A_20m_36LWL-30_32736_2023-07-10_2023-08-01.nc'
    test = xr.open_dataset(infile)
