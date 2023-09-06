
from pathlib import Path
import pandas as pd
from loguru import logger
import numpy as np
from typing import List
import xarray as xr
from rasterio.crs import CRS
import os

from satio.collections import TerrascopeV200Collection
from satio.utils.resample import downsample_n
from satio.timeseries import Timeseries
from satio.grid import S2TileBlocks
from satio import layers

from et_calc.sen_et.collections import (get_S2_products,
                                        create_products_df)
from et_calc.sen_et.io import check_collection
from et_calc.sen_et.utils.masking import (scl_mask,
                                          SCL_MASK_VALUES)

MASK_SETTINGS = {
    'erode_r': 3,
    'dilate_r': 13,
    'mask_values': SCL_MASK_VALUES,
    'multitemporal': False}

BLOCK_SIZE = 1024

S2_GRID = layers.load('s2grid')

S2_10M_BANDS = ['B02', 'B03', 'B04', 'B08']
S2_20M_BANDS = ['B05', 'B06', 'B07', 'B11', 'B12',
                'SCENECLASSIFICATION']


def get_processing_blocks(tile, blocks: List):

    splitter = S2TileBlocks(BLOCK_SIZE, s2grid=S2_GRID)
    processingblocks = splitter.blocks(tile)

    # Subset on the desired blocks
    processingblocks = processingblocks.loc[blocks]

    return processingblocks


def load_mask(coll):

    logger.info(f"L2A collection products: {coll.df.shape[0]}")

    logger.info('SCL: loading')
    scl_ts = coll.load_timeseries('SCENECLASSIFICATION')

    scl_ts = scl_ts.upsample()

    logger.info("SCL: preparing mask")
    mask, obs, valid_before, valid_after = scl_mask(
        scl_ts.data, **MASK_SETTINGS)

    mask_20 = downsample_n(mask.astype(np.uint16), 1) == 1

    mask_dict = {10: mask,
                 20: mask_20}

    return mask_dict


def load_data(collection, resolution,
              timeseries=None,
              no_data=32767,
              dtype=None):
    """
    Load Timeseries from the collection and merge with `timeseries` if
    given.
    `dtype` allows optional explicit casting of the loaded data
    """

    if resolution == 10:
        bands_to_load = S2_10M_BANDS
        bands_scaling = dict.fromkeys(bands_to_load, 0.0001)
        bands_dtype = dict.fromkeys(bands_to_load, 'uint16')
    elif resolution == 20:
        bands_to_load = S2_20M_BANDS
        bands_scaling = dict.fromkeys(bands_to_load, 0.0001)
        bands_dtype = dict.fromkeys(bands_to_load, 'uint16')
        bands_scaling['SCENECLASSIFICATION'] = 1  # This band is not scaled!
        bands_dtype['SCENECLASSIFICATION'] = 'uint8'  # This band is different!
    else:
        raise ValueError(f'Resolution {resolution} not supported!')

    logger.info(f'Loading bands: {bands_to_load}')
    bands_ts = collection.load_timeseries(*bands_to_load)

    # Nodata to zero
    bands_ts.data[bands_ts.data == no_data] = 0
    # additionally also values >= 21.000 to 0
    bands_ts.data[bands_ts.data >= 21000] = 0

    # Force dtype
    if dtype is not None:
        bands_ts.data = bands_ts.data.astype(dtype)

    # Merge with other timeseries
    if timeseries is None:
        timeseries = bands_ts
    else:
        timeseries = timeseries.merge(bands_ts)

    return timeseries, bands_scaling, bands_dtype


def filter_collection(coll, tile, bounds, epsg, start_date, end_date):
    """Filter input collections both spatially and temporally."""

    # Filter collections spatially
    coll = coll.filter_bounds(bounds, epsg)

    # Filter temporal collections
    # Need to add one day to end_date
    # because temporal filtering excludes "end_date"!
    end_date = (pd.to_datetime(end_date) +
                pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    coll = coll.filter_dates(start_date, end_date)

    # If we have S2 or S3 products,
    # additionally filter on the tile
    coll = coll.filter_tiles(tile)

    check_collection(coll, 'S2', start_date, end_date)
    logger.info('-' * 50)

    return coll


def get_s2_col(tile, epsg, start_date, end_date):

    # we use a buffer of 15 days before and after!
    start_s2 = (pd.to_datetime(start_date) -
                pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    end_s2 = (pd.to_datetime(end_date) +
              pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    s2products = get_S2_products(tile, start_s2, end_s2)
    s2_products_df = create_products_df(s2products, tile, epsg)

    TSS2coll = TerrascopeV200Collection(s2_products_df)

    return TSS2coll


def _arr_to_xarr(arr, bounds, timestamps, resolution, name=None):

    if arr.ndim == 1:
        # single pixel
        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=-1)

    dims = ['timestamp', 'y', 'x']
    dims = {k: arr.shape[i] for i, k in enumerate(dims)}

    center_shift = resolution / 2
    xmin, xmax = (bounds[0] + center_shift), (bounds[2] - center_shift)
    ymin, ymax = (bounds[1] + center_shift), (bounds[3] - center_shift)

    x = np.linspace(xmin,
                    xmax,
                    dims['x'])

    y = np.linspace(ymin,
                    ymax,
                    dims['y'])

    coords = {'timestamp': timestamps,
              'x': x, 'y': y}

    da = xr.DataArray(arr,
                      coords=coords,
                      dims=dims,
                      name=name,
                      attrs={'resolution': resolution})

    return da


def ts_to_xarr(ts, bounds, epsg, resolution, products):

    xds_dict = {band: _arr_to_xarr(ts.data[i],
                                   bounds,
                                   ts.timestamps,
                                   resolution,
                                   name=band)
                for i, band in enumerate(ts.bands)}

    xds_dict.update({'epsg': epsg,
                     'bounds': bounds,
                     'products': products,
                     'bands': ts.bands})

    return xds_dict


def data_to_xarr(data, prefix='S2-L2A'):

    arr = data['data']
    bounds = data['bounds']
    xmin, ymin, xmax, ymax = bounds
    resolution = data['resolution']
    tile = data['tile']
    block = data['block']

    attrs = {a: data[a] for a in ['epsg',
                                  'resolution',
                                  'bounds',
                                  'start_date', 'end_date',
                                  'tile', 'block']}

    attrs['crs'] = CRS.from_epsg(data['epsg']).to_proj4()

    dims = ['band', 'timestamp', 'y', 'x']
    dims = {k: arr.shape[i] for i, k in enumerate(dims)}

    if arr.shape[2] > 1:

        x = np.linspace(xmin + resolution / 2,
                        xmax - resolution / 2,
                        dims['x'])

        y = np.linspace(ymax - resolution / 2,
                        ymin + resolution / 2,
                        dims['y'])
    else:
        # Special case where we have only one pixel
        # Set the center coordinate
        x = [xmin + (xmax - xmin) / 2]
        y = [ymin + (ymax - ymin) / 2]

    timestamps = data['timestamps']
    bands = data['bands']
    coords = {'band': bands,
              'timestamp': timestamps,
              'x': x,
              'y': y}

    da = xr.DataArray(arr,
                      coords=coords,
                      dims=dims,
                      attrs=attrs)

    # Some things we need to do to make GDAL
    # and other software recognize the CRS
    # cfr: https://github.com/pydata/xarray/issues/2288
    da.coords['spatial_ref'] = 0
    da.coords['spatial_ref'].attrs['spatial_ref'] = CRS.from_epsg(
        data['epsg']).wkt
    da.coords['spatial_ref'].attrs['crs_wkt'] = CRS.from_epsg(
        data['epsg']).wkt
    da.attrs['grid_mapping'] = 'spatial_ref'

    # Now we convert DataArray to Dataset to set
    # band-specific metadata
    ds = da.to_dataset(dim='band')
    ds.attrs['name'] = f"{prefix}_{resolution}m_{tile}-{block}"
    ds.attrs['grid_mapping'] = 'spatial_ref'

    # Attributes of DataArray were copied to Dataset at global level
    # we need them in the individual DataArrays of the new Dataset as well
    for band in ds.data_vars:
        ds[band].attrs = ds.attrs
        ds[band].attrs['grid_mapping'] = 'spatial_ref'

    # Apply scale factor
    # ATTENTION: at this point, the data is still in Float32, which
    # is needed for proper scaling. By constructing the proper
    # NetCDF4 encoding, the data will be scaled again and stored
    # in the desired dtype. Upon loading, Xarray will recognize the
    # encoding and unscale again to float32.
    # Note that by passing 'mask_and_scale=False' or
    # 'decode_cf=False', you can instruct Xarray to leave the
    # compressed values untouched, if you like to work on int16 e.g.

    for band in ds.data_vars:
        ds[band].values = ds[band]*data['bands_scaling'][band]

    # Construct the NetCDF4 encoding
    # TODO: check if we need to define _FillValue
    encoding = dict()
    for band in data['bands']:
        if type(data['bands_dtype']) is str:
            dtype = data['bands_dtype']
        else:
            dtype = data['bands_dtype'][band]
        encoding[band] = dict(dtype=dtype,
                              scale_factor=data['bands_scaling'][band])

    # Set the encoding for the dataset
    ds.encoding = encoding

    return ds


def to_file(dataset, filename, encoding):
    if os.path.exists(filename):
        os.remove(filename)
    # Write to file
    # NOTE: when using default netcdf4 engine, it appears
    # the resulting .nc files are sometimes corrupt
    # this has not been observed with h5netcdf engine (yet)
    dataset.to_netcdf(filename, encoding=encoding,
                      engine='h5netcdf')


def decode_to_file(data, outfolder):

    # Convert to xarray
    ds = data_to_xarr(data)

    # Check if we actually have any timestamps
    # which is very rarely not the case
    if len(ds['timestamp']) == 0:
        logger.warning(('No timestamps in dataset!'
                        ' Skipping file.'))
        return

    ds_basename = (ds.attrs['name']
                   + f'_{ds.epsg}'
                   + f'_{ds.start_date}'
                   + f'_{ds.end_date}'
                   + '.nc')

    # Set output directory
    filename = (Path(outfolder) / data['tile'] /
                str(data['block']) / ds_basename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Set _FillValue
    for band in ds.encoding.keys():
        ds.encoding[band]['_FillValue'] = 0

    # Get the encoding with renamed band values
    encoding = ds.encoding

    # Write to encoded NetCDF
    try:
        logger.info(f'Writing to: {filename}')
        to_file(ds, filename, encoding)
    except Exception:
        # If we end up here, it means we couldn't
        # write successfully after a few attempts
        # Remove the bad file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        logger.error(f'Could not write to file: {filename}')
    finally:
        ds.close()


def generate_s2_data(tile, blocks, start_date, end_date, outdir):

    # get processing blocks
    proc_blocks = get_processing_blocks(tile, blocks)
    processing_tuples = [(f'{row.tile}_{row.block_id:03d}', row)
                         for id, row in proc_blocks.iterrows()]

    # generate tile collection one time...
    epsg = processing_tuples[0][1].epsg
    s2_col = get_s2_col(tile, epsg, start_date, end_date)

    for tup in processing_tuples:

        block_id, block = tup
        t = block.tile
        bounds = block.bounds
        epsg = block.epsg
        logger.info(f'Processing block {block_id}')

        # filter collection
        s2_col_filtered = filter_collection(s2_col, t, bounds, epsg,
                                            start_date, end_date)
        # products = s2_col_filtered.products

        # load data
        for resolution in [10, 20]:

            data, bands_scaling, bands_dtype = load_data(
                s2_col_filtered, resolution, dtype=np.uint16)
            timestamps = list(data.timestamps)
            data = data.to_xarray()

            # # add mask to timeseries
            # mask_ts = Timeseries(np.expand_dims(mask_dict[resolution], 0),
            #                      ts.timestamps,
            #                      ['SCL'])
            # ts = ts.merge(mask_ts)

            # Wrap everything in dictionary
            data = dict(
                data=data.values,
                bands=list(data.bands.values),
                bounds=bounds,
                resolution=resolution,
                epsg=epsg,
                start_date=start_date,
                end_date=end_date,
                tile=t,
                block=block.block_id,
                timestamps=timestamps,
                bands_scaling=bands_scaling,
                bands_dtype=bands_dtype
            )

            # convert and export to netCDF
            decode_to_file(data, outdir)

        logger.info('Done!')


if __name__ == '__main__':

    # Download raw S2 data for tile 36LWL, block 30,
    # from Oct 2022 till Jul 2023

    tile = '36LWL'
    blocks = [30]
    start_date = '2022-10-01'
    end_date = '2023-08-01'
    outdir = Path('/data/users/Private/jeroendegerickx/eoafrica/'
                  'exercise_parcel_monitoring/s2_data')

    generate_s2_data(tile, blocks, start_date, end_date, outdir)

    # Download pre-processed S2 data for tile 36LWL, block 30,
    # starting from Oct 2019 till Jul 2023

    # tile = '36LWL'
    # block = 30
    # start_date = '2019-10-01'
    # end_date = '2023-08-01'
