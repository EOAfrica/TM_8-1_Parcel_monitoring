import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries

# custom functions
from rescale import rescale_ts
from mask import mask_ts
from composite import composite_ts
from interpolate import interpolate_ts
from features import tsteps
from phenology import (detect_seasons, visualize_seasons)


if __name__ == '__main__':

    # Select the correct data file and open it
    # infile = '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/s2_data/36LWL/30/demo/S2-L2A_20m_36LWL-30_32736_2023-07-10_2023-08-01.nc'
    infile = '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/s2_data/36LWL/30/S2-L2A_20m_36LWL-30_32736_2022-10-01_2023-08-01.nc'
    ds = xr.open_dataset(infile)
    # inspect the content of the file
    ds

    ##
    # Plot the first SWIR band to see what the data looks like
    fig, ax = plt.subplots()
    b11 = ds[['B11']].to_array().values[0, :, 200, 200]
    time = ds.coords['timestamp'].values
    ax.plot(time, b11, '-o')
    plt.xticks(rotation=45, ha='right')
    plt.title('B11 original')
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/b11.png')

    ##
    # We will first process the 20 m data and after pre-processing upsample the data to 10m
    # so it can be combined with the 10 m data.

    # First step will be cloud masking. For this, we need the scene classification band, which is available at 20m resolution.
    # We convert it to a data array for further processing...
    scl_20 = ds[['SCENECLASSIFICATION']].to_array()

    # within the scene classification layer, the following values point to clouds/shadows and should be masked:
    scl_mask_values = [1, 3, 8, 9, 10, 11]

    # create mask (True = not to be masked, False = to be masked)
    mask_20 = np.logical_not(scl_20.isin(scl_mask_values))
    mask_20.attrs = scl_20.attrs.copy()

    # NOTE that the quality of the scene classification layer for sentinel-2 is not always optimal.
    # Several enhancements are possible before applying this layer for masking. An example can be found here:m
    # https://github.com/dzanaga/satio-pc/blob/main/satio_pc/preprocessing/clouds.py#L60
    # This is however beyond the scope of this exercise.

    # Get the 20 m bands from the file
    ts_20 = ds[['B05', 'B06', 'B07', 'B11', 'B12']].to_array()

    # Apply the mask to the data
    ts_20_masked = mask_ts(ts_20, mask_20)
    fig, ax = plt.subplots()
    b11 = ts_20_masked.sel(variable='B11').values[:, 200, 200]
    time = ds.coords['timestamp'].values
    ax.plot(time, b11, '-o')
    plt.xticks(rotation=45, ha='right')
    plt.title('B11 masked')
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/b11_msk.png')

    # Now we create a 10-daily summary of the data through temporal compositing.
    # The result of this step will be that you end up with a time series having one observation every 10 days.
    # Each observation is the median of all available observations within the 10 day window.
    # This makes the time series more uniform (number of available observations highly depends on the location!)
    ts_20_comp = composite_ts(ts_20_masked, freq=10, window=20, mode='median')
    fig, ax = plt.subplots()
    b11 = ts_20_comp.sel(variable='B11').values[:, 200, 200]
    time = ts_20_comp.coords['timestamp'].values
    ax.plot(time, b11, '-o')
    plt.xticks(rotation=45, ha='right')
    plt.title('B11 composited')
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/b11_comp.png')

    # As a final step, we linearly interpolate any remaining gaps in the time series
    # (for some 10 day windows, there was not a single valid observation...)
    ts_20_intp = interpolate_ts(ts_20_comp)
    fig, ax = plt.subplots()
    b11 = ts_20_intp.sel(variable='B11').values[:, 200, 200]
    time = ts_20_comp.coords['timestamp'].values
    ax.plot(time, b11, '-o')
    plt.xticks(rotation=45, ha='right')
    plt.title('B11 interpolated')
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/b11_interp.png')

    # Finally, we upsample the 20 m data to 10 m
    ts_20_fin = rescale_ts(ts_20_intp, scale=2, order=0)
    ts_20_fin

    ##
    # Now we prepare the 10 m data

    # Select the correct data file and open it
    infile = '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/s2_data/36LWL/30/S2-L2A_10m_36LWL-30_32736_2022-10-01_2023-08-01.nc'
    # infile = < YOUR CODE HERE >
    ds = xr.open_dataset(infile)
    ds

    ##
    # Now we pre-process the 10 m data

    # convert the mask to 10 m resolution
    mask_10 = rescale_ts(mask_20, scale=2, order=0)

    # Get the 10 m bands from the file
    ts_10 = ds[['B02', 'B03', 'B04', 'B08']].to_array()

    # Apply the mask to the data
    ts_10_masked = mask_ts(ts_10, mask_10)

    # Apply temporal compositing
    ts_10_comp = composite_ts(ts_10_masked, freq=10, window=20, mode='median')

    # Apply linear interpolation
    ts_10_fin = interpolate_ts(ts_10_comp)
    ts_10_fin

    # plot comparison between original NIR band and the result of pre-processing
    fig, ax = plt.subplots()
    b08 = ts_10.sel(variable='B08').values[:, 200, 200]
    time = ds.coords['timestamp'].values
    ax.plot(time, b08, '-o')
    time = ts_10_fin.coords['timestamp'].values
    b08 = ts_10_fin.sel(variable='B08').values[:, 200, 200]
    ax.plot(time, b08, '-or')
    plt.xticks(rotation=45, ha='right')
    plt.title('NIR band pre-processing')
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/b08_preproc.png')

    # As a final step, we merge all bands together

    ts_fin = xr.concat([ts_10_fin, ts_20_fin], dim='variable')

    # free up some memory
    del ts_20, ts_20_fin, ts_20_comp, ts_20_intp, ts_20_masked
    del ts_10, ts_10_fin, ts_10_comp, ts_10_masked

    ##
    # Compute NDVI and visualize
    b08 = ts_fin.sel(variable='B08').values
    b04 = ts_fin.sel(variable='B04').values
    ndvi = (b08 - b04) / (b08 + b04)

    # Visualize spatially for one particular date
    fig, ax = plt.subplots()
    ndviplot = plt.imshow(ndvi[10, ...])
    fig.colorbar(ndviplot, ax=ax)
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/ndvi_image.png')

    # Visualize temporally for one pixel
    time = ts_fin.coords['timestamp'].values
    fig, ax = plt.subplots()
    ax.plot(time, ndvi[:, 200, 200], '-or')
    plt.xticks(rotation=45, ha='right')
    plt.title('NDVI for pixel (200,200)')
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/ndvi_timeseries.png')

    ##
    # Using NDVI throughout the season to delineate individual fields
    # Clip a portion from the image
    ndvi_sub = ndvi[:, 0:200, 0:200]
    fig, ax = plt.subplots()
    plt.imshow(ndvi[12, ...])
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/ndvi_sub.png')

    # Resample NDVI temporally to six equidistant timesteps
    ndvi_tsteps = tsteps(ndvi_sub)

    # plot these 6 timesteps
    for i in range(6):
        fig, ax = plt.subplots()
        feat_plt = ax.imshow(np.squeeze(ndvi_tsteps[i, ...]))
        fig.colorbar(feat_plt, ax=ax)
        ax.set_title(f'NDVI tstep {i}')
        plt.savefig(
            f'/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/ndvi_sub_tstep{i}.png')

    # use 3 out of 6 timesteps for parcel delineation
    tsteps_selected = [1, 3, 5]
    inputs = np.take(ndvi_tsteps, tsteps_selected, axis=0)
    inputs = np.moveaxis(inputs, 0, -1)

    # apply segmentation algorithms
    # https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
    segments_fz = felzenszwalb(inputs, scale=50, sigma=0.5)
    segments_fz

    # prepare an RGB image for visual reference
    rgb = ts_fin.sel(variable=['B04', 'B03', 'B02']
                     ).values[:, 12, 0:200, 0:200]
    rgb = np.moveaxis(rgb, 0, -1)

    # plot result
    fig, ax = plt.subplots(1, 2, figsize=(10, 10),
                           sharex=True, sharey=True)
    ax[0].imshow(mark_boundaries(inputs, segments_fz))
    ax[0].set_title("Felzenszwalbs segmentation result")
    ax[1].imshow(rgb)
    ax[1].set_title('RGB image')
    plt.savefig(
        '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/segmentation_result.png')

    ##
    # Phenology detection (start, mid and end of growing seasons)
    # Detection of growing seasons can be done on a pixel-per-pixel basis using the NDVI time series of
    # each individual pixel OR it can be done per parcel using the segments delineated above.
    # Here we will showcase the latter option.

    # compute the average NDVI profile per parcel
    segment_ids = np.unique(segments_fz)
    nsegments = len(segment_ids)
    ntimes = ndvi_sub.shape[0]
    ndvi_segm = np.zeros([ntimes, nsegments], dtype=np.float32)
    for i, id in enumerate(segment_ids):
        msk = segments_fz == id
        msk_3d = np.broadcast_to(msk, ndvi_sub.shape)
        ndvi_masked = np.where(msk_3d == 1, ndvi_sub, np.nan)
        ndvi_segm[:, i] = np.nanmean(ndvi_masked, axis=(1, 2))

    # Now apply phenology detection on this data array
    # first make sure the input shape is 3D (see documentation of the detect_seasons function)
    ndvi_segm = np.expand_dims(ndvi_segm, axis=2)
    time = ts_fin.coords['timestamp'].values
    nseas, sos, mos, eos = detect_seasons(ndvi_segm, time)

    # visualize output for one pixel
    outfile = '/data/users/Private/jeroendegerickx/eoafrica/exercise_parcel_monitoring/pheno_detection.png'
    x, y = 5, 0
    visualize_seasons(nseas, sos, mos, eos, x, y,
                      ndvi_segm, time, outfile=outfile)
