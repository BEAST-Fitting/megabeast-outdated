#!/usr/bin/env python
#
# reorder the BEAST results to be in ra/dec bins
#
# optimized BEAST runs are done by sets of stars in source density bins
#   sorted by flux and subdivided into smaller files
#   this allow for the BEAST grid to be cut and speeds up the fitting
# 
# but this is non-ideal for almost any analysis of the results
#   especially the MegaBEAST!
#
# History: based on work by Heddy Arab prior to 2016
#          reordered code and flow to optimize for speed by Karl Gordon (09/16)

import os
import glob
import math

import argparse
import numpy as np

from astropy import wcs
from astropy.table import Table

def setup_spatial_regions(cat_filename,
                          pix_size=10.0):
    """
    The spatial regions are setup via a WCS object
        
    Parameters
    ----------
    cat_filename : string
       filename of catalog

    pix_size : float
       size of pixels/regions in arcsec

    Returns
    -------
    wcs_info: astropy WCS object
    """
    
    # read in the catalog file
    cat = Table.read(cat_filename)

    # min/max ra
    min_ra = cat['RA'].min()
    max_ra = cat['RA'].max()
    min_dec = cat['DEC'].min()
    max_dec = cat['DEC'].max()

    # ra/dec delta values
    dec_delt = pix_size/3600.
    ra_delt = dec_delt

    # compute the number of pixels and 
    n_y = np.fix(np.round((max_dec - min_dec)/dec_delt))
    n_x = np.fix(np.round(math.cos(0.5*(max_dec+min_dec)*math.pi/180.)*
                          (max_ra-min_ra)/ra_delt)) 

    # ra delta should be negative
    ra_delt *= -1.

    print('# of x & y pixels = ', n_x, n_y)

    w = wcs.WCS(naxis = 2)
    w.wcs.crpix = np.asarray([n_x, n_y], dtype = float) / 2.
    w.wcs.cdelt = [ra_delt, dec_delt]
    w.wcs.crval = np.asarray([(min_ra+max_ra), (min_dec+max_dec)]) / 2.
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return w

def regions_for_objects(ra,
                        dec,
                        wcs_info):
    """
    Generate the x,y coordinates for each object based on the input
    ra/dec and already created WCS information.
        
    Parameters
    ----------
    ra : array of float
       right ascension of the objects

    dec : array of float
       declination of the objects

    wcs_info: astropy WCS object
       previously generated WCS object based on the full catalog

    Returns
    -------
    dictonary of:

    x : int array
      x values of regions

    y : int array
      y values of regions

    name : str array
      string array composed of x_y
    """

    # generate the array needed for fast conversion
    world = np.empty((len(ra),2),float)
    world[:,0] = ra
    world[:,1] = dec

    # convert
    pixcrd = wcs_info.wcs_world2pix(world, 1)

    # get the arrays to return
    x = pixcrd[:,0].astype(int)
    y = pixcrd[:,1].astype(int)
    xy_name = [None]*len(ra)

    x_str = x.astype(np.string_)
    y_str = y.astype(np.string_)
                  
    for k in range(len(x)):
        xy_name[k] = str(x[k]) + '_' + str(y[k])
    
    # return the results as a dictonary
    #   values are truncated to provide the ids for the subregions
    return {'x': x, 'y': y, 'name': xy_name}

if __name__ == '__main__':

    # command line params to specify the run directory
    #   and any other needed parameters

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--bricknum", 
                        help="PHAT brick num shortcut")
    parser.add_argument("-s","--stats_filename", 
                        help="Filename of the full run stats")
    parser.add_argument("-r","--region_filebase", 
                        help="Filebase of the run regions")
    parser.add_argument("-o","--output_filebase", 
                        help="Filebase to use for output")
    args = parser.parse_args()

    if args.bricknum:
        brick = str(args.bricknum)
        cat_filename = '/astro/dust_kg2/harab/toothpick_results/v1_1/b' + \
            brick + '_stats_v1_1.fits'
        reg_filebase = '/astro/dust_kg2/kgordon/BEAST_production/b' + \
            brick + '/b' + brick 
        out_dir = '/astro/dust_kg2/kgordon/BEAST_production/b' + \
            brick + '/spatial'
        out_filebase = out_dir + '/b' + brick
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    elif (args.stats_filename and args.reg_filebase):
        cat_filename = args.stats_filename    
        reg_filebase = args.region_filebase
        out_filebase = args.out_filebase
    else:
        parser.print_help()
        exit()

    # read in the full brick catalog and setup the spatial subdivided regions
    wcs_info = setup_spatial_regions(cat_filename)

    # find all the subdivided BEAST files for this brick
    sub_files = glob.glob(reg_filebase + '*_stats.fits')

    # loop over the files and output to the appropriate spatial region files
    # in loop:
    #      read in the locations of each star and calculated spatial region
    #      append the 1D pdfs
    #      append the nD pdfs
    #      append the completeness function (??)

    for cur_file in sub_files:
        # read in the stats file
        cur_cat = Table.read(cur_file)
        
        # determine the subregions for all the objects
        xy_vals = regions_for_objects(cur_cat['RA'],
                                      cur_cat['DEC'],
                                      wcs_info)

        # get the unique xy regions
        xy_names = np.squeeze(xy_vals['name'])
        uniq_xy_names, rindxs = np.unique(xy_names, 
                                          return_inverse=True)

        # loop over the unique xy regions
        for k, uxy_name in enumerate(uniq_xy_names):

            # get the indexes for the objects in this region
            indxs, = np.where(rindxs == k)
            print(uxy_name, len(indxs), xy_names[indxs])

            # create region directory if it does not exist
            reg_dir = out_filebase + '_' + uxy_name
            if not os.path.exists(reg_dir):
                os.makedirs(reg_dir)

            # write the stats info
            reg_stats_file = out_filebase + '_' + uxy_name + '_stats.fits'
            print(reg_stats_file)
            #cur_cat_region = cur_cat[indxs]
            cur_cat[indxs].write(reg_stats_file)
            #cur_cat_region.write(reg_stats_file)

            if k > 10:
                exit()

        exit()