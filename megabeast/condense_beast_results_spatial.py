#!/usr/bin/env python
#
# condense the spatially reordered BEAST results to single
#   stats, pdf1d, and lnp files per spatial pixel
#
# History: 
#  created Oct 2016 by Karl Gordon

import os
import glob
import math

import h5py
from tqdm import trange, tqdm

import argparse
import numpy as np

from astropy.io import fits
from astropy.table import Table, Column, vstack

def condense_stats_files(bname,
                         cur_dir,
                         out_dir):

    # get all the stats files
    stats_files = glob.glob(cur_dir + '*_stats.fits')

    # loop through the stats files, building up the output table
    cats_list = []
    for cur_stat in stats_files:
        cur_cat = Table.read(cur_stat)

        # get the source density and subregion name
        #  in other words, the reordering tag
        #  bpos is the location after the 2nd underscore of pix coords
        #  epos is before the _stats.fits ending string
        bpos = cur_stat.find(bname+'/')+len(bname)+1
        bpos = cur_stat.find('_',bpos)
        bpos = cur_stat.find('_',bpos+1) + 1
        epos = cur_stat.find('_stats')
        reorder_tag = cur_stat[bpos:epos]
            
        # add the reorder tag to each entry in the current catalog
        n_entries = len(cur_cat)
        cur_cat.add_column(Column([reorder_tag] * n_entries,
                                  name='reorder_tag'))

        # append to list
        cats_list.append(cur_cat)
    
    # concatenate all the small catalogs together
    full_cat = vstack(cats_list)
        
    # output the full pixel catalog
    full_cat.write(out_dir+'/' + bname + '_stats.fits', overwrite=True)

def condense_pdf1d_files(bname,
                         cur_dir,
                         out_dir):

    # get all the files
    pdf1d_files = glob.glob(cur_dir + '*_pdf1d.fits')



if __name__ == '__main__':

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--bricknum", 
                        help="PHAT brick num shortcut" + \
                        " (superceeds other input)")
    parser.add_argument("-d","--filedir", 
                        help="Directory to condense results")
    args = parser.parse_args()

    if args.bricknum:
        brick = str(args.bricknum)
        out_dir = '/astro/dust_kg2/kgordon/BEAST_production/b' + \
            brick + '/spatial'
        if not os.path.exists(out_dir):
            print(out_dir + ' directory does not exist')
            exit()
    elif (args.stats_filename and args.reg_filebase):
        out_dir = args.filedir
    else:
        parser.print_help()
        exit()

    # get the list of directories
    #    each directory is a different pixel
    pix_dirs = glob.glob(out_dir + '/*/')

    # loop over each subdirectory and condense the files as appropriate
    for cur_dir in tqdm(pix_dirs, desc='spatial regions'):
        # get the base name
        bname = cur_dir[cur_dir.find('spatial/')+8:-1]

        # process that catalog (stats) files
        condense_stats_files(bname, cur_dir, out_dir)

        # process the pdf1d files
        condense_pdf1d_files(bname, cur_dir, out_dir)

