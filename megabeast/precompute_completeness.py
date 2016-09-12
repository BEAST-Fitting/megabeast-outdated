import numpy as np
import re
import pyfits
from beast.core.grid import FileSEDGrid
from astropy.io import fits

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split('(\d+)', text)]

def completeness(sedgrid,full_idx, bins,  key='Av'):
    return np.histogram(sedgrid.grid[key][full_idx], bins = bins, normed =True)#, weights= sedgrid.grid['weight'])

def storeGridNames(dir_grid):
    import os
    import glob

    os.chdir(dir_grid)

    grid_files = glob.glob("*sed_trim.grid_indexs*")
    os.chdir('/astro/dust_kg/harab/beast/projects/dust_mapping/')
    grid_files.sort(key=natural_keys)
    return grid_files


# Computes and saves in a fits table the completeness
# Non-generic: Brick and parameter are hard-coded (Nov 15 version Rv completeness) 
brick = 'b15'
dir_grid = '/astro/dust_kg2/kgordon/BEAST_production/%s/' % brick

grid_files = storeGridNames(dir_grid)
full_grid = FileSEDGrid('/astro/dust_kg2/kgordon/BEAST_production/BEAST_production_seds.grid.hd5')

save_completeness = []
pdf1d = pyfits.open(dir_grid + '%s_sd0-1_sub0_pdf1d.fits' % brick)
bins = pdf1d[4].data[-1] # Av = 1, Rv = 4

bins_for_hist = np.zeros(len(bins) + 1)
bins_for_hist[:-1] = bins
bins_for_hist[-1] = bins[-1] + 0.15

for grid_name in grid_files:
    grid = pyfits.open(dir_grid + grid_name)
    full_idx = grid[1].data['fullgrid_idx']
    comp, bins2 = completeness(full_grid, full_idx, bins_for_hist, key='Av')
    save_completeness.append(comp)
save_completeness.append(bins2[:-1])

outname = 'Rv_completeness/%s_Rv_completeness.fits' % brick
fits.writeto(outname, np.zeros((2,2)),clobber=True)
for k in range(len(save_completeness)):
    if k < range(len(save_completeness))[-1]-1:
        extname = grid_files[k].split('_')[1] + '_' + grid_files[k].split('_')[2]
    else:
        extname = 'Rv_bins'
    hdu = fits.PrimaryHDU(save_completeness[k])
    pheader = hdu.header
    pheader.set('EXTNAME',extname)
    fits.append(outname,save_completeness[k],header=pheader)

        
    


