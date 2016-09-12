import pyfits
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from astropy import wcs
from astropy.io import fits
from beast.core.vega import Vega


def star_position_map(cat, pix_size = 5.):
    """
    
    HA 05/21/15
    """
 
    # Setting map fame
    min_ra = cat['ra'].min()
    max_ra = cat['ra'].max()
    min_dec = cat['dec'].min()
    max_dec = cat['dec'].max()

    #Compute number of pixel alog each axis pix_size in arcsec
    dec_delt = pix_size/3600.
    n_y = np.fix(np.round((max_dec - min_dec)/dec_delt))
    ra_delt = dec_delt
    n_x = np.fix(np.round(math.acos(0.5*(max_dec+min_dec)*math.pi/180.)*(max_ra-min_ra)/ra_delt)) #!!! BUG NOT ACOS BUT COS - kept for consistency in current version (not a big difference in PHAT) !!!
    ra_delt *= -1.

    print '# of x & y pixels = ', n_x, n_y

    ra_limits = min_ra + ra_delt*np.arange(0,n_x+1, dtype=float)
    dec_limits = min_dec + dec_delt*np.arange(0,n_y+1, dtype=float)
    
    cdelt = [ra_delt, dec_delt]
    crpix = np.asarray([n_x, n_y], dtype = float) / 2.
    crval = np.asarray([(min_ra + max_ra), (min_dec+max_dec)]) / 2.

    w = wcs.WCS(naxis = 2)
    w.wcs.crpix = crpix
    w.wcs.cdelt = cdelt
    w.wcs.crval = crval
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    #with Vega() as v:
    #    vega_f,vega_flux,lamb = v.getFlux(['HST_ACS_WFC_F475W'])
    #min_flux = vega_flux*10**(-0.4*max_mag) 
    #good,= np.where((cat['chi2min'] < max_chisq) & (cat['HST_ACS_WFC_F475W'] > min_flux))
    #print len(good)
    #cat = cat[good]
    N_stars = len(cat)
    print N_stars

    world = np.zeros((N_stars,2),float)
    world[:,0] = cat['ra']
    world[:,1] = cat['dec']
    print 'working on converting ra, dec to pix x,y'
    pixcrd = w.wcs_world2pix(world, 1)
    pix_x = pixcrd[:,0]
    pix_y = pixcrd[:,1]
    D = {}

    for i in range(n_x):
        print 'x = %s out of %s' % (str(i+1), str(n_x))
        for j in range(n_y):
		code = np.str(i) + '-' + np.str(j)
		indxs ,= np.where((pix_x > i) & (pix_x <= i+1) & (pix_y > j) & (pix_y <= j+1))
		D[code] = indxs
	    
    return D

if __name__ == '__main__':
	
    brick = 'b02'
    dir_cat = '/astro/dust_kg2/harab/toothpick_results/v1_1/'
    cat = pyfits.getdata(dir_cat + '%s_stats_v1_1.fits' % brick)
    ii,=np.where(cat['ra'] > 1.)
    D2 = star_position_map(cat[ii], pix_size = 10.)  # Pixel size = 10 arcsec
    outname= 'star_positions_v1.1/%s_star_positions_10arcsec.p' % brick
    with open(outname,"wb") as f:
        pickle.dump(D2,f)
    f.close()

