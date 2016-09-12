import os
import glob
import re
import pyfits
import numpy as np
from beast.core.vega import Vega
import smtplib
from email.mime.text import MIMEText


#-------------------------- UTILITY FUNCTIONS ------------------------------
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split('(\d+)', text)]

def findStar(star, brick):
    """
    Find the source density bin, the subfile and the position of a full brick catalog star
    INPUTS:
    -------
           star: integer
                 Index of the star in the full brick catalog
           brick: string
                 Name of the brick (example: 'b21')
    OUTPUT:
    -------
           tuple
           (source density bin, subfile, star index in subfile)  All indexes start to 0
    HA
    """
    import os
    import glob
    direct_data = '/astro/dust_kg2/kgordon/BEAST_production/%s/' % (brick)
    direct_work = '/astro/dust_kg/harab/beast/projects/dust_mapping/'
    os.chdir(direct_data)
    stats_files = glob.glob("*stats*")
    stats_files.sort(key=natural_keys)
    all_sds = []
    if brick == 'b15':
        len_file = 12500 # Number of lines in split files for b15
    else:
        len_file = 6250  # Number of lines in split files
    for nn in stats_files:
        all_sds.append(np.int(nn.split('_')[1].split('-')[0].split('d')[1]))
    sds = list(set(all_sds))
    first_index_tot = np.zeros(len(sds),dtype =np.int)
    for ek, i in enumerate(sds[:-1]):
        stats_files_sd = glob.glob("*sd%s-*stats*" % i)
        stats_files_sd.sort(key=natural_keys)
        N_subs = len(stats_files_sd)
        last_file_sd = pyfits.open(stats_files_sd[-1])
        if i != sds[-1]:
            if N_subs == 1:
                first_index_tot[ek+1] = first_index_tot[ek] + len(last_file_sd[1].data)
            else:
                first_index_tot[ek+1] = first_index_tot[ek] + (N_subs-1)*len_file + len(last_file_sd[1].data)
    lt ,= np.where(first_index_tot <= star)
    sd = sds[lt[-1]]
    index_sd = star - first_index_tot[lt[-1]]
    sub = index_sd // len_file
    ind = index_sd % len_file
    os.chdir(direct_work)
    return (sd, sub, ind)

#------------------------------------------------------------------------------


def number_of_pixel_in_map(cat, pix_size = 5.):
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
    n_x = np.fix(np.round(math.acos(0.5*(max_dec+min_dec)*math.pi/180.)*(max_ra-min_ra)/ra_delt)) #!!! BUG SHOULD BE COS NOT ACOS kept in this version for consistency (not a big difference in PHAT)!!!
    ra_delt *= -1.
    return n_x, n_y


if __name__ == '__main__':
    import pickle

    print datetime.datetime.now()
    brick = 'b12'
    dir_brick = '/astro/dust_kg2/kgordon/BEAST_production/%s/' % brick
    dir_cat = '/astro/dust_kg2/harab/toothpick_results/v1_1/'
    star_positions_outname = 'star_positions_v1.1/%s_star_positions_10arcsec.p' % brick
    precomp_completeness = pyfits.open('Av_completeness/%s_Av_completeness.fits'% brick) # See precomputing_completeness.py
    cat = pyfits.getdata(dir_cat + '%s_stats_v1_1_full.fits' % brick)
    with open(star_positions_outname,"rb") as f:
        star_positions = pickle.load(f)       # See precompute_star_positions.py
    f.close()

    # number of pixel in each dimension
    nx, ny = number_of_pixel_in_map(cat,pix_size=10.)
    #nx = 77 #154   # numbers form map making function 
    #ny = 50 #101

    
    max_chi2 = 35.    # Chi-square filter
    max_mag  = 27.6   # Magnitude filter
    with Vega() as v:
        vega_f,vega_flux,lamb = v.getFlux(['HST_ACS_WFC_F475W'])
    min_flux = vega_flux*10**(-0.4*max_mag)  

    y1 = raw_input("Enter y1:")
    y2 = raw_input('Enter y2:')
    print y1, y2
    print nx, ny
    for absc in range(nx):
        print "i= %s out of %s" % (np.str(absc), np.str(nx-1))
        for ordi in range(np.int(y1),np.int(y2)):#range(ny):

            #PART1: PREPARE VARIABLES (Read and store the 1d PDFs and completeness) 
            code = np.str(absc) + '-' + np.str(ordi) # keyword in star position dictionary
            index = star_positions[code]             # Find the star indexes in that pixels
            good,= np.where((cat['chi2min'][index] < max_chi2) & (cat['HST_ACS_WFC_F475W'][index] > min_flux) & (cat['Rv_Exp'][index] != 0.)) # Keep only good fits Rv_Exp != 0 to remove bad pdfs (Nan values everywhere)
            ind = index[good]
            if len(ind) > 0:
                pdfs = np.zeros((len(ind),50))          # Contains BEAST 1d PDFs
                comp = np.zeros((len(ind),50))          # Contains Completness function for each star (but only different for each BEAST running file)
                sd = np.zeros(len(ind),dtype=np.int8)   # Source density bin name
                sub = np.zeros(len(ind),dtype=np.int8)  # Subfile name
                pos = np.zeros(len(ind),dtype=np.int16) # Star index in subfile
                for k in range(len(ind)):
                    sd[k], sub[k], pos[k] = findStar(ind[k], brick)   # Find the BEAST running files and the star position in those files
                sd2open, whichsd = np.unique(sd,return_inverse=True)  
                for m in range(len(sd2open)):
                    index_sd,= np.where(sd2open[whichsd] == sd2open[m])
                    sub2open, whichsub = np.unique(sub[whichsd == m], return_inverse=True)
                    for n in range(len(sub2open)):
                        pdf1d_filename = '%s_sd%s-%s_sub%s_pdf1d.fits' % (brick, np.str(sd[index_sd][n]), np.str(sd[index_sd][n]+1), np.str(sub2open[n]))
                        pdf1d = pyfits.open(dir_brick + pdf1d_filename)   # Open the 1d PDF file
                        index_pos,= np.where((sd == sd2open[m]) & (sub == sub2open[n])) 
                        pdfs[index_pos] = pdf1d[1].data[pos[index_pos]]   # Store the star PDF
                        comp[index_pos] = precomp_completeness[sub2open[n]+1].data[:] # Store the completeness (+1 because first line in completeness file is blank) 
                        xx = pdf1d[1].data[-1] # Store the histogram x-axis (same for all stars)
                        pdf1d.close()          # Close file
                  
            if len(ind) > 0:
                np.save('variables/%s/completeness_%s_%s.npy' % (brick, absc, ordi), comp)
                np.save('variables/%s/pdfs_%s_%s.npy' % (brick, absc, ordi), pdfs)

    #Send email
    me = 'arab@stsci.edu'
    msg = MIMEText('')
    msg['Subject'] = 'Maximum likelihood run %s col %s-%s completed' % (brick, y1, y2)#'Python notification: run over'
    msg['From'] = me
    msg['To'] = me

    s = smtplib.SMTP('localhost')
    s.sendmail(me, [me], msg.as_string())
    s.quit()
            
