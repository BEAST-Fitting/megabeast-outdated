import os
import glob
import re
import pyfits
import emcee
import numpy as np
from scipy import stats
import scipy.optimize as op
from scipy.integrate import simps, trapz
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
#----------------------------- FITTING FUNCTIONS ------------------------------
def lognorm(x, max_pos, sigma=0.5, N=1.):
    sqrt_2pi = 1. / np.sqrt(2 * np.pi)
    mu = np.log(max_pos) + sigma**2
    log_x = np.log(x)
    domain_mask = x > 0
    normalization = sqrt_2pi / (x * sigma)
    return np.where(domain_mask, N * normalization * np.exp(-0.5 * ((log_x - mu) / sigma)**2), 0)


def mixture(xs, max_pos1, max_pos2, sigma1=0.5, sigma2=0.5, N1 = 1., N2=1.):
    """
    Computes model for posterior distribution in a pixel: Mixture of 2 lognormal functions
    INPUTS:
    -------
          xs: numpy.1darray
              Array of A(V) (x-axis)
          max_pos1, max_pos2: floats
              Position of the lognormal function's maximum respectively component 1 and 2
          sigma1, sigma2: floats
              Sigma of the lognormal function resp. component 1 and 2
          N1, N2: floats
              Multiplicative factor for each lognormal

     OUTPUT:
     -------
          Mixture model: (LOGNORM1 + LOGNORM2) / INTEGRAL(LOGNORM1 + LOGNORM2)

    HA 05/2015
    """
    pointwise = lognorm(xs, max_pos1, sigma=sigma1, N=N1) + lognorm(xs, max_pos2, sigma=sigma2, N=N2)
    normalization = np.trapz(pointwise, x=xs)
    return pointwise / normalization

def lnprior(phi,xs):
    """
    Defines the prior info in log for A(V) mapping using a 2-component lognormal pixel posterior distribution
    INPUTS:
    -------
           phi: np.1darray
               vector of fitting parameters
    OUTPUT:
    -------
                     {  = 0          if possible value
           ln(Prior) {
                     {  = -infinite  if rejected value
    HA 05/2015
    """
    max_pos1, max_pos2, sigma1, sigma2, N1, N2 = phi
    if 0.05 <= sigma1 < 2 and 0.05 <= sigma2 < 2 and 0 <= max_pos1 < 2 and 0 <= max_pos2 < 3 and max_pos1 < max_pos2 and 0<= N1 <= 1 and 0<= N2 <= 1:
        return 0.0
    return -np.inf


def lnlike(phi, x, compl, marg_pdfs):
    """
    Computes the log(likelihood) for a given set of parameters in a given pixel (no prior info)    
    INPUTS:
    -------
           phi: np.1darray
               vector of fitting parameters
           x: numpy.1darray
              Array of A(V) (x-axis)
           compl: numpy.1darray
              Completeness function in A(V)
           marg_pdfs: numpy.Ndarray
              1d PDFs for each star in a given pixel (from BEAST)
    OUTPUT:
    -------
           Ln(likelihood) = ln(integral(Model * completeness/integral(completeness) * beast1dPDF/integral(beast1dPDF)) summed for all star in the considered pixel
    HA 05/2015
    """
    max_pos1, max_pos2, sigma1, sigma2, N1, N2 = phi
    pix_pPDF = mixture(x, max_pos1, max_pos2, N1=N1, N2=N2, sigma1=sigma1, sigma2=sigma2)
    gu = np.empty(len(marg_pdfs))
    var = pix_pPDF[None, :] * marg_pdfs
    gu = np.log(np.trapz(var, x=x, axis=1))
    return np.sum(gu, dtype=np.float64)

def lnprob(phi, x, marg_pdfs, compl):
    """
    Computes ln(Likelihood) with prior information
    INPUTS:
    -------
           phi: np.1darray
               vector of fitting parameters
           x: numpy.1darray
              Array of A(V) (x-axis)
           compl: numpy.Ndarray
              Completeness function in A(V) for each star
           marg_pdfs: numpy.Ndarray
              1d PDFs for each star in a given pixel (from BEAST)
    OUTPUT:
    -------
                            { -inf  if rejected inputs
           lnlike(inputs) + {
                            { 0     if possible inputs
          
    HA 05/2015
    """
    lp = lnprior(phi, x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(phi, x, compl, marg_pdfs)
#---------------------------------------------------------------------

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
    n_x = np.fix(np.round(math.acos(0.5*(max_dec+min_dec)*math.pi/180.)*(max_ra-min_ra)/ra_delt))
    ra_delt *= -1.
    return n_x, n_y


if __name__ == '__main__':
    brick = raw_input('Enter brick bXX:')
    dir_cat = '/astro/dust_kg2/harab/toothpick_results/v1_1/'
    cat = pyfits.getdata(dir_cat + '%s_stats_v1_1.fits' % brick)
    print datetime.datetime.now()
    dir_variables = 'variables/%s/' % brick
    dir_maxlike = 'max_likelihood_v1.1/'
    # number of pixel in each dimension (5*5 arcsec^2 pixels)
    nx, ny = number_of_pixel_in_map(cat,pix_size=10.)
    pdf1d_ex = pyfits.open('/astro/dust_kg2/kgordon/BEAST_production/b21/b21_sd0-1_sub0_pdf1d.fits') # open a file to get the A(V) bins
    xx = pdf1d_ex[1].data[-1]

    method = raw_input("maxlike or mcmc?")

    if method == 'maxlike':
        # maximum likelihood computed from scipy.optimize.minimize (chi-square minimization)
        res = np.empty((nx,ny,6))
        flag = np.zeros((nx,ny), dtype = bool)
        for absc in range(nx):
            print "i= %s out of %s" % (np.str(absc), np.str(nx-1))
            for ordi in range(ny):
                if not os.path.isfile(dir_variables + 'pdfs_%s_%s.npy' % (np.str(absc), np.str(ordi))):
                    pass
                else:
                    pdfs = np.load(dir_variables + 'pdfs_%s_%s.npy' % (np.str(absc), np.str(ordi)))
                    comp = np.load(dir_variables + 'completeness_%s_%s.npy' % (absc, ordi))
                    pdfs /= np.trapz(pdfs, axis=1, x= xx)[:, None]
                    comp /= np.trapz(comp, axis=1)[:,None]
                    pdfs_wcomp = pdfs * comp
                    chi2 = lambda * args: -2* lnprob(*args)
                    result = op.minimize(chi2,[0.1,0.7,0.5,0.5,1.,0.3], args=(xx,pdfs_wcomp, comp), method = 'Nelder-Mead')  # Chi-square minimization to find initial values
                    res[absc,ordi,:] = result['x']   # Store the results
                    flag[absc, ordi] = result['success']
        np.save('max_likelihood_v1.1/%s_results_hk.npy' % (brick), res) # Save results (out the loop)
        np.save('max_likelihood_v1.1/%s_flag_hk.npy' % (brick), flag)
        print datetime.datetime.now()

    elif method == 'mcmc':
        # MCMC from maximum likelihood result -- Hard-code: number of dimensions, walkers, chain length
        result = np.load('max_likelihood_v1.1/%s_results_hk.npy' % brick)
        ndim, nwalkers = 6, 12
        chain_len = 10000
        dir_write = 'mcmc_v1.1/brick_results/%s/chains/' % brick
        ny_crop1 = raw_input('Enter start crop:')
        ny_crop2 = raw_input('Enter end crop:')
        for absc in range(nx):
            print "i= %s out of %s" % (np.str(absc), np.str(nx-1))
            for ordi in range(np.int(ny_crop1),np.int(ny_crop2)):
                if not os.path.isfile(dir_variables + 'pdfs_%s_%s.npy' % (np.str(absc), np.str(ordi))):
                    pass
                elif os.path.isfile(dir_write + 'chain_%s_%s.npy' % (np.str(absc),np.str(ordi))):
                    pass
                else:
                    pdfs = np.load(dir_variables + 'pdfs_%s_%s.npy' % (np.str(absc), np.str(ordi)))
                    comp = np.load(dir_variables + 'completeness_%s_%s.npy' % (absc, ordi))
                    pdfs /= np.trapz(pdfs, axis=1, x= xx)[:, None]
                    comp /= np.trapz(comp, axis=1)[:,None]
                    pdfs_wcomp = pdfs * comp
                    init = [result[absc,ordi] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] # Use result['x'] if not re-loading results
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xx,pdfs_wcomp, comp))
                    #print("Running MCMC...")
                    sampler.run_mcmc(init, chain_len, rstate0=np.random.get_state())
                    #print("Done pixel %s out of %s" % (np.str(ordi - np.int(ny_crop1)),np.str(np.int(ny_crop2) - np.int(ny_crop1))))
                    np.save(dir_write + "chain_%s_%s.npy" % (np.str(absc),np.str(ordi)), sampler.chain)

    #Send email
    me = 'arab@stsci.edu'
    msg = MIMEText('')
    if method == 'mcmc':
        msg['Subject'] = '%s brick %s col %s-%s completed' % (method, brick, ny_crop1, ny_crop2)#'Python notification: run over'
    else:
        msg['Subject'] = '%s brick %s completed' % (method, brick)
    msg['From'] = me
    msg['To'] = me

    s = smtplib.SMTP('localhost')
    s.sendmail(me, [me], msg.as_string())
    s.quit()

                        
        
            
