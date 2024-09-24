#!/usr/bin/env python
# coding: utf-8

# In[2]:


# IMPORT BLOCK

import numpy as np
import scipy.integrate as integ
import scipy.special as spec
import scipy.optimize as optim
from scipy.ndimage import map_coordinates
import scipy.optimize as optim
import vegas
import matplotlib.pyplot as plt

import disSat as dis
import disSat.dark_matter.models.wdm as wdm
from colossus.cosmology import cosmology
import colossus.halo as halo

class CartesianGridInterpolator:
    """
    This class works as scipy.interpolate.RegularGridInterpolator, but
    it's optimized for equally-spaced grids.

    Obtained from
    https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_regular_grid.html
    """
    def __init__(self, points, values, method='linear'):
        self.limits = np.array([[min(x), max(x)] for x in points])
        self.values = np.asarray(values, dtype=float)
        self.order = {'linear': 1, 'cubic': 3, 'quintic': 5}[method]

    def __call__(self, xi):
        """
        `xi` here is an array-like (an array or a list) of points.

        Each "point" is an ndim-dimensional array_like, representing
        the coordinates of a point in ndim-dimensional space.
        """
        # transpose the xi array into the ``map_coordinates`` convention
        # which takes coordinates of a point along columns of a 2D array.
        xi = np.asarray(xi).T

        # convert from data coordinates to pixel coordinates
        ns = self.values.shape
        coords = [(n-1)*(val - lo) / (hi - lo)
                  for val, n, (lo, hi) in zip(xi, ns, self.limits)]

        # interpolate
        return map_coordinates(self.values, coords,
                               order=self.order,
                               cval=np.nan)  # fill_value
    
# Subhalo data block

dwarfs_updated = {'.Boötes I': {'reff': 191.0,'refferr': [5.0, 5.0],'sigma': 5.1,'sigerr': [0.8, 0.7]},
                  '.Boötes II': {'reff': 38.7,'refferr': [5.1, 5.1],'sigma': 2.9,'sigerr': [1.2, 1.6]},
                  '.Canes Venatici I': {'reff': 452.0,'refferr': [13.0, 13.0],'sigma': 7.6,'sigerr': [0.4, 0.4]},
                  '.Canes Venatici II': {'reff': 70.7,'refferr': [11.2, 11.2],'sigma': 4.6,'sigerr': [1.0, 1.0]},
                  '.Carina': {'reff': 308.0,'refferr': [3.0, 3.0],'sigma': 6.6,'sigerr': [1.2, 1.2]},
                  '.Coma Berenices': {'reff': 72.1,'refferr': [3.8, 3.8],'sigma': 4.6,'sigerr': [0.8, 0.8]},
                  '.Draco': {'reff': 214.0,'refferr': [2.0, 2.0],'sigma': 9.1,'sigerr': [1.2, 1.2]},
                  '.Fornax': {'reff': 838.0,'refferr': [3.0, 3.0],'sigma': 11.7,'sigerr': [0.9, 0.9]},
                  '.Hercules': {'reff': 216.0,'refferr': [17.0, 17.0],'sigma': 5.1,'sigerr': [0.9, 0.9]},
                  '.Leo I': {'reff': 270.0,'refferr': [2, 2],'sigma': 9.2,'sigerr': [0.4, 0.4]},
                  '.Leo II': {'reff': 171.0,'refferr': [2.0, 2.0],'sigma': 7.4,'sigerr': [0.4, 0.4]},
                  '.Leo IV': {'reff': 114.0,'refferr': [13.0, 13.0],'sigma': 3.4,'sigerr': [0.9, 1.3]},
                  '.Leo V': {'reff': 49.0,'refferr': [16.0, 16.0],'sigma': 2.3,'sigerr': [1.6, 3.2]},
                  '.LMC': {'reff': 2697.0,'refferr': [115, 115],'sigma': 30.0,'sigerr': [2.5, 2.5]},
                  '.Sculptor': {'reff': 280.0,'refferr': [1.0, 1.0],'sigma': 9.2,'sigerr': [1.1, 1.1]},
                  '.Segue 1': {'reff': 24.2,'refferr': [2.8, 2.8],'sigma': 3.7,'sigerr': [1.1, 1.4]},
                  '.Sextans': {'reff': 413.0,'refferr': [3.0, 3.0],'sigma': 7.9,'sigerr': [1.3, 1.3]},
                  '.SMC': {'reff': 1106.0,'refferr': [77, 77],'sigma': 16.5,'sigerr': [1, 1]},
                  '.Ursa Major I': {'reff': 234.0,'refferr': [10.0, 10.0],'sigma': 7.0,'sigerr': [1.0, 1.0]},
                  '.Ursa Major II': {'reff': 128.0,'refferr': [5.0, 5.0],'sigma': 5.6,'sigerr': [1.4, 1.4]},
                  '.Ursa Minor': {'reff': 405.0,'refferr': [21.0, 21.0],'sigma': 9.5,'sigerr': [1.2, 1.2]},
                  '.Willman 1': {'reff': 27.7,'refferr': [2.4, 2.4],'sigma': 4.0,'sigerr': [0.8, 0.8]}}

sigmas = []
reffs = []
sigerr_upper = []
sigerr_lower = []
refferr_upper = []
refferr_lower = []
names = []

for dwarf, properties in dwarfs_updated.items():
    name = dwarf
    sigma = properties['sigma']
    reff = properties['reff']
    sigerr = properties['sigerr']
    refferr = properties['refferr']
    if sigma is not None and reff is not None:
        names.append(name)
        sigmas.append(sigma)
        reffs.append(reff)
        if isinstance(sigerr, list):
            sigerr_upper.append(sigerr[1])
            sigerr_lower.append(sigerr[0])
        else:
            sigerr_upper.append(sigerr)
            sigerr_lower.append(sigerr)
        if isinstance(refferr, list):
            refferr_upper.append(refferr[1])
            refferr_lower.append(refferr[0])
        else:
            refferr_upper.append(refferr)
            refferr_lower.append(refferr)

sigmas= np.array(sigmas)
reffs= np.array(reffs) / 1000
sigerr_upper = np.array(sigerr_upper)
sigerr_lower = np.array(sigerr_lower)
refferr_upper = np.array(refferr_upper) / 1000
refferr_lower = np.array(refferr_lower) / 1000
sigerr = ((sigerr_upper + sigerr_lower) / 2) + 1e-240
refferr = ((refferr_upper + refferr_lower) / 2) + 1e-240
    
# Establish constants
scatter_concentration = 0.16
scatter_rhalf2D = 0.234
scatter_SI = 0.18
log10_to_log = np.log(10)
cosmoP18 = cosmology.setCosmology('planck18')
log10_to_log = 1/np.log(10)
tin = float(cosmoP18.age(1))
h = cosmoP18.h
rhoc_zin = cosmoP18.rho_c(1)*h**2

def neg2lnL(trmass=9,alpha=1.31,Mhof=10**8.35,beta=0.96,sigma=0.15,gamma=0,mcorethres=10**9,sigma_CO=1,y_c=1,M_mw=10**12):
    """
    Input parameters:
    
    1. trmass (mass of WDM particles) (default: 9keV)
    2. alpha (the halo occupation fraction parameter) (default: 1.31)
    3. Mhof (the halao occupation fraction parameter) (default: 10**8.35)
    4. beta (the power law relation for the SMHM relation) (default: 0.96)
    5. sigma (a constant in the scatter of the SMHM relation) (default: 0.15)
    6. gamma (a constant in the scatter of the SMHM relation) (default: 0)
    7. mcorethres (the mass where the density profile of a galaxy switches from NFW to coreNFW) (default: 10**9)
    8. sigma_CO (included to account for the anisotropy of of the satellite distribution) (default: 1)
    9. y_c (interpolates between NFW and coreNFW profiles to account for tidal disruption) (default: 1)
    10. M_mw (mass of Milky Way galaxy + halo) (default: 10**12)
    
    Output:
    
    -2ln(L) which is the likelihood function to obtain subhalos with the given input parameters
    """
    # Compute some starter constants that rely on initial parameters
    WDM1 = wdm.WDM(trmass)
    M_half_mode = dis.dark_matter.models.wdm.helper.half_mode_mass(WDM1.mWDM)
    Rvir_mw = halo.mass_so.M_to_R(M_mw*cosmoP18.h, 0, '200c') / cosmoP18.h
    c_mw = 9
    r_s_mw = Rvir_mw/c_mw

    # MAKE INTEGRATION FUNCTIONS
    
    def P_M(logM):
        """ Returns the probability to have a halo mass M """
        M = np.exp(logM)
        tffactor = (1 + (4.2*M_half_mode/M)**2.5)**-0.2
        hofprob = .5 + .5*spec.erf(alpha*np.log10(M/Mhof))
        P_M_value = (M**(-1.84)) * hofprob * tffactor
        return P_M_value

    def P_Reff(logR_eff, logM_star):
        """ Returns the probability to have log(half-light radius) logR_eff given a log(stellar mass) logM_star """
        R_eff = np.exp(logR_eff)
        M_star = np.exp(logM_star)
        scatter=0.234*log10_to_log
        mean=np.log(0.0077624712 * (M_star**0.268))
        return (1/(R_eff*scatter*(np.sqrt(2*np.pi))))*np.exp((-(np.log(R_eff)-mean)**2)/(2*(scatter**2)))

    # Make function for P(c)

    # Load the data file only once and set up the 2D interpolator
    data = np.loadtxt('c200WDM_results.txt')
    M_values = np.logspace(7, 12, 100)
    trmass_values = np.linspace(1, 12, 100)
    interpolator = CartesianGridInterpolator((np.log10(M_values), trmass_values), data)
    # Function to retrieve the interpolated data point for given trmass and M values
    def get_c200WDM(trmass_target, M_target):
        """ Returns the concentration corresponding to a thermal mass trmass_target and halo mass M_target """
        # Use the 2D interpolator to find the c200WDM value for the given trmass and M
        c200WDM_value = interpolator([[np.log10(M_target), trmass_target]])[0]
        return c200WDM_value

    def P_c(logc, logM):
        """ Returns the probability to have log(concentration) logc given a halo mass M """
        M = np.exp(logM)
        c = np.exp(logc)
        scatter=0.16*log10_to_log
        mean = np.log(get_c200WDM(trmass, M))
        P_c_value = (1/(c*scatter*(np.sqrt(2*np.pi))))*np.exp((-(np.log(c)-mean)**2)/(2*(scatter**2)))
        return P_c_value

    # Make function for P(M*)
    r_c_GK_raw, integral_GK_raw = np.loadtxt("disSat/data/radial_distributions/gk17-menc.dat", unpack=True)

    def nfw(rs, rc, n0):
        """ Returns the integral of r**2 * n_NFW(r), from 0 to rc, where
        n_NFW(r) = n0/(r/rs (1+r/rs)**2)
        """
        return (n0 * rs**3) * (np.log((rc/rs) + 1) - (rc/(rc+rs)))

    def P_cc(logM_star):
        """ Returns the observation probability for log(stellar mass) logM_star """
        M_star = np.exp(logM_star)
        L_WDM = M_star / 2
        r_c_WDM = 1.5 * (L_WDM**0.51)
        if r_c_WDM < Rvir_mw:
            n_NFW = nfw(r_s_mw, r_c_WDM, 1)
            C_rbottom_WDM = (1-y_c)*np.interp(r_c_WDM, r_c_GK_raw, integral_GK_raw) + y_c*n_NFW

            C_rtop_WDM = nfw(r_s_mw, Rvir_mw, 1)

            C_r_WDM = C_rtop_WDM / C_rbottom_WDM
        else:
            C_r_WDM = 1

        Omega = 3.65
        C_omega = ((4*np.pi) / Omega)*sigma_CO

        C_WDM=C_r_WDM*C_omega
        Pobs_WDM=1/C_WDM
        threshold = 4e5
        if M_star > threshold:
               Pobs_WDM=1
        return Pobs_WDM

    def P_smhm(logM_star, logM):
        """ Stellar mass-halo mass relation """
        M = np.exp(logM)
        M_star = np.exp(logM_star)
        N = .046
        M0 = 1.5e12
        mean = np.log(M * N * ((M / M0)**beta))
        scatter = sigma + (gamma * np.log10(M/(10**11)))
        P_smhm_value = (1 / (M_star * scatter * (np.sqrt(2 * np.pi)))) * np.exp((-(np.log(M_star) - mean)**2) / (2 * (scatter**2)))
        return P_smhm_value

    def P_Mstar(logM_star, logM):
        """ Returns the probability to have log(stellar mass) logM_star given a halo mass M """
        return P_cc(logM_star)*P_smhm(logM_star, logM)

    # Make P_tot function
    def P_tot(logM, logM_star, logR_eff, logc):
        """ Returns the differential probability to have
              log(halo mass) logM
              log(stellar mass) logM_star
              log(half-light radius) logR_eff
              log(concentration) logc
        """
        return P_M(logM)*P_Reff(logR_eff, logM_star)*P_c(logc, logM)*P_Mstar(logM_star, logM)*np.exp(logM)*np.exp(logM_star)*np.exp(logc)*np.exp(logR_eff)

    # Create probsamp1 to return sampled arrays of M, m_star, R_eff, and c
    def P_M_SAMP(logM):
        """ Returns the probability to have a log(halo mass) logM to be used in probabilistic sampling"""
        M=np.exp(logM)
        tffactor=WDM1.transfer_function(mass=M, mWDM=WDM1.mWDM)
        hofprob = .5 + .5*spec.erf(alpha*np.log10(M/Mhof))
        P_M_value = (M**(-.84)) * hofprob * tffactor
        return P_M_value

    def smhm_mean_scatter(logM):
        """Return the smhm mean(stellar mass) and scatter given a log(halo mass) logM"""
        M = np.exp(logM)
        N = .046
        M0 = 1.5e12
        mean = np.log(M * N * ((M / M0)**beta))
        scatter = sigma + (gamma * np.log10(M/(10**11)))
        return mean, scatter

    def probsamp1(M):
        """
        Input: array of presampled mass values (example: M=np.geomspace(1e7,1e12,100000))

        Output: array of WDM halo masses, stellar masses, concentrations and 2D halflight radii 
        values sampled based on subhalo mass function, transfer function, halo occupation fraction,
        and observation probability.
        """
        # Compute M200_WDM
        Pm = [P_M_SAMP(np.log(M_)) for M_ in M]
        Pm /= np.sum(Pm)
        M200_WDM = np.random.choice(M, len(M), p=Pm)

        # Compute m_stellar
        mean_M_star, scatter_M_star = smhm_mean_scatter(np.log(M200_WDM))
        m_stellarWDM = np.random.lognormal(mean=mean_M_star, sigma=(scatter_M_star * log10_to_log), size=len(M200_WDM))

        # Compute concentration
        c200_WDM_median = [get_c200WDM(trmass, M_) for M_ in M]
        c200_WDM = np.random.lognormal(mean=np.log(c200_WDM_median), sigma=(scatter_concentration * log10_to_log), size=len(M200_WDM))

        # Compute P_obs
        Pobs_WDM=[P_cc(np.log(Mstar_)) for Mstar_ in m_stellarWDM]

        # Compute R_eff
        scatter_R_eff=0.234*log10_to_log
        mean_R_eff=np.log(0.0077624712 * m_stellarWDM**0.268)
        Reff_WDM = np.random.lognormal(mean=mean_R_eff, sigma=scatter_R_eff, size=len(M200_WDM))

        # Sample everything based on observation probability    
        z = np.random.rand(len(M200_WDM))
        Mfin = M200_WDM[z<Pobs_WDM]
        Mstarfin = m_stellarWDM[z<Pobs_WDM]
        Refffin = Reff_WDM[z<Pobs_WDM]
        cfin = c200_WDM[z<Pobs_WDM]

        # Return sampled mass array
        return Mfin, Mstarfin, Refffin, cfin

    # Create a function to give sigLOS given WDM halo mass, density profile, concentration, stellar mass, and halflight radius
    def find_normalization_constant(M_halo, r_halo, rs):
        """Find the normalization constant n0 given the mass of the halo and its radius."""
        term1 = np.log((r_halo / rs) + 1)
        term2 = r_halo / (r_halo + rs)
        n0 = M_halo / (rs**3 * (term1 - term2))
        return n0

    def menc_nfw(rs, rc, M_halo, r_halo):
        return nfw(rs, rc, find_normalization_constant(M_halo, r_halo, rs))

    def menc_corenfw(rs, rhalf, halomass, rvir):
        G = 4.5171031e-39 
        ETA,KAPPA = 3.,0.04
        #fCORENFW = lambda x: (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))  # x = r/rc
        fCORENFW = lambda x: np.tanh(x)
        GYR = 3600*24*365.25*1e9 # seconds in a Gyr
        tSF = tin
        tSF *= GYR
        tDYN = 2*np.pi*np.sqrt((rs)**3/G/(menc_nfw(rs, rhalf, halomass, rvir)))
        q = KAPPA * tSF / tDYN
        n = fCORENFW(q)
        Rc = ETA * rhalf * 0.75  # coreNFW core radius, in kpc
        suppression = fCORENFW(rhalf/Rc)**n
        return menc_nfw(rs, rhalf, halomass, rvir)*suppression

    def get_virial_radius_at_infall(M):
        return (M/(4/3*np.pi*200*rhoc_zin))**(1./3.)

    def sigLOS(halomass, concentration, r_eff, M_star):
        G = 4.3009e-3 / 1000  #kpc * (km/s)^2 * (Msun)^-1
        rvir = get_virial_radius_at_infall(halomass)
        rs = rvir / concentration
        rhalf = r_eff / 0.75
        if halomass < mcorethres:
            Menc = menc_nfw(rs, rhalf, halomass, rvir)
        else:
            Menc = menc_corenfw(rs, rhalf, halomass, rvir)
        sigLOSvalue = np.sqrt((G*(Menc+(M_star/2)))/(4*r_eff))
        return sigLOSvalue

    # Compute the mu integrand
    def sigma_gaus(logM, logM_star, logR_eff, logc, siglosobs, deltasigma):
        """The gaussian for sigLOS in equation B2"""
        M = np.exp(logM)
        M_star = np.exp(logM_star)
        R_eff = np.exp(logR_eff)
        c = np.exp(logc)
        if M < mcorethres:
            prof='NFW'
        else:
            prof='coreNFW'
        siglos = sigLOS(M, c, R_eff, M_star)
        return (1/(np.sqrt(2*np.pi)*deltasigma))*np.exp((-(siglosobs-siglos)**2)/(2*(deltasigma**2)))

    def reff_gaus(logR_eff, reffobs, deltareff):
        """The gaussian for R_eff in equation B2"""
        R_eff = np.exp(logR_eff)
        return (1/(np.sqrt(2*np.pi)*deltareff))*np.exp((-(reffobs-R_eff)**2)/(2*(deltareff**2)))

    def mu_integrand(logM, logM_star, logR_eff, logc, siglosobs, reffobs, deltasigma, deltareff):
        """Returns the integrand of mu to be integrated"""
        return P_tot(logM, logM_star, logR_eff, logc)*sigma_gaus(logM, logM_star, logR_eff, logc, siglosobs, deltasigma)*reff_gaus(logR_eff, reffobs, deltareff)
 
    # Define integration function to compute mu
    def integratemu(siglosobs, reffobs, deltasigma, deltareff):
        """ Returns mu given siglosobs, reffobs, deltasigma, and deltareff """
        if reffobs < .04:
            M, m_star, R_eff, c = probsamp1(np.geomspace(1e7,1e8,2000000))
        else:
            M, m_star, R_eff, c = probsamp1(np.geomspace(1e6,1e13,200000))
        x = np.stack((np.log(M), np.log(m_star), np.log(R_eff), np.log(c)), axis=-1)
        delta_range = 3
        reff_range = [reffobs - delta_range*deltareff, reffobs + delta_range*deltareff]
        map = vegas.AdaptiveMap([np.log(np.array([1e7,3e10])),np.log(np.array([1e2,5e9])),np.log(np.array(reff_range)),np.log(np.array([1, 12]))]) 
        probs1 = np.array([])
        for k in range(len(x)):
            prob1 = (mu_integrand(x[k][0],x[k][1],x[k][2],x[k][3],siglosobs,reffobs,deltasigma,deltareff))
            probs1 = np.append(probs1, prob1)
        def integrand1(x):
            return (mu_integrand(x[0],x[1],x[2],x[3],siglosobs,reffobs,deltasigma,deltareff))
        map.adapt_to_samples(x, probs1, nitn=6)
        integ = vegas.Integrator(map, alpha=0.2)
        r1 = integ(integrand1, neval=20000, nitn=4)
        return r1
    
    # Compute the normalization of mu values
    def munormintegrand(logM_star, logM):
        """Returns the normalization of mu given logM, logM_star, logR_eff, logc"""
        return 0.000854*0.8*M_mw*P_M(logM)*P_Mstar(logM_star, logM)*np.exp(logM)*np.exp(logM_star)
    
    a = np.log(1e7)
    b = np.log(1e12)
    def smhm(logM):
        M = np.exp(logM)
        N = 0.046
        M0 = 1.5e12
        mean = np.log(M * N * ((M / M0)**beta))
        return mean
    c = smhm(a)
    d = smhm(b)
    munorm, munormerr = integ.dblquad(munormintegrand, a, b, c, d)
    
    # Define negative binomial distribution to compute P(N_obs)
    def fGAMMA(x):
        """The gamma function of a negative binomial distribution"""
        return spec.gamma(x)
    def p_negbin(Nmean, scatter):
        """The p parameter of a negative binomial distribution"""
        return (1/(1+(scatter*Nmean)))
    def r_negbin(scatter):
        """The r parameter of a negative binomial distribution"""
        return 1/scatter
    def negativebinomial(N, Nmean, scatter):
        """Returns the negative binomial distribution given a mean and intrinsic scatter"""
        p = p_negbin(Nmean, scatter)
        r = r_negbin(scatter)
        return ((fGAMMA(N + r) / (fGAMMA(r)*fGAMMA(N+1)))*(p**r)*((1-p)**N))
    
    # Compute mu normalization, N_obs, and P(N_obs)
    N = len(sigmas)
    N_obs_error = munormerr
    N_obs = munorm
    Nmean = N_obs
    P_Nobs = negativebinomial(N, Nmean, scatter_SI)
    
    # Compute neg2ln(L)
    neg2lnL_Nobs = -2*np.log(P_Nobs)
    return neg2lnL_Nobs

