baseDir = '/home/jonathan/Dropbox/jonathanmain/CGM/rapidCoolingCGM/'
import sys

base_pyDir = baseDir+'../pysrc/'
figDir = baseDir+'figures/'
presentation_figDir = baseDir+'doc/ChicagoJointMeetingJun18_figs/'
dataDir = baseDir+'data/'
pyobjDir = baseDir+'pyobj/'

import pdb, glob, string
import h5py
import scipy, numpy as np
from numpy import log as ln, log10 as log, e, pi
from scipy import integrate, interpolate
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
import pylab as pl
import my_utils as u
from scipy import optimize

mu = 0.62
X = 0.7
gamma = 5/3.
tHubble=13.6*un.Gyr
ne2nH = 1.2

Ez_flatUniverse = lambda z,Omega_M: (Omega_M*(1+z)**3 + 1-Omega_M)**0.5
Tremonti04 = lambda logMstar: -1.492 + 1.847*logMstar - 0.08026*logMstar**2 
AndrewsMartini13 = lambda logMstar: 8.798 - log(1+(10.**8.901/10**logMstar)**0.64)
mass_metallicity_z0_AndrewsMartini13 = lambda logMstar: 10.**(AndrewsMartini13(logMstar) - 8.69)
mass_metallicity_z0 = lambda logMstar: 10.**(Tremonti04(logMstar) - Tremonti04(10.8))
mass_metallicity_z2 = lambda logMstar: mass_metallicity_z0(logMstar) / 2. #Erb+06, Sanders+18
mass_metallicity = lambda logMstar,z: mass_metallicity_z0(logMstar) * 10**(-0.3 * log(1+z) / log(3.))
mass_metallicity_z007_Maiolino08 = lambda logMstar: -0.0864 *(logMstar - 11.18)**2 + 9.04 
mass_metallicity_z07_Maiolino08 = lambda logMstar:  -0.0864 *(logMstar - 11.57)**2 + 9.04
mass_metallicity_z22_Maiolino08 = lambda logMstar:  -0.0864 *(logMstar - 12.38)**2 + 8.99
mass_metallicity_z35_Maiolino08 = lambda logMstar:  -0.0864 *(logMstar - 11.35)**2 + 8.79


BehrooziFile = '/home/jonathan/research/separate/umachine-dr1/data/smhm/params/smhm_true_med_cen_params.txt'
BehrooziDataDir = '/home/jonathan/research/separate/umachine-dr1/data/'

def Behroozi_params(z, parameter_file=BehrooziFile):
    param_file = open(parameter_file, "r")
    param_list = []
    allparams = []
    for line in param_file:
        param_list.append(float((line.split(" "))[1]))
        allparams.append(line.split(" "))
    
    if (len(param_list) != 20):
        print(("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list)))
        quit()
    
    names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
    params = dict(list(zip(names, param_list)))
    
    
    #Print SMHM relation
    a = 1.0/(1.0+z)
    a1 = a - 1.0
    lna = ln(a)
    zparams = {}
    zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + z*params['M_1_Z']
    zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + z*params['EFF_0_Z']
    zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + z*params['ALPHA_Z']
    zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + z*params['BETA_Z']
    zparams['delta'] = params['DELTA']
    zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + z*params['GAMMA_Z'])
    
    smhm_max = 14.5-0.35*z
    if (params['CHI2']>200):
        print('#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')
    ms = 0.05 * np.arange(int(10.5*20),int(smhm_max*20+1),1)
    dms = ms - zparams['m_1'] 
    dm2s = dms/zparams['delta']
    sms = zparams['sm_0'] - log(10**(-zparams['alpha']*dms) + 10**(-zparams['beta']*dms)) + zparams['gamma']*np.e**(-0.5*(dm2s*dm2s))
    return ms,sms

def MgalaxyBehroozi(lMhalo, z, parameter_file=BehrooziFile):
    ms,sms = Behroozi_params(z,parameter_file)
    lMstar = scipy.interpolate.interp1d(ms, sms, fill_value='extrapolate')(lMhalo)
    return 10.**lMstar*un.Msun
def MgalaxyBehroozi_extrapolate(lMhalo, z, polydeg=2,parameter_file=BehrooziFile):
    log_z_plus_1s = np.arange(0.7,1,.025)
    _lMgalaxies = [log(MgalaxyBehroozi(lMhalo,10**log_z_plus_1-1.,parameter_file=parameter_file).value) for log_z_plus_1 in log_z_plus_1s]
    poly = np.poly1d(np.polyfit(log_z_plus_1s, _lMgalaxies, polydeg))
    return 10.**poly(log(z+1))*un.Msun

def MhaloBehroozi(lMstar, z, parameter_file=BehrooziFile):
    ms,sms = Behroozi_params(z,parameter_file)
    lMhalo = scipy.interpolate.interp1d(sms, ms, fill_value='extrapolate')(lMstar)
    return 10.**lMhalo*un.Msun
    
    
        
    
def halomasserrfunc(logmstars,logmhalo,z):
    ### Error function to solve for Mhalo/Mstars
    a=1./(z+1.)
    M10=11.590
    M11=1.195
    N10=0.0351
    N11=-0.0247
    beta10=1.376
    beta11=-0.826
    gamma10=0.608
    gamma11=0.329
    logM1= M10 + M11*(1.-a)
    N= N10 + N11*(1.-a)
    beta= beta10+beta11*(1.-a)
    gamma= gamma10+gamma11*(1.-a)
    mhalo=10**logmhalo
    mstar=10**logmstars
    M1=10**logM1
    return 2.*N/((mhalo/M1)**(-beta)+(mhalo/M1)**gamma) - mstar/mhalo
    

def gas_frac(Mstar):
    return 1.4*(Mstar/(1e10*un.Msun))**-0.4

def MgalaxyMoster(lMhalo, z):
    Mstar = 10.**optimize.brentq(halomasserrfunc,6.0,15.0,args=(lMhalo,z)) * un.Msun
    Mgal = Mstar * (1+gas_frac(Mstar))
    return Mgal
def emissivity(logT, element,ionization,wlRange=[6.2,24.8]):    
    logTs_base = np.arange(0.05,9.,.1)
    logT_less = logTs_base[logTs_base<=logT][-1]
    logT_more = logTs_base[logTs_base>logT][0]  
    vals = np.zeros(2)
    for i,_logT in enumerate((logT_less, logT_more)):
        ite = pyatomdb.spectrum.get_index( 10**_logT, teunits='K', logscale=False,filename='/home/jonathan/research/separate/atomdb/apec_line.fits')    
        llist = pyatomdb.spectrum.list_lines(wlRange, index=ite,linefile='/home/jonathan/research/separate/atomdb/apec_line.fits')
        vals[i] = sum([x[2] for x in llist if x[4]==element and x[5]==ionization])
    return np.interp(logT,np.array([logT_less, logT_more]),vals)
def emissivity_old(logT, element,ionization,wlRange=[6.2,24.8]):    
    ite = pyatomdb.spectrum.get_index( 10**logT, teunits='K', logscale=False,filename='/home/jonathan/research/separate/atomdb/apec_line.fits')    
    llist = pyatomdb.spectrum.list_lines(wlRange, index=ite,linefile='/home/jonathan/research/separate/atomdb/apec_line.fits')
    return sum([x[2] for x in llist if x[4]==element and x[5]==ionization])

def BehrooziRange(z):
    if z<=0.1: return 2e10,5e15
    if z==1: return 5e10,1.5e14
    if z==2: return 8e10,8e13
    if z==3: return 8e10,3e13
    if z==4: return 8e10,1.5e13
    if z==5: return 5e10,5e12
    if z==6: return 5e10,2e12
    if z==7: return 5e10,1.5e12
    if z==8: return 3e10,5e11
    if z==9: return 3e10,3e11
    if z==10: return 2e10,1.5e11
    if z<10:
        return BehrooziRange(int(z+1))
def MgalaxyBehroozi13(lMhalo):
    lMhalos = Behroozi13[:,0]
    lMstars = Behroozi13[:,1] + Behroozi13[:,0]
    return 10.**np.interp(lMhalo,lMhalos,lMstars)*un.Msun

def Behroozi_file(s):
    s2 = s
    if s=='qf': s2+='_hm'
    file_names = sorted(glob.glob(BehrooziDataDir+'%ss/%s_a*.dat'%(s,s2)))
    zs = np.zeros(len(file_names))    
    for ifn,fn in enumerate(file_names):
        a = float(fn[fn.index('%s_a'%s2)+(len(s2)+2):-4])
        zs[ifn] = a**-1. - 1.
        data = np.genfromtxt(fn)
        if ifn==0:
            Ms = data[:,0]
            res = np.zeros((len(file_names),len(Ms)))
        if s!='qf': res[ifn,:] = data[:,1]
        else: res[ifn,:] = data[:,1] * (data[:,-1]>0) + -1*(data[:,-1]==0)
    return res,Ms,zs
Behroozi_quenchedFractions = lambda : Behroozi_file('qf')

        
try: 
    
    SFH_Behroozi13_dat = np.genfromtxt('/home/jonathan/research/separate/sfh_z0_z8/release-sfh_z0_z8_052913/sfh/sfh_release.dat')
    zs_plus_1 = np.unique(SFH_Behroozi13_dat[:,0])
    lMhalos_Behroozi13 = np.unique(SFH_Behroozi13_dat[:,1])
    lSFR = SFH_Behroozi13_dat[:,2].reshape((lMhalos_Behroozi13.shape[0],zs_plus_1.shape[0]))
    SFH_Behroozi13 = lambda z,lMhalo: interpolate.RectBivariateSpline(lMhalos_Behroozi13,log(zs_plus_1),lSFR)(lMhalo,log(1.+z))
    
    B18 = np.genfromtxt('/home/jonathan/research/separate/umachine-dr1/data/sfhs/sfrs_output.txt')
    scale_factor_B18 = B18[:,0]
    lMhalo_B18 = B18[:,1]
    scale_factor_B18_dic = {0: 1.000412,0.1:0.911185,1:0.501122,2:0.334060,2.3: 0.298623, 3.5:0.217623,4:0.202435,
                            5:0.161935,6:0.141685,7:0.123123,8:0.112998,10:0.089373}
except:
    print('Behroozi+18 not loaded')
    

Mout_to_SFR_Chisholm = lambda Mstar: 0.76*(Mstar/(1e10*un.Msun))**-0.43 #eq. 16 in Chisholm+17
SFR_to_Min_Chisholm = lambda Mstar: 1. / (1.+Mout_to_SFR_Chisholm(Mstar))


def RodriguesPuebla16_Mvir_vs_z(Mvir0,z):
    a = (1.+z)**-1.
    a0 = lambda Mvir0: 0.592 - log((10**15.7*cosmo.h**-1*un.Msun/Mvir0)**0.113 + 1)
    g = lambda Mvir0,a: 1+np.e**(-3.676*(a-a0(Mvir0)))
    M13 = lambda z: 10**13.6*cosmo.h**-1*un.Msun*(1.+z)**2.755*(1+z/2.)**-6.351*np.e**(-0.413*z)
    f = log(Mvir0/M13(0.)) * g(Mvir0,1.) / g(Mvir0,a)
    return M13(z) * 10.**f
def RodriguesPuebla16_dMvir_dt(Mvir,z):
    alpha0, alpha1, alpha2, beta0, beta1, beta2 = 0.975, 0.300, -0.224, 2.677, -1.708, 0.661 #instantaneous
    # alpha0, alpha1, alpha2, beta0, beta1, beta2 = 1.,0.329,-0.206, 2.73, -1.828, 0.654 #dynamic-time avg        
    E = (1.-cosmo.Om0+cosmo.Om0*(1.+z)**3.)**0.5
    a = (1.+z)**-1.
    alpha = alpha0+alpha1*a+alpha2*a**2
    beta = 10.**(beta0+beta1*a+beta2*a**2)
    return beta*(Mvir/(1e12*cosmo.h**-1.*un.Msun))**alpha*E
def Mstar(z): #eq. 31 in Rodriguez-Puebla+16
    y = z/(1.+z)
    return 10.**(12.68 - 0.084*y**0.01 - 5.33*y**1.92 - 8.22*y**7.8)*un.Msun

def Moster18(M,z):
    ## integrated efficiencies
    #z, M1,epsilon_N,beta,gamma,Msigma,sigma0,alpha0 = np.array([
        #[0.1, 11.75, 0.12, 1.75, 0.57, 10.35, 0.20, 1.10],
        #[0.5, 11.80, 0.14, 1.70, 0.58, 10.25, 0.10, 0.50],
        #[1.0, 11.90, 0.15, 1.60, 0.60, 10.15, 0.08, 0.45],
        #[2.0, 11.95, 0.16, 1.55, 0.62, 10.05, 0.05, 0.35],
        #[4.0, 12.05, 0.18, 1.50, 0.64, 9.95, 0.03, 0.30],
        #[8.0, 12.10, 0.24, 1.30, 1.64, 9.85, 0.02, 0.10]]).transpose()
    #Table 6
    M0 = 11.339
    M_z = 0.692
    epsilon0 = 0.005
    epsilon_z = 0.689
    beta_0 = 3.344
    beta_z = -2.079
    gamma_0 = 0.966
    
    M1 = 10.**(M0 + M_z * z / (1+z))
    epsilon_N = epsilon0 + epsilon_z * z/(1+z)
    beta = beta_0 + beta_z * z/(1+z)
    gamma = gamma_0
    return 2*epsilon_N * ( (M/M1)**-beta + (M/M1)**gamma ) ** -1

def COG(wl0,f,EW,b):
    Ns = 10.**np.arange(8.,20.,0.01)*un.cm**-2
    tau0 = (1.497e-2*un.cm**2*un.s**-1*Ns*f*wl0/b).to('')
    us_base = np.arange(-4,4,0.01)*b
    us = us_base.repeat(len(Ns)).reshape((len(us_base),len(Ns)))
    wls = wl0 * (1+us/cons.c)
    dwls = np.pad((wls[2:,:]-wls[:-2,:]).value/2.,((1,1),(0,0)),mode='edge')*wl0.unit
    tau_nu = tau0 * np.e**-((us/b)**2)
    EWs = ((1.-np.e**-tau_nu)*dwls).sum(axis=0)
    return 10.**np.interp(log(EW.value),log(EWs.to(EW.unit).value),log(Ns.value))*un.cm**-2
def COGtest(): #compare to fig. 9.2 in Draine's book
    wl = 1215.7*un.angstrom
    W_no_unit = 10.**np.arange(-6,-2,.01)
    EW = W_no_unit*wl
    f = 0.4164
    for b in np.array([1.,2.,5.,10.,20])*un.km/un.s:
        Ns = COG(wl,f,EW,b)
        pl.plot((Ns*f*wl).to('cm**-1'),W_no_unit)
    pl.loglog()
