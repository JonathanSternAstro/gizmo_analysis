import h5py, numpy as np, glob, string, os, pdb, time
from workdirs import *

import os
from numpy import log10 as log
import scipy
import scipy.stats
from scipy import interpolate
from scipy.spatial.transform import Rotation
import scipy.ndimage.filters
import astropy, astropy.convolution
from astropy import constants as cons, units as un
from astropy.cosmology import Planck15 as cosmo


import my_utils as u
import projectPlotBasics as pB
import matplotlib, matplotlib.colors
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pylab as pl
import projectBasics as pB2

t_ratio_virialization = 2.5
Z_solar = 0.0129
gamma = 5/3.
mu = 0.62
ne2nH = 1.2
X = 0.7; He2H = 0.1; Y = 4.*He2H * X

gamma_FG09_zs = np.arange(7)
gamma_FG09 = 0.04,0.3,0.65,0.55,0.45,0.35,0.27
Gamma12 = lambda z: 10.**np.interp(z,gamma_FG09_zs,log(gamma_FG09))
def alpha_Ha(T4):
    if 0.1<T4<3:
        return 1.17e-13 * T4**(-0.942-0.031*ln(T4))
    else:
        return 0

def iSnapshot(z,fn=pyobjDir+'snapshot_zs_stampede.npz'):
    zs =np.load(fn)['zs']
    if 'stampede' in fn: tot = 600
    else: tot = 277
    return tot - u.searchsortedclosest(zs,z)
def z_from_iSnapshot(iSnapshot,stampede=True):
    if stampede: 
        fn=pyobjDir+'snapshot_zs_stampede.npz'
        ind = 600-iSnapshot
    else: 
        fn=pyobjDir+'snapshot_zs.npz'         
        ind = 277-iSnapshot
    zs =np.load(fn)['zs']    
    return zs[ind]

def LambdaFunc(z):
    fns = np.array(glob.glob(tables_dir+'z_?.???.hdf5'))
    zs = np.array([float(fn[-10:-5]) for fn in fns])
    fn = fns[zs.argsort()][u.searchsortedclosest(sorted(zs), z)]
    
    f=h5py.File(fn,'r')
    iHe = u.searchsortedclosest(f['Metal_free']['Helium_mass_fraction_bins'][:],Y)    
    H_He_Cooling  = f['Metal_free']['Net_Cooling'][iHe,...]
    Tbins         = f['Metal_free']['Temperature_bins'][...]
    nHbins        = f['Metal_free']['Hydrogen_density_bins'][...]
    Metal_Cooling = f['Total_Metals']['Net_cooling'][...] 
    
    f_H_He = interpolate.RegularGridInterpolator((log(Tbins), log(nHbins)),
                                                    H_He_Cooling, 
                                                    bounds_error=False, fill_value=None)
    f_Z = interpolate.RegularGridInterpolator((log(Tbins), log(nHbins)),
                                                    Metal_Cooling, 
                                                    bounds_error=False, fill_value=None)
    
    
    return lambda T,nH,Z2Zsun,f_H_He=f_H_He,f_Z=f_Z: (
        f_H_He((log(T), log(nH))) + f_Z((log(T), log(nH))) * Z2Zsun )


class Snapshot_meta:
    AHF_backup_dir = homedir+'other_repositories/ahf_wrapper/halo_files/'

    #simDir_Daniel = '/projects/b1026/anglesd/FIRE/%s_%s_sn1dy300ro100ss/' #%(self.galaxyname,self.res)
    #simDir_Robert = '/scratch/02750/robfeld/MassiveFIRE2/production_runs/B762_N1024_z6_TL%05d_baryon_toz6_HR/' #%int(self.galaxyname)
    #simDir_Xiangcheng = '/scratch/02688/xchma/HiZFIRE/runs/%s/' # %self.galaxyname
    #simDir_Xiangcheng_05Myr_snapshots = '/scratch/02688/xchma/bhacc/B172H003/%s/' # %self.galaxyname
    #simDir_FIRE = '/scratch/projects/xsede/GalaxiesOnFIRE/%s/%s_res%d/output/' #self.simgroup, self.galaxyname, self.res    

    #AHF_Dir = simDir+'AHF/'
    #halo_filename = glob.glob(AHF_Dir+"%d/*.AHF_halos"%self.iSnapshot)[0]                    
    #halo_filename = glob.glob(AHF_Dir+"snap%03dRPep.z*.AHF_halos"%self.iSnapshot)[0]
    #self.main_halo_AHF_filename = AHF_Dir+'halo_00000_smooth.dat'
    #my AHF dirs:
    #'/scratch/04613/tg839127/ahf/%s/output/' #simgroup
    #'/projects/b1026/jonathan/AHF/%s/'  #simname
    
    # Zach AHF dirs
    #'/scratch/03057/zhafen/%s/AHFHalos/' #simname
    
    # Robert AHF dirs
    # '/scratch/02750/robfeld/MassiveFIRE2/analysis/%s/halo/' #simname
    def __init__(self,galaxyname,simgroup,res,
                 snapshotDir,AHF_fn,redshift=None,iSnapshot=None,smooth_AHF= True,snapshot_in_directory=True,subhalo_AHF_filename_template=None):
        """
          galaxyname: 'h113, 'h206', 'h29', or 'h2'
        """
        #            
        self.galaxyname = galaxyname
        self.simgroup = simgroup
        self.res = res
        if smooth_AHF:            
            try:
                haloFinderAllSnapshots = np.genfromtxt(AHF_fn,skip_header=1,invalid_raise=False)
            except:
                halo_fn = AHF_fn.split('/')[-1]
                alt_AHF_fn = self.AHF_backup_dir + '%s/%s_res%d/%s'%(self.simgroup,self.galaxyname,self.res,halo_fn)
                haloFinderAllSnapshots = np.genfromtxt(alt_AHF_fn,skip_header=1,invalid_raise=False)
                subhalo_AHF_filename_template = None
            if iSnapshot!=None:
                haloFinderRes = haloFinderAllSnapshots[::-1][u.searchsortedclosest(haloFinderAllSnapshots[::-1,0],iSnapshot)]
                self.iSnapshot = iSnapshot        
            else:
                haloFinderRes = haloFinderAllSnapshots[u.searchsortedclosest(haloFinderAllSnapshots[:,1],redshift)]	                        
                self.iSnapshot = haloFinderRes[0]        
            self.z = haloFinderRes[1]
            center = haloFinderRes[7:10]
            centerV = haloFinderRes[10:13]
            mvir = haloFinderRes[5]
            rvir = haloFinderRes[13]                        
        else:
            self.iSnapshot = iSnapshot
            fns = glob.glob(subhalo_AHF_filename_template%self.iSnapshot)
            if len(fns)==1:
                haloFinderAllHalos = np.genfromtxt(fns[0],skip_header=1,invalid_raise=False)
                self.z = float(fns[0].split('/')[-1].split('z')[-1][:-10])
                ind = haloFinderAllHalos[:,3].argmax()
                center = haloFinderAllHalos[ind,5:8]
                centerV = haloFinderAllHalos[ind,8:11]
                rvir = haloFinderAllHalos[ind,11]
                mvir = haloFinderAllHalos[ind,3]
            else:
                print('Error: Number of halo files found: %d'%len(fns))
        if snapshot_in_directory is True:
            self.fn=snapshotDir+'snapdir_%03d/snapshot_%03d.0.hdf5'%(self.iSnapshot,self.iSnapshot)               
        elif snapshot_in_directory is False: 
            self.fn=snapshotDir+'snapshot_%03d.hdf5'%self.iSnapshot              
        elif snapshot_in_directory<self.iSnapshot:
            self.fn=snapshotDir+'snapdir_%03d/snapshot_%03d.0.hdf5'%(self.iSnapshot,self.iSnapshot)               
        else:
            self.fn=snapshotDir+'snapshot_%03d.hdf5'%self.iSnapshot              
 

        print(self.fn)
        self.f = h5py.File(self.fn)
        self.h = self.f['Header'].attrs['HubbleParam']
        self.a = (self.z+1)**-1.
        self.center = center*un.kpc / self.h * self.a 
        self.centerV = centerV*un.km/un.s 
        self.mvir = mvir / self.h * un.Msun
        self.rvir = rvir*un.kpc/self.h* self.a 
        self.sub_halo_centers = None
        self.sub_halo_rvirs = None
        self.sub_halo_gasMasses = None
        if False: #subhalo_AHF_filename_template!=None:            
            fns = glob.glob(subhalo_AHF_filename_template%self.iSnapshot)
            if len(fns)==1:
                self.add_subhalos(fns[0])            

    def add_subhalos(self,subhalo_AHF_filename,max_dsubhalo=4.,min_distance=0.001):
        """
        subhalo_AHF_filename = AHF_Dir+"snap%03dRpep..z%.3f.AHF_halos"%(self.iSnapshot,self.z)            
        """
        haloFinderAllHalos = np.genfromtxt(subhalo_AHF_filename,skip_header=1,invalid_raise=False)
        centers = haloFinderAllHalos[:,5:8]*un.kpc / self.h * self.a 
        distance = ((centers - self.center)**2).sum(axis=1)**0.5
        inds = (distance < max_dsubhalo * self.rvir) & (distance > min_distance*self.rvir)        
        self.sub_halo_centers = centers[inds,:]
        self.sub_halo_rvirs = haloFinderAllHalos[inds,11]*un.kpc / self.h * self.a 
        self.sub_halo_gasMasses = haloFinderAllHalos[inds,53]*un.Msun / self.h 
        
                    
class h5py_dic:
    def __init__(self,fs,non_subhalo_inds=None,pr=True):
        self.dic = {}
        self.fs = fs
        self.non_subhalo_inds = non_subhalo_inds
        self.pr = pr
    def __getitem__(self,k):
        if type(k)==type((None,)):
            particle,field = k
        else:
            particle = 'PartType0'
            field = k
        if (particle,field) not in self.dic:
            if particle in self.fs[0].keys():                 
                arr = self.fs[0][particle][field][...]                
                for f in self.fs[1:]:
                    new_arr = f[particle][field][...]                    
                    arr = np.append(arr, new_arr,axis=0)                
            else:
                arr = []
            self.dic[(particle,field)] = arr
            if self.pr: print('loaded %s, %s'%(particle, field))
        if self.non_subhalo_inds is None:
            return self.dic[(particle,field)]
        else:
            return self.dic[(particle,field)][self.non_subhalo_inds]

        
class dummySimulation:
    def __init__(self,mvir,rvir,name,z,simgroup,res):
        self.mvir = mvir
        self.rvir = rvir  
        self.galaxyname = name
        self.z = z
        self.h = None
        self.a = (self.z+1)**-1
        self.res = res
        self.simgroup = simgroup
        self.sub_halo_centers = None
        self.sub_halo_rvirs = None
        self.sub_halo_gasMasses = None          
class Snapshot:
    all_fields = 'Masses','Coordinates','Velocities','InternalEnergy','Metallicity','Density', #'Potential
    def __init__(self,sim=None,pr=True,loadAll=True,filter_subhalos=False,fn=None):
        self.sim = sim
        
        self.filter_subhalos = filter_subhalos
        if fn == None: 
            fs = [sim.f]
            self.fn = sim.fn
        else: 
            self.fn = fn
            fs = [h5py.File(self.fn)]  
            self.sim.h = fs[0]['Header'].attrs['HubbleParam']          

        for ifn in range(1,fs[0]['Header'].attrs['NumFilesPerSnapshot']):
            new_fn = self.fn[:-6] + '%d.hdf5'%ifn
            if pr: print(new_fn)
            fs.append(h5py.File(new_fn))
        if self.filter_subhalos:
            self.dic = h5py_dic(fs,non_subhalo_inds=None) #for loading all coordinates
            self.dic = h5py_dic(fs,non_subhalo_inds=self.no_subhalo_inds())
        else:
            self.dic = h5py_dic(fs)
        if pr: t_start = time.time()
        if loadAll:
            for particle in ('PartType0','PartType1','PartType2','PartType4'):            
                if particle=='PartType0': fields = self.all_fields
                else: fields = self.all_fields[:2]
                for field in fields:                
                    _ = self.dic[(particle,field)]
                    #if pr: print('%d: loaded %s, %s: %d/%d'%(time.time()-t_start, particle, field,
                                                             #arr.shape[0],sim.f['Header'].attrs['NumPart_Total'][int(particle[-1])]))
            if pr: print('finished loading arrays in %d seconds'%(time.time()-t_start))           
    def masses(self,iPartType=0): #in Msun
        _masses = self.dic[('PartType%d'%iPartType,'Masses')]
        if len(_masses): _masses = _masses * 1e10/self.sim.h
        return _masses


    def coords(self,iPartType=0): #in kpc
        return self.dic[('PartType%d'%iPartType,'Coordinates')]/self.sim.h*self.sim.a - self.sim.center.value
    def vs(self,iPartType=0): # in km/s
        return self.dic[('PartType%d'%iPartType,'Velocities')]*self.sim.a**0.5 - self.sim.centerV.value
    def vx(self,iPartType=0): # in km/s
        return self.vs(iPartType)[:,0]
    def vy(self,iPartType=0): # in km/s
        return self.vs(iPartType)[:,1]
    def vz(self,iPartType=0): # in km/s
        return self.vs(iPartType)[:,2]
    def E_kin(self):
        return 0.5*(self.vs()**2).sum(axis=1)
    def E_grav(self, rs_Phi, Phi):
        return np.interp(self.rs(), rs_Phi, Phi)
    def E_total(self,rs_Phi,Phi):
        return self.E_kin() + self.E_grav(rs_Phi, Phi)
    def j_c(self,rs_Phi,vc,Phi): # eq. 5 in El Badry 2018
        E_totals = self.E_total(rs_Phi, Phi)
        E_rc = 0.5*vc**2 + Phi
        j_c = vc * rs_Phi
        return np.interp(E_totals, E_rc,j_c)        # assumes E_rc monotonically increasing
    def circularity(self,jvec,rs_Phi,vc,Phi,project=True):
        if project:
            #j_z = self.rotated_vector(jvec=jvec,vector_str='js',edge_on=True)[:,1]
            j_z = (self.js() * jvec/np.linalg.norm(jvec)).sum(axis=1)
        else:
            j_z = (self.js()**2).sum(axis=1)**0.5
        return j_z / self.j_c(rs_Phi, vc, Phi) 
        
    def cs(self): # in km/s
        return self.dic[('PartType0','InternalEnergy')][:]**0.5 * (10/9.) # 10/9. should be within square root
    def SFRs(self): # in msun/yr
        return self.dic[('PartType0','StarFormationRate')][:]
    def StarFormationTimes(self): # scale factor
        return self.dic[('PartType4','StellarFormationTime')][:]
    def Ts(self): # in K
        epsilon = self.dic[('PartType0','InternalEnergy')][:] #energy per unit mass
        return (un.km**2/un.s**2 * cons.m_p / cons.k_B).to('K').value * (2./3* self.mus()) * epsilon 
    def mus(self):
        return (1 + 4*self.y_heliums()) / (1+self.y_heliums()+self.ne2nHs()) 
    def y_heliums(self): #number of helium atoms per hydrogen atom
        Y = self.dic[('PartType0','Metallicity')][:,1]
        return Y / (4*(1-Y))
    def ne2nHs(self):
        return self.dic[('PartType0','ElectronAbundance')][:]
    def log_Ts(self):
        return log(self.Ts())
    def t_cool(self):
        Lambda = LambdaFunc(self.sim.z)
        LAMBDA = Lambda(self.Ts(),self.nHs(),self.Z2Zsuns())
        return ((cons.k_B*un.K/(un.erg/un.s)).to('Myr').value * 
                3.5*self.Ts() / (self.nHs()*LAMBDA))
    def rad_per_unit_volume(self):
        Lambda = LambdaFunc(self.sim.z)
        nHs = self.nHs() * (self.nHs()<0.1)
        LAMBDA = Lambda(self.Ts(),self.nHs(),self.Z2Zsuns())
        return self.nHs()**2*LAMBDA
    def rad_per_unit_volume_max_nH(self): #bug fix after Paper III
        Lambda = LambdaFunc(self.sim.z)
        nHs = self.nHs() * (self.nHs()<0.1) + 0.1 * (self.nHs()>=0.1)            
        LAMBDA = Lambda(self.Ts(),nHs,self.Z2Zsuns())
        return self.nHs()**2*LAMBDA
    def rad_per_unit_mass(self):
        return self.rad_per_unit_volume() / self.rho()
    def rad_per_unit_mass_max_nH(self):
        return self.rad_per_unit_volume_max_nH() / self.rho()        
    def nHTs(self): #  in cm^-3 K
        return self.nHs() * self.Ts()
    def log_nHTs(self):
        return log(self.nHTs())
    def log_nHs(self):
        return log(self.nHs())
    def Ks(self): # in keV cm^2
        return (cons.k_B*un.K/(ne2nH*un.cm**-3)**(2/3.)).to('keV*cm**2').value * self.Ts()/self.nHs()**(2/3.)        
    def log_Ks(self):
        return log(self.Ks())
    def rs(self,iPartType=0):
        return ((self.coords(iPartType)**2).sum(axis=1))**0.5
    def cos_theta(self,z_vec):
        z_vec = z_vec / np.linalg.norm(z_vec)
        normed_coords = (self.coords().T / np.linalg.norm(self.coords(),axis=1)).T
        return np.dot(normed_coords,z_vec)  
    def sin_theta(self,z_vec):
        return (1-self.cos_theta(z_vec)**2)**0.5  
    def r2rvirs(self,iPartType=0):
        return self.rs(iPartType) / self.sim.rvir.value    
    def vrs(self):
        vs = self.vs() 
        coords = self.coords() 
        return (vs[:,0]*coords[:,0] + vs[:,1]*coords[:,1] + vs[:,2]*coords[:,2])/ self.rs()    
    def vtans(self):
        v_vec = self.vs() 
        vr = self.vrs()        
        r_vec = self.coords().T / self.rs()
        vtan_vec = v_vec - (vr * r_vec).T
        return (vtan_vec**2).sum(axis=1)**0.5
    def js(self): 
        vs = self.vs() 
        coords = self.coords() 
        return np.array([coords[:,1] * vs[:,2] - coords[:,2] * vs[:,1],
                         coords[:,2] * vs[:,0] - coords[:,0] * vs[:,2],
                         coords[:,0] * vs[:,1] - coords[:,1] * vs[:,0]]).T
    def v_phi(self,z_vec):
        js = self.js()
        z_vec /= np.linalg.norm(z_vec)
        j_z = (z_vec[0]*js[:,0] + z_vec[1]*js[:,1] + z_vec[2]*js[:,2])
        return j_z / (self.rs()*self.sin_theta(z_vec))
    def v_theta(self,z_vec):
        z_vec /= np.linalg.norm(z_vec)
        vs = self.vs()
        v_z = vs[:,0] * z_vec[0] + vs[:,1] * z_vec[1] + vs[:,2] * z_vec[2]        
        vtheta2 = (self.vs()**2).sum(axis=1) - self.v_phi(z_vec)**2 - self.vrs()**2
        vtheta2 *= vtheta2>0 #avoid nans
        return -np.sign(v_z) * vtheta2**0.5
    def Js(self): 
        vs = self.vs() 
        coords = self.coords() 
        return (np.array([coords[:,1] * vs[:,2] - coords[:,2] * vs[:,1],
                         coords[:,2] * vs[:,0] - coords[:,0] * vs[:,2],
                         coords[:,0] * vs[:,1] - coords[:,1] * vs[:,0]])*self.masses()).T
    def machs(self):
        return self.vrs()/self.cs()
    def isSubsonic(self):
        return np.abs(self.machs())<1
    def subsonicAndInflow(self):
        return ((self.machs()<0) * ( (self.machs()<-1)*-1.5 + (self.machs()>-1)*-0.5 ) + 
                (self.machs()>0) * ( (self.machs()>1)*1.5 + (self.machs()<1)*0.5 ))
    def Z2Zsuns(self):
        return self.dic[('PartType0','Metallicity')][:,0] / Z_solar
    def X(self):
        return 1 - self.dic[('PartType0','Metallicity')][:,0] - self.dic[('PartType0','Metallicity')][:,1]
    def rho(self): # in g/cm^3
        return ((un.Msun/un.kpc**3).to('g/cm**3') *
                self.dic[('PartType0','Density')] * 1e10/self.sim.h * self.sim.h**3 / self.sim.a**3)
    def fHIs(self):
        return self.dic[('PartType0','NeutralHydrogenAbundance')]
    def volume(self): # in kpc^3
        return un.Msun.to('g') * un.cm.to('kpc')**3 * self.masses() / self.rho()
    def nHs(self): # in cm^-3
        return self.X() *self.rho() / cons.m_p.to('g').value     
    def nHIs(self):        
        return self.nHs()*self.fHIs()
    def HImasses(self):
        return self.masses() * self.fHIs()        
    def Phi(self):
        return self.dic[('PartType0','Potential')]*(1.+self.sim.z)
    def no_subhalo_inds(self):
        if os.path.exists(self.filename()):
            return np.load(self.filename())['coords']
        coords = self.coords() 
        shalo_centers = self.sim.sub_halo_centers.value
        goods = np.ones(coords.shape[0],dtype=bool)
        for i in range(shalo_centers.shape[0]):
            distances_from_subhalo = ((coords - shalo_centers[i,:])**2).sum(axis=1)
            goods &= distances_from_subhalo > self.sim.sub_halo_rvirs[i].value**2
        np.savez(self.filename(),coords=goods)
        return goods
    def filename(self):
        base_fn = pyobjDir + 'nosubhalos_'+'_'.join((self.galaxyname,
                                                self.simgroup, 
                                                self.res, 
                                                'z%.4f'%self.z))
        return base_fn + '.npz'        
    def rotated_vector(self,jvec,vector_str,edge_on = True,rot_j=0,reverse=False):           
        """
        rotate such that jvec is directed either 
        along the y-axis (edge_on = true) or along the z-axis (edge_on = False)
        and then apply an additional rotation of rot_j around the y or z-axis. 
        """
        new_orientation = (np.array([0,0,1]), np.array([0,1,0]))[edge_on]
        rotvec = np.cross(jvec/(jvec**2).sum()**0.5,new_orientation)
        sin_theta = (rotvec**2).sum()**0.5
        rotvec_normed = -rotvec / sin_theta * np.arcsin(sin_theta)        
        if not reverse:
            rot1 = Rotation.from_rotvec(rotvec_normed)
            #rot1.apply(jvec) #(should be equal to new_orientation)
            rot2 = Rotation.from_rotvec(new_orientation*rot_j)
            rot = rot2 * rot1            
        else:
            rot1 = Rotation.from_rotvec(-rotvec_normed)            
            rot2 = Rotation.from_rotvec(-new_orientation*rot_j)
            rot = rot1 * rot2
        return rot.apply(getattr(self,vector_str)())
    def isShielded(self,_maxT=None):       #_maxT to avoid confusion with maxT in Snapshot_profiler.profile1D()
        nHss = 0.0123 * (self.Ts()/1e4)**0.173 * Gamma12(self.sim.z)**(2/3)
        cond = self.nHs() > nHss
        if _maxT!=None:
            cond &= self.Ts()<_maxT
        return cond
    def isCool(self,_maxT):
        return self.Ts()<_maxT
    
    def ionFractions(self, element='O',ionizationLevel=5,fn=os.getenv('HOME')+'/.trident/hm2012_hrB.h5'):
        F = h5py.File(fn,'r')
        logT        = np.array(F[element].attrs['Temperature'])
        lognH       = np.array(F[element].attrs['Parameter1'])
        z_hm2012    = np.array(F[element].attrs['Parameter2'])   
        ind_z         = u.searchsortedclosest(z_hm2012,self.sim.z)                
        log_f_tab   = np.array(F[element])[ionizationLevel,:,ind_z,:]
        func = interpolate.RectBivariateSpline(lognH,logT, log_f_tab)
        res = func.ev(log(self.nHs()),log(self.Ts()))
        F.close()
        return res
    def Halpha_emission(self): 
        n_electrons = self.nHs()*self.ne2nHs()
        n_protons   = self.nHs() * (1-self.fHIs())
        alphas      = alpha_Ha(self.Ts()/1e4)
        photon_energy = cons.h.to('erg s').value*3e18/6564.6 
        return n_electrons * n_protons * alphas * photon_energy
def midbins(bins):
    return (bins[1:]+bins[:-1])/2.
class Snapshot_profiler:
    useSaved = True
    log_r2rvir_bins = np.arange(-3,0.5,.01)
    mach_bins = np.concatenate([-10.**np.arange(-0.2,2.5,.1)[::-1],np.linspace(-10.**-0.3,10.**-0.3,11),10.**np.arange(-0.2,2.5,.1)])
    vr_bins = np.linspace(-1200,1200,601)
    cs_bins = 10.**np.arange(-1.,3,.1)
    T_bins = 10.**np.arange(2.,10.,.01)
    nHT_bins = 10.**np.arange(-1,10.,.01)
    K_bins = 10.**np.arange(-4,4.,.01)
    J_bins = 10.**np.arange(1.,6.,.01)
    t_cool_bins = 10.**np.arange(-2,7,.01)
    
    def __init__(self,snapshot,Rcirc2Rvir = 0.1):        
        self.Rcirc2Rvir = Rcirc2Rvir
        self.snapshot = snapshot
        self._saveDic = {}
        if not isinstance(snapshot,tuple):
            self.mvir = self.snapshot.sim.mvir.value
            self.rvir = self.snapshot.sim.rvir.value
            self.sub_halo_centers = self.snapshot.sim.sub_halo_centers 
            self.sub_halo_rvirs = self.snapshot.sim.sub_halo_rvirs
            self.sub_halo_gasMasses = self.snapshot.sim.sub_halo_gasMasses
            if self.sub_halo_centers!=None:
                self.sub_halo_centers -= self.snapshot.sim.center
                self.sub_halo_cetners = self.sub_halo_centers.value 
                self.sub_halo_rvirs = self.sub_halo_rvirs.value
                self.sub_halo_gasMasses = self.sub_halo_gasMasses.value
            self.galaxyname = self.snapshot.sim.galaxyname
            self.simgroup = self.snapshot.sim.simgroup#short()
            self.res = str(self.snapshot.sim.res)
            self.z =self.snapshot.sim.z
            self.filter_subhalos = snapshot.filter_subhalos
        else:
            if len(snapshot)==4:
                self.galaxyname, self.simgroup, self.res, self.z = snapshot
                self.filter_subhalos = False
            else:
                self.galaxyname, self.simgroup, self.res, self.z,self.filter_subhalos  = snapshot
            self.load()
        #assert(isinstance(self.snapshot,Snapshot))
    def filename(self):
        base_fn = pyobjDir + 'profiler_'+'_'.join((self.galaxyname,
                                                self.simgroup, 
                                                self.res, 
                                                'z%.4f'%self.z))
        if self.filter_subhalos:
            base_fn += '_nosubhalos'
        return base_fn + '.npz'
        
    def runAll(self):
        """
        creates profiles for saving
        """
        Mtotal = self.massProfile()
        self.profile1D_multiple(['machs','Z2Zsuns','Ts','log_Ts','vrs','nHTs','log_nHTs','Ks','nHs','isSubsonic','E_kin','rad_per_unit_mass'], 'MW')
        self.profile1D_multiple(['machs','Z2Zsuns','Ts','log_Ts','vrs','nHTs','log_nHTs','Ks','nHs','isSubsonic','E_kin','rad_per_unit_volume'], 'VW')
        self.profile1D_multiple(['machs','Z2Zsuns','Ts','log_Ts','vrs','nHTs','log_nHTs','Ks','nHs'], 'MW',2)
        self.profile1D_multiple(['machs','Z2Zsuns','Ts','log_Ts','vrs','nHTs','log_nHTs','Ks','nHs'], 'VW',2)
        self.SFRprofile()
        for weight in ('MW','HI','mol','SFR'):
            self.profile1D('v_phi', weight,power=1,z_vec=self.central_jvec(maxR2Rvir_disk=0.05,weight=weight))
            self.profile1D('v_phi', weight,power=2,z_vec=self.central_jvec(maxR2Rvir_disk=0.05,weight=weight))
            self.jvec_Profile(weight=weight)
        self.circularityProfile('MW',power=1)
        self.circularityProfile('MW',power=2)
        self.subsonic_fraction('MW')
        self.subsonic_fraction('VW')
        self.HImassProfile()
        self.MdotProfile(direction='out',suffix='_out')
        self.MdotProfile(direction='in',suffix='_in')
        
        #self.MdotProfile([1e5,1e10],suffix='_hot')
        #self.MdotProfile([0,1e5],suffix='_cool')
    def run_PaperIII(self,maxR2Rvir_disk=0.05,short=False,all_jvecs=True):
        """
        all calculations used in Paper III
        """
        Mtotal = self.massProfile()
        self.profile1D_multiple(['log_Ts','isSubsonic'], 'VW')
        self.profile1D_multiple(['Z2Zsuns','vrs'], 'MW')      
        self.median_T('VW')

        if short: return
        self.HImassProfile()
        self.SFRprofile()
        zbins, dt = self.sfh_zbins()
        self.stellar_ages(zbins)
        for weight in ('MW','HI','mol','SFR')[:(1,4)[all_jvecs]]:
            self.jvec_Profile(weight=weight)
            self.profile1D('v_phi', weight,power=1,z_vec=self.central_jvec(maxR2Rvir_disk=maxR2Rvir_disk,weight=weight),lazy=(weight!='HI'))
            self.profile1D('v_phi', weight,power=2,z_vec=self.central_jvec(maxR2Rvir_disk=maxR2Rvir_disk,weight=weight),lazy=(weight!='HI'))
        
        
    def run_tcool_comparison(self,minT2Tvir=0.5):
        Mtotal = self.massProfile()
        self.profile1D_multiple(['Ts','t_cool','rad_per_unit_mass'], 'MW')
        self.profile1D_multiple(['t_cool'], 'VW')
        self.profile1D_multiple(['Ts','t_cool','rad_per_unit_mass'], 'MW',minT=minT2Tvir*self.Tvir())
        self.profile1D_multiple(['t_cool'], 'VW',minT=minT2Tvir*self.Tvir())
    def run_HotAccretion(self):
         self.MdotProfile(T_range=[1e5,1e10],suffix='_hot_nodisc2')
         self.MdotProfile(T_range=[5e3,1e5], suffix='_cool_nodisc5')
         self.MdotProfile(T_range=[10,5e3],  suffix='_cool_nodisc6')
         self.SFRprofile()[::-1].cumsum()[::-1]
         self.profile1D('rad_per_unit_mass_max_nH','MW',minT=1e5)
         self.profile1D_multiple(['nHTs','vrs'],'VW')
         self.profile1D_multiple(['nHTs','vrs'],'MW')
         self.MdotProfile(isEdot=True,lazy=False)
    def run_extendedDLA(self):
        if not self.isSaved('cool_fraction_multiT_VW'):
            self.cool_fraction('VW')
            self.cool_fraction('MW')
        for maxT in (None,3e4):
            self.profile1D('nHs','VW',maxT=maxT)
            self.profile1D('nHs','MW',maxT=maxT)
        self.profile2D('vrs','MW',self.vr_bins)
        self.profile2D('vtans','MW',self.vr_bins)
        self.profile2D('vrs','HI',self.vr_bins)
        self.profile2D('vtans','HI',self.vr_bins)
        jvec = self.central_jvec(maxR2Rvir_disk=0.05,weight='HI')
        self.profile2D('v_phi', 'HI',self.vr_bins,z_vec=jvec)
        self.profile2D('v_theta', 'HI',self.vr_bins,z_vec=jvec)
    def tofile(self,overwrite=False,maxsize=None):
        self.load(loadAll=True, overwrite=overwrite)
        tosavedic = dict(self._saveDic)
        if maxsize!=None:
            ks = list(tosavedic.keys())
            for k in ks:
                if tosavedic[k].flatten().shape[0]>maxsize:
                    tosavedic.pop(k)
            
        np.savez(self.filename(),
                 rs_midbins=self.rs_midbins(),
                 rvir = self.rvir,
                 sub_halo_centers = self.sub_halo_centers,
                 sub_halo_rvirs = self.sub_halo_rvirs,
                 sub_halo_gasMasses = self.sub_halo_gasMasses,
                 **tosavedic)
    def load(self,loadAll=False,overwrite=False):
        #print('checking for file %s'%self.filename())
        if not os.path.exists(self.filename()): return
        try:
            files = np.load(self.filename(),allow_pickle=True)
        except AttributeError:
            return 
        #print('file includes: %s'%files.files)
        if isinstance(self.snapshot,tuple):
            self.mvir = files['mvir']
            self.rvir = files['rvir']
            if 'sub_halo_centers' in files.files:
                self.sub_halo_centers = files['sub_halo_centers']
                self.sub_halo_rvirs = files['sub_halo_rvirs']
                if 'sub_halo_gasMasses' in files.files: 
                    self.sub_halo_gasMasses = files['sub_halo_gasMasses']
                else:
                    self.sub_halo_gasMasses = None
            else:
                self.sub_halo_centers = None
                self.sub_halo_rvirs = None
                self.sub_halo_gasMasses = None
        if loadAll:
            for f in files.files:        
                if (f not in self._saveDic) or overwrite:
                    if f not in ('rs_midbins','mvir','rvir','sub_halo_centers','sub_halo_rvirs','sub_halo_gasMasses'):
                        try:
                            self._saveDic[f] = files[f]
                        except:
                            pass
        files.close()
    def isSaved(self,save_name):
        if not self.useSaved: return False
        if save_name in self._saveDic: return True
        if os.path.exists(self.filename()) and save_name in np.load(self.filename()).files: return True
        return False
    def save(self,save_name,val):
        print('saved %s'%save_name)
        self._saveDic[save_name] = val
    def get_saved(self,save_name):
        if save_name in self._saveDic.keys():
            return self._saveDic[save_name]
        if os.path.exists(self.filename()): 
            files = np.load(self.filename())
            if save_name in files.files:
                self._saveDic[save_name] = files[save_name]
                return self._saveDic[save_name]
        raise KeyError(save_name)
        
        

    def rs_midbins(self):
        return 10.**midbins(self.log_r2rvir_bins)*self.rvir
    def drs_midbins(self):
        return np.pad((self.rs_midbins()[2:]-self.rs_midbins()[:-2])/2., 1, 'edge')
    def massProfile(self,iPartTypes = (0,1,2,4),minT=None):
        total_mass = np.zeros(self.log_r2rvir_bins.shape[0]-1)
        for iPartType in iPartTypes:     
            if minT==None or iPartType!=0:
                save_name = 'massProfile%d'%iPartType
                inds = Ellipsis
            else:
                save_name = 'massProfile%d_minT'%iPartType
                inds = self.snapshot.Ts()>minT
            if not self.isSaved(save_name):
                if len(self.snapshot.dic[('PartType%d'%iPartType,'Masses')])==0: continue
                hist,_,_ = scipy.stats.binned_statistic(log(self.snapshot.r2rvirs(iPartType)[inds]),
                                                            self.snapshot.masses(iPartType)[inds],
                                                            statistic='sum',
                                                            bins=self.log_r2rvir_bins)
                self.save(save_name,hist)            
            total_mass+=self.get_saved(save_name)
        return total_mass
    def HImassProfile(self):
        save_name = 'HImassProfile'
        if not self.isSaved(save_name):
            hist,_,_ = scipy.stats.binned_statistic(log(self.snapshot.r2rvirs(0)),
                                                        self.snapshot.HImasses(),
                                                        statistic='sum',
                                                        bins=self.log_r2rvir_bins)
            self.save(save_name,hist)            
        return self.get_saved(save_name)
    
    def vc(self):
        return (((cons.G*un.Msun / un.kpc)**0.5).to('km/s').value * 
                (self.massProfile().cumsum() /self.rs_midbins())**0.5 )
    def Phi(self):
        return (-self.vc()**2 / self.rs_midbins() * self.drs_midbins())[::-1].cumsum()[::-1]
    def Tc(self):
        return (((gamma-1)*mu*cons.m_p*(un.km/un.s)**2/cons.k_B).to('K').value *
                self.vc()**2)
    def Tvir(self):
        return ((0.5*mu*cons.m_p*cons.G*un.Msun/un.kpc/cons.k_B).to('K').value *
                    (self.mvir/self.rvir))        
    def t_ff(self):
        return ((un.kpc / (un.km/un.s)).to('Myr') * 2**0.5 * self.rs_midbins() / self.vc())
    def nH(self,avoidGalaxy=False,useP=True,useTvir=False,limit_nH=True,maxR2Rvir=1.):
        if useP:
            if useTvir: T = self.Tvir()
            else: T = self.Tc()
            nH = X*mu * self.P2k_CGM(maxR2Rvir=maxR2Rvir) / T
        else:
            if avoidGalaxy: rho = self.rhoProfile_CGM()
            else: rho = self.rhoProfile()                
            nH =  X*rho / cons.m_p.to('g').value                            
        if limit_nH:
            ind = np.searchsorted(self.log_r2rvir_bins,log(0.2))
            limit = nH[ind]*(self.rs_midbins()/self.rs_midbins()[ind])**-3.
            nH[:ind] = np.array([nH[:ind],limit[:ind]]).min(axis=0)
        return nH
    def t_cool(self,avoidGalaxy=False,use_actual_temperature=False,useP=True,f_Tc=1,useTvir=False,limit_nH=True):        
        coeff = (mu*X*(gamma-1))**-1
        Lambda = LambdaFunc(self.z)
        nH = self.nH(avoidGalaxy,useP,useTvir,limit_nH)
        if useTvir:
            T = self.Tvir()
        elif not use_actual_temperature:
            T = f_Tc * self.Tc()
        else:
            T = 10.**self.profile1D('log_Ts','VW')
        LAMBDA = Lambda(T,nH,self.Z2Zsun())
        tcools = (cons.k_B*un.K/(un.erg/un.s)).to('Myr').value * coeff*T / (nH*LAMBDA)       
        return tcools
    def t_flow(self):
        return self.rs_midbins() / -self.profile1D('vrs','MW') * (un.kpc/(un.km/un.s)).to('Myr')    
    def Z2Zsun(self):
        return self.thickerShells(self.profile1D('Z2Zsuns','MW'),d=5,weighting='MW')
    def thickerShells(self,arr,d,weighting):
        #if True:  # remove this after bug fix
            #return astropy.convolution.convolve(arr, np.ones((d)), boundary='extend',normalize_kernel=False)
        if weighting=='MW':
            weights = self.rhoProfile()
        elif weighting=='VW':
            weights = 4*np.pi * self.rs_midbins()**2 * self.drs_midbins()
        else:
            weights = self.get_saved('j_vec_weight_%s'%weighting)             
            
        a = astropy.convolution.convolve(arr*weights, np.ones((d)), boundary='extend',normalize_kernel=False)
        b = astropy.convolution.convolve(weights, np.ones((d)), boundary='extend',normalize_kernel=False)
        return a/b
        
    def atRcirc(self,arr):
        return np.interp(log(self.Rcirc2Rvir*self.rvir), log(self.rs_midbins()), arr)        
    def atRvir(self,arr,fraction_of_Rvir=0.5):
        return np.interp(log(fraction_of_Rvir*self.rvir), log(self.rs_midbins()), arr)        
    def gasMassProfile(self,minT=None):
        return self.massProfile(iPartTypes=(0,),minT=minT)
    def jvec_Profile(self,weight='MW'):
        suffix = ('','_%s'%weight)[weight!='MW']
        if not self.isSaved('j_vec_x'+suffix):
            if weight=='MW': weightvals = self.snapshot.masses()
            if weight=='HI': weightvals = self.snapshot.HImasses()
            if weight=='mol': weightvals = self.snapshot.masses() * (self.snapshot.nHs() > 1000.)
            if weight=='SFR': weightvals = self.snapshot.SFRs()                 
            
            js = (self.snapshot.js().T * weightvals).T
            hist,_,_ = scipy.stats.binned_statistic(log(self.snapshot.r2rvirs()),
                                                            [weightvals]+[js[:,i] for i in range(3)],
                                                            statistic='sum',
                                                            bins=self.log_r2rvir_bins)        
            self.save('j_vec_x'+suffix, hist[1,:] / hist[0,:])
            self.save('j_vec_y'+suffix, hist[2,:] / hist[0,:])
            self.save('j_vec_z'+suffix, hist[3,:] / hist[0,:])
            self.save('j_vec_weight'+suffix, hist[0,:])
        return self.get_saved('j_vec_x'+suffix), self.get_saved('j_vec_y'+suffix), self.get_saved('j_vec_z'+suffix)
    def SFRprofile(self):
        if not self.isSaved('sfrs'):
            sfrs = self.snapshot.SFRs()
            hist,_,_ = scipy.stats.binned_statistic(log(self.snapshot.r2rvirs()),
                                                            sfrs,
                                                            statistic='sum',
                                                            bins=self.log_r2rvir_bins)        
            self.save('sfrs', hist)
        return self.get_saved('sfrs')
    def sfh_zbins(self,dt=0.01,maxz=10):
        zs = np.arange(0,maxz+0.5e-3,1e-3)
        ages = np.arange(cosmo.age(maxz).value,cosmo.age(0.).value,dt)
        zbins = np.interp(ages,cosmo.age(zs)[::-1].value,zs[::-1])
        return zbins[zbins<maxz][::-1], dt
    
    def stellar_ages(self,zbins,max_r2rvir=0.1):
        if not self.isSaved('stellar_ages'):            
            z_SF = self.snapshot.StarFormationTimes()**-1-1
            masses = self.snapshot.masses(4)
            inds = self.snapshot.rs(4) < max_r2rvir * self.rvir
            # TODO: fix for mass loss
            hist,x,_ = scipy.stats.binned_statistic(z_SF[inds],masses[inds],statistic='sum',bins=zbins)   
            self.save('stellar_ages', hist)            
        return self.get_saved('stellar_ages')
    
    def diskiness(self,max_r2rvir):
        j_vec = self.jvec_Profile()
        j_vec_sum = np.array([(j_vec_part*self.gasMassProfile())[self.rs_midbins() < max_r2rvir*self.rvir].sum() for j_vec_part in j_vec])
        j_sum = np.array
        
    def jProfile(self):
        j_vec_x,j_vec_y,j_vec_z = self.jvec_Profile()
        J_scalar = (j_vec_x**2+j_vec_y**2+j_vec_z**2)**0.5
        return J_scalar
    def jzProfile(self,maxR2Rvir_disk=0.05,weight='MW'):
        j_vec_x,j_vec_y,j_vec_z = self.jvec_Profile(weight)
        jDisc = self.central_jvec(maxR2Rvir_disk=maxR2Rvir_disk,weight=weight)
        jDisc /= np.linalg.norm(jDisc)
        j_z = (j_vec_x*jDisc[0] + j_vec_y*jDisc[1] + j_vec_z*jDisc[2])
        return j_z
    def jProfile2D(self,weight):
        k = 'j_2D_%s'%weight
        if not self.isSaved(k):
            js = (self.snapshot.Js()**2).sum(axis=1)**0.5 / self.snapshot.masses()
            self.profile2D('j', weight, self.J_bins,js)
        return self.get_saved(k), self.get_saved(k+'_bins_x'), self.get_saved(k+'_bins_y')
        
    def Rcirc(self,min_r2rvir=0.1,max_r2rvir=1,lazy=True):
        if not self.isSaved('Rcirc') or not lazy:
            
            if False:
                j_vec_x,j_vec_y,j_vec_z = self.jvec_Profile()                
                inds = (10.**midbins(self.log_r2rvir_bins) > min_r2rvir) & (10.**midbins(self.log_r2rvir_bins) < max_r2rvir)
                j = (j_vec_x[inds].sum()**2 + j_vec_y[inds].sum()**2 + j_vec_z[inds].sum()**2)**0.5
            else: # old calculation
                inds = (self.snapshot.r2rvirs() > min_r2rvir) & (self.snapshot.r2rvirs() < max_r2rvir)
                j = ((self.snapshot.Js()[inds,:].sum(axis=0)**2).sum(axis=0))**0.5 / self.snapshot.masses()[inds].sum()
            j_circ =self.vc() * self.rs_midbins()
            Rcirc = np.interp(j,j_circ, self.rs_midbins())
            self.save('Rcirc',(Rcirc,min_r2rvir, max_r2rvir))
        return self.get_saved('Rcirc')[0]
    def Rcirc2(self,factor,max_r2rvir=None,min_r2rvir=None,lazy=True,use_min_low=False):
        if not self.isSaved('Rcirc2') or not lazy:
            js = self.jProfile()
            j_circ =self.vc() * self.rs_midbins()
            if max_r2rvir!=None: max_ind = (self.log_r2rvir_bins < log(max_r2rvir)).nonzero()[0][-1]
            else: max_ind = None
            if min_r2rvir!=None: min_ind = (self.log_r2rvir_bins > log(min_r2rvir)).nonzero()[0][0]
            else: min_ind = 0
            if use_min_low:
                low_AM_inds = (js < factor*j_circ)[min_ind:max_ind].nonzero()[0] + min_ind
                if len(low_AM_inds): Rcirc = self.rs_midbins()[low_AM_inds[0]]
                else: Rcirc = np.nan
            else:
                high_AM_inds = (js > factor*j_circ)[min_ind:max_ind].nonzero()[0] + min_ind
                if len(high_AM_inds): Rcirc = self.rs_midbins()[high_AM_inds[-1]]
                else: Rcirc = np.nan
                
            self.save('Rcirc2',Rcirc)
        return self.get_saved('Rcirc2')
    
    def rhoProfile(self):
        return ((un.Msun/un.kpc**3).to('g/cm**3') * 
                self.gasMassProfile() / (4*np.pi*self.rs_midbins()**2*self.drs_midbins()))            
        
    def rhoProfile_CGM(self,f=0.5,min_mass_ratio=10**-2.5):        
        sub_inds = self.sub_halo_gasMasses.nonzero()
        distances = (self.sub_halo_centers[sub_inds]**2).sum(axis=1)**0.5 / self.rvir
        goods =  self.log_r2rvir_bins > log(2*self.Rcirc2Rvir)
        goods &= self.log_r2rvir_bins < log(1.)
        for i,sub_rvir in enumerate(self.sub_halo_rvirs[sub_inds]):
            if self.sub_halo_gasMasses[sub_inds][i]/self.mvir>min_mass_ratio:
                goods &= (10.**self.log_r2rvir_bins < distances[i] - sub_rvir*f/self.rvir) | (10.**self.log_r2rvir_bins > distances[i] + sub_rvir*f/self.rvir)
        new_goods = goods[1:] & goods[:-1]
        if len(goods.nonzero()[0])<20: 
            #print(self.galaxyname, self.z,'good CGM points: %d'%len(goods.nonzero()[0]))
            return np.nan*np.ones(self.rs_midbins().shape)
            
        rhos = 10.**pl.poly1d(pl.polyfit(log(self.rs_midbins()/self.rvir)[new_goods],
                                         log(self.rhoProfile()[new_goods]),1))(log(self.rs_midbins()/self.rvir))
        
        return rhos
    
    def delta(self):
        return self.rhoProfile() /(cosmo.Ob(self.z)*cosmo.critical_density(self.z).to('g*cm**-3').value)
    
    def profile1D_multiple(self,attrs,weight,power=1,minT=None,maxT=None,
                           costheta=None,lazy=True,*args,**kwargs): 
        if minT==None and maxT==None:
            if costheta!=None:
                k = lambda attr,weight=weight,power=power: '%s%s_%s_costheta_%.2f_%.2f'%(attr,('','2')[power==2],weight,costheta[0],costheta[1])
            else:
                k = lambda attr,weight=weight,power=power: '%s%s_%s'%(attr,('','2')[power==2],weight)
        elif minT!=None:
            k = lambda attr,weight=weight,power=power: '%s%s_%s_minT_%d'%(attr,('','2')[power==2],weight,minT)
        elif maxT!=None:
            k = lambda attr,weight=weight,power=power: '%s%s_%s_maxT_%d'%(attr,('','2')[power==2],weight,maxT)
        new_attrs = [attr for attr in attrs if (not lazy) or (not self.isSaved(k(attr)))]
        if len(new_attrs):
            if weight=='VW': weightvals = self.snapshot.volume()
            if weight=='MW': weightvals = self.snapshot.masses()
            if weight=='HI': weightvals = self.snapshot.HImasses()
            if weight=='mol': weightvals = self.snapshot.masses() * (self.snapshot.nHs() > 1000.)
            if weight=='SFR': weightvals = self.snapshot.SFRs()                 
            
            if minT!=None:
                weightvals *= self.snapshot.Ts()>minT
            if maxT!=None:
                weightvals *= self.snapshot.Ts()<maxT
            if costheta!=None: #limit to disk plane
                weightvals *= ((self.snapshot.cos_theta()>costheta[0]) & (self.snapshot.cos_theta()<costheta[1]))
            values = [weightvals] + [getattr(self.snapshot,attr)(*args,**kwargs)**power*weightvals for attr in new_attrs]        
            hist,_,_ = scipy.stats.binned_statistic(log(self.snapshot.r2rvirs()),
                                                    values, 
                                                    statistic='sum',
                                                    bins=self.log_r2rvir_bins)
            normed_hist = [hist[i,:]/hist[0,:] for i in range(1,hist.shape[0])]
            for iattr,attr in enumerate(new_attrs):
                self.save(k(attr),normed_hist[iattr])
        return [self.get_saved(k(attr)) for attr in attrs]
    def profile1D(self,attr,weight,power=1,minT=None,maxT=None,costheta=None,lazy=True,*args,**kwargs):
        return self.profile1D_multiple([attr], weight,power,minT,maxT,costheta,lazy,*args,**kwargs)[0]
            
    def profile2D(self,attr,weight,bins,arr=None,*args,**kwargs):
        k = lambda attr,weight=weight: attr+'_2D_%s'%weight
        if not self.isSaved(k(attr)):
            #weights = self.snapshot.f['PartType0']['Masses'] * (self.snapshot.cs()>10.)
            if weight=='VW': weightvals = self.snapshot.volume()
            if weight=='MW': weightvals = self.snapshot.masses()       
            if weight=='HI': weightvals = self.snapshot.HImasses()
            if arr is None: arr = getattr(self.snapshot,attr)(*args,**kwargs)
            m,x,y,_ = scipy.stats.binned_statistic_2d(log(self.snapshot.r2rvirs()),
                                                      arr,
                                                      weightvals,
                                                      statistic='sum',
                                                      bins=(self.log_r2rvir_bins,bins))
            self.save(k(attr), m)
            self.save(k(attr)+'_bins_x',x)
            self.save(k(attr)+'_bins_y',y) 
        return self.get_saved(k(attr)),self.get_saved(k(attr)+'_bins_x'),self.get_saved(k(attr)+'_bins_y')
    def smooth(self,arr,polydeg=5):
        goods = ~np.isnan(arr) * ~np.isinf(arr)
        return pl.poly1d(pl.polyfit(log(self.rs_midbins()[goods]),arr[goods],polydeg))(log(self.rs_midbins()))
    def central_jvec(self,maxR2Rvir_disk=0.05,weight='MW'):
        if weight=='MW': weights = self.gasMassProfile()
        else: weights =  self.get_saved('j_vec_weight_%s'%weight)             
        weights = weights * (self.rs_midbins()/self.rvir<maxR2Rvir_disk)
        jvec = np.nansum(np.array(self.jvec_Profile(weight))*weights,axis=1) / np.nansum(weights)                   
        return jvec
    def circularityProfile(self,weight,power=1,jvec=None,maxR2Rvir_disk=0.05):
        if jvec is None: jvec = self.central_jvec(maxR2Rvir_disk)
        return self.profile1D('circularity',weight,power,jvec,self.rs_midbins(),self.vc(),self.Phi())
        
        
    def slab(self,attr,r_max2Rvir,width2r_max,jvec=None, edge_on = True,rot_j=0,pixels=600,vals=None,lazy=True):        
        attr_str = attr+'_'
        if attr=='log_Ts': attr_str = ''
        k = 'slab_%s%.2f_%.2f_%.2f_%s'%(attr_str,r_max2Rvir,width2r_max,rot_j,('face_on','edge_on')[edge_on])
        if not self.isSaved(k) or not lazy:
            if jvec is None: jvec = self.central_jvec()
            coords = self.snapshot.rotated_vector(jvec=jvec,vector_str='coords',edge_on=edge_on,rot_j=rot_j)
            r_max = r_max2Rvir*self.snapshot.sim.rvir.value
            width = width2r_max * r_max
            inds = ((coords[:,2] < width) & (coords[:,2] > -width) &
                    (coords[:,1] < r_max) & (coords[:,1] > -r_max) &
                    (coords[:,0] < r_max) & (coords[:,0] > -r_max) )
            if vals is None: vals = getattr(self.snapshot,attr)()
            m,x,y,_ = scipy.stats.binned_statistic_2d(coords[inds,0],
                                                      coords[inds,1],
                                                      vals[inds],
                                                      statistic='mean',
                                                      bins=pixels)
            self.save(k,m)
            self.save(k+'_x',x)
            self.save(k+'_y',y)
        return self.get_saved(k), self.get_saved(k+'_x'), self.get_saved(k+'_y')
    def subsonic_fraction(self,weight):
        k = 'subsonic_fraction_'+weight
        if not self.isSaved(k):
            m,x,y = self.profile2D('machs',weight,self.mach_bins)
            y = (y[1:]+y[:-1])/2.
            x = (x[1:]+x[:-1])/2.                        
            subsonic_inds = abs(y)<1
            self.save(k, m[:,subsonic_inds].sum(axis=1) / m.sum(axis=1) )
        return self.get_saved(k)
    def cool_fraction(self,weight,maxTs=[None,1.5e4,3e4,5e4],lazy=True):
        k = 'cool_fraction_multiT_'+weight
        if not lazy or not self.isSaved(k):
            isCoolArr = [None]*(2*len(maxTs)-1)
            for iT,maxT in enumerate(maxTs):
                isCoolArr[iT] = self.profile1D('isShielded',weight,_maxT=maxT,lazy=lazy)
            for iT,maxT in enumerate(maxTs[1:]):
                isCoolArr[len(maxTs)+iT] = self.profile1D('isCool',weight,_maxT=maxT,lazy=lazy)            
            self.save(k, np.array(isCoolArr))
        return self.get_saved(k)
    
    def median_T(self,weight):
        k = 'median_T'+weight
        if not self.isSaved(k):
            m,x,y = self.profile2D('Ts',weight,self.T_bins)
            y = (y[1:]+y[:-1])/2.
            Tmed = np.array([np.interp(0.5, m[i,:].cumsum() / m[i].sum(), y) for i in range(m.shape[0])])
            self.save(k, Tmed)
        return self.get_saved(k)
    def MdotProfile(self,T_range=None,nH_range=None,direction='net',suffix='',isEdot=False,lazy=True):
        k = '%cdot'%('ME'[isEdot])+suffix
        if (not lazy) or (not self.isSaved(k)):
            goods = np.ones(self.snapshot.Ts().shape).astype(bool)
            if T_range!=None:
                goods &= (self.snapshot.Ts() > T_range[0]) & (self.snapshot.Ts() < T_range[1])                
            if nH_range!=None:
                goods &= (self.snapshot.nHs() > nH_range[0]) & (self.snapshot.nHs() < nH_range[1])
            if direction=='out':                
                goods = self.snapshot.vrs()>0
            if direction=='in':               
                goods = self.snapshot.vrs()<0
            all_values = self.snapshot.vrs() * self.snapshot.masses()
            if isEdot: 
                all_values *= (self.snapshot.dic[('PartType0','InternalEnergy')][:]
				+self.snapshot.E_kin() ) #energy per unit mass
            values = all_values[goods]
            Mvs_total,_,_ = scipy.stats.binned_statistic(log(self.snapshot.r2rvirs()[goods]),
                                                    values, 
                                                    statistic=np.nansum,
                                                    bins=self.log_r2rvir_bins)
            dRs = np.pad((self.rs_midbins()[2:] - self.rs_midbins()[:-2])/2.,1,mode='edge')
            res = Mvs_total/dRs * (un.Msun*un.km/un.s/un.kpc).to('Msun/yr')
            if isEdot:
                res = res * (un.km**2/un.s**2).to('erg/Msun')
            self.save(k, res)
        return self.get_saved(k)
    def LcoolProfile(self,T_range,suffix='',lazy=True):
        k = 'Lcool'+suffix
        if not self.isSaved(k) or not lazy:
            _values = self.snapshot.rad_per_unit_mass() * self.snapshot.masses()
            goods = (self.snapshot.Ts() > T_range[0]) & (self.snapshot.Ts() < T_range[1])
            values = _values[goods]
            L_total,_,_ = scipy.stats.binned_statistic(log(self.snapshot.r2rvirs()[goods]),
                                                    values, 
                                                    statistic='sum',
                                                    bins=self.log_r2rvir_bins)
            self.save(k, L_total * (un.Msun/un.g).to(''))
        return self.get_saved(k)
    def Mdot_coolProfile(self,T_range,suffix='',lazy=True):
        g = self.vc()**2 / self.rs_midbins()
        return (self.LcoolProfile(T_range=T_range,suffix=suffix,lazy=lazy) / g /self.drs_midbins()) * (un.erg/un.s*(un.km/un.s)**-2).to('Msun/yr')
    def P2k_CGM(self,maxR2Rvir=1.):
        ind = np.searchsorted(self.log_r2rvir_bins,log(maxR2Rvir))
        dPs = self.rhoProfile()*self.vc()**2/self.rs_midbins()*self.drs_midbins() 
        dPs[ind:] = np.zeros(dPs.shape[0]-ind)
        Ps = dPs[::-1].cumsum()[::-1]
        return Ps * (un.g*un.cm**-3*un.km**2/un.s**2/cons.k_B).to('cm**-3*K').value
        
    def sigma_turb(self,weight='MW',minT=None):
        tmp=np.array([self.profile1D('v%s'%d,weight,power=2,minT=minT) - self.profile1D('v%s'%d,weight,minT=minT)**2 for d in ('rs','_phi','_theta')])
        return (tmp.sum(axis=0)/3.)**0.5
    def sigma_log_rho(self,weight='MW'):
        return (self.profile1D('log_nHs',weight,power=2) - self.profile1D('log_nHs',weight)**2)**0.5        
class Simulation:
    def __init__(self,galaxyname,simgroup,resolution,minMass01Rvir=None,profiles=None,dummy=False,Rcirc2Rvir=0.1,pr=True):
        self.galaxyname = galaxyname
        self.simgroup = simgroup
        self.resolution = resolution
        self.pr = pr
        if not dummy: #just for zs0
            if profiles==None:        
                all_profiles = [Snapshot_profiler((self.galaxyname, self.simgroup, self.resolution, z),
                                                   Rcirc2Rvir=Rcirc2Rvir) for z in self.zs0()]
                if minMass01Rvir!=None:
                    self.profiles = [prof for prof in all_profiles 
                                 if prof.isSaved('log_Ts_VW') and prof.gasMassProfile().cumsum()[u.searchsorted(prof.rs_midbins(),0.1*prof.rvir)]>minMass01Rvir
                                 ] #!!!!!!!!!!
                else:
                    self.profiles = [prof for prof in all_profiles if prof.isSaved('log_Ts_VW')]
                
            else:
                self.profiles = profiles
        self.dic = {}
        filename_Mstars = glob.glob(Mstardir+'*_'+self.galaxyname+'_*')
        if len(filename_Mstars)==1:
            f = np.load(filename_Mstars[0])
            self.z_Mstars, self.Mstars = f['z'], f['Mstar']
            f.close()
        else: self.z_Mstars, self.Mstars = None, None
    def __str__(self):
        return '%s_%s_%s'%(self.galaxyname,self.simgroup,self.resolution)
    def __repr__(self):
        return self.__str__()
    def shortname(self):
        if 'HL03' in self.galaxyname: return 'z5m13b'
        if 'HL09' in self.galaxyname: return 'z5m13a'
        if self.galaxyname[0]=='h': return {'h206':'m13A1','h29':'m13A2','h113':'m13A4','h2':'m13A8'}[self.galaxyname]
        else: return self.galaxyname
    def zs0(self):
        fns = glob.glob(pyobjDir + 'profiler_'+'_'.join((self.galaxyname,self.simgroup, self.resolution, 'z*.npz')))
        return sorted([float(fn[::-1][4:fn[::-1][1:].index('z')+1][::-1]) for fn in fns])
    def zs(self):
        return np.array([prof.z for prof in self.profiles])
    def z_ind(self,z):
        return np.searchsorted(self.zs(), z)    
    def mvirs(self):
        return np.array([prof.mvir for prof in self.profiles])
    def rvirs(self):
        return np.array([prof.rvir for prof in self.profiles])
    def Rcircs(self,byshell=True):
        if byshell:
            return np.array([prof.Rcirc2(0.5,lazy=False,min_r2rvir=0.01,max_r2rvir=1) for prof in self.profiles])
        else:
            return np.array([prof.Rcirc(lazy=False) for prof in self.profiles])
    def z_cross(self,isRcirc=True,t_ratio_virialization=t_ratio_virialization,lazy=True,isLast=True,r2Rvir=None):
        k = 'z_cross_at_'+('Rvir','Rcirc')[isRcirc]
        if (k not in self.dic) or (not lazy):
            if r2Rvir==None: r2Rvir = (0.5,0.1)[isRcirc]
            ind_tratio = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
            t_ratio = np.array([(prof.t_cool()/prof.t_ff())[ind_tratio] for prof in self.profiles])            
            if isLast:
                ind_z_cross = (t_ratio<t_ratio_virialization).nonzero()[0][0]
            else:
                ind_z_cross = (t_ratio>t_ratio_virialization).nonzero()[0][-1]
            if ind_z_cross == 0:
                self.dic[k] = None
            else:                
                self.dic[k] = self.zs()[ind_z_cross-(0,1)[isLast]]
            #if isRcirc:
                #tcools = np.array([prof.atRcirc(prof.t_cool()) for prof in self.profiles])
                #tffs   = np.array([prof.atRcirc(prof.t_ff()) for prof in self.profiles])
            #else:
                #tcools = np.array([prof.atRvir(prof.t_cool()) for prof in self.profiles])
                #tffs   = np.array([prof.atRvir(prof.t_ff()) for prof in self.profiles])                
            #zs = self.zs()            
            #log_t_ratios_poly = self.smooth(log(tcools / tffs / t_ratio_virialization))
            #iz_crosses = ((np.sign(log_t_ratios_poly[1:]*log_t_ratios_poly[:-1])==-1)*(zs[1:]<maxz)).nonzero()[0]
            #if len(iz_crosses):
                #iz_cross = iz_crosses[ind_z_cross]
                #self.dic[k] = 10.**np.interp(0, log_t_ratios_poly[iz_cross:iz_cross+2], log(1+zs[iz_cross:iz_cross+2]))-1
            #else:
                #self.dic[k] = None
        return self.dic[k]
    def at_z_cross(self,ys,isRcirc=True):
        z_cross = self.z_cross(isRcirc=isRcirc)
        if z_cross==None: return None            
        return np.interp(log(1+z_cross),log(1+self.zs()),ys)
    def smooth(self,arr,polydeg=10):
        goods = ~np.isnan(arr) * ~np.isinf(arr) * (arr!=None)
        return pl.poly1d(pl.polyfit(log(1+self.zs()[goods]),arr[goods],polydeg))(log(1+self.zs()))
    def quantity_vs_z(self,attr,isRcirc,*args,**kwargs):
        k = attr + '_at_'+('Rvir','Rcirc')[isRcirc]
        if k not in self.dic:
            if isRcirc: 
                self.dic[k] = np.array([prof.atRcirc(getattr(prof,attr)(*args,**kwargs)) for prof in self.profiles])
            else:
                self.dic[k] = np.array([prof.atRvir(getattr(prof,attr)(*args,**kwargs)) for prof in self.profiles])
        return self.dic[k]        
    def tcool_Rvirs(self):
        return np.array([prof.atRvir(prof.t_cool()) for prof in self.profiles])
    def tff_Rvirs(self):
        return np.array([prof.atRvir(prof.t_ff()) for prof in self.profiles])
    def tcool_Rcircs(self):
        return np.array([prof.atRcirc(prof.t_cool()) for prof in self.profiles])
    def tff_Rcircs(self):
        return np.array([prof.atRcirc(prof.t_ff()) for prof in self.profiles])
    
        
    def sfhs(self,time_window=0.3,frac_of_tH=None,newversion=False):
        _zs = np.arange(0,20,1e-3)        
        ages = cosmo.age(_zs)
        self.sfh = {}
        if not newversion:            
            if self.simgroup=='Daniel':
                fns = glob.glob(basedir+'alex_sfr/%s/*'%self.galaxyname)
            elif self.simgroup=='Xiangcheng':
                fns = glob.glob(basedir+'alex_sfr/L172_ref9_HL00009_ref13/*')
            else:
                fns = glob.glob(basedir+'alex_sfr/%s_res%s%s/*'%(self.galaxyname,self.resolution,('','_md')[self.simgroup=='md']))
            if len(fns)==0: 
                self.sfh=None
                return
            fn = fns[0]            
            npz = np.load(fn)
            ts = (npz['time_edges'][1:] + npz['time_edges'][:-1])/2.            
            self.sfh['sfrs']  = npz['sfrs']
            self.sfh['zs']    = np.interp(-ts, -ages, _zs)
        else:
            zbins,dt = self.profiles[0].sfh_zbins()
            sfrs = np.zeros(zbins.shape[0]-1)
            profs = [prof for prof in self.profiles if prof.isSaved('stellar_ages')]
            for iprof,prof in enumerate(profs[:-1]):                                
                inds = (zbins>prof.z) & (zbins<profs[iprof+1].z)
                non_zero_bins = inds.nonzero()[0]
                if len(non_zero_bins) and (inds.nonzero()[0][-1]<len(inds)-1):
                    inds[inds.nonzero()[0][-1]+1] = True
                sfrs[inds[:-1]] = prof.stellar_ages(zbins)[inds[:-1]] 
            self.sfh['sfrs'] = sfrs / (dt*1e9)
            self.sfh['zs'] = (zbins[1:]+zbins[:-1])/2.
            self.sfh['sfrs'] = self.sfh['sfrs'][self.sfh['zs']>self.zs()[0]]
            self.sfh['zs']   =   self.sfh['zs'][self.sfh['zs']>self.zs()[0]]            
            ts = cosmo.age(self.sfh['zs']).value
            
            
        self.sfh['means'] = np.zeros(ts.shape[0])
        self.sfh['std']   = np.zeros(ts.shape[0])
        self.sfh['std_log']   = np.zeros(ts.shape[0])
        for it,t in enumerate(ts):
            if frac_of_tH!=None:
                window = t*frac_of_tH
            else: 
                window = time_window
            inds = (ts<t+window/2.) & (ts > t-window/2.)
            self.sfh['means'][it]   = self.sfh['sfrs'][inds].mean()
            self.sfh['std'][it]     = (self.sfh['sfrs'][inds]    /self.sfh['means'][it]).std()	
            self.sfh['std_log'][it] = (log(self.sfh['sfrs'][inds])).std()	

class Snapshot_plotter:
    def __init__(self,profiler,others=None):
        self.profiler=profiler
        if others!=None:
            self.others = others
    def snapshot_analysis_single(self,attr,weight,bins,yls=None,show1D=False,
                                 vmin=-2,vmax = 0.,symlog=True,islog=False,tup=None):
        if tup is None:
            m,x,y = self.profiler.profile2D(attr,weight,bins)
        else:
            m,x,y = tup
        x = midbins(x); y = midbins(y)
        pl.figure(figsize=(3.,3.))
        pl.subplots_adjust(hspace=0.3,wspace=0.25,bottom=0.2)
        ax = pl.subplot(111)
                
        
        logm = log(m.T / m.T.sum(axis=0));logm[np.isinf(logm)] = -100        
        mesh = pl.pcolormesh(x,y,logm,cmap=matplotlib.cm.gnuplot_r,zorder=-100,vmin=vmin,vmax=vmax)
        mesh.cmap.set_under('w')
        
        if show1D:
            pl.plot(midbins(self.profiler.log_r2rvir_bins), self.profiler.profile1D(attr,weight),c='k',lw=2)
        
        pl.xlim(-1.5,0.3)
        pl.xlabel(r'$\log\ r / R_{\rm vir}$')
        if symlog:
            ax.set_yscale('symlog',linthreshy=1.,linscaley=0.5)
            ax.yaxis.set_major_formatter(u.arilogformatter)        
        if islog:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(u.arilogformatter)        
        pl.ylabel(r'Mach number $\mathcal{M}_r$')
        #[pl.axhline(x,c='.3',lw=1,zorder=1000) for x in (0,)]
        
        if yls!=None:
            pl.ylim(*yls)
        #ymax = pl.ylim()[1]*2
        #j = -1.4
        ##pl.axvline(j+log(rvirs[ifn]),c='.3',ls='-',zorder=1e5,lw=0.5)
        #pl.text(j+log(self.snapshot.sim.rvir),ymax,r'$r_{\rm circ}$',ha='center')
        #pl.text(j+log(self.snapshot.sim.rvir),ymax/2.,r'|',ha='center',va='center')
        
        ##pl.axvline(log(rvirs[ifn]),c='.3',ls='-',zorder=1e5,lw=0.5)
        #pl.text(log(self.snapshot.sim.rvir),ymax,r'$r_{\rm vir}$',ha='center')
        #pl.text(log(self.snapshot.sim.rvir),ymax/2.,r'|',ha='center',va='center')
        #pl.text(log(220),35,r'${\rm m12i},\,z=%s$'%self.snapshot.sim.z,fontsize=7,ha='right')
                
        cbar = pl.colorbar(mesh,orientation='vertical',ax=ax,pad=0.025,ticks=[-1, -2, -3])
        cbar.set_label(r'$\log\ M/M_{\rm shell}(r)$')
        #cbar.ax.tick_params(labelsize=7) 
        #pl.savefig(FIRE.figDir+'FIRE_single_snapshot.png',bbox_inches='tight',dpi=600)    
    def snapshot_analysis_multiple(self,weight,vmin=-2,vmax=0.5,layout=None):
        pl.figure(figsize=(pB.fig_width_full,6))
        pl.subplots_adjust(hspace=0.3,wspace=0.6)
        axs = [pl.subplot(2,3,iPanel+1) for iPanel in range(6)]
        for iPanel in range(6):
            ax = axs[iPanel]; pl.sca(ax)
            if iPanel==0: 
                m,x,y = self.profiler.profile2D('Ts',weight,Snapshot_profiler.T_bins)
                d_log_ybin = 2
            if iPanel==1: 
                m,x,y = self.profiler.profile2D('nHTs',weight,Snapshot_profiler.nHT_bins)
                d_log_ybin = 2
            if iPanel==2: 
                m,x,y = self.profiler.profile2D('machs',weight,Snapshot_profiler.mach_bins)
                d_log_ybin = 1
            if iPanel==3:
                m,x,y = self.profiler.profile2D('t_cool',weight, Snapshot_profiler.t_cool_bins)
                d_log_ybin = 2
            if iPanel==4:
                m,x,y = self.profiler.jProfile2D(weight)
                d_log_ybin = 2
            if iPanel==5:
                m,x,y = None,midbins(Snapshot_profiler.log_r2rvir_bins),None
            else:
                x = midbins(x); y = midbins(y)
                logm = log(m.T / m.T.sum(axis=0))+d_log_ybin;logm[np.isinf(logm)] = -100        
                mesh = pl.pcolormesh(x,y,logm,cmap=matplotlib.cm.plasma_r,zorder=-100,vmin=vmin,vmax=vmax)
                mesh.cmap.set_under('w')
            if iPanel==0:
                pl.ylabel(r'$T\ [{\rm K}]$')            
                ax.set_yscale('log'); ax.yaxis.set_major_formatter(u.arilogformatter)
                ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
                if layout in (1,2): pl.ylim(3e3,3e6)
                if layout in (3,4): pl.ylim(3e3,2e7)
            if iPanel==1:
                pl.ylabel(r'$n_{\rm H}T\ [{\rm cm}^{-3} {\rm K}]$')
                ax.set_yscale('log'); ax.yaxis.set_major_formatter(u.arilogformatter)
                ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                if layout in (1,2): pl.ylim(0.05,1e4)
                if layout in (3,4): pl.ylim(10,1e6)
            if iPanel==2:
                pl.ylabel(r'$\mathcal{M}_{\rm r}$')
                [pl.axhline(x,c='.5',lw=0.5,zorder=1000) for x in (-1,1)]
                ax.set_yscale('symlog',linthreshy=0.5,linscaley=0.5)
                ax.yaxis.set_major_formatter(u.arilogformatter)
                ax.yaxis.set_major_locator(ticker.FixedLocator([-10,-1,0,1,10]))
                if layout in (1,2): pl.ylim(-30,30)
                if layout in (3,4): pl.ylim(-60,60)
            if iPanel==3:
                pl.ylabel(r'$t_{\rm cool}\ [{\rm Myr}]$')
                #pl.plot(x,self.profiler.t_cool(),c='k')
                pl.plot(x,self.profiler.t_ff(),ls='--',c='k')
                pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                ax.set_yscale('log'); ax.yaxis.set_major_formatter(u.arilogformatter)
                ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=10,numticks=10))
                if layout==1: pl.annotate(r'$t_{\rm ff}$',(log(0.8),1000),(log(1.),300),arrowprops=pB.slantlinepropsblack,ha='left',va='top')
                if layout==2: pl.text(log(0.04),200,r'$t_{\rm ff}$')
                
                if layout in (1,2): pl.ylim(0.01,1e6)
                if layout in (3,4): pl.ylim(0.01,3e4)
            if iPanel==4:                
                pl.plot(x,self.profiler.jProfile(),lw=2,c='k')
                pl.plot(x,self.profiler.rs_midbins()*self.profiler.vc(),ls='--',c='k')
                pl.ylabel(r'$\left|\vec{j}\right|\ [{\rm kpc}\ {\rm km}{\rm s}^{-1}]$')
                if layout in (1,2):
                    pl.annotate(r'',(log(1.),(None,1500,2200)[layout]),(log(0.8),1000),arrowprops=pB.slantlinepropsblack,ha='right',va='top')
                    pl.text(log(0.8),1300,r'$\left|\Sigma \vec{j}\right|}$',ha='right',va='top')
                    pl.annotate(r'',(log(1.),(None,4.5e4,3e4)[layout]),(log(0.8),6e4),arrowprops=pB.slantlinepropsblack,ha='right',va='bottom')
                    pl.text(log(0.8),5e4,r'$v_{\rm c} r$',ha='right',va='bottom')
                ax.set_yscale('log'); ax.yaxis.set_major_formatter(u.arilogformatter)
                if layout in (1,2): pl.ylim(500,1e5)
                if layout in (3,4): pl.ylim(100,1e5)
            if iPanel==5:
                Mdot = self.profiler.profile1D('vrs','MW')*self.profiler.gasMassProfile()/self.profiler.drs_midbins()
                Mdot *= (un.km/un.s/un.kpc).to('yr**-1')
                pl.plot(x,Mdot,c='k')
                pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                pl.ylabel(r'${\dot M_{\rm r}}\ [{\rm M}_{\odot}\ {\rm yr}^{-1}]$')
                if layout==1: ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                if layout==2: ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
                if layout==4: 
                    pl.ylim(-400,300)
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
                if layout==3: 
                    pl.ylim(-600,100)
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))

            if layout in (1,2): pl.xlim(-1.5,0.3)
            if layout in (3,4): pl.xlim(-2,0.3)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.))    
            ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(log(np.array([0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2,3]))))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,pos: u.arilogformat(10**x)))    
            if iPanel>=3: pl.xlabel(r'$r/R_{\rm vir}$')

            
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[0.,-1, -2, -3],shrink=0.5,aspect=40)
        cbar.ax.set_xticklabels([r'$1$', r'$0.1$', r'$0.01$', r'$0.001$'])  # horizontal colorbar
        
        #cbar = pl.colorbar(mesh,orientation='vertical',ax=ax,pad=0.025,)
        if weight=='MW': cbar.set_label(r'$\log\ M/M_{\rm shell}(r)\ [{\rm dex}^{-1}]$')
        if weight=='VW': cbar.set_label(r'$\log\ {\rm solid\ angle/4\pi} [{\rm dex}^{-1}]$')

        #j = -1.4
        ##pl.axvline(j+log(rvirs[ifn]),c='.3',ls='-',zorder=1e5,lw=0.5)
        #pl.text(j+log(self.snapshot.sim.rvir),ymax,r'$r_{\rm circ}$',ha='center')
        #pl.text(j+log(self.snapshot.sim.rvir),ymax/2.,r'|',ha='center',va='center')

        ##pl.axvline(log(rvirs[ifn]),c='.3',ls='-',zorder=1e5,lw=0.5)
        #pl.text(log(self.snapshot.sim.rvir),ymax,r'$r_{\rm vir}$',ha='center')
        #pl.text(log(self.snapshot.sim.rvir),ymax/2.,r'|',ha='center',va='center')
        #pl.text(log(220),35,r'${\rm m12i},\,z=%s$'%self.snapshot.sim.z,fontsize=7,ha='right')

        
        pl.savefig(figDir+'FIRE_single_snapshot%d_%s.png'%(layout,self.profiler.galaxyname),bbox_inches='tight',dpi=600)    
    def compare_snapshots(self,weight,vmin=-2,vmax=0.5,layout=None,nPanels=4,present=False):
        fig = pl.figure(figsize=(pB.fig_width_half*(1.,1.5)[present],(5,3)[nPanels==1]*(1,0.75)[present]))
        pl.subplots_adjust(hspace=0.1,wspace=0.5)
        profs = [self.profiler]+self.others; nSims = len(profs)
        axs = [pl.subplot(nPanels,nSims,iPanel+1) for iPanel in range(nPanels*nSims)]
        for isim,prof in enumerate(profs):
            for iPanel in range(nPanels):
                ax = axs[isim+nSims*iPanel]; pl.sca(ax)                
                if iPanel==0: 
                    m,x,y = prof.profile2D('Ts',weight,Snapshot_profiler.T_bins)                    
                    d_log_ybin = 2
                if iPanel==1: 
                    m,x,y = prof.profile2D('nHTs',weight,Snapshot_profiler.nHT_bins)
                    d_log_ybin = 2
                if iPanel==10:
                    m,x,y = prof.profile2D('machs',weight,Snapshot_profiler.mach_bins)
                    d_log_ybin = 1
                if iPanel==10:
                    m,x,y = prof.profile2D('t_cool',weight, Snapshot_profiler.t_cool_bins)
                if iPanel==10:
                    m,x,y = prof.jProfile2D(weight)
                x = midbins(x); y = midbins(y)
                if iPanel!=2:                    
                    m = np.apply_along_axis(lambda m: np.convolve(m, np.ones((10,))/10, mode='same'), axis=1, arr=m)
                
                    
                #m = np.convolve(m,,axis=1,mode='same')#[:,:10*(len(y)//10)].reshape((len(x),len(y)//10,10)).sum(axis=2)
                logm = log(m.T / m.T.sum(axis=0))+d_log_ybin;logm[np.isinf(logm)] = -100        
                mesh = pl.pcolormesh(x,y,logm,cmap=matplotlib.cm.plasma_r,zorder=-100,vmin=vmin,vmax=vmax)
                #logm = m.T / m.T.sum(axis=0)*10**d_log_ybin
                #mesh = pl.pcolormesh(x,y,logm,cmap=matplotlib.cm.plasma_r,zorder=-100,vmin=0.01,vmax=1)
                mesh.cmap.set_under('w')
                Rcirc_loc = log(prof.Rcirc2(0.5)/prof.rvir)
                #Rcirc_loc = log(prof.Rcirc()/prof.rvir)
                pl.axvline(Rcirc_loc,lw=0.5,c='k',ls='--')
                
                if iPanel==0:
                    ylabel = r'$T\ [{\rm K}]$'
                    if layout==1: pl.ylim(3e3,1e7)
                    if layout==2: pl.ylim(3e3,1e8)
                    pl.plot(x,prof.Tc(),c='k',ls='--')
                    ax.set_yscale('log')
                    ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                    if isim==0:
                        if layout==1:
                            pl.annotate(r'',(log(1.),1e6),(log(1.1),1.5e6),arrowprops=pB.slantlinepropsblack)
                            pl.annotate(r'$T_{\rm c}$',(log(1.),1e6),(log(1.),2e6),fontsize=pB.infigfontsize)
                            pl.annotate(r'',(Rcirc_loc,5e6),(Rcirc_loc+log(1.5),5e6),arrowprops=pB.slantlinepropsblack)
                            pl.annotate(r'$R_{\rm circ}$',(Rcirc_loc+log(1.1),5e6),(Rcirc_loc+log(1.5),4.25e6),fontsize=pB.infigfontsize)                        
                        if layout==2:
                            pl.annotate(r'',(log(0.7),4.5e6),(log(0.8),9e6),arrowprops=pB.slantlinepropsblack)
                            pl.annotate(r'$T_{\rm c}$',(log(0.7),4.5e6),(log(0.75),1e7),fontsize=pB.infigfontsize)
                            pl.annotate(r'',(Rcirc_loc,4e7),(Rcirc_loc+log(1.5),3.4e7),arrowprops=pB.slantlinepropsblack)
                            pl.annotate(r'$R_{\rm circ}$',(Rcirc_loc+log(1.1),4e7),(Rcirc_loc+log(1.5),2.4e7),fontsize=pB.infigfontsize)                        
                if iPanel==1:
                    ylabel = r'$n_{\rm H}T\ [{\rm cm}^{-3} {\rm K}]$'
                    if layout==1: pl.ylim(0.1,1e4)
                    if layout==2: pl.ylim(1,1e7)
                    ax.set_yscale('log')
                    ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                if iPanel==10:
                    pl.ylabel(r'$\mathcal{M}_{\rm r}$')
                    [pl.axhline(x,c='.5',lw=0.5,zorder=1000) for x in (-1,1)]
                    ax.set_yscale('symlog',linthreshy=0.5,linscaley=0.5)
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    ax.yaxis.set_major_locator(ticker.FixedLocator([-10,-1,0,1,10]))
                    if layout in (1,2): pl.ylim(-30,30)
                if iPanel==10:
                    ylabel = r'$t_{\rm cool}\ [{\rm Myr}]$'
                    pl.plot(x,prof.t_cool(),c='k')
                    pl.plot(x,prof.t_ff(),ls='--',c='k')
                    #pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                    pl.text(Rcirc_loc,0.1,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
                    if layout==1: 
                        pl.annotate(r'$t_{\rm ff}$',(log(0.8),1000),(log(1.),300),arrowprops=pB.slantlinepropsblack,ha='left',va='top',fontsize=pB.infigfontsize)
                        if isim==0:
                            pl.annotate(r'',(log(0.04),0.9),(log(0.045),0.3),arrowprops=pB.slantlinepropsblack,ha='left',fontsize=pB.infigfontsize)
                            pl.annotate(r'$\langle t_{\rm cool}\rangle$',(log(0.04),0.8),(log(0.04),0.075),ha='left',fontsize=pB.infigfontsize)
                        pl.ylim(0.03,1e6)
                    if layout==2 and isim==0: 
                        pl.annotate(r'$t_{\rm ff}$',(log(0.8),300),(log(1.),100),arrowprops=pB.slantlinepropsblack,ha='left',va='top',fontsize=pB.infigfontsize)
                        pl.annotate(r'',(log(0.04),0.09),(log(0.045),0.03),arrowprops=pB.slantlinepropsblack,ha='left',fontsize=pB.infigfontsize)
                        pl.annotate(r'$\langle t_{\rm cool}\rangle$',(log(0.04),0.27),(log(0.04),0.025),ha='left',fontsize=pB.infigfontsize)
                        pl.ylim(0.01,3e4)
                if iPanel==10:
                    pl.plot(x,prof.jProfile(),lw=2,c='k')
                    pl.plot(x,prof.rs_midbins()*self.profiler.vc(),ls='--',c='k')
                    ylabel = r'$\left|\vec{j}\right|\ [{\rm kpc}\ {\rm km}{\rm s}^{-1}]$'
                    if layout in (1,2):
                        pl.annotate(r'',(log(1.),(None,1500,2200)[layout]),(log(0.8),1000),arrowprops=pB.slantlinepropsblack,ha='right',va='top')
                        pl.text(log(0.8),1300,r'$\left|\Sigma \vec{j}\right|}$',ha='right',va='top')
                        pl.annotate(r'',(log(1.),(None,4.5e4,3e4)[layout]),(log(0.8),6e4),arrowprops=pB.slantlinepropsblack,ha='right',va='bottom')
                        pl.text(log(0.8),5e4,r'$v_{\rm c} r$',ha='right',va='bottom')
                    if layout in (1,2): pl.ylim(100,1e5)
                    if layout in (3,4): pl.ylim(100,1e5)                    
                
                
                if ax.is_first_row() and not present:                    
                    pl.title((r'$10^{%.1f}{\rm M}_\odot,\ z=%s$'%(log(prof.mvir),(('%d','%.1f')[layout-1])%prof.z,) +
                              '\n' +
                              r'$t_{\rm cool}^{(s)}%st_{\rm ff}\ {\rm at}\ R_{\rm circ}$'%('><'[isim])),
                              fontsize=pB.infigfontsize)
                    
                    #pl.text(0.95,0.05,prof.galaxyname,ha='right',va='bottom',fontsize=pB.infigfontsize,transform=ax.transAxes)
                if ax.is_first_col() or ax.is_last_col():
                    pl.ylabel(ylabel)
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    if ax.is_last_col():
                        pl.ylabel(ylabel)
                        ax.yaxis.set_label_position('right')
                        ax.yaxis.set_ticks_position('right')
                        ax.yaxis.set_ticks_position('both')
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                if layout==1: pl.xlim(log(0.03),log(1.75))
                if layout==2: pl.xlim(-2.3,log(1.75))                    
                ax.xaxis.set_major_locator(ticker.FixedLocator([-2.0001,-1.0001,0.0001]))    
                ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,3]))))
                if ax.is_last_row():
                    if layout==1: toshow = np.array(log([0.003,0.01,0.03,0.1,0.3,1]))
                    if layout==2: toshow = np.array(log([0.01,0.1,1]))
                    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',u.nSignificantDigits(10.**x,1,True))[x in toshow]))                        
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
                    pl.xlabel(r'$r/R_{\rm vir}$')
                else:
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    
        if not present:
            cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[0.,-1, -2, -3],
                               aspect=40,fraction=0.015,pad=(0.15,0.25)[nPanels==1])
            cbar.ax.set_xticklabels([r'$1$', r'$0.1$', r'$0.01$', r'$0.001$'])  # horizontal colorbar
            
            if weight=='MW': cbar.set_label(r'$\log\ M/M_{\rm shell}(r)\ [{\rm dex}^{-1}]$')
            if weight=='VW': cbar.set_label(r'fraction of volume at radius $r\ [{\rm dex}^{-1}]$')
    
        #j = -1.4
        ##pl.axvline(j+log(rvirs[ifn]),c='.3',ls='-',zorder=1e5,lw=0.5)
        #pl.text(j+log(self.snapshot.sim.rvir),ymax,r'$r_{\rm circ}$',ha='center')
        #pl.text(j+log(self.snapshot.sim.rvir),ymax/2.,r'|',ha='center',va='center')
    
        ##pl.axvline(log(rvirs[ifn]),c='.3',ls='-',zorder=1e5,lw=0.5)
        #pl.text(log(self.snapshot.sim.rvir),ymax,r'$r_{\rm vir}$',ha='center')
        #pl.text(log(self.snapshot.sim.rvir),ymax/2.,r'|',ha='center',va='center')
        #pl.text(log(220),35,r'${\rm m12i},\,z=%s$'%self.snapshot.sim.z,fontsize=7,ha='right')
    
        
        pl.savefig(figDir+'FIRE_compare_snapshot%d_%s_nPanels%d.png'%(layout,weight,nPanels),bbox_inches='tight',dpi=300)    
 
    def snapshot_image_and_2Dprofile(self,img_fns,weights=('VW','MW'),vmin=-2.,vmax=0.5,layout=1,show_cbars=True):
        profs = [self.profiler]+self.others
        fig = pl.figure(figsize=(pB.fig_width_full*1.05,(None,2.675,5.)[len(profs)]))
        pad = (None,0.25,0.12)[len(profs)]
        top_right = (None,0.875,0.93)[len(profs)]
        spec = matplotlib.gridspec.GridSpec(ncols=7, nrows=len(profs), figure=fig,hspace=0.35,wspace=0.1,left=0.025,right=0.93,bottom=0.02,top=0.95)
        spec2 = matplotlib.gridspec.GridSpec(ncols=13, nrows=len(profs), figure=fig,hspace=0.5,wspace=0.4,left=0.025,right=0.93,bottom=0.02,top=top_right)
        
        if len(img_fns):            
            axs = []
            for iprof,prof in enumerate(profs):
                ax = fig.add_subplot(spec[iprof,:3])
                axs.append(ax)
                img = pl.imread(img_fns[iprof])
                ax.imshow(img,aspect='equal',zorder=-100)
                ax.axis('off')
                if prof.galaxyname=='m12i': pl.text(0.,1.1,r'${\rm m12i}$, $z=0$, $M_{12}=1.1$, $t_{\rm cool}^{\rm (s)}/t_{\rm ff}(0.1R_{\rm vir})=16$',transform=ax.transAxes,fontsize=pB.infigfontsize)
                if prof.galaxyname=='m11d': pl.text(0.,1.1,r'${\rm m11d}$, $z=0$, $M_{12}=0.3$, $t_{\rm cool}^{\rm (s)}/t_{\rm ff}(0.1R_{\rm vir})=0.2$',transform=ax.transAxes,fontsize=pB.infigfontsize)
                if prof.galaxyname=='h2': pl.text(0.,1.1,r'${\rm m13A8}$, $z=2.5$, $M_{12}=1.3$, $t_{\rm cool}^{\rm (s)}/t_{\rm ff}(0.1R_{\rm vir})=0.4$',transform=ax.transAxes,fontsize=pB.infigfontsize)
                if prof.galaxyname=='h206': pl.text(0.,1.1,r'${\rm m13A1}$, $z=2.5$, $M_{12}=2.1$, $t_{\rm cool}^{\rm (s)}/t_{\rm ff}(0.1R_{\rm vir})=5$',transform=ax.transAxes,fontsize=pB.infigfontsize)
            
        if layout==1: 
            cbar_range = 10**3.,10**6.
            cbar_labels = [r'$10^%d$'%d for d in range(3,7)]
        if layout==2: 
            cbar_range = 10**3.,10**7.
            cbar_labels = [r'$10^%d$'%d for d in range(3,8)]
        if show_cbars:
            cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(*cbar_range),cmap='RdBu_r')
            cmap.set_array([])
            cbar = pl.colorbar(cmap,orientation='horizontal',ax=axs,aspect=50,fraction=0.1,pad=pad,cmap='RdBu_r')
            cbar.ax.set_xticklabels(cbar_labels)  # horizontal colorbar            
            cbar.set_label(r'temperature $[{\rm K}]$')

        axs = []
        for iprof,prof in enumerate(profs):           
            for iweight,weight in enumerate(weights):
                ax = fig.add_subplot(spec2[iprof,7+3*iweight:10+3*iweight])
                axs.append(ax)
                m,x,y = prof.profile2D('Ts',weight,Snapshot_profiler.T_bins)                    
                d_log_ybin = 2
                x = midbins(x); y = midbins(y)
                m = np.apply_along_axis(lambda m: np.convolve(m, np.ones((10,))/10, mode='same'), axis=1, arr=m)
                #m = np.convolve(m,,axis=1,mode='same')#[:,:10*(len(y)//10)].reshape((len(x),len(y)//10,10)).sum(axis=2)
                logm = log(m.T / m.T.sum(axis=0))+d_log_ybin;logm[np.isinf(logm)] = -100        
                mesh = pl.pcolormesh(x,y,logm,cmap=matplotlib.cm.plasma_r,zorder=-100,vmin=vmin,vmax=vmax)
                #logm = m.T / m.T.sum(axis=0)*10**d_log_ybin
                #mesh = pl.pcolormesh(x,y,logm,cmap=matplotlib.cm.plasma_r,zorder=-100,vmin=0.01,vmax=1)
                mesh.cmap.set_under('w')
                
                ylabel = r'$T\ [{\rm K}]$'
                if layout==1: pl.ylim(1e3,1e7)
                if layout==2: pl.ylim(1e3,1e8)                
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                
                
                if iprof==0:
                    pl.title('%s-weighted'%('volume','mass')[weight=='MW'],fontsize=pB.infigfontsize)                    
                #if ax.is_first_row() and not present:                    
                    #pl.title((r'$10^{%.1f}{\rm M}_\odot,\ z=%s$'%(log(prof.mvir),(('%d','%.1f')[layout-1])%prof.z,) +
                              #'\n' +
                              #r'$t_{\rm cool}^{(s)}%st_{\rm ff}\ {\rm at}\ R_{\rm circ}$'%('><'[isim])),
                              #fontsize=pB.infigfontsize)
                    
                    ##pl.text(0.95,0.05,prof.galaxyname,ha='right',va='bottom',fontsize=pB.infigfontsize,transform=ax.transAxes)
                
                ax.yaxis.set_major_formatter(u.arilogformatter)
                pl.ylabel(ylabel)
                if iweight==1:
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.set_ticks_position('right')
                    ax.yaxis.set_ticks_position('both')
                    x,y=log(2.),prof.Tvir()
                    if prof.galaxyname=='m12i': 
                        x+=0.05; y/=2.25
#                       pl.annotate('',(log(1.8),prof.Tvir()),(x,y),arrowprops=pB.slantlinepropsblack,zorder   
                        pl.arrow(x-0.02,y*1.25,-0.05,1.5e5,clip_on=False,zorder=1000)
                    pl.text(x,y,r'$T_{\rm vir}$',va='center',fontsize=8)
                    pl.plot([log(1.6),log(1.9)],[prof.Tvir(),prof.Tvir()],c='k',clip_on=False)
                if layout==1: pl.xlim(log(0.03),log(1.75))
                if layout==2: pl.xlim(-2.3,log(1.75))                    
                ax.xaxis.set_major_locator(ticker.FixedLocator([-2.0001,-1.0001,0.0001]))    
                ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,3]))))
                toshow = np.array(log([0.01,0.1,1]))
                ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',r'$%s$'%u.nSignificantDigits(10.**x,1,True))[x in toshow]))                        
                ax.xaxis.set_major_formatter(ticker.NullFormatter())
                pl.xlabel(r'$r/R_{\rm vir}$',fontsize=pB.infigfontsize)
                ax.tick_params(labelsize=pB.infigfontsize)
        if show_cbars:
            cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(10**-2,10**0.5),cmap=matplotlib.cm.plasma_r)
            cmap.set_array([])
            cbar = pl.colorbar(cmap,orientation='horizontal',ax=axs,ticks=[1.,0.1,0.01,0.001],aspect=50,fraction=0.1,pad=pad)
            cbar.ax.set_xticklabels([r'$1$', r'$0.1$', r'$0.01$', r'$0.001$'])  # horizontal colorbar            
            cbar.set_label(r'fraction of shell mass or volume [dex$^{-1}$]')
        if len(profs)==1:
            pl.savefig(figDir+'snapshot_image_and_2Dprofile_%s.png'%prof.galaxyname,dpi=400,bbox_inches='tight')
        else:
            pl.savefig(figDir+'snapshot_image_and_2Dprofile.png',dpi=400)
    def pressure_images(self,img_fn):
        fig = pl.figure(figsize=(pB.fig_width_half,2.675))
        spec = matplotlib.gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
        
        ax = fig.add_subplot(spec[0,0])
        img = pl.imread(img_fn)
        ax.imshow(img,aspect='equal',zorder=-100)
        ax.axis('off')
        if self.profiler.galaxyname=='m12i': 
            pl.text(0.78,1.1,r'$z=0$,    $t_{\rm cool}^{\rm (s)}/t_{\rm ff}=9$',transform=ax.transAxes,fontsize=pB.infigfontsize,ha='center')
            pl.text(0.22,1.1,r'$z=0.6$,   $t_{\rm cool}^{\rm (s)}/t_{\rm ff}=0.6$',transform=ax.transAxes,fontsize=pB.infigfontsize,ha='center')
           
        cbar_range = 10**-1,10.
        cbar = pl.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(*cbar_range), cmap='viridis'),
                           orientation='horizontal',ax=ax,aspect=40,fraction=0.1,pad=0.15,cmap='viridis')
        cbar.ax.set_xticklabels([r'$0.1$', r'$1$', r'$10$'])  # horizontal colorbar            
        cbar.set_label(r'$P/\langle P(r)\rangle$')
        
        pl.savefig(figDir+'pressure_images_%s.png'%self.profiler.galaxyname,dpi=600)

    def compare_snapshots_1D(self,layout,suffix='',weights=('VW','MW')):
        fig = pl.figure(figsize=(pB.fig_size_half[0]*0.7,pB.fig_size_half[1]))
        pl.subplots_adjust(hspace=0.1,wspace=0.5)
        profs = [self.profiler]+self.others; nSims = len(profs)
        ax = pl.subplot(111)
        for isim,prof in enumerate(profs):
            #Rcirc_loc = log(prof.Rcirc2(0.5)/prof.rvir)
            #Rcirc_loc = log(prof.Rcirc()/prof.rvir)
            if layout==1: Rcirc_loc = log(0.12)
            if layout==2: Rcirc_loc = log(0.1)
            if layout==3: Rcirc_loc = log(0.045)
            if layout==4: Rcirc_loc = log(0.04)
            
            pl.axvline(Rcirc_loc,lw=0.5,c='k',ls='--')
            xs = prof.rs_midbins()/prof.rvir
            for weight in weights:
                pl.plot(log(xs),prof.profile1D('log_Ts',weight),c='k',ls=('--','-')[weight=='VW'],
                        label=('by mass','by vol.')[weight=='VW'])
            u.mylegend(loc=('lower right','upper right')[0],handlelength=1.7,fontsize=pB.infigfontsize)
            pl.ylabel(r'$\langle \log\ (T/{\rm K})\rangle$')
            if layout in (1,2): pl.ylim(3.,6.5)
            if layout in (3,4): pl.ylim(3.,7.1)
            if layout!=4: pl.text(Rcirc_loc,3.2,r'$R_{\rm circ}$',ha='left')
            else: pl.text(Rcirc_loc,6.4,r'$R_{\rm circ}$',ha='left')
            #if layout in (1,2):                
                #pl.annotate(r'',(Rcirc_loc,6.7),(Rcirc_loc+log(1.5),6.5),arrowprops=pB.slantlinepropsblack)
                #pl.annotate(r'$R_{\rm circ}$',(Rcirc_loc+log(1.1),6.7),(Rcirc_loc+log(1.5),log(4.25e6)),fontsize=pB.infigfontsize)                        
            #if layout in (3,4):
                #pl.annotate(r'',(Rcirc_loc,3.),(Rcirc_loc+log(1.5),log(3.4e7)),arrowprops=pB.slantlinepropsblack)
                #pl.annotate(r'$R_{\rm circ}$',(Rcirc_loc+log(1.1),7.6),(Rcirc_loc+log(1.5),log(2.4e7)),fontsize=pB.infigfontsize)                        
            if layout in (1,2): pl.xlim(log(0.03),log(1.75))
            if layout in (3,4): pl.xlim(-2.3,log(1.75))                    
            ax.xaxis.set_major_locator(ticker.FixedLocator([-2.0001,-1.0001,0.0001]))    
            ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,3]))))
            if layout in (1,2): toshow = np.array(log([0.003,0.01,0.03,0.1,0.3,1]))
            if layout in (3,4): toshow = np.array(log([0.01,0.1,1]))
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',u.nSignificantDigits(10.**x,1,True))[x in toshow]))                        
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            pl.xlabel(r'$r/R_{\rm vir}$')
        
        
        pl.savefig(figDir+'FIRE_1D_profile%d%s.png'%(layout,suffix),bbox_inches='tight',dpi=300)    
    def compare_snapshots_1D_b(self,layout,suffix='',weight='VW'):
        fig = pl.figure(figsize=(pB.fig_size_half[0]*0.7,pB.fig_size_half[1]))
        pl.subplots_adjust(hspace=0.1,wspace=0.5)
        profs = [self.profiler]+self.others; nSims = len(profs)
        ax = pl.subplot(111)
        for isim,prof in enumerate(profs):
            if layout==1: Rcirc_locs = log(0.1), log(0.12)
            if layout==3: Rcirc_locs = log(0.04), log(0.045)
            for Rcirc_loc in Rcirc_locs:
                pl.axvline(Rcirc_loc,lw=0.5,c='k',ls='--')
            pl.text(Rcirc_loc,(-1.45,-1.9)[layout==3],r'$R_{\rm circ}$',ha='left',fontsize=8)
            c = ('k','.5')[isim]
            xs = prof.rs_midbins()/prof.rvir            
            pl.plot(log(xs),prof.profile1D('log_Ts',weight)-log(prof.Tvir()),c=c,
                        label=(r'$t_{\rm cool}>t_{\rm ff}$',r'$t_{\rm cool}<t_{\rm ff}$')[isim])
            u.mylegend(loc=('lower right','upper right')[0],handlelength=1.7,fontsize=pB.infigfontsize)
            pl.ylabel(r'$\langle \log\ (T/T_{\rm vir})\rangle_{V}$')
            if layout in (1,2): pl.ylim(-1.5,0.5)
            if layout in (3,4): pl.ylim(-2,0.5)
            if layout in (1,2): pl.xlim(log(0.03),log(1.75))
            if layout in (3,4): pl.xlim(-2.3,log(1.75))                    
            ax.xaxis.set_major_locator(ticker.FixedLocator([-2.0001,-1.0001,0.0001]))    
            ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,3]))))
            if layout in (1,2): toshow = np.array(log([0.003,0.01,0.03,0.1,0.3,1]))
            if layout in (3,4): toshow = np.array(log([0.01,0.1,1]))
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',u.nSignificantDigits(10.**x,1,True))[x in toshow]))                        
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            pl.xlabel(r'$r/R_{\rm vir}$')
        
        
        pl.savefig(figDir+'FIRE_1D_profile%d%s.png'%(layout,suffix),bbox_inches='tight',dpi=300)    
    def compare_snapshots_statement(self,weight,vmin=-1.5,vmax=0.5,layout=None,nPanels=1):
        pl.figure(figsize=(pB.fig_width_full*2/3.,2))
        pl.subplots_adjust(hspace=0.1,wspace=0.1)
        profs = [self.profiler]+self.others; nSims = len(profs)
        axs = [pl.subplot(nPanels,nSims,iPanel+1) for iPanel in range(nPanels*nSims)]
        for isim,prof in enumerate(profs):
            for iPanel in range(nPanels):
                ax = axs[isim+nSims*iPanel]; pl.sca(ax)                
                if iPanel==0: 
                    m,x,y = prof.profile2D('Ts',weight,Snapshot_profiler.T_bins)                    
                x = midbins(x); y = midbins(y)
                d_log_ybin = 2
                m = np.apply_along_axis(lambda m: np.convolve(m, np.ones((10,))/10, mode='same'), axis=1, arr=m)
                #m = np.convolve(m,,axis=1,mode='same')#[:,:10*(len(y)//10)].reshape((len(x),len(y)//10,10)).sum(axis=2)
                logm = log(m.T / m.T.sum(axis=0))+d_log_ybin;logm[np.isinf(logm)] = -100        
                mesh = pl.pcolormesh(x,y,logm,cmap=matplotlib.cm.plasma_r,zorder=-100,vmin=vmin,vmax=vmax)
                mesh.cmap.set_under('w')
                Rcirc_loc = log(prof.Rcirc2(0.5)/prof.rvir)
                ymax=1e7
                pl.text(Rcirc_loc,ymax*2,r'$r_{\rm circ} $',ha='center',fontsize=pB.infigfontsize)
                pl.text(Rcirc_loc,ymax,r'|',ha='center',va='center',fontsize=pB.infigfontsize)
                #pl.text(0.,ymax*2,r'$r_{\rm vir}$',ha='center',fontsize=pB.infigfontsize)
                #pl.text(0.,ymax,r'|',ha='center',va='center',fontsize=pB.infigfontsize)
                
                if iPanel==0:
                    ylabel = r'$T\ [{\rm K}]$'
                    if layout==1: pl.ylim(3e3,1e7)
                    if layout==2: pl.ylim(3e3,5e7)
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                if ax.is_first_row():
                    sMh = (('%.1f','%d')[isim])%log(prof.mvir)
                    pl.text(0.95,0.95,
                            r'$10^{%s}{\rm M}_\odot,\ z=%d$'%(sMh,prof.z)
                            ,ha='right',va='top',fontsize=pB.infigfontsize,transform=ax.transAxes)
                if ax.is_first_col():
                    pl.ylabel(ylabel)
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                if layout==1: pl.xlim(log(0.03),0.31)
                if layout==2: 
                    if isim==0: pl.xlim(-1.5,0.31)
                    if isim==1: pl.xlim(-2,0.31)
                ax.xaxis.set_major_locator(ticker.FixedLocator([-1.0001,0.0001]))    
                ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]))))
                if ax.is_last_row():
                    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',u.nSignificantDigits(10.**x,1,True))[x in np.array(log([0.01,0.03,0.1,0.3,1]))]))                        
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
                    pl.xlabel(r'$r/r_{\rm vir}$')
                else:
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    
            
        cbar = pl.colorbar(mesh,orientation='vertical',ax=axs,ticks=[0.,-1, -2, -3])
        cbar.ax.set_xticklabels([r'$1$', r'$0.1$', r'$0.01$', r'$0.001$'])  # horizontal colorbar
        
        if weight=='MW': cbar.set_label(r'$\log\ M/M_{\rm shell}(r)\ [{\rm dex}^{-1}]$')
        if weight=='VW': cbar.set_label(r'fraction of volume at radius $r\ [{\rm dex}^{-1}]$')
    
    
        
        pl.savefig(figDir+'FIRE_compare_snapshot%d_%s.png'%(layout,weight),bbox_inches='tight',dpi=600)    
    def mach(self,layout=None,weights=('MW','VW')):
        pl.figure(figsize=(pB.fig_width_full,3))
        pl.subplots_adjust(hspace=0.1,wspace=0.1)
        profs = [self.profiler]+self.others; nSims = len(profs)
        for isim in range(nSims):
            ax = pl.subplot(1,nSims,isim+1)
            for weight in weights:
                machs,x,y = profs[isim].profile2D('machs',weight,Snapshot_profiler.mach_bins)
                y = (y[1:]+y[:-1])/2.
                x = (x[1:]+x[:-1])/2.
                subsonic_inds = abs(y)<1
                f = machs[:,subsonic_inds].sum(axis=1) / machs.sum(axis=1)
                pl.plot(x,f,label=(r'by vol.',r'by mass')[weight=='MW'])
            if isim==0: pl.legend(handlelength=1,fontsize=pB.infigfontsize)
            pl.xlim(log(0.03),0.31)
            if ax.is_first_col():
                pl.ylabel(r'$f(\left|\mathscr{M}_{\rm r}\right|<1)$')
                pl.ylabel(r'subsonic fraction')
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            pl.title(r'$M_{\rm h}=10^{%.1f}{\rm M}_\odot,\ z=%d$'%(log(profs[isim].mvir),profs[isim].z))
            pl.text(0.95,0.95,profs[isim].galaxyname,ha='right',va='top',fontsize=pB.infigfontsize,transform=ax.transAxes)
            Rcirc_loc = log(profs[isim].Rcirc2(0.5)/profs[isim].rvir)
            pl.axvline(Rcirc_loc,lw=0.5,c='k',ls='--')
            t_ratio = log(profs[isim].t_cool()/profs[isim].t_ff()/2.)
            Rsonic = x[((t_ratio[:-1]*t_ratio[1:])<0).nonzero()[0][0]]
            if Rsonic>Rcirc_loc:
                pl.axvline(Rsonic,lw=0.5,c='k',ls='--')
                pl.text(Rsonic+log(1.05),0.05,r'$t_{\rm cool}=t_{\rm ff}$',fontsize=pB.infigfontsize)
            ax.xaxis.set_major_locator(ticker.FixedLocator([-1.0001,0.0001]))    
            ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]))))
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',u.nSignificantDigits(10.**x,1,True))[x in np.array(log([0.01,0.04,0.1,0.3,0.6,1,2]))]))                        
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            pl.text(Rcirc_loc,0.045,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
            pl.xlabel(r'$r/R_{\rm vir}$')
            pl.ylim(0,1.1)
            pl.savefig(figDir+'FIRE_subsonic_fraction%d.pdf'%layout,bbox_inches='tight')
    def mach2(self,layout=None,weights=('MW','VW')):
        pl.figure(figsize=(pB.fig_width_full,3))
        pl.subplots_adjust(hspace=0.1,wspace=0.1)
        profs = [self.profiler]+self.others; nSims = len(profs)
        for iweight,weight in enumerate(weights):
            ax = pl.subplot(1,len(weights),iweight+1)
            for isim in range(nSims):
                machs,x,y = profs[isim].profile2D('machs',weight,Snapshot_profiler.mach_bins)
                y = (y[1:]+y[:-1])/2.
                x = (x[1:]+x[:-1])/2.
                subsonic_inds = abs(y)<1
                f = machs[:,subsonic_inds].sum(axis=1) / machs.sum(axis=1)
                pl.plot(x,f,label=profs[isim].galaxyname)
            if isim==0: pl.legend(handlelength=1,fontsize=pB.infigfontsize)
            pl.xlim(log(0.03),0.31)
            if ax.is_first_col():
                pl.ylabel(r'$f(\left|\mathscr{M}_{\rm r}\right|<1)$')
                pl.ylabel(r'subsonic fraction')
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            pl.title(weight)
            #Rcirc_loc = log(profs[isim].Rcirc2(0.5)/profs[isim].rvir)
            #pl.axvline(Rcirc_loc,lw=0.5,c='k',ls='--')
            #t_ratio = log(profs[isim].t_cool()/profs[isim].t_ff()/2.)
            #Rsonic = x[((t_ratio[:-1]*t_ratio[1:])<0).nonzero()[0][0]]
            #if Rsonic>Rcirc_loc:
                #pl.axvline(Rsonic,lw=0.5,c='k',ls='--')
                #pl.text(Rsonic+log(1.05),0.05,r'$t_{\rm cool}=t_{\rm ff}$',fontsize=pB.infigfontsize)
            ax.xaxis.set_major_locator(ticker.FixedLocator([-1.0001,0.0001]))    
            ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]))))
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',u.nSignificantDigits(10.**x,1,True))[x in np.array(log([0.01,0.04,0.1,0.3,0.6,1,2]))]))                        
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            #pl.text(Rcirc_loc,0.045,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
            pl.xlabel(r'$r/R_{\rm vir}$')
            pl.ylim(0,1.1)
            pl.savefig(figDir+'FIRE_subsonic_fraction_2.pdf',bbox_inches='tight')
    def inflow(self):
        pl.figure(figsize=(pB.fig_width_full,3))
        pl.subplots_adjust(hspace=0.1,wspace=0.1)
        profs = [self.profiler]+self.others; nSims = len(profs)
        for isim in range(nSims):
            ax = pl.subplot(1,nSims,isim+1)
            for weight in 'VW','MW':
                machs,x,y = profs[isim].profile2D('machs',weight,Snapshot_profiler.mach_bins)
                y = (y[1:]+y[:-1])/2.
                x = (x[1:]+x[:-1])/2.
                inflow_inds = y<0
                f = machs[:,inflow_inds].sum(axis=1) / machs.sum(axis=1)
                pl.plot(x,f,label=weight)
            if isim==0: pl.legend()
            pl.xlim(log(0.03),0.31)
            if ax.is_first_col():
                pl.ylabel(r'$f(v_{\rm r}<0)$')
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())                
            Rcirc_loc = log(profs[isim].Rcirc2(0.5)/profs[isim].rvir)
            pl.axvline(Rcirc_loc,lw=0.5,c='k',ls='--')
            
            
    def j_and_tcool(self):
        pl.figure(figsize=(pB.fig_width_full,4))
        pl.subplots_adjust(wspace=0.1)
        profs = [self.profiler]+self.others
        for iPanel in range(4):
            ax=pl.subplot(2,2,iPanel+1)
            _profs = (profs[:3],profs[3:])[iPanel%2]
            for iprof,prof in enumerate(_profs):    
                xs = prof.rs_midbins()/prof.rvir
                c='brykc'[iprof]
                Rcirc2Rvir = prof.Rcirc2(0.5)/prof.rvir
                pl.axvline(Rcirc2Rvir,c=c,lw=0.5)
                if iPanel<2:                    
                    pl.plot(xs,(prof.vc()*prof.rs_midbins()),c=c,ls=':')
                    pl.plot(xs,prof.jProfile(),label=prof.galaxyname,c=c)
                else:                    
                    minind = u.searchsorted(xs,Rcirc2Rvir)
                    pl.plot(xs,prof.t_ff(),c=c,ls=':')
                    pl.plot(xs[minind:],prof.t_cool()[minind:],c=c)
                    
            pl.loglog()
            pl.xlim(0.01,2)        
            
            ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))        
            ax.yaxis.set_major_formatter(u.arilogformatter)
            ax.xaxis.set_major_formatter(u.arilogformatter)
            ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))        
            if iPanel==0:
                pl.ylim(8e2,2e5)
                pl.ylabel(r'$\left|\Sigma \vec{j}\right|\ [{\rm kpc\ km\ s^{-1}}]$')
            if iPanel==1:
                pl.ylim(0.8e2,2e4)
            if iPanel<2:                
                u.mylegend(loc='upper left',handlelength=0.5,fontsize=pB.infigfontsize)
            if iPanel>=2:
                pl.ylim(10,1e6)
                if iPanel==2:
                    pl.ylabel(r'$t_{\rm cool}^{sh}\ [{\rm Myr}]$')
                pl.xlabel(r'$r / R_{\rm vir}$')        
            if ax.is_last_col():
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=10,numticks=10))
            ax.yaxis.set_major_formatter(u.arilogformatter)
            ax.xaxis.set_major_formatter(u.arilogformatter)
        pl.savefig(figDir+'FIRE_j_and_tcool.pdf',bbox_inches='tight')        
    def j(self,z,colorDic,layout,minParticles=1000):
        pl.figure(figsize=(pB.fig_width_full,2.5))
        pl.subplots_adjust(wspace=0.35,hspace=0.6,bottom=0.05,left=0.075,right=0.925)
        profs = [self.profiler]+self.others
        for iPanel in range((2,3)[layout]):
            ax=pl.subplot(1,(2,3)[layout],iPanel+1)
            if z==0: _profs = (profs[:3],profs[3:6])[iPanel]
            if z==2: _profs = (profs[:2],profs[2:4])[iPanel]
            for iprof,prof in enumerate(_profs):    
                xs = prof.rs_midbins()/prof.rvir
                c=colorDic[prof.galaxyname]
                Rcirc2Rvir = prof.Rcirc2(0.5,max_r2rvir=0.5)/prof.rvir
                pl.axvline(Rcirc2Rvir,c=c,lw=0.5)
                pl.plot(xs,(prof.vc()*prof.rs_midbins()),c=c,ls=':',lw=0.75)
                if False: #prof.res!='HR':
                    minind = (prof.gasMassProfile()/float(prof.res)<minParticles).nonzero()[0][-1]+1
                else:
                    minind = 0
                pl.plot(xs[minind:],prof.jProfile()[minind:],label=prof.galaxyname,c=c)
                    
            pl.loglog()
            if z==0: pl.xlim(0.01,2)        
            if z==2: pl.xlim(0.005,2)        
            
            ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))        
            ax.yaxis.set_major_formatter(u.arilogformatter)
            ax.xaxis.set_major_formatter(u.arilogformatter)
            ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))        
            
            u.mylegend(loc='upper left',handlelength=0.5,fontsize=pB.infigfontsize)
            mvirs = np.array([prof.mvir for prof in _profs])
            print(iPanel,mvirs, [prof.galaxyname for prof in _profs])
            pl.title(r'$z=%s,\ M_{\rm halo}=10^{%.1f}-10^{%.1f}{\rm M}_\odot,\ t_{\rm cool}^{(s)}%st_{\rm ff}\ {\rm at}\ R_{\rm circ}$'%(
                ('%d'%z,'%.1f'%prof.z)[z==2],
                log(mvirs.min()),log(mvirs.max()),
                '><'[iPanel]), fontsize=pB.infigfontsize)
            if iPanel==0: 
                pl.ylabel(r'$\left|\Sigma \vec{j}\right|\ [{\rm kpc\ km\ s^{-1}}]$')
                if z==0: pl.text(0.1,400,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
                if z==2: pl.text(0.05,150,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
                pl.annotate(r'$v_{\rm c}r$',(0.8,4e4),(0.6,8e4),arrowprops=pB.slantlinepropsblack,
                            ha='right',va='center',fontsize=pB.infigfontsize)
                if z==0: pl.ylim(0.5e3,2e5)  
                if z==2: pl.ylim(100,1e5)  
                
            if iPanel==1:
                #pl.text(0.105,70,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
                if layout==0: pl.ylabel(r'$\left|\Sigma \vec{j}\right|\ [{\rm kpc\ km\ s^{-1}}]$')
                if z==0: pl.ylim(0.5e2,2e4)                
                if z==2: pl.ylim(100,2e4)
                #pl.annotate(r'$v_{\rm c}r$',(0.8,4e3),(0.6,8e3),arrowprops=pB.slantlinepropsblack,
                                        #ha='right',va='center',fontsize=pB.infigfontsize)
            if iPanel==2:
                if z==2: pl.ylim(5,1e4)
                
            pl.xlabel(r'$r / R_{\rm vir}$')        
            ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=10,numticks=10))
            ax.yaxis.set_major_formatter(u.arilogformatter)
            ax.xaxis.set_major_formatter(u.arilogformatter)
        pl.savefig(figDir+'FIRE_j_z%d.pdf'%z,bbox_inches='tight')        
    def mach3(self,weights=('MW','VW')):
        pl.figure(figsize=(pB.fig_width_full,3))
        pl.subplots_adjust(hspace=0.1,wspace=0.1)
        for iPanel in range(2):
            profs = [self.profiler]+self.others
            ax = pl.subplot(1,2,iPanel+1)
            _profs = (profs[:3],profs[3:])[iPanel]
            for iprof,prof in enumerate(_profs):    
                for weight in weights:
                    machs,x,y = prof.profile2D('machs',weight,Snapshot_profiler.mach_bins)
                    y = (y[1:]+y[:-1])/2.
                    x = (x[1:]+x[:-1])/2.
                    subsonic_inds = abs(y)<1
                    f = machs[:,subsonic_inds].sum(axis=1) / machs.sum(axis=1)
                    pl.plot(x,f,ls=('-','--')[weight=='MW'],label=(prof.galaxyname,'_')[weight=='MW'],c='bry'[iprof])
            
            pl.xlim(log(0.03),0.31)    
            pl.ylabel(r'subsonic fraction')
            pl.axhline(1.,c='k',ls=':')
            if iPanel==0:
                pl.legend(handlelength=1,fontsize=pB.infigfontsize)
            if iPanel==1:
                pl.legend(handlelength=1,fontsize=pB.infigfontsize,loc='lower right')
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
        #     pl.title(r'$M_{\rm h}=10^{%.1f}{\rm M}_\odot,\ z=%d$'%(log(profs[isim].mvir),profs[isim].z))
        #     pl.text(0.95,0.95,profs[isim].galaxyname,ha='right',va='top',fontsize=pB.infigfontsize,transform=ax.transAxes)
        #     Rcirc_loc = log(profs[isim].Rcirc2(0.5)/profs[isim].rvir)
        #     pl.axvline(Rcirc_loc,lw=0.5,c='k',ls='--')
        #     t_ratio = log(profs[isim].t_cool()/profs[isim].t_ff()/2.)
        #     Rsonic = x[((t_ratio[:-1]*t_ratio[1:])<0).nonzero()[0][0]]
        #     if Rsonic>Rcirc_loc:
        #         pl.axvline(Rsonic,lw=0.5,c='k',ls='--')
        #         pl.text(Rsonic+log(1.05),0.05,r'$t_{\rm cool}=t_{\rm ff}$',fontsize=pB.infigfontsize)
            ax.xaxis.set_major_locator(ticker.FixedLocator([-1.0001,0.0001]))    
            ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]))))
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',u.nSignificantDigits(10.**x,1,True))[x in np.array(log([0.01,0.04,0.1,0.3,0.6,1,2]))]))                        
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
        #     pl.text(Rcirc_loc,0.045,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
            pl.xlabel(r'$r/R_{\rm vir}$')
            pl.ylim(0,1.1)
        pl.savefig(figDir+'FIRE_subsonic_fraction_z0.pdf',bbox_inches='tight')

    def Ts(self,z,colorDic,layout,weights=('MW','VW')):
        fig = pl.figure(figsize=(pB.fig_width_full,3.5))
        pl.subplots_adjust(left=0.075,wspace=0.05,hspace=0.05,right=0.925)
        for iPanel in range((4,6)[layout]):
            profs = [self.profiler]+self.others
            ax = pl.subplot(2,(2,3)[layout],iPanel+1)
            if z==0: _profs = (profs[:3],profs[3:6])[iPanel%2]
            if z==2: _profs = (profs[:2],profs[2:4])[iPanel%2]
            mvirs = np.array([prof.mvir for prof in _profs])
            for iprof,prof in enumerate(_profs):    
                c=colorDic[prof.galaxyname]
                for weight in weights:
                    m,x,y = prof.profile2D(('Ts','machs')[iPanel//(2,3)[layout]],weight,Snapshot_profiler.mach_bins)
                    y = (y[1:]+y[:-1])/2.
                    x = (x[1:]+x[:-1])/2.
                    
                    if iPanel<(2,3)[layout]: subsonic_inds = y>1e5
                    else: subsonic_inds = abs(y)<1
                    f = m[:,subsonic_inds].sum(axis=1) / m.sum(axis=1)
                    #Tcs = prof.Tc()
                    #f = np.zeros(x.sape[0])
                    #for i in range(x.shape[0]):
                        #f[i] = m[i,y>Tcs[i]].sum() / m[i].sum()
                    pl.plot(x,f,ls=('-','--')[weight=='MW'],
                            label=(prof.galaxyname,'_')[weight=='MW' or iPanel==0],c=c)
                Rcirc2Rvir = prof.Rcirc2(0.5)/prof.rvir
                pl.axvline(log(Rcirc2Rvir),c=c,lw=0.5)                
            if z==0: pl.xlim(log(0.03),0.31)  
            if z==2: pl.xlim(log(0.01),log(2.6))  
            ax.xaxis.set_major_locator(ticker.FixedLocator([-1.0001,0.0001]))    
            ax.xaxis.set_minor_locator(ticker.FixedLocator(log(np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]))))            
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            #     pl.text(Rcirc_loc,0.045,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
            if ax.is_last_row():              
                pl.xlabel(r'$r/R_{\rm vir}$')
                ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: (r'',u.nSignificantDigits(10.**x,1,True))[x in np.array(log([0.01,0.03,0.1,0.3,0.6,1,2,3]))]))                        
                u.mylegend(handlelength=1.5,fontsize=pB.infigfontsize)
                pl.ylabel(r'subsonic fraction')            
            else:
                pl.ylabel(r'hot fraction')
                pl.title(r'$z=%s,\ M_{\rm halo}=10^{%.1f}-10^{%.1f}{\rm M}_\odot,\ t_{\rm cool}^{(s)}%st_{\rm ff}\ {\rm at}\ R_{\rm circ}$'%(
                    ('%d'%z,'%.1f'%prof.z)[z==2],
                    log(mvirs.min()),log(mvirs.max()),
                    '><'[iPanel]), fontsize=pB.infigfontsize)
                ax.xaxis.set_major_formatter(ticker.NullFormatter())
            pl.ylim(0,1.1)
            
            if iPanel==0:
                if z==0: pl.text(log(0.13),0.05,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
                if z==2: pl.text(log(0.05),0.05,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
                pl.plot([0,1],[-1,-1],ls='-',c='k',label='by volume')
                pl.plot([0,1],[-1,-1],ls='--',c='k',label='by mass')
                
                u.mylegend(handlelength=1.5,fontsize=pB.infigfontsize,
                           loc='lower right',bbox_to_anchor=(0.4,0.0,0.5,0.5),bbox_transform=ax.transAxes)
            if iPanel%2==1: 
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
        
        #     pl.title(r'$M_{\rm h}=10^{%.1f}{\rm M}_\odot,\ z=%d$'%(log(profs[isim].mvir),profs[isim].z))
        #     pl.text(0.95,0.95,profs[isim].galaxyname,ha='right',va='top',fontsize=pB.infigfontsize,transform=ax.transAxes)
        #     Rcirc_loc = log(profs[isim].Rcirc2(0.5)/profs[isim].rvir)
        #     pl.axvline(Rcirc_loc,lw=0.5,c='k',ls='--')
        #     t_ratio = log(profs[isim].t_cool()/profs[isim].t_ff()/2.)
        #     Rsonic = x[((t_ratio[:-1]*t_ratio[1:])<0).nonzero()[0][0]]
        #     if Rsonic>Rcirc_loc:
        #         pl.axvline(Rsonic,lw=0.5,c='k',ls='--')
        #         pl.text(Rsonic+log(1.05),0.05,r'$t_{\rm cool}=t_{\rm ff}$',fontsize=pB.infigfontsize)
        #pl.text(0.02,0.5,r'hot gas () fraction',rotation=90,ha='center',va='center',transform=fig.transFigure)
        #pl.text(0.98,0.5,r'subsonic fraction',rotation=90,ha='center',va='center',transform=fig.transFigure)
        
        pl.savefig(figDir+'FIRE_hot_fraction_z%d.pdf'%z,bbox_inches='tight')
    def image(self,ax,attr,r_circles,r_max2Rvir,width2r_max,edge_on=True,rot_j=0.,**kwargs):
        m,x,y = self.profiler.slab(attr,r_max2Rvir,width2r_max,edge_on=edge_on,rot_j=rot_j)        
        pl.sca(ax)
        pl.pcolormesh(x,y,m.T,**kwargs)
        pl.axis('off')
        for ir,r_circle in enumerate(r_circles):
            circle = pl.Circle((0, 0), r_circle, edgecolor='kw'[ir],facecolor='None',ls='--',lw=0.5)
            ax.add_artist(circle)        
class Sim_plotter:
    def __init__(self,sim):
        self.sim = sim
    def mach_profile(self):    
        
        profs = self.sim.profiles[:450:3]
        for CF in False,True:            
            ax = pl.subplot(1,2,CF+1)
            for iprof,prof in enumerate(profs):
                if not CF and prof.z < self.sim.z_cross(): continue
                if CF and prof.z > self.sim.z_cross(): continue
                xs = prof.rs_midbins()/prof.rvir
                mach = prof.profile1D('machs','VW')
                pl.plot(xs,mach,c=pB.cmap(iprof*1./len(profs)),label=(r'$z=%.2f$'%prof.z,'_nolegend_')[iprof%10!=0])
                #sig_vr = (prof.profile1D('machs','VW',2)-vr**2)**0.5
                #pl.fill_between(xs, (vr-sig_vr),(vr+sig_vr),facecolor='b',alpha=0.2)
            ax.set_xscale('log')
            ax.set_yscale('symlog',linthreshy=0.1,linscaley=0.25)
            pl.ylim(-10,10)
            pl.xlim(0.05,3)
            #pl.legend(ncol=2)
    def T2Tc_profile(self,weight,Tvir):       
        pl.figure()
        profs = self.sim.profiles[::-1]
        log_t_ratio = log(self.sim.tcool_Rcircs()/self.sim.tff_Rcircs())[::-1]
        ax = pl.subplot(111)
        ax.set_facecolor('.1')     
        for iprof,prof in enumerate(profs[5:-5]):
            xs = prof.rs_midbins()/prof.rvir
            #logTs = prof.smooth(prof.profile1D('log_Ts',weight),10)            
            logTs = np.nanmedian(np.array([profs[iprof+i].profile1D('log_Ts',weight)
                                           for i in range(-5,5)]),axis=0)#prof.smooth(prof.profile1D('log_Ts',weight),10)                        
            Tcs = Tvir(prof)#prof.Tc()
            c = pB.cmap(iprof*1./len(profs))
            c = matplotlib.cm.coolwarm((log_t_ratio[iprof]/5.+0.5))
            pl.plot(xs,logTs-log(Tcs),c=c,label=(r'$z=%.2f$'%prof.z,'_nolegend_')[iprof%10!=0],alpha=0.6)
            #sig_vr = (prof.profile1D('machs',weight,2)-vr**2)**0.5
            #pl.fill_between(xs, (vr-sig_vr),(vr+sig_vr),facecolor='b',alpha=0.2)
        ax.set_xscale('log')
        pl.xlim(0.1,1)
        pl.ylim(-1.,0.25)
        #pl.legend(ncol=2)
    def sigP_profile(self,weight,Tvir):       
        pl.figure()
        profs = self.sim.profiles[:][::-1]
        log_t_ratio = log(self.sim.tcool_Rcircs()/self.sim.tff_Rcircs())[::-1]
        ax = pl.subplot(111)
        ax.set_facecolor('.6')        
        for iprof,prof in enumerate(profs[5:-5]):
            xs = prof.rs_midbins()/prof.rvir
            #sig_Ps = np.nanmedian(np.array([(profs[iprof+i].profile1D('log_nHTs',weight,2) -
                                         #profs[iprof+i].profile1D('log_nHTs',weight)**2)**0.5 for i in range(-1,1)]),axis=0)#prof.smooth(prof.profile1D('log_Ts',weight),10)            
            sig_Ps = np.nanmedian(np.array([(profs[iprof+i].profile1D('log_nHTs',weight,2) -
                                            profs[iprof+i].profile1D('log_nHTs',weight)**2)**0.5 for i in range(-5,5)]),axis=0)#prof.smooth(prof.profile1D('log_Ts',weight),10)            
            c = pB.cmap(1.-iprof*1./len(profs))
            c = matplotlib.cm.RdBu((log_t_ratio[iprof]/5.+0.5))
            pl.plot(xs,sig_Ps,c=c,label=(r'$z=%.2f$'%prof.z,'_nolegend_')[iprof%10!=0],alpha=0.6)
        #ax.set_xscale('log')
        pl.xlim(0.05,3)
        pl.xlim(0.0,1)
        pl.ylim(0,1.)
        #pl.legend(ncol=2)
        
    def velocities_presentation(self):    
        f=pl.figure(figsize=(fig_width_full,5))    
        pl.subplots_adjust(left=0.13,right=0.99)        
        ax = pl.subplot(111)
        vc = vcs[simshort][goods,iR]
        vr = vrs[simshort][:,iR]
        sig_vr = (vr2s[simshort][:,iR]-vr**2)**0.5
        pl.fill_between(1+zs3[simshort], (vr-sig_vr)/vc,(vr+sig_vr)/vc,facecolor='b',alpha=0.2)
        pl.plot(1+zs3[simshort], vr/vc,c='b')
        if vcools!=None:
            pl.plot(1+zs3[simshort],vcools[simshort][goods,iR]/vc,c='k',ls=':')    
        pl.semilogx()
        if simgroup!='Daniel':         pl.xlim(1,6)
        else:         pl.xlim(2,8)
        pl.axhline(1.,c='.5',lw=0.5,zorder=1000)
        pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
        pl.axhline(-1.,c='.5',lw=0.5,zorder=1000)
        ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,7,8,9,10,11]))
        ax.xaxis.set_minor_locator(ticker.NullLocator())        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: (r'$%d$'%(x-1),r'')[x==7]))
        pl.xlabel(r'${\rm redshift}$',fontsize=18)
        pl.ylim(-1,1.5)
        pl.text(1+z_cross,1.51,'x',color='k',fontsize=16,ha='center')
        pl.axvline(1+z_cross,ls=':',c='k')
        pl.text(0.05,0.05,templates[simshort],fontsize=12,transform=ax.transAxes,color='k')
        pl.ylabel(r'$v_r / v_{\rm circ}$',fontsize=18)        
        pl.text(0.05,0.9,r'$0.1\ R_{\rm vir}$',fontsize=14,transform=ax.transAxes)
        ax.tick_params('both',labelsize=14,which='both')
        pl.savefig(FIRE.presentation_figDir+'vr2vc_Rcirc_in_FIRE_presentation.pdf')
    def inner_halo_images(self,iSnapshots,props,weights):        
        rvirs = [self.sim.rvirs()[u.searchsorted(self.sim.zs(), z_from_iSnapshot(i))] for i in iSnapshots]
        pl.figure(figsize=(pB.fig_width_full*1.1,4))
        pl.subplots_adjust(wspace=0.1,hspace=0.,top=0.9,bottom=0,left=0,right=1)        
        for iprop,prop in enumerate(props):                
            weight=weights[iprop]
            fns = [figDir+'FIREstudio_images/slab_FIREstudio_m12i_md_7100_%03d_rot0_0_%s_by_%s.png'%(i,prop,weight) for i in iSnapshots]
            axs = [None] * len(iSnapshots)
            for ifn,fn in enumerate(fns):                
                ax = axs[ifn] = pl.subplot(2,len(fns),ifn+1+len(fns)*iprop)
                img = pl.imread(fn)
                ax.imshow(img,aspect='equal',zorder=-100)
                ax.axis('off')
                if ax.is_first_row():
                    tratio = (0.25,1,4,16)[ifn]
                    pl.text(0.5,1.15,r'$t_{\rm cool}^{({\rm s})} / t_{\rm ff}=%s$'%u.nSignificantDigits(
                        tratio,2,True),
                            ha='center',va='bottom',fontsize=10,transform=ax.transAxes)
                    pl.plot([0.5,1],[1.025,1.025],lw=0.5,c='k',clip_on=False,transform=ax.transAxes)
                    pl.text(0.75,1.045,r'$0.2\,R_{\rm vir}$',transform=ax.transAxes,fontsize=7,ha='center')
            if prop=='logTs':
                cbar_range = 1e3,1e6
                ticks = [1e3,1e4,1e5,1e6]
                ticklabels = [r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$']
                label = r'temperature $[{\rm K}]$'
                cmap = 'RdBu_r'
            if prop=='lognHTs':
                cmap='viridis'            
                cbar_range = 0.1,10
                ticks = [0.1,1,10]
                ticklabels = [r'$0.1$', r'$1$', r'$10$']
                label = r'$P/\langle P(r)\rangle$'
            
            cbar = pl.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(*cbar_range), 
                                                            cmap=cmap),
                       orientation='vertical',ax=axs,fraction=0.1,shrink=0.8,pad=0.02,cmap=cmap,ticks=ticks)
            cbar.set_label(label,fontsize=10)
            cbar.ax.set_yticklabels(ticklabels,fontsize=10)
        pl.savefig(figDir+'m12i_multiple_images.png',dpi=400,bbox_inches='tight')        


    
Nsmooth = 41
class Sims_plotter:    
    color_fn = pyobjDir+'sim_colors'
    def __init__(self,sims,sortby=None,reverse=False):
        if sortby=='z_cross':
            inds = np.array([(sim.z_cross(),-1)[sim.z_cross()==None] for sim in sims]).argsort()        
            self.sims = np.array(sims)[inds]
        elif sortby=='name':
            self.sims = sorted(sims,key=lambda sim: sim.shortname())
        else:
            self.sims = sims
        if reverse: self.sims = self.sims[::-1]
        self.c_sims = dict([(sim.galaxyname,pB.cmap(isim*1./len(self.sims))) for isim,sim in enumerate(self.sims)])
        
    def mvir_and_Rcirc(self):
        fig = pl.figure(figsize=(pB.fig_width_full,3))
        pl.subplots_adjust(hspace=0.1,wspace=0.1)
        for iPanel in range(2):
            ax=pl.subplot(1,2,iPanel+1)
            for isim,sim in enumerate(self.sims):
                c=pB.cmap(isim*1./len(self.sims))
                if iPanel==0: 
                    ys = sim.mvirs()
                if iPanel==1: 
                    ys = 10.**sim.smooth(log(sim.Rcircs(byshell=True)/sim.rvirs()),polydeg=10)
                    #ys = sim.Rcircs(byshell=True)/sim.rvirs()
                tmp = (sim.mvirs()<1e10).nonzero()[0]
                if len(tmp): max_ind=tmp[0]
                else: max_ind=None
                if iPanel==0:
                    pl.plot(1.+sim.zs()[:max_ind],ys[:max_ind],lw=0.75,label=sim.shortname(),c=c)
                else:
                    pl.plot(1.+sim.zs()[:max_ind],ys[:max_ind],',',label=sim.shortname(),c=c)
                    ysmooth = 10.**np.convolve(log(ys),np.ones(30)/30.,mode='same')
                    pl.plot(1.+sim.zs()[:max_ind],ys[:max_ind],lw=1.5,label=sim.shortname(),c=c)
                for isRcirc in True,False:
                    if sim.z_cross(isRcirc=isRcirc)!=None:
                        if iPanel==0: y = 10.**sim.at_z_cross(log(ys),isRcirc=isRcirc)
                        if iPanel==1: y = 10.**sim.at_z_cross(log(ysmooth),isRcirc=isRcirc)
                        pl.plot([1+sim.z_cross(isRcirc=isRcirc)],[y],marker='+x'[isRcirc],c=c,ms=16,mew=2)
            pl.loglog()
            if iPanel==0:
                pl.ylabel(r'$M_{\rm halo}\ [{\rm M}_\odot]$')
                pl.ylim(1e10,1e13)
                zs_plus_1 = 10.**np.arange(0,1.3,.01)
                Mvirs = 10.**np.arange(10.,13.,.01)
                zs_plus_1_mesh, Mvirs_mesh = np.meshgrid(zs_plus_1,Mvirs)
                Ez = (u.Delta_c(zs_plus_1_mesh)/102.*(1-cosmo.Om0+cosmo.Om0*zs_plus_1_mesh**3))**0.5
                vvirs = 128 * (Mvirs_mesh/1e12)**(1/3.)*Ez*un.km/un.s
                Tvirs = (0.62*cons.m_p*vvirs**2/(2*cons.k_B)).to('K').value
                #pl.contour(zs_plus_1,Mvirs,Tvirs,10.**np.array([5,6,7,8,9]),zorder=-1000,colors='.3',linewidths=0.3,linestyles='-')
                    
                #pl.legend(ncol=2,fontsize=pB.infigfontsize,loc='lower left',handlelength=0.8,columnspacing=0.5)
            if iPanel==1:
                pl.ylabel(r'$R_{\rm circ} / R_{\rm vir}$')
                pl.ylim(0.01,1)
                
                
            pl.xlim(1,6.)
            ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,8,11,16]))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(u.arilogformatter)
            pl.xlabel(r'$z$')
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '%d'%(x-1)))
            ax2=pl.twiny()
            ax2.set_xscale('log')            
            zs_arr = np.arange(0.,10.,0.01)
            ages = cosmo.age(zs_arr).value
            ages_to_show = 0.5,1.,2,5.,10.
            zs_to_show = np.interp(ages_to_show,ages[::-1],zs_arr[::-1])
            # ax2.xaxis.set_major_locator(ticker.FixedLocator(1+zs_to_show))
            ax2.xaxis.set_minor_locator(ticker.NullLocator())
            pl.xticks(1+zs_to_show,[u.nSignificantDigits(x,1,True) for x in ages_to_show])
            pl.xlabel('time [Gyr]')
            pl.xlim(1.,6.)
            if iPanel%2==1:
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
        pl.savefig(figDir + 'mvir_and_Rcirc.pdf',bbox_inches='tight')    
    def smooth(self,arr,N=Nsmooth):
        return astropy.convolution.convolve(arr, np.ones((N)), boundary='extend')
    def desample(self,arr,N=5):
        size = arr.shape[0]
        newarr = arr[:(size//N) * N].reshape((size//N,N))
        return newarr.mean(axis=1)
    def Rcircs(self):
        fig = pl.figure(figsize=(pB.fig_width_full,6))
        pl.subplots_adjust(hspace=0.1,wspace=0.1)
        for isim,sim in enumerate(self.sims):
            ax=pl.subplot(4,4,isim+1)
            AM_2D = np.array([prof.jProfile() / (prof.vc() * prof.rs_midbins()) for prof in sim.profiles])
            mesh = pl.pcolormesh(Snapshot_profiler.log_r2rvir_bins,log(1+sim.zs()),AM_2D,cmap=matplotlib.cm.gnuplot_r,zorder=-100,
                                 vmin=0,vmax=1)
            pl.text(0.1,0.1,sim.shortname(),transform=ax.transAxes)
            pl.xlim(-2,0)
            pl.ylim(0,log(6.))
            #ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,8,11,16]))
            #ax.xaxis.set_minor_locator(ticker.NullLocator())
            #if ax.is_first_col():                
                #pl.ylabel(r'$R_{\rm circ} / R_{\rm vir}$')
                #ax.yaxis.set_major_formatter(u.arilogformatter)
            #elif ax.is_last_col():
                #ax.yaxis.set_label_position('right')
                #ax.yaxis.set_ticks_position('right')
                #ax.yaxis.set_ticks_position('both')
                #ax.yaxis.set_major_formatter(u.arilogformatter)
            #else:
                #ax.yaxis.set_major_formatter(ticker.NullFormatter())            
            #if ax.is_last_row():                
                #pl.xlabel(r'$z$')
                #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '%d'%(x-1)))
            #else:
                #ax.xaxis.set_major_formatter(ticker.NullFormatter())
        pl.savefig(figDir + 'Rcircs.pdf',bbox_inches='tight')    
        
        
    def mvir(self):
        fig = pl.figure(figsize=(pB.fig_width_half,4))
        pl.subplots_adjust(hspace=0.1,wspace=0.1,bottom=0.07)
        ax=pl.subplot(111)
        for isim,sim in enumerate(self.sims):
            c = self.c_sims[sim.galaxyname]
            ys = sim.mvirs()
            pl.plot(1.+sim.zs(),ys,lw=0.5,c=c)
            for isRcirc in True,False:
                if sim.z_cross(isRcirc=isRcirc)!=None:
                    y = 10.**sim.at_z_cross(log(ys),isRcirc=isRcirc)
                    pl.plot([1+sim.z_cross(isRcirc=isRcirc)],[y],ls='none',marker='+x'[isRcirc],c=c,ms=8,mew=2)


        pl.loglog()
        pl.ylabel(r'$M_{\rm halo}\ [{\rm M}_\odot]$')
        pl.ylim(0.7e10,1e13)
        #pl.legend(ncol=2,fontsize=pB.infigfontsize,loc='lower left',handlelength=0.8,columnspacing=0.5,markerscale=2)
        for isRcirc in False,True:
            label=r'$t_{\rm cool}^{(s)}=t_{\rm ff}$ at $%sR_{\rm vir}$'%(('','0.1 ')[isRcirc])
            pl.plot([15],[1e9],ls='none',marker='+x'[isRcirc],c='k',ms=6,mew=2,label=label)
        u.mylegend(fontsize=pB.infigfontsize,loc='lower left',labelspacing=0.,handletextpad=0.)
            
        pl.xlim(1,11.)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.5,1,2,3,4,5,6,8,11,16]))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(u.arilogformatter)
        pl.xlabel(r'${\rm redshift}$')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '%d'%(x-1)))
        ax2=pl.twiny()
        ax2.set_xscale('log')            
        zs_arr = np.arange(0.,10.,0.01)
        ages = cosmo.age(zs_arr).value
        ages_to_show = 0.5,1.,2,5.,10.
        zs_to_show = np.interp(ages_to_show,ages[::-1],zs_arr[::-1])
        # ax2.xaxis.set_major_locator(ticker.FixedLocator(1+zs_to_show))
        ax2.xaxis.set_minor_locator(ticker.NullLocator())
        pl.xticks(1+zs_to_show,[u.nSignificantDigits(x,1,True) for x in ages_to_show])
        pl.xlabel('time [Gyr]')
        pl.xlim(1.,11.)
        pl.savefig(figDir + 'mvir_all.pdf',bbox_inches='tight')    
    def mvir_bytime(self,max_log_t_ratio=1.5,isRcirc=True):
        fig = pl.figure(figsize=(pB.fig_width_half,4))
        pl.subplots_adjust(hspace=0.1,wspace=0.1,bottom=0.07)
        
        ax=pl.subplot(111)        
        
        for isim,sim in enumerate(self.sims):
            c = self.c_sims[sim.galaxyname]
            xs = cosmo.age(sim.zs()).value
            ys = sim.mvirs()
            
            log_t_ratio = sim.smooth(log(sim.quantity_vs_z('t_cool',isRcirc)/sim.quantity_vs_z('t_ff',isRcirc)))
            
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)            
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = pl.Normalize(-max_log_t_ratio, max_log_t_ratio)
            lc = matplotlib.collections.LineCollection(segments, cmap='coolwarm', norm=norm)
            lc.set_array(log_t_ratio)
            lc.set_linewidth(1.5)
            line = ax.add_collection(lc)
            #pl.plot(xs,ys,lw=0.5,c=c)
            for isRcirc2 in True,:
                if sim.z_cross(isRcirc=isRcirc2)!=None:
                    y = 10.**sim.at_z_cross(log(ys),isRcirc=isRcirc2)
                    c = pl.get_cmap('coolwarm')(norm(10.**sim.at_z_cross(log(log_t_ratio),isRcirc=isRcirc2)))
                    pl.plot([cosmo.age(sim.z_cross(isRcirc=isRcirc2)).value],[y],ls='none',
                            marker='+x'[isRcirc2],c=c,ms=5,mew=1)


        pl.semilogy()
        pl.ylabel(r'$M_{\rm halo}\ [{\rm M}_\odot]$')
        pl.ylim((1e9,1e10)[isRcirc],1e13)
        #pl.legend(ncol=2,fontsize=pB.infigfontsize,loc='lower left',handlelength=0.8,columnspacing=0.5,markerscale=2)
        for isRcirc2 in True,:
            label=r'$t_{\rm cool}^{(s)}=2t_{\rm ff}$ ${\rm at}$ $%sR_{\rm vir}$'%(('','0.1 ')[isRcirc2])
            pl.plot([1e11],[1e9],ls='none',marker='+x'[isRcirc2],c=pl.get_cmap('coolwarm')(norm(0.3)),ms=6,mew=2,label=label)
        u.mylegend(fontsize=pB.infigfontsize,loc='lower right',labelspacing=0.,handletextpad=0.)
            
        ax.yaxis.set_major_formatter(u.arilogformatter)
        pl.xlim(0,14.)
        pl.xlabel(r'${\rm time\ [Gyr]}$')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.))
        
        cbar = fig.colorbar(line, ax=ax,ticks=[-1.,0.,1])#,shrink=0.5,aspect=40)
        cbar.ax.set_yticklabels([r'$0.1$',r'$1$', r'$10$'])
        cbar.set_label(r'$t_{\rm cool}^{\rm (s)} / t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1')[isRcirc]))
        pl.text(0.3,5e12,r"m13's")
        pl.text(9,2e12,r"m12's")
        pl.text(10,2.5e10,r"m11's")
        
        
        
        ax2=pl.twiny()
        zs_to_show = np.array([0.,0.5,1,2,3,5])
        ages_to_show = cosmo.age(zs_to_show).value
        pl.xticks(ages_to_show,[u.nSignificantDigits(x,1,True) for x in zs_to_show])       
        pl.xlim(0.,14.)
        pl.xlabel(r'${\rm redshift}$')

        pl.savefig(figDir + 'mvir_all_bytime%s.pdf'%(('_Rvir','')[isRcirc]),bbox_inches='tight')    
    def t_ratio_vs_mvir(self,isRcirc=True):
        fig = pl.figure(figsize=(pB.fig_width_half,pB.fig_width_half*1.))
        pl.subplots_adjust(bottom=0.04,top=0.98,right=0.9,left=0.12)
        
        ax=pl.subplot(111)        
        
        zs_to_show = (0.,2.,5.)
        x_zs = {}; y_zs = {}
        for z in zs_to_show:                
            x_zs[z] = []; y_zs[z] = []        
        for isim,sim in enumerate(self.sims):            
            xs = 10.**self.smooth(log(sim.mvirs()))
            ys = 10.**self.smooth(log(sim.quantity_vs_z('t_cool',isRcirc)/sim.quantity_vs_z('t_ff',isRcirc)),
                                  N=31)
            #pl.plot(xs,ys,c='k',lw=0.5)
            for z in zs_to_show:                
                if min(sim.zs())<=z:
                    x_zs[z].append( 10.**np.interp(log(1+z),log(1+sim.zs()),log(xs)) )
                    y_zs[z].append( 10.**np.interp(log(1+z),log(1+sim.zs()),log(ys)) )
        for iz,z in enumerate(zs_to_show):            
            pl.scatter(x_zs[z],y_zs[z],marker = '*os^v'[iz],s=50,label=r'$z=%d$'%z,
                       edgecolors='k',facecolors='kwww'[iz])
            
        #Mhalos = 10.**np.arange(10,14.,.01)
        #pl.plot(Mhalos,2*0.12*1.28*(Mhalos/1e12)**1.1*2**0.5)

        pl.loglog()
        pl.xlabel(r'$M_{\rm halo}\ [{\rm M}_\odot]$')
        pl.xlim(0.5e10,1e13)
        ax.yaxis.set_major_formatter(u.arilogformatter)
        pl.ylim(0.004,40)
        pl.ylabel(r'$t_{\rm cool}^{({\rm s})} / t_{\rm ff}\ {\rm at}\ 0.1 R_{\rm vir}$')
        #pl.fill_between([0.5e10,1e13],1,4,facecolor='grey',zorder=-100,alpha=0.2)
        pl.axhline(2.,c='grey',lw=0.5)
        pl.text(0.7e10,2.45,r'${\rm virialization,\ disk\ settles}$',ha='left',va='center',fontsize=9)
        pl.legend(frameon=True,ncol=1,columnspacing=0.5,
                  handletextpad=0.0,borderpad=0.3,loc='lower right',fontsize=10)
       
        
        pl.savefig(figDir + 'mvir_vs_t_ratio.pdf',bbox_inches='tight')    
    def param_figs(self,isRcirc,showOther=False,max_log_t_ratio=1.5):
        fig = pl.figure(figsize=(pB.fig_width_half,7))
        pl.subplots_adjust(left=0.2,hspace=0.1,right=0.85,bottom=0.06)
        for iPanel in range(4):
            ax=pl.subplot(4,1,iPanel+1)            
            for isim,sim in enumerate(self.sims):
                xs = 1.+sim.zs()
                c = self.c_sims[sim.galaxyname]
                if iPanel==0:  
                    ys = 10.**sim.smooth(log(sim.quantity_vs_z('vc',isRcirc)))
                if iPanel==2: 
                    #ys = 10.**sim.smooth(log(sim.delta_Rcircs()))                    
                    nHs = sim.quantity_vs_z('rhoProfile_CGM',isRcirc)*X/cons.m_p.to('g').value
                    ys = 10.**sim.smooth(log(nHs))                    
                if iPanel==1: 
                    ys = 10.**sim.smooth(log(sim.quantity_vs_z('Z2Zsun')))
                    #ys2 = np.array([prof.atRcirc(prof.profile1D('Z2Zsuns','VW')) for prof in sim.profiles])                    
                    #pl.plot(1+sim.zs(),10.**sim.smooth(log(ys2)),lw=0.5,c=c,ls='--')
                if iPanel==3:
                    ys = 10.**sim.smooth(log(sim.quantity_vs_z('t_cool',isRcirc)))
                    if isim==0:                        
                        t_ffs = 10.**sim.smooth(log(sim.quantity_vs_z('t_ff',isRcirc)))
                        pl.plot(1+sim.zs(),t_ffs,c='k',ls=':',lw=1,zorder=-100)
                log_t_ratio = sim.smooth(log(sim.quantity_vs_z('t_cool',True)/sim.quantity_vs_z('t_ff',True)))                
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)            
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = pl.Normalize(-max_log_t_ratio, max_log_t_ratio)
                lc = matplotlib.collections.LineCollection(segments, cmap='coolwarm', norm=norm)
                lc.set_array(log_t_ratio)
                lc.set_linewidth(1.5)
                line = ax.add_collection(lc)
                
                #pl.plot(1.+sim.zs(),ys,lw=0.5,c=c,label=sim.shortname())
                #for isRcirc2 in False,True:
                    #if sim.z_cross(isRcirc=isRcirc2)!=None:
                        #if isRcirc2==isRcirc: 
                            #c2=c 
                            #alpha=1.
                        #else: 
                            #if not showOther: continue                                 
                            #c2 = '.8'
                            #alpha=0.5
                        #y = 10.**sim.at_z_cross(log(ys),isRcirc=isRcirc2)
                        #pl.plot([1+sim.z_cross(isRcirc=isRcirc2)],[y],marker='+x'[isRcirc2],c=c2,ms=8,mew=2,alpha=alpha)
            pl.loglog()
            Rstr = ('','0.1 ')[isRcirc]
            if iPanel==0:
                pl.ylabel(r'$v_{\rm c}(%sR_{\rm vir})\ [{\rm km}\ {\rm s}^{-1}]$'%Rstr)
                pl.ylim(30,800) 
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: ('',r'$%d$'%x)[x in (50,200,400,800)]))
                ax3 = pl.twinx()
                ax3.set_yscale('log')
                Tc_lims =(((gamma-1)*mu*cons.m_p*(un.km/un.s)**2/cons.k_B).to('K').value *
                                      np.array([30,1000])**2) 
                pl.ylim(Tc_lims)
                pl.ylabel(r'$T_{\rm c}\ [{\rm K}]$')
                ax3.yaxis.set_major_locator(ticker.LogLocator(numticks=6,numdecs=6))
                
                #pl.legend(ncol=2,fontsize=pB.infigfontsize,loc='lower left',handlelength=0.8,columnspacing=0.5,
                          #markerscale=2)
            if iPanel==2:
                _zs = np.arange(0,15,.01)
                for delta in 10.**np.arange(0,8):
                    pl.plot(1+_zs,delta*X*cosmo.Ob(_zs)*(cosmo.critical_density(_zs)/cons.m_p).to('cm**-3').value,
                            c='.7',ls='-',lw=0.3,zorder=-102)
                    if 0.1<delta<1e4:
                        x = 9
                        y = 1e-4*delta
                    elif delta==1e4: x = 4.5
                    elif delta==1e5: x = 2
                    else: continue                        
                    pl.text(x,y,r'%s'%u.arilogformat(delta),bbox=dict(facecolor='w',edgecolor='w'),
                            fontsize=pB.infigfontsize,color='.3',ha='center',zorder=-100)
                pl.text(8.5,6e-4,r'$\delta_{\rm baryons}=$',ha='right',va='bottom',color='.3')
                pl.ylabel(r'$n_{\rm H}(%sR_{\rm vir})\ [{\rm cm}^{-3}]$'%Rstr)
                pl.ylim(1e-4,1) 
            if iPanel==1:
                pl.ylabel(r'$Z(%s{R_{\rm vir}})\ [{\rm Z}_\odot]$'%Rstr)
                if isRcirc: pl.ylim(0.02,3)
                else: pl.ylim(0.02,3)
                    
            if iPanel==3:
                pl.ylabel(r'$t_{\rm cool}^{(s)}(%sR_{\rm vir})\ [{\rm Myr}]$'%Rstr)
                pl.ylim(1,1.e4)
                _zs = 10.**np.arange(0,1.5,.01)-1
                pl.plot(1+_zs,cosmo.age(_zs).to('Myr'),c='.7',ls='-',lw=0.3,zorder=-102)
                pl.annotate(r'$t_{\rm ff}$',(6.,(125,50)[isRcirc]),(7.5,(200,90)[isRcirc]),ha='center',fontsize=pB.infigfontsize,va='bottom')
                pl.annotate(r'',(6.,(100,40)[isRcirc]),(7.,(200,90)[isRcirc]),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                pl.annotate(r'$t_{\rm Hubble}$',(6.,1250),(7,2000),ha='center',fontsize=pB.infigfontsize,va='bottom')
                pl.annotate(r'',(6.,1000),(7.,2000),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                
                
            pl.xlim(1,8.)
            ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,7,8,11,16]))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            if ax.is_first_row():
                ax2=pl.twiny()
                ax2.set_xscale('log')            
                zs_arr = np.arange(0.,30.,0.01)
                ages = cosmo.age(zs_arr).value
                ages_to_show = 0.2,0.5,1.,2,5.,10.
                zs_to_show = np.interp(ages_to_show,ages[::-1],zs_arr[::-1])
                # ax2.xaxis.set_major_locator(ticker.FixedLocator(1+zs_to_show))
                ax2.xaxis.set_minor_locator(ticker.NullLocator())
                pl.xticks(1+zs_to_show,[u.nSignificantDigits(x,1,True) for x in ages_to_show])
                pl.xlim(1,8.)
                pl.xlabel('time [Gyr]')
            if ax.is_last_row():
                pl.xlabel(r'$z$')
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '%d'%(x-1)))
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_locator(ticker.LogLocator(numticks=7,numdecs=7))
            ax.yaxis.set_major_formatter(u.arilogformatter)
                
                
        pl.savefig(figDir + 'param_figs_at_R%s.pdf'%(('vir','circ')[isRcirc]))    
    def param_figs_by_time(self,isRcirc,showOther=False,max_log_t_ratio=1.5):
        fig = pl.figure(figsize=(pB.fig_width_half,9.5))
        pl.subplots_adjust(left=0.2,hspace=0.1,right=0.85,bottom=0.06)
        log_t_ff_zs_p1 = np.arange(0,1,.01)
        t_ffs = np.zeros((len(self.sims),log_t_ff_zs_p1.shape[0]))
        nHs_all = np.zeros((len(self.sims),log_t_ff_zs_p1.shape[0]))
        axs = []
        for iPanel in range(4):
            ax=pl.subplot(4,1,iPanel+1)
            axs.append(ax)
            for isim,sim in enumerate(self.sims):
                xs = cosmo.age(sim.zs())
                if iPanel==0:  
                    ys = 10.**sim.smooth(log(sim.quantity_vs_z('vc',isRcirc)))
                if iPanel==2: 
                    #ys = 10.**sim.smooth(log(sim.delta_Rcircs()))                    
                    nHs = sim.quantity_vs_z('rhoProfile_CGM',isRcirc)*X/cons.m_p.to('g').value
                    nHs_all[isim,:] = np.interp(log_t_ff_zs_p1, log(1+sim.zs()), nHs,left=np.nan,right=np.nan)
                    ys = 10.**sim.smooth(log(nHs))                    
                if iPanel==1: 
                    ys = 10.**sim.smooth(log(sim.quantity_vs_z('Z2Zsun',isRcirc)))
                    #ys2 = np.array([prof.atRcirc(prof.profile1D('Z2Zsuns','VW')) for prof in sim.profiles])                    
                    #pl.plot(1+sim.zs(),10.**sim.smooth(log(ys2)),lw=0.5,c=c,ls='--')
                if iPanel==3:
                    ys = 10.**sim.smooth(log(sim.quantity_vs_z('t_cool',isRcirc)))
                    t_ff = 10.**sim.smooth(log(sim.quantity_vs_z('t_ff',isRcirc)))
                    t_ffs[isim,:] = np.interp(log_t_ff_zs_p1, log(1+sim.zs()), t_ff,left=np.nan,right=np.nan)
                        
                log_t_ratio = sim.smooth(log(sim.quantity_vs_z('t_cool',True)/sim.quantity_vs_z('t_ff',True)))                
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)            
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = pl.Normalize(-max_log_t_ratio, max_log_t_ratio)
                lc = matplotlib.collections.LineCollection(segments, cmap='coolwarm', norm=norm)
                lc.set_array(log_t_ratio)
                lc.set_linewidth(1.5)
                line = ax.add_collection(lc)
                
            pl.semilogy()
            Rstr = ('0.5','0.1 ')[isRcirc]
            if iPanel==0:
                yls = (30,700)
                pl.ylabel(r'$v_{\rm c}(%sR_{\rm vir})\ [{\rm km}\ {\rm s}^{-1}]$'%Rstr)
                pl.ylim(*yls) 
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: ('',r'$%d$'%x)[x in (50,200,400,800)]))
                ax3 = pl.twinx()
                ax3.set_yscale('log')
                Tc_lims =((mu*cons.m_p*(un.km/un.s)**2/(gamma*cons.k_B)).to('K').value * np.array(yls)**2) 
                pl.ylim(Tc_lims)
                pl.ylabel(r'$T_{\rm c.f.}\ [{\rm K}]$')
                ax3.yaxis.set_major_locator(ticker.LogLocator(numticks=6,numdecs=6))
            if iPanel==1:
                pl.ylabel(r'$Z(%s{R_{\rm vir}})\ [{\rm Z}_\odot]$'%Rstr)
                if isRcirc: pl.ylim(0.02,3)
                else: pl.ylim(0.01,1)
            if iPanel==2:
                _zs = np.arange(0,15,.01)[::-1]
                ages = cosmo.age(_zs)
                deltas = 10.**np.arange(0,7)   
                for delta in deltas:
                    nHs = delta*X*cosmo.Ob(_zs)*(cosmo.critical_density(_zs)/cons.m_p).to('cm**-3').value
                    pl.plot(ages,nHs,c='.5',ls='-',lw=0.5,zorder=-102)
                    if delta>=1000:
                        x = 13
                        y = np.interp(x,ages,nHs)
                        if delta==1000: y=1.6e-4
                        if delta==1e6: pl.text(x-0.5,y,r'$\delta_{\rm b}=$',ha='right',color='.3',va='center') 
                    else:
                        y = 1.6e-4
                        x = np.interp(y,nHs[::-1],ages[::-1])
                    pl.text(x,y,r'%s'%u.arilogformat(delta),bbox=dict(facecolor='w',edgecolor='w'),
                            fontsize=pB.infigfontsize,color='.3',va='center',ha='center',zorder=-100)
                    
                pl.ylabel(r'$n_{\rm H}(%sR_{\rm vir})\ [{\rm cm}^{-3}]$'%Rstr)
                if isRcirc: pl.ylim(1e-4,1) 
                else: pl.ylim(1e-5,0.1)
                
                median_nHs = np.nanmedian(nHs_all,axis=0)
                _zs2 = 10.**log_t_ff_zs_p1[::-1]-1
                print(['%d'%x for x in np.interp(np.arange(0,14), cosmo.age(_zs2), 
                        median_nHs[::-1]/(X*cosmo.Ob(_zs2)*(cosmo.critical_density(_zs2)/cons.m_p).to('cm**-3').value))])
                    
            if iPanel==3:
                pl.ylabel(r'$t_{\rm cool}^{(s)}(%sR_{\rm vir})\ [{\rm Myr}]$'%Rstr)
                _zs = 10.**np.arange(0,1.5,.01)-1
                pl.plot(cosmo.age(_zs),cosmo.age(_zs).to('Myr'),c='.5',ls='-',lw=0.5,zorder=-102)
                median_tff = np.nanmedian(t_ffs,axis=0)
                pl.plot(cosmo.age(10.**log_t_ff_zs_p1-1),median_tff,c='k',ls=':',lw=1,zorder=-100)
                print('t_ff max / median:',np.nanmax(t_ffs/median_tff))
                print('t_ff median / min:',np.nanmax(median_tff/t_ffs))
                if isRcirc:
                    pl.annotate(r'$t_{\rm ff}$',(1.5,70),(0.8,100),ha='center',fontsize=pB.infigfontsize,va='bottom')                    
                    pl.annotate(r'',(1.2,40),(1.,100),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                    pl.ylim(1,1e4)
                else:
                    pl.annotate(r'$t_{\rm ff}$',(1.5,350),(0.8,500),ha='center',fontsize=pB.infigfontsize,va='bottom')                    
                    pl.annotate(r'',(1.2,200),(1.,500),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                    pl.ylim(10,1e5)
                #pl.annotate(r'$t_{\rm Hubble}$',(6.,1250),(7,2000),ha='center',fontsize=pB.infigfontsize,va='bottom')
                #pl.annotate(r'',(6.,1000),(7.,2000),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                
                
            pl.xlim(0,14.)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2.))               
            if ax.is_last_row():
                pl.xlabel(r'${\rm time\ [Gyr]}$')
                
            else:
                ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_locator(ticker.LogLocator(numticks=7,numdecs=7))
            ax.yaxis.set_major_formatter(u.arilogformatter)
            
            if ax.is_first_row():
                ax2=pl.twiny()
                zs_to_show = np.array([0.,0.5,1,2,3,5])
                ages_to_show = cosmo.age(zs_to_show).value
                pl.xticks(ages_to_show,[u.nSignificantDigits(x,1,True) for x in zs_to_show])       
                pl.xlim(0.,cosmo.age(0.).value)
                pl.xlabel(r'${\rm redshift}$')  
        cbar = fig.colorbar(line, ax=axs,ticks=[-1.,0.,1],orientation='horizontal',
                            pad=0.1,shrink=1,aspect=30)
        cbar.ax.set_yticklabels([r'$0.1$',r'$1$', r'$10$'])
        cbar.set_label(r'$t_{\rm cool}^{\rm (s)} / t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1')[isRcirc]))
        
        pl.savefig(figDir + 'param_figs_by_time_at_R%s.pdf'%(('vir','circ')[isRcirc]),bbox_inches='tight')    
        return cosmo.age(10.**log_t_ff_zs_p1-1),median_tff
    def param_figs_by_time_withMvirRcirc(self,isRcirc,showOther=False,max_log_t_ratio=1.5,t_ratio_cross=2,savefig=True):
        fig = pl.figure(figsize=(pB.fig_width_full*1.2,7.5))
        pl.subplots_adjust(left=0.2,wspace=0.35,hspace=0.1,right=0.85,bottom=0.06)
        log_t_ff_zs_p1 = np.arange(0,1,.01)
        t_ffs = np.zeros((len(self.sims),log_t_ff_zs_p1.shape[0]))
        nHs_all = np.zeros((len(self.sims),log_t_ff_zs_p1.shape[0]))
        axs = []
        norm = pl.Normalize(-max_log_t_ratio, max_log_t_ratio)
        for iPanel in range(6):
            ax=pl.subplot(3,2,iPanel+1)
            axs.append(ax)
            for isim,sim in enumerate(self.sims):
                xs = cosmo.age(sim.zs())
                if iPanel==0:  
                    ys = sim.mvirs()
                    #for isRcirc2 in True,:
                        #if sim.z_cross(isRcirc=isRcirc2)!=None:
                            #y = 10.**sim.at_z_cross(log(ys),isRcirc=isRcirc2)
                            #c = pl.get_cmap('coolwarm')(norm(log(t_ratio_cross)))
                            #pl.plot([cosmo.age(sim.z_cross(isRcirc=isRcirc2)).value],[y],ls='none',
                                    #marker='+x'[isRcirc2],c=c,ms=5,mew=1)                    
                if iPanel==1:  
                    ys = 0.1*sim.rvirs()
                if iPanel==3:  
                    ys = 10.**self.smooth(log(sim.quantity_vs_z('vc',isRcirc)))
                if iPanel==2: 
                    ys = 10.**self.smooth(log(sim.quantity_vs_z('Z2Zsun',isRcirc)))
                    #ys2 = np.array([prof.atRcirc(prof.profile1D('Z2Zsuns','VW')) for prof in sim.profiles])                    
                    #pl.plot(1+sim.zs(),10.**sim.smooth(log(ys2)),lw=0.5,c=c,ls='--')
                if iPanel==4: 
                    #ys = 10.**self.smooth(log(sim.delta_Rcircs()))                    
                    nHs = sim.quantity_vs_z('nH',isRcirc)
                    nHs_all[isim,:] = np.interp(log_t_ff_zs_p1, log(1+sim.zs()), nHs,left=np.nan,right=np.nan)
                    ys = 10.**self.smooth(log(nHs))                                    
                if iPanel==5:
                    ys = 10.**self.smooth(log(sim.quantity_vs_z('t_cool',isRcirc)))
                    t_ff = 10.**self.smooth(log(sim.quantity_vs_z('t_ff',isRcirc)))
                    t_ffs[isim,:] = np.interp(log_t_ff_zs_p1, log(1+sim.zs()), t_ff,left=np.nan,right=np.nan)
                        
                log_t_ratio = self.smooth(log(sim.quantity_vs_z('t_cool',True)/sim.quantity_vs_z('t_ff',True)))                
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)            
                segments = np.concatenate([points[:-1], points[1:]], axis=1)                
                lc = matplotlib.collections.LineCollection(segments, cmap='coolwarm', norm=norm)
                lc.set_array(log_t_ratio)
                lc.set_linewidth(1.5)
                line = ax.add_collection(lc)
                
            pl.semilogy()
            Rstr = ('0.5','0.1 ')[isRcirc]
            if iPanel==0:
                pl.ylabel(r'$M_{\rm halo}\ [{\rm M}_\odot]$')
                pl.ylim((1e9,1e10)[isRcirc],1e13)
                #pl.legend(ncol=2,fontsize=pB.infigfontsize,loc='lower left',handlelength=0.8,columnspacing=0.5,markerscale=2)
                #for isRcirc2 in True,:                    
                    #label=r'$t_{\rm cool}^{(s)}=2t_{\rm ff}$ ${\rm at}$ $%sR_{\rm vir}$'%(('','0.1 ')[isRcirc2])
                    #pl.plot([1e11],[1e9],ls='none',marker='+x'[isRcirc2],c=pl.get_cmap('coolwarm')(norm(0.3)),ms=6,mew=2,label=label)
                #pl.legend(fontsize=pB.infigfontsize,loc=(0.45,0.006),labelspacing=0.,handletextpad=0.,frameon=False)
                pl.text(2,5e12,r"m13's",fontsize=pB.infigfontsize,ha='center')
                pl.text(9,2e12,r"m12's",fontsize=pB.infigfontsize)
                pl.text(11,3.5e11,r"m11's",fontsize=pB.infigfontsize)
                
            if iPanel==1:
                pl.ylabel(r'$0.1 R_{\rm vir}\ [{\rm kpc}]$')
                pl.ylim(1,100) 
                
                axins = inset_axes(ax, width="100%", height="100%", loc='lower right',
                                   bbox_to_anchor=(0.5,0.175,0.5,0.05),bbox_transform=ax.transAxes)
                cbar = fig.colorbar(line,cax=axins,ticks=[-1,0,1],orientation='horizontal')
                axins.set_xticklabels([r'$0.1$',r'$1$', r'$10$'],fontsize=pB.infigfontsize)
                minorticks = log(np.concatenate([np.arange(0.03,0.1,0.01),
                                                 np.arange(0.2,1,0.1),
                                                 np.arange(2,10,1),
                                                 np.arange(20,40,10)]))
                axins.xaxis.set_ticks(minorticks, minor=True)        
                pl.text(0.5,-1.7,r'$t_{\rm cool}^{\rm (s)} / t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1')[isRcirc]),
                                 fontsize=pB.infigfontsize,va='top',ha='center',transform=axins.transAxes)
                axins.set_xlim(-1.5,1.5)
                pl.sca(ax)
                
            if iPanel==3:
                yls = (30,700)
                pl.ylabel(r'$v_{\rm c}(%sR_{\rm vir})\ [{\rm km}\ {\rm s}^{-1}]$'%Rstr)
                pl.ylim(*yls) 
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: ('',r'$%d$'%x)[x in (50,200,400,800)]))
                ax3 = pl.twinx()
                ax3.set_yscale('log')
                Tc_lims =((mu*cons.m_p*(un.km/un.s)**2/(gamma*cons.k_B)).to('K').value * np.array(yls)**2) 
                pl.ylim(Tc_lims)
                pl.ylabel(r'$T^{(s)}(%sR_{\rm vir})\ [{\rm K}]$'%Rstr)
                ax3.yaxis.set_major_locator(ticker.LogLocator(numticks=6,numdecs=6))
            if iPanel==2:
                pl.ylabel(r'$Z(%s{R_{\rm vir}})\ [{\rm Z}_\odot]$'%Rstr)
                if isRcirc: pl.ylim(0.02,3)
                else: pl.ylim(0.01,1)
            if iPanel==4:
                _zs = np.arange(0,15,.01)[::-1]                
                ages = cosmo.age(_zs)
                deltas = 10.**np.arange(-2,4)   
                for delta in deltas:
                    nHs = delta*(u.Delta_c(_zs)*cosmo.critical_density(_zs)/cons.m_p).to('cm**-3').value
                    pl.plot(ages,nHs,c='.5',ls='-',lw=0.5,zorder=-102)
                    if delta>=10:
                        x = 13                   
                        y = np.interp(x,ages,nHs)
                        if delta==1000:
                            y = 0.35
                            pl.text(x-0.7,y*1.2,r'$\rho_{\rm H}^{(s)}/(\Delta_{\rm c}\rho_{\rm crit})=$',ha='right',color='.3',va='center') 
                            
                    else:
                        x = 2
                        y = np.interp(x,ages,nHs)
                        if delta==1:
                            y = 0.01                        
                    if delta not in (1,1e3):
                        pl.text(x,y,r'%s'%u.arilogformat(delta),bbox=dict(facecolor='w',edgecolor='w'),
                            fontsize=pB.infigfontsize,color='.3',va='center',ha='center',zorder=-100)
                    else:
                        pl.text(x,y,r'%s'%u.arilogformat(delta),
                            fontsize=pB.infigfontsize,color='.3',va='center',ha='center',zorder=-100)
                        
                    
                pl.ylabel(r'$n_{\rm H}^{(s)}(%sR_{\rm vir})\ [{\rm cm}^{-3}]$'%Rstr)
                if isRcirc: pl.ylim(1e-4,1) 
                else: pl.ylim(1e-5,0.1)
                
                median_nHs = np.nanmedian(nHs_all,axis=0)
                _zs2 = 10.**log_t_ff_zs_p1[::-1]-1
                print('typical overdensities:')
                _zs3 = np.array([0,2,5])
                print(['z=%d: %d'%(_zs3[i],x) for i,x in enumerate(np.interp(_zs3, _zs2[::-1], 
                        median_nHs/((cosmo.critical_density(_zs2)/cons.m_p).to('cm**-3').value)))])
                    
            if iPanel==5:
                pl.ylabel(r'$t_{\rm cool}^{(s)}(%sR_{\rm vir})\ [{\rm Myr}]$'%Rstr)
                _zs = 10.**np.arange(0,1.5,.01)-1
                pl.plot(cosmo.age(_zs),cosmo.age(_zs).to('Myr'),c='.5',ls='-',lw=0.5,zorder=-102)
                median_tff = np.nanmedian(t_ffs,axis=0)
                pl.plot(cosmo.age(10.**log_t_ff_zs_p1-1),median_tff,c='k',ls=':',lw=1,zorder=-100)
                print('t_ff max / median:',np.nanmax(t_ffs/median_tff))
                print('t_ff median / min:',np.nanmax(median_tff/t_ffs))
                if isRcirc:
                    pl.annotate(r'$t_{\rm ff}$',(1.5,70),(0.8,100),ha='center',fontsize=pB.infigfontsize,va='bottom')                    
                    pl.annotate(r'',(1.2,40),(1.,100),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                    pl.ylim(1,1e4)
                else:
                    pl.annotate(r'$t_{\rm ff}$',(1.5,350),(0.8,500),ha='center',fontsize=pB.infigfontsize,va='bottom')                    
                    pl.annotate(r'',(1.2,200),(1.,500),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                    pl.ylim(10,1e5)
                #pl.annotate(r'$t_{\rm Hubble}$',(6.,1250),(7,2000),ha='center',fontsize=pB.infigfontsize,va='bottom')
                #pl.annotate(r'',(6.,1000),(7.,2000),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                
                
            pl.xlim(0,14.)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2.))               
            if ax.is_last_row():
                pl.xlabel(r'${\rm time\ [Gyr]}$')                
            else:
                ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_locator(ticker.LogLocator(numticks=7,numdecs=7))
            ax.yaxis.set_major_formatter(u.arilogformatter)
            
            if ax.is_first_row():
                ax2=pl.twiny()
                zs_to_show = np.array([0.,0.5,1,2,3,5])
                ages_to_show = cosmo.age(zs_to_show).value
                pl.xticks(ages_to_show,[u.nSignificantDigits(x,1,True) for x in zs_to_show])       
                pl.xlim(0.,cosmo.age(0.).value)
                pl.xlabel(r'${\rm redshift}$')  
        if savefig:
            pl.savefig(figDir + 'param_figs_by_time_at_R%s_withMvir_and_Rvir.pdf'%(('vir','circ')[isRcirc]),
                   bbox_inches='tight')    
        return cosmo.age(10.**log_t_ff_zs_p1-1),median_tff


    def param_figs_by_time_withMvirRcirc2(self,isRcirc,showOther=False,max_log_t_ratio=1.5,t_ratio_cross=2):
        fig = pl.figure(figsize=(pB.fig_width_full*1.2,3.5))
        pl.subplots_adjust(left=0.2,wspace=0.35,hspace=0.1,right=0.85,bottom=0.06)
        log_t_ff_zs_p1 = np.arange(0,1,.01)
        t_ffs = np.zeros((len(self.sims),log_t_ff_zs_p1.shape[0]))
        nHs_all = np.zeros((len(self.sims),log_t_ff_zs_p1.shape[0]))
        axs = []
        norm = pl.Normalize(-max_log_t_ratio, max_log_t_ratio)
        for iPanel in range(4):
            ax=pl.subplot(2,2,iPanel+1)
            axs.append(ax)
            for isim,sim in enumerate(self.sims):
                xs = cosmo.age(sim.zs())
                if iPanel==1:  
                    ys = 10.**self.smooth(log(sim.quantity_vs_z('vc',isRcirc)))
                if iPanel==0: 
                    ys = 10.**self.smooth(log(sim.quantity_vs_z('Z2Zsun',isRcirc)))
                    #ys2 = np.array([prof.atRcirc(prof.profile1D('Z2Zsuns','VW')) for prof in sim.profiles])                    
                    #pl.plot(1+sim.zs(),10.**sim.smooth(log(ys2)),lw=0.5,c=c,ls='--')
                if iPanel==2: 
                    #ys = 10.**self.smooth(log(sim.delta_Rcircs()))                    
                    nHs = sim.quantity_vs_z('nH',isRcirc)
                    nHs_all[isim,:] = np.interp(log_t_ff_zs_p1, log(1+sim.zs()), nHs,left=np.nan,right=np.nan)
                    ys = 10.**self.smooth(log(nHs))                                    
                if iPanel==3:
                    ys = 10.**self.smooth(log(sim.quantity_vs_z('t_cool',isRcirc)))
                    t_ff = 10.**self.smooth(log(sim.quantity_vs_z('t_ff',isRcirc)))
                    t_ffs[isim,:] = np.interp(log_t_ff_zs_p1, log(1+sim.zs()), t_ff,left=np.nan,right=np.nan)
                        
                log_t_ratio = self.smooth(log(sim.quantity_vs_z('t_cool',True)/sim.quantity_vs_z('t_ff',True)))                
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)            
                segments = np.concatenate([points[:-1], points[1:]], axis=1)                
                lc = matplotlib.collections.LineCollection(segments, cmap='coolwarm', norm=norm)
                lc.set_array(log_t_ratio)
                lc.set_linewidth(1.5)
                line = ax.add_collection(lc)
                
            pl.semilogy()
            Rstr = ('0.5','0.1 ')[isRcirc]
                
            if iPanel==1:
                yls = (30,300)
                pl.ylabel(r'$v_{\rm c}(%sR_{\rm vir})\ [{\rm km}\ {\rm s}^{-1}]$'%Rstr)
                pl.ylim(*yls) 
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: ('',r'$%d$'%x)[x in (50,200,400,800)]))
                
                axins = inset_axes(ax, width="100%", height="100%", loc='lower right',
                                   bbox_to_anchor=(0.5,0.25,0.5,0.075),bbox_transform=ax.transAxes)
                cbar = fig.colorbar(line,cax=axins,ticks=[-1,0,1],orientation='horizontal')
                axins.set_xticklabels([r'$0.1$',r'$1$', r'$10$'],fontsize=pB.infigfontsize)
                minorticks = log(np.concatenate([np.arange(0.03,0.1,0.01),
                                                 np.arange(0.2,1,0.1),
                                                 np.arange(2,10,1),
                                                 np.arange(20,40,10)]))
                axins.xaxis.set_ticks(minorticks, minor=True)        
                pl.text(0.5,-1.7,r'$t_{\rm cool}^{\rm (s)} / t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1')[isRcirc]),
                                 fontsize=pB.infigfontsize,va='top',ha='center',transform=axins.transAxes)
                axins.set_xlim(-1.5,1.5)
                pl.sca(ax)
                
            if iPanel==0:
                pl.ylabel(r'$Z(%s{R_{\rm vir}})\ [{\rm Z}_\odot]$'%Rstr)
                if isRcirc: pl.ylim(0.02,3)
                else: pl.ylim(0.01,1)
                pl.text(13.5,0.06,'low res',color='.3',ha='right')
                pl.text(13.5,0.3,'fiducial',color='.3',ha='right')
                pl.text(13.5,1.35,'no md',color='.3',ha='right')
                pl.text(0.05,0.925,'m12i',transform=ax.transAxes,va='top')
            if iPanel==2:
                _zs = np.arange(0,15,.01)[::-1]                
                ages = cosmo.age(_zs)
                deltas = 10.**np.arange(-2,4)   
                for delta in deltas:
                    nHs = delta*(u.Delta_c(_zs)*cosmo.critical_density(_zs)/cons.m_p).to('cm**-3').value
                    pl.plot(ages,nHs,c='.5',ls='-',lw=0.5,zorder=-102)
                    if delta>=10:
                        x = 13                   
                        y = np.interp(x,ages,nHs)
                        if delta==1000:
                            y = 0.35
                            pl.text(x-0.7,y*1.2,r'$\rho_{\rm H}^{(s)}/(\Delta_{\rm c}\rho_{\rm crit})=$',
                                    ha='right',color='.3',va='center') 
                            
                    else:
                        x = 2
                        y = np.interp(x,ages,nHs)
                        if delta==1:
                            y = 0.01                        
                    if delta not in (1,1e3):
                        pl.text(x,y,r'%s'%u.arilogformat(delta),bbox=dict(facecolor='w',edgecolor='w'),
                            fontsize=pB.infigfontsize,color='.3',va='center',ha='center',zorder=-100)
                    else:
                        pl.text(x,y,r'%s'%u.arilogformat(delta),
                            fontsize=pB.infigfontsize,color='.3',va='center',ha='center',zorder=-100)
                        
                    
                pl.ylabel(r'$n_{\rm H}^{(s)}(%sR_{\rm vir})\ [{\rm cm}^{-3}]$'%Rstr)
                if isRcirc: pl.ylim(1e-4,1) 
                else: pl.ylim(1e-5,0.1)
                
                median_nHs = np.nanmedian(nHs_all,axis=0)
                _zs2 = 10.**log_t_ff_zs_p1[::-1]-1
                print('typical overdensities:')
                _zs3 = np.array([0,2,5])
                print(['z=%d: %d'%(_zs3[i],x) for i,x in enumerate(np.interp(_zs3, _zs2[::-1], 
                        median_nHs/((cosmo.critical_density(_zs2)/cons.m_p).to('cm**-3').value)))])
                    
            if iPanel==3:
                pl.ylabel(r'$t_{\rm cool}^{(s)}(%sR_{\rm vir})\ [{\rm Myr}]$'%Rstr)
                _zs = 10.**np.arange(0,1.5,.01)-1
                pl.plot(cosmo.age(_zs),cosmo.age(_zs).to('Myr'),c='.5',ls='-',lw=0.5,zorder=-102)
                median_tff = np.nanmedian(t_ffs,axis=0)
                pl.plot(cosmo.age(10.**log_t_ff_zs_p1-1),median_tff,c='k',ls=':',lw=1,zorder=-100)
                print('t_ff max / median:',np.nanmax(t_ffs/median_tff))
                print('t_ff median / min:',np.nanmax(median_tff/t_ffs))
                if isRcirc:
                    pl.annotate(r'$t_{\rm ff}$',(1.5,70),(0.8,100),ha='center',fontsize=pB.infigfontsize,va='bottom')                    
                    pl.annotate(r'',(1.2,40),(1.,100),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                    pl.ylim(1,3e4)
                else:
                    pl.annotate(r'$t_{\rm ff}$',(1.5,350),(0.8,500),ha='center',fontsize=pB.infigfontsize,va='bottom')                    
                    pl.annotate(r'',(1.2,200),(1.,500),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                    pl.ylim(10,1e5)
                #pl.annotate(r'$t_{\rm Hubble}$',(6.,1250),(7,2000),ha='center',fontsize=pB.infigfontsize,va='bottom')
                #pl.annotate(r'',(6.,1000),(7.,2000),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                
                
            pl.xlim(0,14.)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2.))               
            if ax.is_last_row():
                pl.xlabel(r'${\rm time\ [Gyr]}$')                
            else:
                ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_locator(ticker.LogLocator(numticks=7,numdecs=7))
            ax.yaxis.set_major_formatter(u.arilogformatter)
            
            if ax.is_first_row():
                ax2=pl.twiny()
                zs_to_show = np.array([0.,0.5,1,2,3,5])
                ages_to_show = cosmo.age(zs_to_show).value
                pl.xticks(ages_to_show,[u.nSignificantDigits(x,1,True) for x in zs_to_show])       
                pl.xlim(0.,cosmo.age(0.).value)
                pl.xlabel(r'${\rm redshift}$')  
        
        pl.savefig(figDir + 'param_figs_by_time_appendix.pdf',bbox_inches='tight')    


    
    def param_comparison(self):
        fig = pl.figure(figsize=(pB.fig_width_full,6))
        pl.subplots_adjust(hspace=0.1,wspace=0.1)
        for iPanel in range(4):
            ax=pl.subplot(3,2,iPanel+1)
            for isim,sim in enumerate(self.sims):
                c=pB.cmap(isim*1./len(self.sims))
                xs = 10.**sim.smooth(log(sim.tcool_Rcircs()))/10.**sim.smooth(log(sim.tff_Rcircs()))
                xs = sim.mvirs()
                if iPanel==0:                     
                    ys = 10.**sim.smooth(log(sim.vc_Rcircs()))
                    norm = np.array([prof.atRvir(prof.vc()) for prof in sim.profiles])
                if iPanel==1: 
                    #ys = 10.**sim.smooth(log(sim.delta_Rcircs()))                    
                    nHs = np.array([prof.atRcirc(prof.rhoProfile())*X/cons.m_p.to('g').value for prof in sim.profiles])
                    ys = 10.**sim.smooth(log(nHs))                    
                    norm = 6000*X*cosmo.Ob(sim.zs())*(cosmo.critical_density(sim.zs())/cons.m_p).to('cm**-3').value
                if iPanel==2: 
                    ys = 10.**sim.smooth(log(sim.Z_Rcircs()))
                    Zism = np.array([prof.profile1D('Z2Zsuns','MW')[100] for prof in sim.profiles])
                    norm = 10.**sim.smooth(log(Zism))
                if iPanel==3:
                    ys = 10.**sim.smooth(log(sim.tcool_Rcircs()))
                    norm = 10.**sim.smooth(log(sim.tff_Rcircs()))
                pl.plot(xs,ys/norm,'.',lw=0.75,c=c,label=sim.shortname())
                #y = 10.**sim.at_z_cross(log(ys/norm))
                #pl.plot([1+sim.z_cross()],[y],marker='x',c=c,ms=8,mew=2,)
            pl.loglog()
            if iPanel==0:
                pl.ylabel(r'$v_{\rm c}(0.1 R_{\rm vir})\ [{\rm km}\ {\rm s}^{-1}]$')
                pl.ylim(0.5,2) 
                pl.legend(ncol=2,fontsize=pB.infigfontsize,loc='upper left',handlelength=0.8,columnspacing=0.5,
                          markerscale=2)
            if iPanel==1:                
                pl.annotate(r'$\delta_{\rm baryons}=10^4$',(2.1,0.03),(1.8,0.05),ha='center',va='bottom')
                pl.annotate(r'',(2.2,0.02),(1.8,0.05),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                pl.ylabel(r'$n_{\rm H}(0.1 R_{\rm vir})\ [{\rm cm}^{-3}]$')
                #pl.ylim(1e-4,10) 
            if iPanel==2:
                pl.ylabel(r'$Z(0.1 {R_{\rm vir}})\ [{\rm Z}_\odot]$')
                pl.ylim(0.1,2)
                
            if iPanel==3:
                pl.ylabel(r'$\langle t_{\rm cool}\rangle(0.1 R_{\rm vir})\ [{\rm Myr}]$')
                #pl.ylim(0.1,1e4)
                pl.annotate(r'$t_{\rm ff}$',(8.,50),(9.5,90),ha='center',fontsize=pB.infigfontsize,va='bottom')
                pl.annotate(r'',(8.,40),(9.,90),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
            pl.axhline(1.,c='k',ls=':')
                
                
            #pl.xlim(1,16.)
            #ax.xaxis.set_major_locator(ticker.FixedLocator([1,2,3,4,5,6,8,11,16]))
            #ax.xaxis.set_minor_locator(ticker.NullLocator())
            #ax.xaxis.set_major_formatter(ticker.NullFormatter())
            if iPanel%2==1:
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(ticker.LogLocator(numticks=6,numdecs=6))
            ax.yaxis.set_major_formatter(u.arilogformatter)
        pl.savefig(figDir + 'param_comparison.pdf',bbox_inches='tight')    

    def Ts_profile(self,weight):       
        pl.figure(figsize=(pB.fig_width_full,8))
        pl.subplots_adjust(hspace=0.08,wspace=0.04)
        for isim,sim in enumerate(self.sims):            
            #pl.title(sim)
            profs = np.array(sim.profiles)            
            log_t_ratio = log(sim.tcool_Rcircs()/sim.tff_Rcircs())
            log_mvirs = log(np.array([prof.mvir for prof in profs]))
            saveTs = []
            dlogM = 0.3
            mvirs_to_show = np.arange(11.551,12.9,dlogM)
            for imvir,mvir_to_show in enumerate(mvirs_to_show):
                ax = pl.subplot(len(self.sims),len(mvirs_to_show),1+imvir+isim*(len(mvirs_to_show)))
                if imvir == len(mvirs_to_show)-1:
                    for Ts,c,xs in saveTs:
                        pl.plot(xs,Ts,c=c) 
                else:
                    inds = (log_mvirs < mvir_to_show+dlogM/2.) & (log_mvirs > mvir_to_show-dlogM/2.)
                    if len(inds.nonzero()[0])!=0:
                        all_Ts = np.array([prof.profile1D('log_Ts',weight) for prof in profs[inds]])
                        Ts = np.nanmean((all_Ts.T).T,axis=0)#prof.smooth(prof.profile1D('log_Ts',weight),10)                                
                        var_Ts = np.array([prof.profile1D('log_Ts',weight,power=2) for prof in profs[inds]]) - all_Ts**2                                          
                        sig_Ts = np.nansum(var_Ts/var_Ts.shape[0],axis=0)**0.5                    
                        
                        prof = profs[inds][inds.nonzero()[0].shape[0]//2]
                        xs = prof.rs_midbins()/prof.rvir
                        c = matplotlib.cm.inferno((imvir)/len(mvirs_to_show))
                        
                        pl.plot(xs,Ts,c=c,label=(r'$\log\ M_h=%.2f$'%log(prof.mvir),'_nolegend_')[False])
                        saveTs.append((Ts,c,xs))
                        for j in -1,1:
                            pl.plot(xs,Ts+j*sig_Ts,c=c,lw=0.5)
                pl.xlim(0.05,2)
                pl.semilogx()
                pl.ylim(4,(6.8,7.7)[isim>=2])
                
                #pl.axvline(0.1,c='.5',lw=1.,ls=':')
                #pl.axvline(1,c='.5',lw=1.,ls=':')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                
                if ax.is_first_col() or ax.is_last_col():                     
                    pl.ylabel(r'$\langle\log\ T\rangle_{\rm V}$')
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                if ax.is_first_row():
                    if imvir == len(mvirs_to_show)-1:
                        pl.title('comparison',fontsize=8)
                    else:
                        pl.title(r'$\log M_{\rm h}=%.1f$'%mvir_to_show,fontsize=8)
                   
                if ax.is_last_col():
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.set_ticks_position('right')
                    ax.yaxis.set_ticks_position('both')
                else:
                    if len(inds.nonzero()[0])!=0:
                        pl.text(0.95,0.05,r'$z=%.1f$'%prof.z,ha='right',fontsize=8,transform=ax.transAxes)
                        pl.axhline(log(prof.Tvir()),c='.5',lw=1.,ls=':')
                if ax.is_first_col():
                    pl.text(0.05,0.85,sim.shortname(),fontsize=8,bbox=dict(edgecolor='k',facecolor='w',lw=0.5),transform=ax.transAxes)
                
                if ax.is_last_row(): 
                    pl.xlabel(r'$r/R_{\rm vir}$')
                    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: u.nSignificantDigits(x,1,True)))    
                    ax.xaxis.set_major_formatter(u.arilogformatter)    
                    
                    
                else:
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
                #pl.legend(ncol=2)
                if imvir==1 and isim==0:
                    pl.annotate(r'$T_{\rm vir}$',(1.,6.1),(0.8,6.3),ha='right',fontsize=pB.infigfontsize,va='bottom')
                    pl.annotate(r'',(1.,6.1),(0.8,6.3),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')
                    
        pl.savefig(figDir + 'Tprofiles.pdf',bbox_inches='tight')    
    def T_medians(self,weight,relative=False):       
        fig = pl.figure(figsize=(pB.fig_width_full,5))
        pl.subplots_adjust(hspace=0.4,wspace=0.04)
        for isim,sim in enumerate(self.sims):  
            #pl.title(sim)
            profs = np.array(sim.profiles)            
            log_t_ratio = log(sim.tcool_Rcircs()/sim.tff_Rcircs())
            log_mvirs = log(np.array([prof.mvir for prof in profs]))
            saveTs = []
            if sim.z_cross()!=None:
                log_Mthres = sim.at_z_cross(log_mvirs)
            else:
                log_Mthres = log(sim.mvirs()[0])
            dlogM = 0.2
            mvirs_to_show = log(np.array([0.1,1/3.,0.5,10.**-0.1,10.**0.1,2.,3.]))+log_Mthres
            for imvir,mvir_to_show in enumerate(mvirs_to_show[1:]):
                inds = (log_mvirs < mvir_to_show+dlogM/2.) & (log_mvirs > mvir_to_show-dlogM/2.)
                if len(inds.nonzero()[0])!=0:                    
                    prof = profs[inds][inds.nonzero()[0].shape[0]//2]
                    xs = prof.rs_midbins()/prof.rvir
                    c = matplotlib.cm.inferno(1-(imvir+1.)/len(mvirs_to_show))
                    log_Tvir = log(prof.Tvir())

                    all_Ts = np.array([prof.profile1D('log_Ts',weight) for prof in profs[inds]])
                    #all_Ts = log(np.array([prof.Tc() for prof in profs[inds]]))
                    if relative:
                        all_Ts -= log(np.array([prof.Tc() for prof in profs[inds]]))
                        
                    
                    Ts = np.nanmedian((all_Ts.T).T,axis=0)#prof.smooth(prof.profile1D('log_Ts',weight),10)                                
                    var_Ts = np.array([prof.profile1D('log_Ts',weight,power=2) for prof in profs[inds]]) - all_Ts**2                                          
                    sig_Ts = np.nansum(var_Ts/var_Ts.shape[0],axis=0)**0.5                    
                    
                    saveTs.append((Ts,c,xs,log_Tvir,mvir_to_show,sig_Ts))
            
            ax = pl.subplot(3,3,1+isim)
            
            for Ts,c,xs,log_Tvir,mvir_to_show,sig_Ts  in saveTs:
                pl.plot(xs,Ts ,c=c,label=r'$%.1f M_{\rm thres}$'%(10.**(mvir_to_show-log_Mthres)))
                pl.axhline((log_Tvir,0)[relative],c=c,ls=':')
                #pl.plot(xs,sig_Ts,c=c,label='%.1f'%mvir_to_show) 
            pl.xlim(0.05,2)
            pl.semilogx()
            if not relative:
                if isim<3: pl.ylim(4,6.65)
                else: pl.ylim(4.5,7.15)
            else:
                pl.ylim(-1,0.1)
            #pl.ylim(0,1)
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
            ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            
            if not relative and isim==0:
                pl.annotate(r'$T_{\rm vir}$',(0.8,6.),(1,6.2),ha='left',fontsize=pB.infigfontsize,va='bottom')
                pl.annotate(r'',(0.8,6.),(1,6.2),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')                
            if isim==4: 
                u.mylegend(handlelength=1.,ncol=7,columnspacing=0.5,bbox_transform=fig.transFigure,fontsize=pB.infigfontsize+3,
                       loc='lower center',numpoints=3,handletextpad=0.1,bbox_to_anchor=(0.5,-0.05))            
            if ax.is_first_col() or ax.is_last_col():                     
                if not relative:
                    pl.ylabel(r'$\langle\log\ T\rangle_{\rm V}$')
                else:
                    pl.ylabel(r'$\langle\log\ T/T_{\rm c}\rangle_{\rm V}$')
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            if ax.is_last_col():
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
            if sim.z_cross()!=None:
                pl.title(sim.shortname()+
                         r', $M_{\rm thres}=%.1f$'%log_Mthres+
                         r', $z_{\rm thres}=%.1f$'%sim.z_cross(), fontsize=pB.infigfontsize,transform=ax.transAxes) #bbox=dict(edgecolor='k',facecolor='w',lw=0.5),
            
            if ax.is_last_row(): 
                pl.xlabel(r'$r/R_{\rm vir}$')
                #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: u.nSignificantDigits(x,1,True)))    
            ax.xaxis.set_major_formatter(u.arilogformatter)    
            #else:
                #ax.xaxis.set_major_formatter(ticker.NullFormatter())
            
        pl.savefig(figDir + 'Tprofiles2%s.pdf'%('','_relative')[relative],bbox_inches='tight')    

    def T_medians_single_redshift(self,weight,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5): 
        fig = pl.figure(figsize=(pB.fig_width_half,7.5))
        pl.subplots_adjust(hspace=0.3,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        axs = [pl.subplot(3,len(zs),i+1) for i in range(3*len(zs))]
        log_t_ratios = np.ones(len(self.sims))*np.nan
        for iRow in range(3):
            for iz,z in enumerate(zs):                
                ax = axs[len(zs)*iRow+iz]; pl.sca(ax)
                for isim,sim in enumerate(self.sims):  
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    log_t_ratios[isim] = log_t_ratio
                    if iRow==0: print('%d %s %.2f'%(isim,sim,log_t_ratio))
                    if abs(log_t_ratio)>max_t_ratio: continue
                    profs = np.array(sim.profiles)[inds]
                    normed_ratio = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratio)
                    xs = np.nanmedian([prof.rs_midbins()/prof.rvir for prof in profs],axis=0)
                    if iRow<2:                        
                        all_logTs = np.array([prof.thickerShells(prof.profile1D('log_Ts',weight),5,weight)
                                              for prof in profs])

                        log_Tvir = np.array([log(prof.Tvir()) for prof in profs])
                            
                        if iRow==1:
                            all_logTs = (all_logTs.T - log_Tvir).T
                        median_Tprofile = np.nanmedian(all_logTs,axis=0)                                                            
                        pl.plot(self.desample(xs),10.**self.desample(median_Tprofile),c=c,lw=1)
                    if iRow==2:    
                        fs = np.array([prof.thickerShells(prof.profile1D('isSubsonic',weight),5,weight)
                                                                      for prof in profs])                        
                        f = np.nanmedian(fs,axis=0)                    
                        pl.plot(xs,1-f,c=c,lw=1.)
                
                if iRow<2:                    
                    pl.loglog()
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    if iRow==0: 
                        if z==0: 
                            pl.title(r"all sims, $z=0$",fontsize=pB.infigfontsize+2)
                            mvirs = np.array([sim.mvirs()[0] for sim in self.sims])/1e12
                            min_Mhalo,max_Mhalo = [f(mvirs[log_t_ratios>log(1.5)]) for f in (min,max)]
                            print(min_Mhalo,max_Mhalo)
                            pl.text(0.3,1.15e6,r'$M_{12} = %.1f-%.1f$'%(min_Mhalo,max_Mhalo),color=pB.cmap(0.75),fontsize=pB.infigfontsize)
                            min_Mhalo,max_Mhalo = [f(mvirs[log_t_ratios<log(1.5)]) for f in (min,max)]
                            pl.text(0.17,2.e4,r'$M_{12} = %.2f-%.1f$'%(min_Mhalo,max_Mhalo),color=pB.cmap(0.25),fontsize=pB.infigfontsize)
                        else:
                            pl.text(0.5,1.05,r'$z=%d$'%z,transform=ax.transAxes,ha='center')
                        pl.ylim(10.**4.,10.**(7.,6.5)[z==0])
                        ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
                        ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=5,numticks=5,subs=range(2,10)))
                        ylabel = r'$\langle T\rangle$'
                        
                    if iRow==1:                    
                        pl.axhline(1.,c='k',lw=0.5)
                        pl.ylim(0.1,3.)                                
                        ylabel = r'$\langle T\rangle/T_{\rm vir}$'
                if iRow==2:    
                    pl.semilogx()
                    pl.ylim(0.,1.)        
                    pl.axhline(0.5,c='k',lw=0.5)
                    ylabel = r'supersonic fraction'
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: r'$%.1f$'%x))
                if ax.is_first_col():
                    pl.ylabel(ylabel)  
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_formatter(u.arilogformatter)    
                pl.xlim(0.03,3)                
                if ax.is_last_row():                    
                    pl.xlabel(r'$r/R_{\rm vir}$')
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=1,aspect=30,
                           pad=0.1)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        minorticks = log(np.concatenate([np.arange(0.03,0.1,0.01),
                                         np.arange(0.2,1,0.1),
                                         np.arange(2,10,1),
                                         np.arange(20,40,10)]))
        cbar.ax.xaxis.set_ticks(minorticks, minor=True)        
        
        pl.savefig(figDir + 'Tprofiles_z%d.pdf'%z,bbox_inches='tight')    
    def T_medians_single_sim(self,weight,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5,log_t_ratio_snapshot_range=None): 
        fig = pl.figure(figsize=(pB.fig_width_half,7.5))
        pl.subplots_adjust(hspace=0.3,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        axs = [pl.subplot(3,len(self.sims),i+1) for i in range(3*len(self.sims))]        
        for iRow in range(3):
            for isim,sim in enumerate(self.sims):  
                normed_ratios = np.ones(len(zs))*np.nan
                ax = axs[len(self.sims)*iRow+isim]; pl.sca(ax)
                for iz,z in enumerate(zs):  
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    if log_t_ratio_snapshot_range!=None and (log_t_ratio<log_t_ratio_snapshot_range[0] or log_t_ratio>log_t_ratio_snapshot_range[1]): continue                         
                    normed_ratios[iz] = log_t_ratio
                    profs = np.array(sim.profiles)[inds]
                    if iRow==0: print('%d %.2f %.2f %.2g'%(iz,z,log_t_ratio,profs[0].mvir))                           
                    if abs(log_t_ratio)>max_t_ratio: continue
                    normed_ratios[iz] = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratios[iz])
                    xs = np.nanmedian([prof.rs_midbins()/prof.rvir for prof in profs],axis=0)
                    if iRow<2:                        
                        all_logTs = np.array([prof.thickerShells(prof.profile1D('log_Ts',weight),5,weight)
                                              for prof in profs])
                        log_Tvir = np.array([log(prof.Tvir()) for prof in profs])
                        if iRow==1:
                            all_logTs = (all_logTs.T - log_Tvir).T
                        median_Tprofile = np.nanmedian(all_logTs,axis=0)                                                            
                        pl.plot(xs,10.**median_Tprofile,c=c,lw=1.)
                    if iRow==2:    
                        fs = np.array([prof.thickerShells(prof.profile1D('isSubsonic',weight),5,weight)
                                                                      for prof in profs])                        
                        #self.profile1D_multiple(['log_Ts',''], 'VW')
                        #fs = np.nan*np.ones((len(profs),len(xs)))
                        #good_profs = [prof for prof in profs if prof.isSaved('subsonic_fraction_%s'%weight)]
                        #if len(good_profs)==0: continue
                        #for iprof,prof in enumerate(good_profs): 
                            #fs[iprof,:] = prof.subsonic_fraction(weight)
                        f = np.nanmedian(fs,axis=0)                    
                        pl.plot(xs,1-f,c=c,lw=1.)

                
                if iRow<2:                    
                    pl.loglog()
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    if iRow==0: 
                        if sim.shortname()=='m12i': 
                            pl.title(r'm12i, $0<z<1$, $M_{\rm halo}\approx10^{12}{\rm M}_\odot$',fontsize=pB.infigfontsize+2)
                        else: pl.title(sim.shortname(),fontsize=pB.infigfontsize+2)
                        if zs[0]<1:
                            pl.ylim(10.**4.5,10.**6.3)
                            pl.text(1.8,6e5,r'$z=%d$'%u.iround(zs[-1]),
                                    color=pl.get_cmap('coolwarm')(normed_ratios[len(zs)-1]),
                                    ha='right',fontsize=pB.infigfontsize)
                            pl.text(1.8,5e4,r'$z=%d$'%zs[0],
                                    color=pl.get_cmap('coolwarm')(normed_ratios[0]),
                                    ha='right',fontsize=pB.infigfontsize)
                            #pl.text(1.5,5e5,r'$z=%d$'%u.iround(zs[-1]),color=pB.cmap(normed_ratios[len(zs)-1]),ha='right',fontsize=pB.infigfontsize)
                            #pl.text(1.5,6e4,r'$z=%d$'%zs[0],color=pB.cmap(normed_ratios[0]),ha='right',fontsize=pB.infigfontsize)
                        else:
                            pl.ylim(10.**5,2e7)
                            if sim.galaxyname=='m12i':                                
                                pl.text(0.6,5e6,r'$z=%d$'%zs[~np.isnan(normed_ratios)][0],color=pl.get_cmap('coolwarm')([~np.isnan(normed_ratios)][0]),ha='right')
                                pl.text(0.6,4e5,r'$z=%d$'%u.iround(zs[~np.isnan(normed_ratios)][-1]),color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1]),ha='right')                                 
                            if sim.galaxyname=='h206':                                
                                pl.text(0.4,7e6,r'$z=%.1f$'%zs[~np.isnan(normed_ratios)][0],color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0]),ha='right')
                                pl.text(0.6,3e5,r'$z=%.1f$'%(zs[~np.isnan(normed_ratios)][-1]),color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1]),ha='right')
                            if sim.galaxyname=='h2':                                
                                pl.text(1,7.5e6,r'$z=%.1f$'%zs[~np.isnan(normed_ratios)][0],color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0]),ha='right')
                                pl.text(0.7,3e5,r'$z=%.1f$'%zs[~np.isnan(normed_ratios)][-1],color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1]),ha='right')
                        ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
                        ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=5,numticks=5,subs=range(2,10)))
                        ylabel = r'$\langle T\rangle$'
                        
                    if iRow==1:                    
                        pl.axhline(1.,c='k',lw=0.5)
                        if zs[0]<1: pl.ylim(0.05,3.)
                        else: pl.ylim(0.05,4)
                        ylabel = r'$\langle T\rangle/T_{\rm vir}$'
                if iRow==2:    
                    pl.semilogx()
                    pl.ylim(0.,1.)        
                    pl.axhline(0.5,c='k',lw=0.5)
                    ylabel = r'supersonic fraction'
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: r'$%.1f$'%x))
                if ax.is_first_col():
                    pl.ylabel(ylabel)  
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_formatter(u.arilogformatter)    
                pl.xlim((0.03,0.005)[zs[0]>=1],3)
                if ax.is_last_row():                    
                    pl.xlabel(r'$r/R_{\rm vir}$')
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=1,aspect=30,pad=0.1)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        minorticks = log(np.concatenate([np.arange(0.03,0.1,0.01),
                                         np.arange(0.2,1,0.1),
                                         np.arange(2,10,1),
                                         np.arange(20,40,10)]))
        cbar.ax.xaxis.set_ticks(minorticks, minor=True)        
        
        
        pl.savefig(figDir + 'Tprofiles_%s.pdf'%sim,bbox_inches='tight')    
    def T_medians_single_sim_physical_radius(self,weight,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5,log_t_ratio_snapshot_range=None): 
        fig = pl.figure(figsize=(pB.fig_width_half,7.5))
        pl.subplots_adjust(hspace=0.3,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        axs = [pl.subplot(3,len(self.sims),i+1) for i in range(3*len(self.sims))]        
        for iRow in range(3):
            for isim,sim in enumerate(self.sims):  
                normed_ratios = np.ones(len(zs))*np.nan
                ax = axs[len(self.sims)*iRow+isim]; pl.sca(ax)
                for iz,z in enumerate(zs):  
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    if log_t_ratio_snapshot_range!=None and (log_t_ratio<log_t_ratio_snapshot_range[0] or log_t_ratio>log_t_ratio_snapshot_range[1]): continue                         
                    normed_ratios[iz] = log_t_ratio
                    profs = np.array(sim.profiles)[inds]
                    if iRow==0: print('%d %.2f %.2f %.2g'%(iz,z,log_t_ratio,profs[0].mvir))                           
                    if abs(log_t_ratio)>max_t_ratio: continue
                    normed_ratios[iz] = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratios[iz])
                    xs = np.nanmedian([prof.rs_midbins() for prof in profs],axis=0)
                    if iRow<2:                        
                        all_logTs = np.array([prof.thickerShells(prof.profile1D('log_Ts',weight),5,weight)
                                              for prof in profs])
                        log_Tvir = np.array([log(prof.Tvir()) for prof in profs])
                        if iRow==1:
                            all_logTs = (all_logTs.T - log_Tvir).T
                        median_Tprofile = np.nanmedian(all_logTs,axis=0)                                                            
                        pl.plot(xs,10.**median_Tprofile,c=c,lw=1.)
                    if iRow==2:    
                        fs = np.array([prof.thickerShells(prof.profile1D('isSubsonic',weight),5,weight)
                                                                      for prof in profs])                        
                        #self.profile1D_multiple(['log_Ts',''], 'VW')
                        #fs = np.nan*np.ones((len(profs),len(xs)))
                        #good_profs = [prof for prof in profs if prof.isSaved('subsonic_fraction_%s'%weight)]
                        #if len(good_profs)==0: continue
                        #for iprof,prof in enumerate(good_profs): 
                            #fs[iprof,:] = prof.subsonic_fraction(weight)
                        f = np.nanmedian(fs,axis=0)                    
                        pl.plot(xs,1-f,c=c,lw=1.)

                
                if iRow<2:                    
                    pl.loglog()
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    if iRow==0: 
                        if sim.shortname()=='m12i': 
                            pl.title(r'm12i, $0<z<1$, $M_{\rm vir}\approx10^{12}{\rm M}_\odot$',fontsize=pB.infigfontsize+2)
                        else: pl.title(sim.shortname(),fontsize=pB.infigfontsize+2)
                        if zs[0]<1:
                            pl.ylim(10.**4.5,10.**6.3)
                            pl.text(1.8,6e5,r'$z=%d$'%u.iround(zs[-1]),
                                    color=pl.get_cmap('coolwarm')(normed_ratios[len(zs)-1]),
                                    ha='right',fontsize=pB.infigfontsize)
                            pl.text(1.8,5e4,r'$z=%d$'%zs[0],
                                    color=pl.get_cmap('coolwarm')(normed_ratios[0]),
                                    ha='right',fontsize=pB.infigfontsize)
                            #pl.text(1.5,5e5,r'$z=%d$'%u.iround(zs[-1]),color=pB.cmap(normed_ratios[len(zs)-1]),ha='right',fontsize=pB.infigfontsize)
                            #pl.text(1.5,6e4,r'$z=%d$'%zs[0],color=pB.cmap(normed_ratios[0]),ha='right',fontsize=pB.infigfontsize)
                        else:
                            pl.ylim(10.**5,2e7)
                            if sim.galaxyname=='m12i':                                
                                pl.text(0.6,5e6,r'$z=%d$'%zs[~np.isnan(normed_ratios)][0],color=pl.get_cmap('coolwarm')([~np.isnan(normed_ratios)][0]),ha='right')
                                pl.text(0.6,4e5,r'$z=%d$'%u.iround(zs[~np.isnan(normed_ratios)][-1]),color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1]),ha='right')                                 
                            if sim.galaxyname=='h206':                                
                                pl.text(0.4,7e6,r'$z=%.1f$'%zs[~np.isnan(normed_ratios)][0],color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0]),ha='right')
                                pl.text(0.6,3e5,r'$z=%.1f$'%(zs[~np.isnan(normed_ratios)][-1]),color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1]),ha='right')
                            if sim.galaxyname=='h2':                                
                                pl.text(1,7.5e6,r'$z=%.1f$'%zs[~np.isnan(normed_ratios)][0],color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0]),ha='right')
                                pl.text(0.7,3e5,r'$z=%.1f$'%zs[~np.isnan(normed_ratios)][-1],color=pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1]),ha='right')
                        ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
                        ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=5,numticks=5,subs=range(2,10)))
                        ylabel = r'$\langle T\rangle$'
                        
                    if iRow==1:                    
                        pl.axhline(1.,c='k',lw=0.5)
                        if zs[0]<1: pl.ylim(0.05,3.)
                        else: pl.ylim(0.05,4)
                        ylabel = r'$\langle T\rangle/T_{\rm vir}$'
                if iRow==2:    
                    pl.semilogx()
                    pl.ylim(0.,1.)        
                    pl.axhline(0.5,c='k',lw=0.5)
                    ylabel = r'supersonic fraction'
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: r'$%.1f$'%x))
                if ax.is_first_col():
                    pl.ylabel(ylabel)  
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_formatter(u.arilogformatter)    
                pl.xlim(0.03*300,600)                
                if ax.is_last_row():                    
                    pl.xlabel(r'$r$')
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=1,aspect=30,pad=0.1)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        minorticks = log(np.concatenate([np.arange(0.03,0.1,0.01),
                                         np.arange(0.2,1,0.1),
                                         np.arange(2,10,1),
                                         np.arange(20,40,10)]))
        cbar.ax.xaxis.set_ticks(minorticks, minor=True)        
        
        
        pl.savefig(figDir + 'Tprofiles_%s_physical_radius.pdf'%sim,bbox_inches='tight')    
    def nHT_medians_single_sim(self,weight,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5,log_t_ratio_snapshot_range=None): 
        fig = pl.figure(figsize=(pB.fig_width_half,6))
        pl.subplots_adjust(hspace=0.3,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        axs = [pl.subplot(1,len(self.sims),i+1) for i in range(1*len(self.sims))]        
        for iRow in range(1):
            for isim,sim in enumerate(self.sims):  
                normed_ratios = np.ones(len(zs))*np.nan
                ax = axs[len(self.sims)*iRow+isim]; pl.sca(ax)
                for iz,z in enumerate(zs):  
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    if log_t_ratio_snapshot_range!=None and (log_t_ratio<log_t_ratio_snapshot_range[0] or log_t_ratio>log_t_ratio_snapshot_range[1]): continue                         
                    normed_ratios[iz] = log_t_ratio
                    profs = np.array(sim.profiles)[inds]
                    if abs(log_t_ratio)>max_t_ratio: continue
                    normed_ratios[iz] = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratios[iz])
                    
                    
                    all_nHTs = 10.**np.array([prof.thickerShells(prof.profile1D('log_nHTs',weight),5,weight)
                                          for prof in profs])
                    xs = np.nanmedian([prof.rs_midbins()/prof.rvir for prof in profs],axis=0)
                    nHT_Rvir = 10.**np.array([prof.thickerShells(prof.profile1D('log_nHTs',weight),5,weight)[300]
                                      for prof in profs])
                    all_nHTs = (all_nHTs.T / nHT_Rvir).T
                    median_nHTprofile = np.nanmedian(all_nHTs,axis=0)
                    pl.plot(xs,median_nHTprofile,c=c,lw=1.,zorder=100)

                    rs = np.nanmedian([prof.rs_midbins() for prof in profs],axis=0)
                    drs = np.nanmedian([prof.drs_midbins() for prof in profs],axis=0)
                    rhos = np.nanmedian(np.array([prof.rhoProfile()for prof in profs]),axis=0)
                    vcs = np.nanmedian(np.array([prof.vc() for prof in profs]),axis=0)
                    nHT_Rvir_median = np.nanmedian(nHT_Rvir,axis=0)
                    P_grav_offset = (rhos*vcs**2 / rs * drs)[:300][::-1].cumsum()[::-1]
                    nHT_grav_offset = P_grav_offset * (un.g*un.cm**-3*un.km**2/un.s**2 / (2.3*cons.k_B)).to(un.cm**-3*un.K).value
                    nHT_grav = 1+nHT_grav_offset/nHT_Rvir_median
                    if iz==0: pl.plot(xs[:300],nHT_grav,ls='--',lw=3,c='k',zorder=-100,label=r'${\rm HSE}$')
                        

                
                pl.loglog()
                ax.yaxis.set_major_formatter(u.arilogformatter)
                    
                pl.axhline(1.,c='k',lw=0.5)
                pl.ylim(0.5,2000)
                ylabel = r'$P(r) / P(R_{\rm vir})$'
                if ax.is_first_col():
                    pl.ylabel(ylabel)  
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_formatter(u.arilogformatter)    
                pl.xlim(0.05,2)
                if ax.is_last_row():                    
                    pl.xlabel(r'$r/R_{\rm vir}$')
                pl.legend(handlelength=3)
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=1,aspect=30,pad=0.1)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        minorticks = log(np.concatenate([np.arange(0.03,0.1,0.01),
                                         np.arange(0.2,1,0.1),
                                         np.arange(2,10,1),
                                         np.arange(20,40,10)]))
        cbar.ax.xaxis.set_ticks(minorticks, minor=True)        
        
        
        pl.savefig(figDir + 'nHTprofiles_%s.pdf'%sim,bbox_inches='tight')    
    def j_single_redshift(self,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5): 
        fig = pl.figure(figsize=(pB.fig_width_half,6))
        pl.subplots_adjust(hspace=0.3,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        axs = [pl.subplot(2,len(zs),i+1) for i in range(2*len(zs))]
        log_t_ratios = np.ones(len(self.sims))*np.nan
        for iRow in range(2):
            for iz,z in enumerate(zs):                
                ax = axs[len(zs)*iRow+iz]; pl.sca(ax)
                for isim,sim in enumerate(self.sims):  
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    log_t_ratios[isim] = log_t_ratio
                    if abs(log_t_ratio)>max_t_ratio: continue
                    profs = np.array(sim.profiles)[inds]
                    normed_ratio = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratio)
                    xs = profs[0].rs_midbins()/profs[0].rvir
                    if iRow==0:                        
                        all_js = np.array([prof.jProfile() for prof in profs])
                        median_jProfile = np.nanmedian(all_js,axis=0)            
                        j_Keplerian = np.array([prof.vc()*prof.rs_midbins() for prof in profs])
                        median_jKep = np.nanmedian(j_Keplerian,axis=0)            
                        pl.plot(xs,median_jProfile,c=c,lw=1)
                        pl.plot(xs,median_jKep,c=c,ls=':',lw=0.75)            
                    if iRow==1:                        
                        circularity = np.array([prof.circularityProfile('MW') for prof in profs if prof.isSaved('circularity_MW')])
                        if len(circularity)==0: continue
                        median_circularity = np.nanmedian(circularity,axis=0)            
                        pl.plot(xs,median_circularity,c=c,lw=1)
                
                
                if iRow==0: 
                    pl.loglog()
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                    ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=8,numticks=8,subs=range(2,10)))
                    pl.text(0.5,1.05,r'$z=%d$'%z,transform=ax.transAxes,ha='center')
                    if z==0: 
                        #mvirs = np.array([log(sim.mvirs()[0]) for sim in self.sims])
                        #min_lMhalo,max_lMhalo = [f(mvirs[log_t_ratios>log(1.5)]) for f in (min,max)]
                        #pl.text(0.006,2000,r'$10^{%.1f}-10^{%.1f}\,{\rm M}_\odot$'%(min_lMhalo,max_lMhalo),color=pB.cmap(0.75))
                        #min_lMhalo,max_lMhalo = [f(mvirs[log_t_ratios<log(1.5)]) for f in (min,max)]
                        #pl.text(0.08,50,r'$10^{%.1f}-10^{%.1f}\,{\rm M}_\odot$'%(min_lMhalo,max_lMhalo),color=pB.cmap(0.25))
                        pl.ylim(10,1e5)                 
                    else:
                        pl.ylim(1,1e5)
                    ylabel = r'$\left|\Sigma \vec{j}\right|\ [{\rm kpc\ km\ s^{-1}}]$'
                    
                if iRow==1:         
                    pl.semilogx()
                    #pl.ylabel(r'$\left|\Sigma \vec{j}\right| / v_{\rm c} r$')
                    ylabel = r'circularity $j_z/j_{\rm c}(E)$'
                    pl.ylim(0.,1.05)                 
                if ax.is_first_col():
                    pl.ylabel(ylabel)  
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_formatter(u.arilogformatter)    
                if z==0:                    
                    pl.xlim(0.005,2)                
                else:
                    pl.xlim(0.001,2)                
                if ax.is_last_row():                    
                    pl.xlabel(r'$r/R_{\rm vir}$')
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=0.75,aspect=50,
                           pad=0.125)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + 'jProfiles_z%d.pdf'%z,bbox_inches='tight')    
        
    
    def j_single_sim(self,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5,log_t_ratio_snapshot_range=None): 
        fig = pl.figure(figsize=(pB.fig_width_full*1.1,6.5))
        pl.subplots_adjust(hspace=0.1,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        axs = [pl.subplot(2,len(self.sims),i+1) for i in range(2*len(self.sims))]        
        for iRow in range(2):
            for isim,sim in enumerate(self.sims):  
                ax = axs[len(self.sims)*iRow+isim]; pl.sca(ax)
                normed_ratios = np.ones(len(zs))*np.nan
                for iz,z in enumerate(zs):  
                    if iRow==0 and isim<2 and iz%2==1: continue
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    if log_t_ratio_snapshot_range!=None and (log_t_ratio<log_t_ratio_snapshot_range[0] or log_t_ratio>log_t_ratio_snapshot_range[1]): continue                         
                    normed_ratios[iz] = log_t_ratio
                    if abs(log_t_ratio)>max_t_ratio: continue
                    #if iRow==0: print('%d %.2f %.2f'%(iz,z,log_t_ratio))
                    profs = np.array(sim.profiles)[inds]
                    normed_ratios[iz] = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratios[iz])
                    xs = profs[0].rs_midbins()/profs[0].rvir
                    if iRow==0:                        
                        all_js = np.array([prof.jProfile() for prof in profs])
                        median_jProfile = self.smooth(np.nanmedian(all_js,axis=0),N=10)
                        j_Keplerian = np.array([prof.vc()*prof.rs_midbins() for prof in profs])
                        median_jKep = np.nanmedian(j_Keplerian,axis=0)            
                        pl.plot(xs,median_jProfile,c=c,lw=1)
                        pl.plot(xs,median_jKep,c=c,ls='--',lw=0.5)
                    if iRow==1:                        
                        circularity = np.array([prof.circularityProfile('MW') for prof in profs if prof.isSaved('circularity_MW')])
                        if len(circularity)==0: continue
                        median_circularity = np.nanmedian(circularity,axis=0)            
                        pl.plot(xs,median_circularity,c=c,lw=1)

                
                if iRow==0:                    
                    pl.loglog()
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                    ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=8,numticks=8,subs=range(2,10)))
                    pl.text(0.5,1.05,sim.shortname(),transform=ax.transAxes,ha='center')
                    #if not ax.is_first_col():
                        #pl.text(0.075,0.85,r'$%d < z < %.1f$'%(zs[~np.isnan(normed_ratios)][0],zs[~np.isnan(normed_ratios)][-1]),
                                #transform=ax.transAxes,fontsize=pB.infigfontsize)
                    if sim.shortname()=='m12i':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.025,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.1,200,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                    if sim.shortname()=='m12b':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.02,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(1,200,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c,ha='right')
                    if sim.shortname()=='m13A1':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.005,1000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.2,160,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                        
                    
                    pl.ylim(10.,0.2e5)
                    if ax.is_first_col():
                        pl.annotate(r'',(0.15,0.8e4),(0.075,0.8e4),arrowprops=pB.slantlinepropsblack,ha='right',va='bottom')
                        pl.text(0.07,0.825e4,r'$j_{\rm c}(r)$',ha='right',va='center')
                        #pl.text(1.5,6e5,r'$z=%d$'%zs[-1],color=pB.cmap(normed_ratios[len(zs)-1]),ha='right')
                        #pl.text(1.5,6e4,r'$z=%d$'%zs[0],color=pB.cmap(normed_ratios[0]),ha='right')
                        #pl.text(0.6,4e6,r'$z=%d$'%zs[0],color=pB.cmap(normed_ratios[0]),ha='right')
                        #pl.text(0.6,6e5,r'$z=%d$'%u.iround(zs[-1]),color=pB.cmap(normed_ratios[len(zs)-1]),ha='right')
                    ylabel = r'$\left|\vec{j}\right|\ [{\rm kpc\ km\ s^{-1}}]$'                        
                if iRow==1:                    
                    pl.semilogx()
                    #pl.ylabel(r'$\left|\Sigma \vec{j}\right| / v_{\rm c} r$')
                    ylabel = r'circularity $j_z/j_{\rm c}(E)$'
                    pl.ylim(0.,1.05)                
                pl.xlim(0.001,2)                
                if ax.is_first_col():
                    pl.ylabel(ylabel)  
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                if ax.is_last_row():                    
                    ax.xaxis.set_major_formatter(u.arilogformatter)                    
                    pl.xlabel(r'$r/R_{\rm vir}$')
                else:
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
                    
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=0.75,aspect=50,
                           pad=0.125)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + 'jProfiles_%s.pdf'%sim,bbox_inches='tight')    
    def j_single_sim2(self,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5,log_t_ratio_snapshot_range=None): 
        fig = pl.figure(figsize=(pB.fig_width_full*1.1,12))
        pl.subplots_adjust(hspace=0.1,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        axs = [pl.subplot(5,len(self.sims),i+1) for i in range(5*len(self.sims))]        
        for iRow in range(5):
            for isim,sim in enumerate(self.sims):  
                ax = axs[len(self.sims)*iRow+isim]; pl.sca(ax)
                normed_ratios = np.ones(len(zs))*np.nan
                for iz,z in enumerate(zs):  
                    if iRow==0 and isim<2 and iz%2==1: continue
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    if log_t_ratio_snapshot_range!=None and (log_t_ratio<log_t_ratio_snapshot_range[0] or log_t_ratio>log_t_ratio_snapshot_range[1]): continue                         
                    normed_ratios[iz] = log_t_ratio
                    if abs(log_t_ratio)>max_t_ratio: continue
                    #if iRow==0: print('%d %.2f %.2f'%(iz,z,log_t_ratio))
                    profs = np.array(sim.profiles)[inds]
                    normed_ratios[iz] = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratios[iz])
                    xs = profs[0].rs_midbins()/profs[0].rvir
                    if iRow==0:                        
                        all_js = np.array([prof.jProfile() for prof in profs])
                        median_jProfile = self.smooth(np.nanmedian(all_js,axis=0),N=10)
                        j_Keplerian = np.array([prof.vc()*prof.rs_midbins() for prof in profs])
                        median_jKep = np.nanmedian(j_Keplerian,axis=0)            
                        pl.plot(xs,median_jProfile,c=c,lw=1)
                        pl.plot(xs,median_jKep,c=c,ls='--',lw=0.5)
                    if iRow==1:                        
                        all_js = np.array([prof.jProfile() for prof in profs])
                        median_jProfile = self.smooth(np.nanmedian(all_js,axis=0),N=10)
                        j_Keplerian = np.array([prof.vc()*prof.rs_midbins() for prof in profs])
                        median_jKep = np.nanmedian(j_Keplerian,axis=0)            
                        pl.plot(xs,median_jProfile/median_jKep,c=c,lw=1)
                    if iRow==2:                        
                        all_js = np.array([prof.jzProfile() for prof in profs])
                        median_jzProfile = self.smooth(np.nanmedian(all_js,axis=0),N=10)
                        j_Keplerian = np.array([prof.vc()*prof.rs_midbins() for prof in profs])
                        median_jKep = np.nanmedian(j_Keplerian,axis=0)            
                        pl.plot(xs,median_jzProfile/median_jKep,c=c,lw=1)
                    if iRow==4:                        
                        circularity = np.array([prof.circularityProfile('MW') for prof in profs if prof.isSaved('circularity_MW')])
                        if len(circularity)==0: continue
                        median_circularity = np.nanmedian(circularity,axis=0)            
                        pl.plot(xs,median_circularity,c=c,lw=1)
                    if iRow==3:                        
                        vcs = np.nanmedian(np.array([prof.vc() for prof in profs]),axis=0)
                        vrs = np.nanmedian(np.array([(prof.profile1D('vrs','MW',2)) for prof in sim.profiles]),axis=0)
                        pl.plot(xs,vrs/vcs**2,c=c,lw=1)

                
                if iRow==0:                    
                    pl.loglog()
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                    ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=8,numticks=8,subs=range(2,10)))
                    pl.text(0.5,1.05,sim.shortname(),transform=ax.transAxes,ha='center')
                    #if not ax.is_first_col():
                        #pl.text(0.075,0.85,r'$%d < z < %.1f$'%(zs[~np.isnan(normed_ratios)][0],zs[~np.isnan(normed_ratios)][-1]),
                                #transform=ax.transAxes,fontsize=pB.infigfontsize)
                    if sim.shortname()=='m12i':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.025,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.1,200,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                    if sim.shortname()=='m12b':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.02,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(1,200,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c,ha='right')
                    if sim.shortname()=='m13A1':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.005,1000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.2,160,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                        
                    
                    pl.ylim(10.,0.2e5)
                    if ax.is_first_col():
                        pl.annotate(r'',(0.15,0.8e4),(0.075,0.8e4),arrowprops=pB.slantlinepropsblack,ha='right',va='bottom')
                        pl.text(0.07,0.825e4,r'$j_{\rm c}(r)$',ha='right',va='center')
                        #pl.text(1.5,6e5,r'$z=%d$'%zs[-1],color=pB.cmap(normed_ratios[len(zs)-1]),ha='right')
                        #pl.text(1.5,6e4,r'$z=%d$'%zs[0],color=pB.cmap(normed_ratios[0]),ha='right')
                        #pl.text(0.6,4e6,r'$z=%d$'%zs[0],color=pB.cmap(normed_ratios[0]),ha='right')
                        #pl.text(0.6,6e5,r'$z=%d$'%u.iround(zs[-1]),color=pB.cmap(normed_ratios[len(zs)-1]),ha='right')
                    ylabel = r'$\left|\vec{j}\right|\ [{\rm kpc\ km\ s^{-1}}]$'                        
                else:                    
                    pl.semilogx()
                    
                    if iRow==1:
                        ylabel = r'$\left|\vec{j}\right|/j_{\rm c}(r)$'
                        pl.ylim(0.,1.5)  
                    if iRow==2:
                        ylabel = r'$j_z/j_{\rm c}(r)$'
                        pl.ylim(0.,1.5)  
                    if iRow==4:
                        ylabel = r'circularity $j_z/j_{\rm c}(E)$'
                        pl.ylim(0.,1.05)                
                    if iRow==3:
                        ylabel = r'$v_r^2/v_{\rm c}^2$'
                        pl.ylim(0.,1.5)                
                pl.xlim(0.001,2)                
                if ax.is_first_col():
                    pl.ylabel(ylabel)  
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                if ax.is_last_row():                    
                    ax.xaxis.set_major_formatter(u.arilogformatter)                    
                    pl.xlabel(r'$r/R_{\rm vir}$')
                else:
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
                    
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=0.75,aspect=50,
                           pad=0.125)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + 'jProfiles_expanded_%s.pdf'%sim,bbox_inches='tight')    
    def j_single_sim3(self,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5,log_t_ratio_snapshot_range=None,
                      weight='MW',d=5,xls=(0.005,2)): 
        fig = pl.figure(figsize=(pB.fig_width_half,8))
        pl.subplots_adjust(hspace=0.1,wspace=0.05,left=0.15,right=0.9,bottom=0.025,top=0.95)
        nRows = 3
        axs = [pl.subplot(nRows,len(self.sims),i+1) for i in range(nRows*len(self.sims))]
        for iRow in range(nRows):
            for isim,sim in enumerate(self.sims):  
                ax = axs[len(self.sims)*iRow+isim]; pl.sca(ax)
                normed_ratios = np.ones(len(zs))*np.nan
                for iz,z in enumerate(zs):  
                    if isim<2 and iz%2==1: continue                    
                    inds = (abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.) & np.array([(weight=='MW') or (prof.isSaved('j_vec_weight_%s'%weight) and prof.isSaved('v_phi_%s'%weight)) for prof in sim.profiles])                    
                    #inds = (abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.) & np.array([(weight=='MW' and prof.isSaved('j_vec_weight')) or (prof.isSaved('j_vec_weight_%s'%weight) and prof.isSaved('v_phi_%s'%weight)) for prof in sim.profiles])                    
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    if log_t_ratio_snapshot_range!=None and (log_t_ratio<log_t_ratio_snapshot_range[0] or log_t_ratio>log_t_ratio_snapshot_range[1]): continue                         
                    normed_ratios[iz] = log_t_ratio
                    if abs(log_t_ratio)>max_t_ratio: continue
                    if iRow==0: print('%d %.2f %d %.2f'%(iz,z,len(inds.nonzero()[0]),log_t_ratio))
                    profs = np.array(sim.profiles)[inds]
                    normed_ratios[iz] = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratios[iz])
                    desampled_rs = profs[0].rs_midbins()
                    xs = desampled_rs /profs[0].rvir                    
                    v_phis = np.array([prof.thickerShells(prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=0.05,weight=weight)),
                                                          d,weight) for prof in profs])                    
                    median_v_phiProfile = np.nanmedian(v_phis,axis=0)
                    j_Keplerian = np.array([prof.vc()*prof.rs_midbins() for prof in profs])
                    median_jKep = np.nanmedian(j_Keplerian,axis=0)
                    jzs = np.array([prof.thickerShells(prof.jzProfile(weight=weight),d,weight) for prof in profs])
                    median_jzProfile = np.nanmedian(jzs,axis=0)  
                    vcs = np.array([prof.vc() for prof in profs])
                    median_vcProfile = np.nanmedian(vcs,axis=0)
                    if iRow==0:   
                        pl.plot(xs,median_jzProfile,c=c,lw=1,zorder=-iz)
                        pl.plot(xs,median_jKep,c=c,ls='--',lw=0.5,zorder=-iz)
                    if iRow==1:                        
                        pl.plot(xs,median_v_phiProfile/median_vcProfile,c=c,lw=1,zorder=-iz)
                    if iRow==2:                                         
                        v_phis2 = np.array([prof.thickerShells(prof.profile1D('v_phi',weight,power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=0.05,weight=weight)),
                                                               d,weight) for prof in profs])
                        sigma = (v_phis2-v_phis**2)**0.5
                        Vrot2sigma = v_phis/sigma
                        median_Vrot2sigma = np.nanmedian(Vrot2sigma,axis=0)
                        pl.plot(xs,median_Vrot2sigma,c=c,lw=1,zorder=-iz)                        
                        #circularity = np.array([prof.circularityProfile('MW') for prof in profs if prof.isSaved('circularity_MW')])
                        #if len(circularity)==0: continue
                        #median_circularity = np.nanmedian(circularity,axis=0)            
                        #pl.plot(xs,median_circularity,c=c,lw=1,zorder=-iz)
                
                if iRow==0:                    
                    pl.loglog()
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                    ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=8,numticks=8,subs=range(2,10)))
                    ax.xaxis.set_label_position('top')
                    ax.xaxis.set_ticks_position('top')
                    ax.xaxis.set_ticks_position('both')
                    
                    pl.text(0.5,1.2,sim.shortname(),transform=ax.transAxes,ha='center',fontsize=14)
                    #if not ax.is_first_col():
                        #pl.text(0.075,0.85,r'$%d < z < %.1f$'%(zs[~np.isnan(normed_ratios)][0],zs[~np.isnan(normed_ratios)][-1]),
                                #transform=ax.transAxes,fontsize=pB.infigfontsize)
                    if sim.shortname()=='m12i':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.025,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.08,150,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                        pl.annotate(r'',(0.15,0.8e4),(0.075,0.8e4),arrowprops=pB.slantlinepropsblack,ha='right',va='bottom')
                        pl.text(0.07,0.825e4,r'$v_{\rm c}r$',ha='right',va='center')
                        
                    if sim.shortname()=='m12b':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.02,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.1,25,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c,ha='right')
                    if sim.shortname()=='m13A1':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.005,1000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.1,30,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                        
                    
                    pl.ylim(10.,0.2e5)
                    ylabel = r'$\langle j_z\rangle\ [{\rm kpc\ km\ s^{-1}}]$'                        
                else:                    
                    pl.semilogx()
                    
                    if iRow==1:
                        ylabel = r'$\langle V_{\rm rot} \rangle/v_{\rm c}$'
                        pl.ylim(0.,1.5)  
                    if iRow==2:
                        #ylabel = r'circularity $j_z/j_{\rm c}(E)$'
                        ylabel = r'$\langle V_{\rm rot}\rangle / \sigma_{\rm g}$'
                        pl.ylim(0.,10) 
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.))
                pl.xlim(*xls)
                if ax.is_first_col():
                    pl.ylabel(ylabel)                      
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                if ax.is_last_row():
                    pl.xlabel(r'$r/R_{\rm vir}$')
                if ax.is_last_row() or ax.is_first_row():                    
                    ax.xaxis.set_major_formatter(u.arilogformatter)                    
                else:
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
                    
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=0.75,aspect=50,
                           pad=0.08)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + 'jProfiles_expanded_%s_%s.pdf'%(sim,weight),bbox_inches='tight')    
    def j_single_sim4(self,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5,log_t_ratio_snapshot_range=None,weight='MW',N=5): 
        fig = pl.figure(figsize=(pB.fig_width_full*1.1,11))
        pl.subplots_adjust(hspace=0.1,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        nRows = 4
        axs = [pl.subplot(nRows,len(self.sims),i+1) for i in range(nRows*len(self.sims))]
        for iRow in range(nRows):
            for isim,sim in enumerate(self.sims):  
                ax = axs[len(self.sims)*iRow+isim]; pl.sca(ax)
                normed_ratios = np.ones(len(zs))*np.nan
                for iz,z in enumerate(zs):  
                    if isim<2 and iz%2==1: continue                    
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    if len(inds.nonzero()[0])==0: continue
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    if log_t_ratio_snapshot_range!=None and (log_t_ratio<log_t_ratio_snapshot_range[0] or log_t_ratio>log_t_ratio_snapshot_range[1]): continue                         
                    normed_ratios[iz] = log_t_ratio
                    if abs(log_t_ratio)>max_t_ratio: continue
                    #if iRow==0: print('%d %.2f %.2f'%(iz,z,log_t_ratio))
                    profs = np.array(sim.profiles)[inds]
                    normed_ratios[iz] = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                    c = pl.get_cmap('coolwarm')(normed_ratios[iz])
                    desampled_rs = self.desample(profs[0].rs_midbins(),N)
                    xs = desampled_rs /profs[0].rvir                    
                    v_phis = np.array([prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=0.05)) for prof in profs])                    
                    median_v_phiProfile = self.desample(np.nanmedian(v_phis,axis=0),N)
                    j_Keplerian = np.array([prof.vc()*prof.rs_midbins() for prof in profs])
                    median_jKep = self.desample(np.nanmedian(j_Keplerian,axis=0),N)
                    jzs = np.array([prof.jzProfile() for prof in profs])
                    median_jzProfile = self.desample(np.nanmedian(jzs,axis=0),N)   
                    vcs = np.array([prof.vc() for prof in profs])
                    median_vcProfile = self.desample(np.nanmedian(vcs,axis=0),N)
                    if iRow==0:   
                        pl.plot(xs,median_jzProfile,c=c,lw=1,zorder=-iz)
                        pl.plot(xs,median_jKep,c=c,ls='--',lw=0.5,zorder=-iz)
                    if iRow==1:                        
                        pl.plot(xs,median_v_phiProfile/median_vcProfile,c=c,lw=1,zorder=-iz)
                    if iRow==2:
                        v_phis2 = np.array([prof.profile1D('v_phi',weight,power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=0.05)) for prof in profs])
                        sigma = (v_phis2-v_phis**2)**0.5
                        median_sigma = self.desample(np.nanmedian(sigma,axis=0),N)
                        pl.plot(xs,median_sigma /median_vcProfile,c=c,lw=1,zorder=-iz)                        
                    if iRow==3:                                         
                        v_phis2 = np.array([prof.profile1D('v_phi',weight,power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=0.05)) for prof in profs])
                        sigma = (v_phis2-v_phis**2)**0.5
                        Vrot2sigma = v_phis/sigma
                        median_Vrot2sigma = self.desample(np.nanmedian(Vrot2sigma,axis=0),N)
                        pl.plot(xs,median_Vrot2sigma,c=c,lw=1,zorder=-iz)                        
                        #circularity = np.array([prof.circularityProfile('MW') for prof in profs if prof.isSaved('circularity_MW')])
                        #if len(circularity)==0: continue
                        #median_circularity = np.nanmedian(circularity,axis=0)            
                        #pl.plot(xs,median_circularity,c=c,lw=1,zorder=-iz)
                
                if iRow==0:                    
                    pl.loglog()
                    ax.yaxis.set_major_formatter(u.arilogformatter)
                    ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
                    ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=8,numticks=8,subs=range(2,10)))
                    pl.text(0.5,1.05,sim.shortname(),transform=ax.transAxes,ha='center')
                    #if not ax.is_first_col():
                        #pl.text(0.075,0.85,r'$%d < z < %.1f$'%(zs[~np.isnan(normed_ratios)][0],zs[~np.isnan(normed_ratios)][-1]),
                                #transform=ax.transAxes,fontsize=pB.infigfontsize)
                    if sim.shortname()=='m12i':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.025,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.1,30,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                    if sim.shortname()=='m12b':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.02,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.1,25,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c,ha='right')
                    if sim.shortname()=='m13A1':
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                        pl.text(0.005,1000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                        c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                        pl.text(0.1,30,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                        
                    
                    pl.ylim(10.,0.2e5)
                    if ax.is_first_col():
                        pl.annotate(r'',(0.15,0.8e4),(0.075,0.8e4),arrowprops=pB.slantlinepropsblack,ha='right',va='bottom')
                        pl.text(0.07,0.825e4,r'$v_{\rm c}r$',ha='right',va='center')
                        #pl.text(1.5,6e5,r'$z=%d$'%zs[-1],color=pB.cmap(normed_ratios[len(zs)-1]),ha='right')
                        #pl.text(1.5,6e4,r'$z=%d$'%zs[0],color=pB.cmap(normed_ratios[0]),ha='right')
                        #pl.text(0.6,4e6,r'$z=%d$'%zs[0],color=pB.cmap(normed_ratios[0]),ha='right')
                        #pl.text(0.6,6e5,r'$z=%d$'%u.iround(zs[-1]),color=pB.cmap(normed_ratios[len(zs)-1]),ha='right')
                    ylabel = r'$\langle j_z\rangle\ [{\rm kpc\ km\ s^{-1}}]$'                        
                else:                    
                    pl.semilogx()
                    
                    if iRow==1:
                        ylabel = r'$\langle V_{\rm rot} \rangle/v_{\rm c}$'
                        pl.ylim(0.,1.5)  
                    if iRow==2:
                        ylabel = r'$\langle \sigma_{\rm g} \rangle/v_{\rm c}$'
                        pl.ylim(0.,3)  
                    if iRow==3:
                        #ylabel = r'circularity $j_z/j_{\rm c}(E)$'
                        ylabel = r'$\langle V_{\rm rot}\rangle / \sigma_{\rm g}$'
                        pl.ylim(0.,10) 
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.))
                pl.xlim(0.001,2)                
                if ax.is_first_col():
                    pl.ylabel(ylabel)                      
                else:
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                if ax.is_last_row():                    
                    ax.xaxis.set_major_formatter(u.arilogformatter)                    
                    pl.xlabel(r'$r/R_{\rm vir}$')
                else:
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
                    
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=0.75,aspect=50,
                           pad=0.125)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + 'jProfiles_expanded2_%s_%s.pdf'%(sim,weight),bbox_inches='tight')    
    def Vrot_to_sigma(self,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.5,log_t_ratio_snapshot_range=None): 
        fig = pl.figure(figsize=(pB.fig_width_full*1.1,3))
        pl.subplots_adjust(hspace=0.1,wspace=0.05,left=0.15,right=0.9,bottom=0.1)
        axs = [pl.subplot(1,len(self.sims),i+1) for i in range(len(self.sims))]
        for isim,sim in enumerate(self.sims):  
            ax = axs[isim]; pl.sca(ax)
            normed_ratios = np.ones(len(zs))*np.nan
            for iz,z in enumerate(zs):  
                if isim<2 and iz%2==1: continue
                inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                if len(inds.nonzero()[0])==0: continue
                log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                if log_t_ratio_snapshot_range!=None and (log_t_ratio<log_t_ratio_snapshot_range[0] or log_t_ratio>log_t_ratio_snapshot_range[1]): continue                         
                normed_ratios[iz] = log_t_ratio
                if abs(log_t_ratio)>max_t_ratio: continue
                profs = np.array(sim.profiles)[inds]
                normed_ratios[iz] = (log_t_ratio+max_t_ratio)/(2*max_t_ratio)                         
                c = pl.get_cmap('coolwarm')(normed_ratios[iz])
                xs = profs[0].rs_midbins()/profs[0].rvir
            
                v_phis = np.array([prof.profile1D('v_phi', 'HI',power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=0.05)) for prof in profs])
                v_phis2 = np.array([prof.profile1D('v_phi', 'HI',power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=0.05)) for prof in profs])
                sigma = (v_phis2-v_phis**2)**0.5
                Vrot2sigma = v_phis/sigma
                median_Vrot2sigma = np.nanmedian(Vrot2sigma,axis=0)    
                pl.plot(xs,median_Vrot2sigma,c=c,lw=1,zorder=-iz)
            
            pl.semilogx()
            pl.ylim(0,10)
            pl.text(0.5,1.05,sim.shortname(),transform=ax.transAxes,ha='center')
            #if sim.shortname()=='m12i':
                #c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                #pl.text(0.025,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                #c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                #pl.text(0.1,200,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
            #if sim.shortname()=='m12b':
                #c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                #pl.text(0.02,2000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                #c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                #pl.text(1,200,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c,ha='right')
            #if sim.shortname()=='m13A1':
                #c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][0])
                #pl.text(0.005,1000,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][0]),fontsize=pB.infigfontsize,color=c,ha='right')
                #c = pl.get_cmap('coolwarm')(normed_ratios[~np.isnan(normed_ratios)][-1])
                #pl.text(0.2,160,r'$z = %d$'%(zs[~np.isnan(normed_ratios)][-1]),fontsize=pB.infigfontsize,color=c)
                                    
            pl.xlim(0.001,2)                
            if ax.is_first_col():
                pl.ylabel(r'$V_{\rm rot} / \sigma_{\rm g}$')  
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
            ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.xaxis.set_major_formatter(u.arilogformatter)                    
            pl.xlabel(r'$r/R_{\rm vir}$')
                    
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=0.75,aspect=50,
                           pad=0.125)
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + 'Vrot_to_sigma_%s.pdf'%sim,bbox_inches='tight')    


    
    def subsonic_fraction(self,zs,dt=1.*un.Gyr,atRcirc=True,max_t_ratio=1.,weights=('VW','MW')): 
        fig = pl.figure(figsize=(pB.fig_width_full,(None,4,6.5)[len(zs)]))
        pl.subplots_adjust(wspace=0.35,hspace=0.3,left=0.05,right=0.95)
        axs = [pl.subplot(len(zs),2,i+1) for i in range(2*len(zs))]
        for iz,z in enumerate(zs):                
            for iweight,weight in enumerate(weights):                
                ax = axs[2*iz+iweight]; pl.sca(ax)
                for isim,sim in enumerate(self.sims):  
                    inds = abs(cosmo.age(sim.zs())-cosmo.age(z))<dt/2.
                    log_t_ratio = np.nanmedian(log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds])
                    if abs(log_t_ratio)>max_t_ratio: continue
                    normed_ratio = (log_t_ratio+max_t_ratio)/(2*max_t_ratio) 
                    c = pl.get_cmap('coolwarm')(normed_ratio)
                    
                    profs = np.array(sim.profiles)[inds]
                    xs = profs[0].rs_midbins()/profs[0].rvir
                    fs = np.nan*np.ones((len(profs),len(xs)))
                    for iprof,prof in enumerate(profs):
                        if not prof.isSaved('subsonic_fraction_%s'%weight): continue
                        fs[iprof,:] = prof.subsonic_fraction(weight)
                    f = np.nanmedian(fs,axis=0)                    
                    pl.plot(prof.rs_midbins()/prof.rvir,f,c=c)
                #Rcirc2Rvir = prof.Rcirc2(0.5)/prof.rvir
                #pl.axvline(log(Rcirc2Rvir),c=c,lw=0.5)                
                
                pl.semilogx()
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_formatter(u.arilogformatter)    
                pl.ylim(0.,1.1)        
                pl.ylabel(r'subsonic fraction')             
                
                pl.xlim(0.03,2)                
                #if z==0: pl.xlim(log(0.03),0.31)  
                #if z==2: pl.xlim(log(0.01),log(2.6))  
                if ax.is_first_row():
                    #if z==0: pl.text(log(0.13),0.05,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
                    #if z==2: pl.text(log(0.05),0.05,r'$R_{\rm circ}$',fontsize=pB.infigfontsize)
                    if weight=='VW': pl.title('weighted by volume')
                    if weight=='MW': pl.title('weighted by mass')
                    
                if ax.is_last_row():                    
                    pl.xlabel(r'$r/R_{\rm vir}$')
                
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([-10,-11]),np.array([-10,-11]),logm,cmap=pl.get_cmap('coolwarm'),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')        
        cbar = pl.colorbar(mesh,orientation='horizontal',ax=axs,ticks=[-1.,0.,1.],shrink=0.75,aspect=50,
                           pad=(None,0.2,0.15)[len(zs)])
        cbar.ax.set_xticklabels([r'$0.1$',r'$1$',r'$10$'])  
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + 'subsonic_fraction_profiles_z%d.pdf'%z,bbox_inches='tight')    
            
            
            
    def T_medians_Nsnapshots(self,weight,relative=False,N=20,
                             atRcirc=True,max_t_ratio=1.5,
                             minz=-1,maxz=100,log_xaxis=True,nRows=3,nCols=3,xls=(0,1),yls=(4.5,6.5)): 
        fig = pl.figure(figsize=(pB.fig_width_full,6.5))
        pl.subplots_adjust(hspace=0.4,wspace=0.04,left=0.05,right=0.95)
        axs = [pl.subplot(nRows,nCols,1+isim) for isim in range(nRows*nCols)]
        for isim,sim in enumerate(self.sims):  
            inds = (sim.zs()<maxz) & (sim.zs()>minz)
            log_t_ratios = log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds]
            profs = np.array(sim.profiles)[inds]
            saveTs = []
            if sim.z_cross()!=None:
                log_Mthres = sim.at_z_cross(log(sim.mvirs()))
            else:
                log_Mthres = log(sim.mvirs()[0])
            for i in range(len(profs)%N,len(profs),N):
                prof = profs[i+N//2]
                xs = prof.rs_midbins()/prof.rvir
                
                log_Tvir = log(prof.Tvir())
                log_Mratio = log(prof.mvir)-log_Mthres
                all_Ts = np.array([prof.profile1D('log_Ts',weight) for prof in profs[i:i+N]])                
                if relative:
                    all_Ts -= log(np.array([prof.Tc() for prof in profs[i:i+N]]))
                Ts = np.nanmedian(all_Ts,axis=0)#prof.smooth(prof.profile1D('log_Ts',weight),10)                                
                saveTs.append((Ts,xs,log_Tvir,log_Mratio,log_t_ratios[i+N//2]))
            
            ax = axs[isim]
            pl.sca(ax)
            
            for Ts,xs,log_Tvir,log_Mratio,log_t_ratio in saveTs[::-1]:
                if log_t_ratio <-max_t_ratio or log_t_ratio >max_t_ratio: continue
                normed_ratio = (log_t_ratio+max_t_ratio)/(2*max_t_ratio) 
                c = pB.cmap(1-normed_ratio)
                pl.plot(xs,Ts ,c=c,lw=0.75)
                pl.axhline((log_Tvir,0)[relative],c=c,ls=':')
            
            
            if not relative:
                if isim<3: 
                    if atRcirc: pl.ylim(4.5,6.5)
                    else: pl.ylim(4.,6)
                else: pl.ylim(5,7.)
            else:
                if atRcirc: pl.ylim(-1,0.1)
                else: pl.ylim(-2.,0.1)
            #pl.ylim(0,1)
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            
            if log_xaxis:
                pl.semilogx()
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_formatter(u.arilogformatter)    
                pl.xlim(0.05,2)
                if not relative and isim==0:
                    pl.annotate(r'$T_{\rm vir}$',(0.8,6.),(1,6.2),ha='left',fontsize=pB.infigfontsize,va='bottom')
                    pl.annotate(r'',(0.7,6.1),(0.9,6.25),arrowprops=pB.slantlinepropsblack,ha='center',fontsize=pB.infigfontsize,va='bottom')                
            else:
                pl.xlim(*xls)
                pl.ylim(*yls)
            
            #if isim==4: 
                #u.mylegend(handlelength=1.,ncol=7,columnspacing=0.5,bbox_transform=fig.transFigure,fontsize=pB.infigfontsize+3,
                       #loc='lower center',numpoints=3,handletextpad=0.1,bbox_to_anchor=(0.5,-0.05))            
            if ax.is_first_col() or ax.is_last_col():                     
                if not relative:
                    pl.ylabel(r'$\langle\log\ T\rangle_{\rm V}$')
                else:
                    pl.ylabel(r'$\langle\log\ T/T_{\rm c}\rangle_{\rm V}$')
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            if ax.is_last_col():
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
            if sim.z_cross()!=None:
                pl.title(r'$M_{\rm thres}=10^{%.1f},\ $'%log_Mthres+
                         r'$z_{\rm thres}=%.1f$'%sim.z_cross(), fontsize=pB.infigfontsize,transform=ax.transAxes) #bbox=dict(edgecolor='k',facecolor='w',lw=0.5),
            pl.text(0.95,0.05,sim.shortname(),ha='right',fontsize=pB.infigfontsize,transform=ax.transAxes)
            
            if ax.is_last_row(): 
                pl.xlabel(r'$r/R_{\rm vir}$')
                #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: u.nSignificantDigits(x,1,True)))    
            
            #else:
                #ax.xaxis.set_major_formatter(ticker.NullFormatter())
        logm = np.array([[0,1],[0,1]])
        mesh = pl.pcolormesh(np.array([0,1]),np.array([0,1]),logm,cmap=pB.cmap.reversed(
            ),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')
        
        cbar = pl.colorbar(mesh,orientation='vertical',ax=axs,ticks=[-1.,0.,1.],
                           shrink=0.75,aspect=50,pad=0.1)
        cbar.ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])  # horizontal colorbar
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ %sR_{\rm vir}$'%(('','0.1 ')[atRcirc]),fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + 'Tprofiles_Nsnapshots%s%s.pdf'%(('','_relative')[relative],('_atRvir','_atRcirc')[atRcirc]),bbox_inches='tight')    
    def medians_Nsnapshots(self,prop_name,weight,N=20,atRcirc=True,max_t_ratio=1.5,minz=-1,maxz=100,log_xaxis=True):       
        fig = pl.figure(figsize=(pB.fig_width_full,6.5))
        pl.subplots_adjust(hspace=0.4,wspace=0.04,left=0.05,right=0.95)
        axs = [pl.subplot(3,3,1+isim) for isim in range(9)]
        for isim,sim in enumerate(self.sims):  
            inds = (sim.zs()>minz) & (sim.zs()<maxz)
            profs = np.array(sim.profiles)[inds]           
            log_t_ratios = log(sim.quantity_vs_z('t_cool',atRcirc)/sim.quantity_vs_z('t_ff',atRcirc))[inds]
            save_props = []
            if sim.z_cross()!=None:
                log_Mthres = sim.at_z_cross(log(sim.mvirs()))
            else:
                log_Mthres = log(sim.mvirs()[0])
            for i in range(len(profs)%N,len(profs),N):
                prof = profs[i+N//2]
                xs = prof.rs_midbins()/prof.rvir                
                log_Mratio = log(prof.mvir)-log_Mthres                
                if prop_name=='j':
                    all_props = np.array([prof.jProfile()/(prof.vc()*prof.rs_midbins()) for prof in profs[i:i+N] if prof.isSaved('j_vec_x')])
                else:
                    all_props = np.array([prof.profile1D(prop_name,weight) for prof in profs[i:i+N] if prof.isSaved(prop_name+'_'+weight)])
                if all_props.shape[0]==0: continue
                
                props = np.nanmedian(all_props,axis=0)#prof.smooth(prof.profile1D('log_Ts',weight),10)                                
                save_props.append((props,xs,log_Mratio,log_t_ratios[i+N//2]))
            
            ax = axs[isim]
            pl.sca(ax)
            
            for prop,xs,log_Mratio,log_t_ratio in save_props[::-1]:
                if log_t_ratio <-max_t_ratio or log_t_ratio >max_t_ratio: continue
                normed_ratio = (log_t_ratio+max_t_ratio)/(2*max_t_ratio) 
                c = pB.cmap(1-normed_ratio)
                pl.plot(xs,prop,c=c,lw=0.75)
                
            if prop_name=='j': 
                pl.xlim(0.0007,2)
                pl.loglog()
                pl.ylim(0.1,3)
                pl.axhline(1.,c='.5',ls='-',lw=0.5,zorder=-1000)
                pl.text(0.95,0.85,sim.shortname(),ha='right',fontsize=pB.infigfontsize,transform=ax.transAxes)
                ax.yaxis.set_major_formatter(u.arilogformatter)
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,pos: ('',r'$%s$'%u.nSignificantDigits(x,1,True))['%.1f'%x in ('0.2','0.3','0.5','2.0','3.0')]))
            elif prop_name=='isSubsonic': 
                if log_xaxis: pl.xlim(0.03,3)
                else: pl.xlim(0,1.)
                pl.ylim(0,1.1)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            else:                
                pl.axhline(0.,c=c,ls=':')
                pl.xlim(0.05,2)
                pl.ylim(-1.5,1.5)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
                pl.text(0.95,0.05,sim.shortname(),ha='right',fontsize=pB.infigfontsize,transform=ax.transAxes)
            if log_xaxis:
                pl.semilogx()
                ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=6,numticks=6))
                ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=6,numticks=6,subs=range(2,10)))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.xaxis.set_major_formatter(u.arilogformatter)    
            
            if ax.is_first_col() or ax.is_last_col():                     
                if prop_name=='isSubsonic': pl.ylabel(r'subsonic fraction')
                if prop_name=='j': pl.ylabel(r'$\left|\Sigma \vec{j}\right| / (v_{\rm c} r)$')
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            if ax.is_last_col():
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
            if sim.z_cross()!=None:
                pl.title(r'$M_{\rm thres}=10^{%.1f},\ $'%log_Mthres+
                         r'$z_{\rm thres}=%.1f$'%sim.z_cross(), fontsize=pB.infigfontsize,transform=ax.transAxes) #bbox=dict(edgecolor='k',facecolor='w',lw=0.5),
            
            if ax.is_last_row(): 
                pl.xlabel(r'$r/R_{\rm vir}$')
            
        logm = log(np.array([[-1,-1],[-1,-1]]))
        mesh = pl.pcolormesh(np.array([0,-1]),np.array([0,1]),logm,cmap=pB.cmap.reversed(
            ),zorder=-100,
                             vmin=-max_t_ratio,vmax=max_t_ratio)
        mesh.cmap.set_under('w')
        
        cbar = pl.colorbar(mesh,orientation='vertical',ax=axs,ticks=[-1.,0.,1.],
                           shrink=0.75,aspect=50,pad=0.1)
        cbar.ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])  # horizontal colorbar
        cbar.set_label(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ 0.1 R_{\rm vir}$',fontsize=pB.infigfontsize)
        
        pl.savefig(figDir + '%s_profiles_Nsnapshots%s.pdf'%(prop_name,('_atRvir','_atRcirc')[atRcirc]),bbox_inches='tight')    
    def P_medians(self,weight,relative=False):       
        fig = pl.figure(figsize=(pB.fig_width_full,5))
        pl.subplots_adjust(hspace=0.4,wspace=0.04)
        for isim,sim in enumerate(self.sims):  
            #pl.title(sim)
            profs = np.array(sim.profiles)            
            log_t_ratio = log(sim.tcool_Rcircs()/sim.tff_Rcircs())
            log_mvirs = log(np.array([prof.mvir for prof in profs]))
            savePs = []
            if sim.z_cross()!=None:
                log_Mthres = sim.at_z_cross(log_mvirs)
            else:
                log_Mthres = log(sim.mvirs()[0])
            dlogM = 0.2
            mvirs_to_show = log(np.array([0.1,1/3.,0.5,10.**-0.1,10.**0.1,2.,3.]))+log_Mthres
            for imvir,mvir_to_show in enumerate(mvirs_to_show[1:]):
                inds = (log_mvirs < mvir_to_show+dlogM/2.) & (log_mvirs > mvir_to_show-dlogM/2.)
                if len(inds.nonzero()[0])!=0:                    
                    prof = profs[inds][inds.nonzero()[0].shape[0]//2]
                    xs = prof.rs_midbins()/prof.rvir
                    c = matplotlib.cm.inferno(1-(imvir+1.)/len(mvirs_to_show))
                    log_Tvir = log(prof.Tvir())

                    all_Ps = np.array([prof.profile1D('log_nHTs',weight) for prof in profs[inds]])
                    #all_Ts = log(np.array([prof.Tc() for prof in profs[inds]]))
                    if relative:
                        all_Ts -= log(np.array([prof.Tc() for prof in profs[inds]]))
                        
                    
                    Ps = np.nanmedian((all_Ps.T).T,axis=0)#prof.smooth(prof.profile1D('log_Ts',weight),10)                                
                    var_Ps = np.array([prof.profile1D('log_nHTs',weight,power=2) for prof in profs[inds]]) - all_Ps**2                                          
                    sig_Ps = np.nansum(var_Ps/var_Ps.shape[0],axis=0)**0.5                    
                    
                    savePs.append((Ps,c,xs,log_Tvir,mvir_to_show,sig_Ps))
            
            ax = pl.subplot(2,3,1+isim)
            
            for Ps,c,xs,log_Tvir,mvir_to_show,sig_Ps  in savePs:
                pl.plot(xs,Ps ,c=c,label=r'$%.1f M_{\rm thres}$'%(10.**(mvir_to_show-log_Mthres)))
                #pl.axhline((log_Tvir,0)[relative],c=c,ls=':')
                #pl.plot(xs,sig_Ts,c=c,label='%.1f'%mvir_to_show) 
            pl.xlim(0.05,2)
            pl.semilogx()
            #if not relative:
                #if isim<3: pl.ylim(4,6.65)
                #else: pl.ylim(4.5,7.15)
            #else:
                #pl.ylim(-1,0.3)
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=4,numticks=4))
            ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=4,numticks=4,subs=range(2,10)))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            
            if isim==4: 
                u.mylegend(handlelength=1.,ncol=7,columnspacing=0.5,bbox_transform=fig.transFigure,fontsize=pB.infigfontsize+3,
                       loc='lower center',numpoints=3,handletextpad=0.1,bbox_to_anchor=(0.5,-0.05))            
            if ax.is_first_col() or ax.is_last_col():                     
                pl.ylabel(r'$\langle\log\ n_{\rm H}T\rangle_{\rm V}$')
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            if ax.is_last_col():
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_ticks_position('both')
            if sim.z_cross()!=None:
                pl.title(sim.shortname()+
                         r', $M_{\rm thres}=%.1f$'%log_Mthres+
                         r', $z_{\rm thres}=%.1f$'%sim.z_cross(), fontsize=pB.infigfontsize,transform=ax.transAxes) #bbox=dict(edgecolor='k',facecolor='w',lw=0.5),
            
            if ax.is_last_row(): 
                pl.xlabel(r'$r/R_{\rm vir}$')
                #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: u.nSignificantDigits(x,1,True)))    
            ax.xaxis.set_major_formatter(u.arilogformatter)    
            #else:
                #ax.xaxis.set_major_formatter(ticker.NullFormatter())
            
        pl.savefig(figDir + 'Pprofiles2%s.pdf'%('','_relative')[relative],bbox_inches='tight')    
    
    def Ps_profile_old(self,weight):       
        pl.figure(figsize=(pB.fig_width_full,5))        
        ax = pl.subplot(111)
        for iPanel,r2Rvir in enumerate((0.1,0.5)):
            
            for isim,sim in enumerate(self.sims):
                profs = np.array(sim.profiles)            
                log_t_ratio = log(sim.tcool_Rcircs()/sim.tff_Rcircs())
                log_mvirs = log(np.array([prof.mvir for prof in profs]))
                ind = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
                sig_log_nHTs = np.array([(prof.profile1D('log_nHTs','VW',2)[ind] - prof.profile1D('log_nHTs','VW')[ind]**2)**0.5 for prof in profs])
                #smooth_sig_log_nHTs = pl.poly1d(pl.polyfit(log_mvirs,sig_log_nHTs,5))(log_mvirs)
                pl.plot(log_mvirs,sig_log_nHTs,'.',label=(sim.shortname(),'_nolegend_')[iPanel],
                        ls=('-','--')[iPanel],c=pB.cmap(isim/len(self.sims)))
        pl.xlim(11,12.5)
        #pl.xlim(-2,2)
        pl.ylim(0,1.2)
        pl.legend()
        pl.ylabel(r'$\sigma')
    def vs_CFindicator(self,val_str,val_str_show,weight=None,r2Rvir=0.1,N=10,show_t_ratio_at_rvir=False,normalize=1,present=False,
                       _ax=None,savefig=True,nRows=3,nCols=3,minR2Rvir=1e-5,maxR2Rvirs=(0.05,),show_ax2=True,show_axtop=True,
                       Rcirc2Rvirs=(0.1,),_xls=None,suffix='',fig_height=5):       
        if _ax==None: 
            pl.figure(figsize=(pB.fig_width_full*(1.05,1)[present],(3,6,8.5,8.5)[nRows-1]))
            pl.subplots_adjust(hspace=(0.3,0.2)[present],wspace=(0.1,0.35)[val_str in ('vrs','Mdot','log_nHTs') or (val_str=='sfhs' and not normalize)])
            if len(self.sims)==6: pl.subplots_adjust(top=0.95)
        for isim,sim in enumerate(self.sims):
            if _ax==None: ax = pl.subplot(nRows,nCols,isim+1)
            else: ax = _ax
            ind = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
            if val_str=='sfhs': 
                y = sim.sfh['sfrs']
                maxind = len(y)//10*10
                xs = 1+sim.sfh['zs'][5:maxind:10]                
                log_val = y[:maxind].reshape((len(y)//10,10)).mean(axis=1)
            elif val_str=='diskiness':
                xs = 1+sim.zs()
                log_val = np.array([0. for prof in sim.profiles])
            elif val_str=='circularity':
                xs = 1+sim.zs()
                maxR2Rvir = maxR2Rvirs[isim%len(maxR2Rvirs)]
                inds = ((midbins(Snapshot_profiler.log_r2rvir_bins) < log(maxR2Rvir)) & 
                        (midbins(Snapshot_profiler.log_r2rvir_bins) > log(minR2Rvir)) )
                log_val = np.array([np.nan_to_num(prof.circularityProfile(weight)) for prof in sim.profiles
                                    if prof.isSaved('circularity_MW')])
                ms = np.array([prof.gasMassProfile() for prof in sim.profiles
                               if prof.isSaved('circularity_MW')])
                log_val = (log_val * ms)[:,inds].sum(axis=1)/ms[:,inds].sum(axis=1)
            elif val_str=='Mdot': 
                xs = 1+sim.zs()
                log_val = np.array([(prof.gasMassProfile() * prof.profile1D('vrs','MW')/prof.drs_midbins())[ind] * (un.km/un.s/un.kpc).to('yr**-1') for prof in sim.profiles])                
            else: 
                xs = 1+sim.zs()
                log_val = np.array([prof.profile1D(val_str,weight)[ind] for prof in sim.profiles])
                
            if val_str=='vrs': 
                norm = np.array([prof.vc()[ind] for prof in sim.profiles])
            elif val_str=='sfhs': 
                norm = sim.sfh['means'][5:maxind:10]
            elif val_str=='log_Ts':
                log_val -= log(np.array([prof.Tvir() for prof in sim.profiles]))
                norm = 1.
            elif val_str=='log_nHTs':
                log_val = 10.**log_val
                norm = 1.
            elif val_str in ('circularity','Mdot'):
                norm = 1.
            else: 
                norm = np.convolve(log_val, np.ones((N,))/N, mode='same')            
            #smooth_sig_log_nHTs = pl.poly1d(pl.polyfit(log_mvirs,sig_log_nHTs,5))(log_mvirs)
            pl.plot(xs,log_val/norm**normalize,c=pB.niceblue,lw=0.5)
            if False: #val_str=='log_nHTs':
                sig_log_val = (np.array([prof.profile1D(val_str,weight,2)[ind] for prof in sim.profiles]) - log(log_val)**2)**0.5
                pl.fill_between(1+sim.zs(), 10.**(log(log_val)-sig_log_val),10.**(log(log_val)+sig_log_val),facecolor='b',alpha=0.2)
            ax.set_xscale('log')
            if _xls==None:
                if 'm1' in sim.shortname(): xls=1,3
                if 'A' in sim.shortname(): xls=((2,8),(2,6))[present]
                if 'HL' in sim.shortname(): xls=6,12
            else: xls=_xls
            pl.xlim(*xls)
            
            if val_str=='vrs': 
                pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                if normalize:
                    ylabel=r'$\langle v_r/v_{\rm c}\rangle_{V}$'
                    pl.ylim(-1.7,1.7)  
                    pl.axhline(1.,c='.5',lw=0.5,zorder=1000)                    
                    pl.axhline(-1.,c='.5',lw=0.5,zorder=1000)            
                else:
                    ylabel=r'radial velocity at $%.1f R_{\rm vir}\ [{\rm km}~{\rm s}^{-1}]$'%r2Rvir 
                    #[pl.plot(xs,j*norm,c='.5',lw=0.5,zorder=1000) for j in (-1,1)]
                    if 'm12' in sim.shortname():
                        #if present: pl.text(1.175,235,r'$v_{\rm c}$')
                        #if sim.shortname()=='m12i':
                            #pl.text(1.175,235,r'$v_{\rm c}$')
                            #pl.annotate('',(1.3,200.),(1.27,225),arrowprops=pB.slantlinepropsblack) 
                        #pl.text(1.1,-250,r'$-v_{\rm c}$')
                        pl.ylim(-300,300) 
                    elif 'm11' in sim.shortname():
                        #pl.text(1.175,100,r'$v_{\rm c}$')
                        #pl.text(1.1,-250,r'$-v_{\rm c}$')
                        pl.ylim(-130,130) 
                    else:
                        #pl.text(2.5,400,r'$v_{\rm c}$')
                        #pl.text(1.5,-500,r'$-v_{\rm c}$')
                        pl.ylim(-500,500) 
            elif val_str=='Mdot': 
                pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                pl.ylim(-100,100) 
                if 'm12' in sim.shortname():
                    pl.ylim(-50,50) 
                elif 'm11' in sim.shortname():
                    pl.ylim(-10,10) 
                #elif 'm11' in sim.shortname():
                    #pl.ylim(-20,20) 
                #elif 'm11' in sim.shortname():
                    #pl.ylim(-20,20) 
                else:
                    pl.ylim(-300,300) 
                ylabel = r'$\dot{M}\ [{\rm M}_\odot\ {\rm yr}^{-1}]$'
            elif val_str=='log_Ts':
                pl.ylim(-1,1)  
                ylabel = r'$\log T/T_{\rm vir}$'
            elif val_str=='circularity':
                ylabel = r'circularity $(j_z/j_{\rm c}(E))$'
                pl.ylim(0.,1)
            elif val_str=='log_nHTs':
                ylabel = r'$n_{\rm H}T\ [{\rm cm}^{-3}{\rm K}]$'
                ax.set_yscale('log')
                if 'm1' in sim.shortname():
                    pl.ylim(30,5000)
                else:
                    pl.ylim(5e3,0.5e6)
            elif val_str=='sfhs' and not normalize:                
                ax.set_yscale('log')
                ylabel = r'${\rm SFR}\ [{\rm M}_\odot{\rm yr}^{-1}]$'
                pl.plot(xs,sim.sfh['means'][5:maxind:10],c='k',lw=0.5)
                if 'm1' in sim.shortname():
                    pl.ylim(0.5,50)
                else:
                    pl.ylim(3,300)
                
            else: 
                ylabel = r'$  %s / \langle %s\rangle_{300{\rm Myr}}$'%(val_str_show,val_str_show)
                pl.ylim(0.1,10.)  
                ax.set_yscale('log')
            if ax.is_first_col():
                pl.ylabel(ylabel,color=pB.niceblue,fontsize=pB.infigfontsize+2)
                if val_str not in ('log_Ts','vrs','diskiness','circularity','Mdot'):
                    ax.yaxis.set_major_formatter(u.arilogformatter)                
            elif val_str in ('vrs','Mdot') and not present:
                pass
            elif (val_str in ('log_nHTs',) or val_str=='sfhs' and not normalize) and not present:
                ax.yaxis.set_major_formatter(u.arilogformatter)                
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            s = sim.shortname()
            if len(self.sims)==1: s = 'FIRE-'+s
            if present: 
                if val_str == 'sfhs': pl.text(0.025,0.05,s,fontsize=8,transform=ax.transAxes)
                if val_str == 'vrs': pl.text(0.975,0.05,s,fontsize=8,transform=ax.transAxes,ha='right')
            else:
                pl.title(s)
        
            ax.xaxis.set_major_locator(ticker.MultipleLocator())
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            if ax.is_last_row():
                pl.xlabel(r'redshift')   
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '%d'%(x-1)))

            if show_axtop and ax.is_first_row():
                ax_top=pl.twiny()
                ax_top.set_xscale('log')            
                zs_arr = np.arange(0.,10.,0.01)
                ages = cosmo.age(zs_arr).value
                ages_to_show = range(2,14,1)
                zsp1_to_show = 10.**np.interp(log(ages_to_show),log(ages[::-1]),log(1+zs_arr[::-1]))
                # ax2.xaxis.set_major_locator(ticker.FixedLocator(1+zs_to_show))
                ax_top.xaxis.set_minor_locator(ticker.NullLocator())
                pl.xticks(zsp1_to_show,[(r'$%d$'%x,'')[x%2==1 and x>10] for x in ages_to_show])
                pl.xlabel('time [Gyr]')
                pl.xlim(*xls)

            if show_ax2:
                ax2=pl.twinx()
                ind_tratio = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(Rcirc2Rvirs[isim%len(Rcirc2Rvirs)]))                
                t_ratio = np.array([(prof.t_cool()/prof.t_ff())[ind_tratio] for prof in sim.profiles])
                #t_ratio = 10.**np.convolve(log(t_ratio),np.ones(5)/5.,mode='same')
                pl.plot(1+sim.zs(),t_ratio,zorder=100,c='r',lw=(1.,1.5)[present])
                if show_t_ratio_at_rvir:
                    t_ratio2 = sim.tcool_Rvirs()/sim.tff_Rvirs()
                    pl.plot(1+sim.zs(),t_ratio2,zorder=100,c='r',lw=0.5)                
                pl.xlim(*xls)
                pl.ylim(0.03,0.03**-1.)
                
                ax2.set_yscale('log')
                ax2.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
                ax.set_zorder(10)
                ax.patch.set_visible(False)        
            
                if ax.is_last_col():
                    ax2.yaxis.set_major_formatter(u.arilogformatter)    
                    pl.ylabel(r'$t_{\rm cool}/t_{\rm ff}\ {\rm at}\ 0.1 R_{\rm vir}$',color='r',fontsize=pB.infigfontsize+2)
                else:
                    ax2.yaxis.set_major_formatter(ticker.NullFormatter())    
                pl.axhline(1.,c='k',lw=0.5,ls=':')
                ax.set_zorder(ax2.get_zorder()-1) # put ax behind ax2 
                ax.patch.set_visible(False) # hide the 'canvas'             
        if savefig: 
            if present:
                pl.savefig(figDir + '%s_profiles%s_nSims%d%s%s.pdf'%(val_str,('_outer','')[r2Rvir==0.1],len(self.sims),suffix,('','with_tratio')[show_ax2]))
            else:
                pl.savefig(figDir + '%s_profiles%s_nSims%d%s%s.pdf'%(val_str,('_outer','')[r2Rvir==0.1],len(self.sims),suffix,('','with_tratio')[show_ax2]),
                           bbox_inches='tight')
    def vs_CFindicator2(self,val_str,val_str_show,weight,r2Rvir=0.1,N=10,show_t_ratio_at_rvir=False,normalize=1,present=False,
                       _ax=None,savefig=True,nRows=3,nCols=3,minR2Rvir=1e-5,maxR2Rvirs=(0.05,),show_ax2=True,show_axtop=True,
                       Rcirc2Rvirs=(0.1,),xls=None,suffix='',fig_height=5):       
        if _ax==None: 
            fig = pl.figure(figsize=(pB.fig_width_full,fig_height))
            pl.subplots_adjust(left=0.125,hspace=0.2,wspace=0.05)        
        for isim,sim in enumerate(self.sims):
            if _ax==None: ax = pl.subplot(nRows,nCols,isim+1)
            else: ax = _ax
            ind = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
            ind_tratio = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(Rcirc2Rvirs[isim%len(Rcirc2Rvirs)]))                
            t_ratio = np.array([(prof.t_cool()/prof.t_ff())[ind_tratio] for prof in sim.profiles])
            xs = t_ratio
            if val_str=='sfhs': 
                y = sim.sfh['sfrs']
                maxind = len(y)//10*10
                xs = 10.**np.interp(log(1+sim.sfh['zs'][5:maxind:10]),log(1+sim.zs()),log(t_ratio))
                log_val = y[:maxind].reshape((len(y)//10,10)).mean(axis=1)
            elif val_str=='diskiness':
                log_val = np.array([0. for prof in sim.profiles])
            elif val_str=='circularity':                
                maxR2Rvir = maxR2Rvirs[isim%len(maxR2Rvirs)]
                inds = ((midbins(Snapshot_profiler.log_r2rvir_bins) < log(maxR2Rvir)) & (midbins(Snapshot_profiler.log_r2rvir_bins) > log(minR2Rvir)) )
                log_val = np.array([np.nan_to_num(prof.circularityProfile(weight)) for prof in sim.profiles if prof.isSaved('circularity_MW')])
                if len(log_val)==0: continue
                ms = np.array([prof.gasMassProfile() for prof in sim.profiles if prof.isSaved('circularity_MW')])
                log_val = (log_val * ms)[:,inds].sum(axis=1)/ms[:,inds].sum(axis=1)
                #log_val = np.array([np.nanmax(prof.circularityProfile(weight)) for prof in sim.profiles if prof.isSaved('circularity_MW')])
                xs = [x for ix,x in enumerate(xs) if sim.profiles[ix].isSaved('circularity_MW')]
            else: 
                log_val = np.array([prof.profile1D(val_str,weight)[ind] for prof in sim.profiles])
            if val_str=='vrs': 
                norm = np.array([prof.vc()[ind] for prof in sim.profiles])
            elif val_str=='sfhs': 
                norm = sim.sfh['means'][5:maxind:10]
            elif val_str=='log_Ts':
                log_val -= log(np.array([prof.Tvir() for prof in sim.profiles]))
                log_val = 10.**log_val
                norm = 1.
            elif val_str=='circularity':
                norm = 1.
            else: 
                norm = np.convolve(log_val, np.ones((N,))/N, mode='same')            
            #smooth_sig_log_nHTs = pl.poly1d(pl.polyfit(log_mvirs,sig_log_nHTs,5))(log_mvirs)
            pl.plot(xs,log_val/norm**normalize,'o',c='k',ms=(1,2)[present])
            if False:#val_str!='sfhs':
                sig_log_val = (np.array([prof.profile1D(val_str,weight,2)[ind] for prof in sim.profiles]) - log_val**2)**0.5
                pl.fill_between(1+sim.zs(), (log_val-sig_log_val)/norm**normalize,(log_val+sig_log_val)/norm**normalize,facecolor='b',alpha=0.2)
            ax.set_xscale('log')
            pl.xlim(0.02,(50,100)[present])
            ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=7,numticks=7))
            ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=7,numticks=7,subs=range(2,10)))
            if ax.is_last_row():                    
                ax.xaxis.set_major_formatter(u.arilogformatter)    
                xlabel = r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ 0.1 R_{\rm vir}$'
                if nCols<4:
                    pl.xlabel(xlabel,color='k',fontsize=pB.infigfontsize+5)
            else:
                ax.xaxis.set_major_formatter(ticker.NullFormatter())    
            pl.axvline(1.,c='.5',lw=0.5,ls='-')


            if val_str=='vrs': 
                pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                if normalize:
                    ylabel=r'$\langle v_r/v_{\rm c}\rangle_{V}$'
                    pl.ylim(-1.7,1.7)  
                    pl.axhline(1.,c='.5',lw=0.5,zorder=1000)                    
                    pl.axhline(-1.,c='.5',lw=0.5,zorder=1000)            
                else:
                    ylabel=r'radial velocity at $0.1 R_{\rm vir}$'                    
                    if 'm12' in sim.shortname():
                        pl.text(1.175,235,r'$v_{\rm c}$')
                        #pl.text(1.1,-250,r'$-v_{\rm c}$')
                        pl.ylim(-300,300) 
                    elif 'm11' in sim.shortname():
                        pl.ylim(-100,100) 
                    else:
                        pl.text(2,500,r'$v_{\rm c}$')
                        #pl.text(1.5,-500,r'$-v_{\rm c}$')
                        pl.ylim(-600,600) 

            elif val_str=='log_Ts':
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))                
                ax.yaxis.set_major_formatter(u.arilogformatter)
                pl.ylim(0.01,10.)                  
                ylabel = r'$T(%sR_{\rm vir})/T_{\rm vir}$'%r_str
                pl.axhline(1.,c='.5',lw=0.5,ls='-')
            elif val_str=='circularity':
                ylabel = r'circularity ($<0.05 R_{\rm vir}$)'
                pl.ylim(0.,1)
            else: 
                ylabel = r'$  %s / \langle %s\rangle_{300{\rm Myr}}$'%(val_str_show,val_str_show)
                ax.set_yscale('log')
                pl.ylim(0.1,10.)  
                ax.set_yscale('log')
            if ax.is_first_col():
                if nRows>=4 or present:
                    if ax.is_first_row():
                        pl.text(0.05,0.5,ylabel,color='k',rotation=90,fontsize=pB.infigfontsize+5,transform=fig.transFigure,ha='center',va='center')
                        pl.text(0.53,0.06,r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ {\rm at}\ 0.1 R_{\rm vir}$',
                                transform=fig.transFigure,fontsize=pB.infigfontsize+5,ha='center')
                else:
                    pl.ylabel(ylabel)
                if val_str not in ('log_Ts','vrs','diskiness','circularity'):
                    ax.yaxis.set_major_formatter(u.arilogformatter)                
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            if present:
                pl.text(0.975,0.1,sim.shortname()+r', CF at $z < %.1f$'%sim.z_cross(),fontsize=10,transform=ax.transAxes,ha='right')
            elif _ax==None:
                pl.text(0.975,0.1,sim.shortname(),fontsize=pB.infigfontsize,transform=ax.transAxes,ha='right')


        if savefig: pl.savefig(figDir + '%s_profiles%s_nSims%d%s%s.pdf'%(val_str,('_outer','')[r2Rvir==0.1],len(
            self.sims),('','with_tratio')[show_ax2],suffix),bbox_inches='tight')


    def vs_CFindicator3(self,val_str,val_str_show,weight,r2Rvirs=(0.1,),N=10,normalize=1,savefig=True,max_ys=np.inf,
                        minR2Rvir=1e-5,maxR2Rvirs=(0.05,), Rcirc2Rvirs=(0.1,),suffix='',show_groups=False,xls=None,res=None,
                        show_means=True,f_Tc=1.,useTvir=False): 
        
        fig = pl.figure(figsize=((pB.fig_width_full*1.1,pB.fig_width_half)[len(r2Rvirs)==1],
                                 pB.fig_width_half))
        pl.subplots_adjust(wspace=0.3)
        for ir,r2Rvir in enumerate(r2Rvirs):            
            ax = pl.subplot(1,len(r2Rvirs),ir+1)
            if res==None:                
                xs = np.array([]); ys = np.array([]); totals = []            
                for isim,sim in enumerate(self.sims):
                    ind = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
                    ind_tratio = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(Rcirc2Rvirs[isim%len(Rcirc2Rvirs)]))                
                    t_ratio = np.array([(prof.t_cool(f_Tc=f_Tc,useTvir=useTvir)/prof.t_ff())[ind_tratio] for prof in sim.profiles])
                    print(sim,sim.zs()[np.where(t_ratio<=0)])
                    if val_str=='sfhs': 
                        y = sim.sfh['sfrs']
                        maxind = len(y)//10*10
                        t_ratio = 10.**np.interp(log(1+sim.sfh['zs'][5:maxind:10]),log(1+sim.zs()),log(t_ratio))
                        log_val = y[:maxind].reshape((len(y)//10,10)).mean(axis=1)
                    elif val_str=='sfh_std':
                        if sim.sfh==None:
                            totals.append(len(xs))
                            continue
                        y = sim.sfh['std_log']
                        maxind = len(y)//10*10
                        t_ratio = 10.**np.interp(log(1+sim.sfh['zs'][5:maxind:10]),log(1+sim.zs()),log(t_ratio))
                        log_val = y[:maxind].reshape((len(y)//10,10)).mean(axis=1)
                        
                    elif val_str=='diskiness':
                        log_val = np.array([0. for prof in sim.profiles])
                    elif val_str=='circularity':                
                        maxR2Rvir = maxR2Rvirs[isim%len(maxR2Rvirs)]
                        inds = ((midbins(Snapshot_profiler.log_r2rvir_bins) < log(maxR2Rvir)) & (midbins(Snapshot_profiler.log_r2rvir_bins) > log(minR2Rvir)) )
                        log_val = np.array([np.nan_to_num(prof.circularityProfile(weight)) for prof in sim.profiles if prof.isSaved('circularity_MW')])
                        if len(log_val)==0: continue
                        ms = np.array([prof.gasMassProfile() for prof in sim.profiles if prof.isSaved('circularity_MW')])
                        log_val = (log_val * ms)[:,inds].sum(axis=1)/ms[:,inds].sum(axis=1)
                        #log_val = np.array([np.nanmax(prof.circularityProfile(weight)) for prof in sim.profiles if prof.isSaved('circularity_MW')])
                        t_ratio = [x for ix,x in enumerate(t_ratio) if sim.profiles[ix].isSaved('circularity_MW')]
                    elif val_str=='supersonic_fraction':
                        log_val = 1-np.array([prof.subsonic_fraction(weight)[ind] for prof in sim.profiles])                    
                    elif val_str=='actual_temperature':
                        log_val = np.array([(prof.t_cool(use_actual_temperature=True)/prof.t_cool())[ind] for prof in sim.profiles])
                    elif val_str=='T_ratio':
                        log_val = np.array([(prof.profile1D('Ts',weight)/prof.Tc())[ind] for prof in sim.profiles])
                    elif val_str=='Tvir2Tc':
                        log_val = np.array([prof.Tc()[ind]/prof.Tvir() for prof in sim.profiles])
                    elif val_str=='Vrot2sigma':
                        log_val = np.zeros(len(sim.profiles))
                        for iprof,prof in enumerate(sim.profiles):
                            if not prof.isSaved('HImassProfile'): 
                                log_val[iprof] = -1
                                print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                                continue
                            if weight=='HI': weights = prof.HImassProfile()  * (prof.rs_midbins() / prof.rvir <maxR2Rvirs) 
                            else:            weights = prof.gasMassProfile() * (prof.rs_midbins() / prof.rvir <maxR2Rvirs)
                            v_phis  = prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvirs))
                            v_phis2 = prof.profile1D('v_phi', weight,power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvirs))
                            vphi_inds = ~np.isnan(v_phis) & ~np.isnan(v_phis2)
                            v_phi  = (v_phis*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                            v_phi2 = (v_phis2*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                            sigma = (v_phi2-v_phi**2)**0.5
                            log_val[iprof] = v_phi/sigma
                    elif val_str=='sigma2vc':
                        log_val = np.zeros(len(sim.profiles))
                        for iprof,prof in enumerate(sim.profiles):
                            if not prof.isSaved('HImassProfile'): 
                                log_val[iprof] = -1
                                print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                                continue
                            if weight=='HI': weights = prof.HImassProfile()  * (prof.rs_midbins() / prof.rvir <maxR2Rvirs) 
                            else:            weights = prof.gasMassProfile() * (prof.rs_midbins() / prof.rvir <maxR2Rvirs)
                            v_phis  = prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvirs))
                            v_phis2 = prof.profile1D('v_phi', weight,power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvirs))
                            vphi_inds = ~np.isnan(v_phis) & ~np.isnan(v_phis2) & ~np.isnan(prof.vc())
                            v_phi  = (v_phis*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                            v_phi2 = (v_phis2*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                            vc = (prof.vc()*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                            sigma = (v_phi2-v_phi**2)**0.5
                            log_val[iprof] = sigma#/vc
                        t_ratio = t_ratio[log_val>0]
                        log_val = log_val[log_val>0]                        
                        
                    elif val_str=='Vrot2vc':
                        log_val = np.zeros(len(sim.profiles))
                        for iprof,prof in enumerate(sim.profiles):
                            if not prof.isSaved('HImassProfile'): 
                                log_val[iprof] = -1
                                print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                                continue
                            if weight=='HI': weights = prof.HImassProfile()  * (prof.rs_midbins() / prof.rvir <maxR2Rvirs) 
                            else:            weights = prof.gasMassProfile() * (prof.rs_midbins() / prof.rvir <maxR2Rvirs)
                            v_phis  = prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvirs))
                            vphi_inds = ~np.isnan(v_phis)
                            v_phi  = (v_phis*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                            vc = (prof.vc()*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                            log_val[iprof] = v_phi#/vc
                        t_ratio = t_ratio[log_val>0]
                        log_val = log_val[log_val>0]                        
                    elif val_str=='vc':
                        log_val = np.zeros(len(sim.profiles))
                        for iprof,prof in enumerate(sim.profiles):
                            if not prof.isSaved('HImassProfile'): 
                                log_val[iprof] = -1
                                print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                                continue
                            if weight=='HI': weights = prof.HImassProfile()  * (prof.rs_midbins() / prof.rvir <maxR2Rvirs) 
                            else:            weights = prof.gasMassProfile() * (prof.rs_midbins() / prof.rvir <maxR2Rvirs)
                            vphi_inds = ~np.isnan(prof.vc())
                            vc = (prof.vc()*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                            log_val[iprof] = vc
                        t_ratio = t_ratio[log_val>0]
                        log_val = log_val[log_val>0]                        
                        
                    else: 
                        log_val = np.array([prof.profile1D(val_str,weight)[ind] for prof in sim.profiles])                    
                        
                    # set normalization
                    if val_str=='vrs': 
                        norm = np.array([prof.vc()[ind] for prof in sim.profiles])
                    elif val_str=='sfhs': 
                        norm = sim.sfh['means'][5:maxind:10]
                    elif val_str=='log_Ts':
                        ind = np.searchsorted(Snapshot_profiler.log_r2rvir_bins,-1)
                        log_val -= log(np.array([prof.Tvir() for prof in sim.profiles]))
                        log_val = 10.**log_val
                        norm = 1.
                    elif val_str in ('circularity','supersonic_fraction','actual_temperature','T_ratio','Tvir2Tc','Vrot2sigma','sfh_std','Vrot2vc','sigma2vc','vc'):
                        norm = 1.
                    else: 
                        norm = np.convolve(log_val, np.ones((N,))/N, mode='same')   
                    xs = np.concatenate([xs,t_ratio])
                    ys = np.concatenate([ys,log_val/norm**normalize])             
                    totals.append(len(xs))
            else:
                xs,ys,totals=res
                    
            ys[ys>max_ys] = max_ys
            if val_str=='actual_temperature':
                ys[ys<0] = max_ys #net heating
            pl.plot(xs,ys,',',c='.5',zorder=-100)
            if show_means:                
                for ip,p in enumerate((16,50,84)):
                    f = lambda arr,p=p: np.nanpercentile(arr,p,axis=1)
                    bins = 10.**np.arange(-3,2,.1)
                    hist,edges,_ = scipy.stats.binned_statistic(xs,ys,statistic=f,bins=bins)
                    pl.plot((edges[1:]*edges[:-1])**0.5,hist,c='k',zorder=100,lw=(1,2,1)[ip],label=('_','all')[ip==1])
            if show_groups:                
                for igroup,(s,e) in enumerate(((None,5),(5,10),(10,15))):
                    f = lambda arr,p=50: np.nanpercentile(arr,p)
                    if s==None: a = 0
                    else: a = totals[s]
                    b = totals[e]                
                    hist,edges,_ = scipy.stats.binned_statistic(xs[a:b],ys[a:b],statistic=f,bins=10.**np.arange(-3,2,.3))
                    counts,_,_   = scipy.stats.binned_statistic(xs[a:b],ys[a:b],statistic='count',bins=10.**np.arange(-3,2,.3))
                    print(counts)
                    pl.plot((edges[1:]*edges[:-1])[counts>15]**0.5,hist[counts>15],c='bgr'[igroup],zorder=(50,150)[val_str=='Vrot2sigma'],lw=(1.,1.5)[val_str=='Vrot2sigma'],
                            label=('m11s','m12s',r'm13s')[igroup])
                    if ax.is_first_col(): 
                        if val_str=='supersonic_fraction': loc = 'upper right'
                        elif val_str=='Vrot2sigma': loc = 'upper left'
                        elif  val_str=='sfh_std': loc='upper right'
                        else: loc = 'lower right'
                        u.mylegend(loc=loc,fontsize=pB.infigfontsize-2,handlelength=1,labelspacing=0.2,handletextpad=0.5)
                
            
            ax.set_xscale('log')
            if xls==None:                
                if Rcirc2Rvirs[0]==0.1: pl.xlim(0.02,50)
                if Rcirc2Rvirs[0]==0.5: pl.xlim(0.01,100)
            else:
                pl.xlim(*xls)
            ax.xaxis.set_major_locator(ticker.LogLocator(numdecs=8,numticks=8))
            ax.xaxis.set_minor_locator(ticker.LogLocator(numdecs=8,numticks=8,subs=range(2,10)))
            ax.xaxis.set_major_formatter(u.arilogformatter)    
            pl.xlabel(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ \ {\rm at}\ \ %.1f R_{\rm vir}$'%Rcirc2Rvirs[ir])
            if val_str!='Vrot2sigma': pl.axvline(1.,c='.5',lw=0.5,ls='-')
            else: pl.axvline(2.,c='.5',lw=0.5,ls='-')
    
            r_str = {0.1:'0.1',0.2:'0.2',0.5:'0.5',1:'',2:'2'}[r2Rvir]
            if val_str=='vrs': 
                pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                if normalize:
                    ylabel=r'$\langle v_r/v_{\rm c}\rangle_{V}$'
                    pl.ylim(-1.7,1.7)  
                    pl.axhline(1.,c='.5',lw=0.5,zorder=1000)                    
                    pl.axhline(-1.,c='.5',lw=0.5,zorder=1000)            
                else:
                    ylabel=r'radial velocity at $0.1 R_{\rm vir}$'                    
                    if 'm12' in sim.shortname():
                        pl.text(1.175,235,r'$v_{\rm c}$')
                        #pl.text(1.1,-250,r'$-v_{\rm c}$')
                        pl.ylim(-300,300) 
                    elif 'm11' in sim.shortname():
                        pl.ylim(-100,100) 
                    else:
                        pl.text(2,500,r'$v_{\rm c}$')
                        #pl.text(1.5,-500,r'$-v_{\rm c}$')
                        pl.ylim(-600,600) 
            elif val_str=='Vrot2sigma':
                pl.ylim(0,(7,8)[weight=='HI'])
                ylabel = r'$V_{\rm rot} / \sigma_{\rm g}$'                
                pl.xlim(1e-3,100)
            elif val_str=='log_Ts':
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))                                
                ax.yaxis.set_major_formatter(u.arilogformatter)
                pl.ylim(0.02,5.)  
                
                ylabel = r'$\langle T\rangle(%sR_{\rm vir})/T_{\rm vir}$'%r_str
                #pl.text(0.95,0.9,r'',transform=ax.transAxes,ha='right',fontsize=pB.infigfontsize)
                pl.axhline(1.,c='.5',lw=0.5,ls='-')
            elif val_str=='circularity':
                ylabel = r'circularity $(j_z/j_{\rm c}(E))$'
                pl.ylim(0.,1)
            elif val_str=='supersonic_fraction':
                ylabel = r'supersonic fraction at $%sR_{\rm vir}$'%r_str
                pl.ylim(0.,1)
            elif val_str=='actual_temperature':                
                ax.set_yscale('symlog',linthreshy=0.01)
                ylabel = r'$t_{\rm cool}(T)/t_{\rm cool}^{\rm (s)}$'
                pl.ylim(0.001,1000.)                
            elif val_str=='T_ratio':                
                ax.set_yscale('log')
                ylabel = r'$T/T_{\rm c}$'
                pl.ylim(0.03,4.)                
            elif val_str=='Tvir2Tc':                
                ax.set_yscale('log')
                ylabel = r'$T_{\rm c}/T_{\rm vir}$'
                pl.ylim(0.5,5.)
            elif val_str=='sfh_std':
                ylabel = r'$\sigma(\log\ {\rm SFR})\ [{\rm dex}]$'
                pl.ylim(0,0.6)
                pl.xlim(1e-3,100)
            else: 
                ylabel = r'$  %s / \langle %s\rangle_{300{\rm Myr}}$'%(val_str_show,val_str_show)
                ax.set_yscale('log')
                pl.ylim(0.1,10.)  
                ax.set_yscale('log')
            pl.ylabel(ylabel)
            if val_str not in ('log_Ts','vrs','diskiness','circularity','supersonic_fraction','Vrot2sigma','sfh_std'):
                ax.yaxis.set_major_formatter(u.arilogformatter)                
        if savefig: 
            if Rcirc2Rvirs[0]==0.1:                
                pl.savefig(figDir + '%s_vs_t_ratio%s.pdf'%(val_str,suffix),bbox_inches='tight')
            else:
                pl.savefig(figDir + '%s_vs_t_ratio_outer%s.pdf'%(val_str,suffix),bbox_inches='tight')

        return xs,ys,totals
    def vs_CFindicator4(self,val_str,val_str_show,weight=None,r2Rvir=0.1,N=10,show_t_ratio_at_rvir=False,normalize=1,present=False,
                       _ax=None,savefig=True,minR2Rvir=1e-5,maxR2Rvirs=(0.05,),show_ax2=True,show_axtop=True,
                       Rcirc2Rvirs=(0.1,),_xls=None,suffix='',fig_height=5):       
        nCols = len(self.sims)
        if _ax==None: 
            pl.figure(figsize=(pB.fig_width_full*(1.05,1)[present],5))
            pl.subplots_adjust(hspace=1e-3,wspace=(0.1,0.25)[val_str in ('vrs','log_nHTs','Mdot') or (val_str=='sfhs' and not normalize)])
        for isim,sim in enumerate(self.sims):
            if _ax==None: ax = pl.subplot(2,nCols,isim+1)
            else: ax = _ax
            ind = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
            label='_nolegend_'
            if val_str=='sfhs': 
                y = sim.sfh['sfrs']
                maxind = len(y)//10*10
                xs = 1+sim.sfh['zs'][5:maxind:10]                
                log_val = y[:maxind].reshape((len(y)//10,10)).mean(axis=1)
            elif val_str=='diskiness':
                xs = 1+sim.zs()
                log_val = np.array([0. for prof in sim.profiles])
            elif val_str=='circularity':
                xs = 1+sim.zs()
                maxR2Rvir = maxR2Rvirs[isim%len(maxR2Rvirs)]
                inds = ((midbins(Snapshot_profiler.log_r2rvir_bins) < log(maxR2Rvir)) & 
                        (midbins(Snapshot_profiler.log_r2rvir_bins) > log(minR2Rvir)) )
                log_val = np.array([np.nan_to_num(prof.circularityProfile(weight)) for prof in sim.profiles
                                    if prof.isSaved('circularity_MW')])
                ms = np.array([prof.gasMassProfile() for prof in sim.profiles
                               if prof.isSaved('circularity_MW')])
                log_val = (log_val * ms)[:,inds].sum(axis=1)/ms[:,inds].sum(axis=1)
            elif val_str=='Mdot': 
                xs = 1+sim.zs()
                log_val = np.array([(prof.gasMassProfile() * prof.profile1D('vrs','MW')/prof.drs_midbins())[ind] * (un.km/un.s/un.kpc).to('yr**-1') for prof in sim.profiles])                
            
            else: 
                xs = 1+sim.zs()
                log_val = np.array([prof.profile1D(val_str,weight)[ind] for prof in sim.profiles])
                
            if val_str=='vrs': 
                norm = np.array([prof.vc()[ind] for prof in sim.profiles])
            elif val_str=='sfhs': 
                norm = sim.sfh['means'][5:maxind:10]
                if not normalize: label=r"${\rm instantaneous}$"
            elif val_str=='log_Ts':
                log_val -= log(np.array([prof.Tvir() for prof in sim.profiles]))
                norm = 1.
            elif val_str=='log_nHTs':
                log_val = 10.**log_val
                norm = 1.
                label=r"instant"
            elif val_str in ('circularity','Mdot'):
                norm = 1.
            else: 
                norm = np.convolve(log_val, np.ones((N,))/N, mode='same')            
            #smooth_sig_log_nHTs = pl.poly1d(pl.polyfit(log_mvirs,sig_log_nHTs,5))(log_mvirs)
            pl.plot(xs,log_val/norm**normalize,c=pB.niceblue,lw=0.5,label=label)
            if False: #val_str=='log_nHTs':
                sig_log_val = (np.array([prof.profile1D(val_str,weight,2)[ind] for prof in sim.profiles]) - log(log_val)**2)**0.5
                pl.fill_between(1+sim.zs(), 10.**(log(log_val)-sig_log_val),10.**(log(log_val)+sig_log_val),facecolor='b',alpha=0.2)            
            if _xls==None:
                if 'm1' in sim.shortname(): xls=1,3
                if 'A' in sim.shortname(): xls=((2,8),(2,6))[present]
                if 'HL' in sim.shortname(): xls=6,12
            else: xls=_xls
            ax.set_xscale('log')
            pl.xlim(*xls)
            r_str = (('%.1f'%r2Rvir,'')[r2Rvir==1])
            if val_str=='vrs': 
                pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                if normalize:
                    ylabel=r'$\langle v_r/v_{\rm c}\rangle_{V}$'
                    pl.ylim(-1.7,1.7)  
                    pl.axhline(1.,c='.5',lw=0.5,zorder=1000)                    
                    pl.axhline(-1.,c='.5',lw=0.5,zorder=1000)            
                else:
                    ylabel=r'radial velocity at $%.1f R_{\rm vir}\ [{\rm km}~{\rm s}^{-1}]$'%r2Rvir 
                    [pl.plot(xs,j*norm,c='.5',lw=0.5,zorder=1000) for j in (-1,1)]
                    if 'm12' in sim.shortname():
                        if present: pl.text(1.175,235,r'$v_{\rm c}$')
                        if sim.shortname()=='m12i':
                            pl.text(1.175,235,r'$v_{\rm c}$')
                            pl.annotate('',(1.3,200.),(1.27,225),arrowprops=pB.slantlinepropsblack) 
                        #pl.text(1.1,-250,r'$-v_{\rm c}$')
                        pl.ylim(-300,300) 
                    elif 'm11' in sim.shortname():
                        #pl.text(1.175,100,r'$v_{\rm c}$')
                        #pl.text(1.1,-250,r'$-v_{\rm c}$')
                        pl.ylim(-130,130) 
                    else:
                        #pl.text(2.5,400,r'$v_{\rm c}$')
                        #pl.text(1.5,-500,r'$-v_{\rm c}$')
                        pl.ylim(-500,500) 
            elif val_str=='Mdot': 
                pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
                pl.ylim(-100,100) 
                if 'm12' in sim.shortname():
                    pl.ylim(-50,50) 
                elif 'm11' in sim.shortname():
                    pl.ylim(-20,20) 
                else:
                    pl.ylim(-300,300) 
                ylabel = r'$\dot{M}(%s R_{\rm vir})\ [{\rm M}_\odot\ {\rm yr}^{-1}]$'%r_str
                        
            elif val_str=='log_Ts':
                pl.ylim(-1,1)  
                ylabel = r'$\log T/T_{\rm vir}$'
            elif val_str=='circularity':
                ylabel = r'circularity $(j_z/j_{\rm c}(E))$'
                pl.ylim(0.,1)
            elif val_str=='log_nHTs':
                mean_nHT = 10.**np.convolve(log(log_val), np.ones((N,))/N, mode='same')
                pl.plot(xs,mean_nHT,c='k',lw=0.5,label=r'mean')
                if ax.is_first_col():
                    u.mylegend(loc='upper left')
                    pl.text(0.03,0.04,r'$0.1 R_{\rm vir}$',transform=ax.transAxes,fontsize=pB.infigfontsize)
                ylabel = r'$\langle n_{\rm H}T\rangle\ [{\rm cm}^{-3}{\rm K}]$'
                ax.set_yscale('log')
                if r2Rvir==0.1:                    
                    if 'A' in sim.shortname():
                        pl.ylim(5e3,0.5e6)
                    else:
                        pl.ylim(30,5000)
                else:
                    if 'm1' in sim.shortname():
                        pl.ylim(5,200)
                    else:
                        pl.ylim(300,1.2e4)
                    
            elif val_str=='sfhs' and not normalize:                
                ax.set_yscale('log')
                ylabel = r'${\rm SFR}\ [{\rm M}_\odot{\rm yr}^{-1}]$'
                pl.plot(xs,sim.sfh['means'][5:maxind:10],c='k',lw=0.5,label=r'${\rm running\ mean}$')
                if ax.is_first_col():
                    u.mylegend(loc='lower left',handlelength=0.5,fontsize=pB.infigfontsize,handletextpad=0.4)
                if 'm13' not in sim.shortname():
                    pl.ylim(0.5,50)
                else:
                    pl.ylim(3,300)
                
            else: 
                ylabel = r'$  %s / \langle %s\rangle_{300{\rm Myr}}$'%(val_str_show,val_str_show)
                pl.ylim(0.1,10.)  
                ax.set_yscale('log')
            if ax.is_first_col():
                pl.ylabel(ylabel,color=pB.niceblue,fontsize=pB.infigfontsize+2)
                if val_str not in ('log_Ts','vrs','diskiness','circularity','Mdot'):
                    ax.yaxis.set_major_formatter(u.arilogformatter)                
            elif val_str==('vrs','Mdot') and not present:
                pass
            elif (val_str=='log_nHTs' or val_str=='sfhs' and not normalize) and not present:
                ax.yaxis.set_major_formatter(u.arilogformatter)                
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())
            s = sim.shortname()
            if len(self.sims)==1: s = 'FIRE-'+s
            if present: 
                if val_str == 'sfhs': pl.text(0.025,0.05,s,fontsize=8,transform=ax.transAxes)
                if val_str == 'vrs': pl.text(0.975,0.05,s,fontsize=8,transform=ax.transAxes,ha='right')
            else:
                pl.title(s)
            pl.axvline(sim.z_cross()+1,c='k',lw=0.5,ls=':')
            ax.xaxis.set_major_locator(ticker.MultipleLocator())
            ax.xaxis.set_minor_locator(ticker.NullLocator())                        
            ax.xaxis.set_major_formatter(ticker.NullFormatter())


            if show_axtop and ax.is_first_row():
                ax_top=pl.twiny()
                ax_top.set_xscale('log')            
                zs_arr = np.arange(0.,10.,0.01)
                ages = cosmo.age(zs_arr).value
                ages_to_show = range(2,14,1)
                zsp1_to_show = 10.**np.interp(log(ages_to_show),log(ages[::-1]),log(1+zs_arr[::-1]))
                # ax2.xaxis.set_major_locator(ticker.FixedLocator(1+zs_to_show))
                ax_top.xaxis.set_minor_locator(ticker.NullLocator())
                pl.xticks(zsp1_to_show,[(r'$%d$'%x,'')[x%2==1 and x>10] for x in ages_to_show])
                pl.xlabel('time [Gyr]')
                pl.xlim(*xls)

            if show_ax2:
                ax2 = pl.subplot(2,nCols,isim+1+nCols)
                ind_tratio = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(Rcirc2Rvirs[isim%len(Rcirc2Rvirs)]))                
                t_ratio = np.array([(prof.t_cool()/prof.t_ff())[ind_tratio] for prof in sim.profiles])
                #t_ratio = 10.**np.convolve(log(t_ratio),np.ones(5)/5.,mode='same')
                pl.plot(1+sim.zs(),t_ratio,zorder=100,c='r',lw=(1.,1.5)[present])
                if show_t_ratio_at_rvir:
                    t_ratio2 = sim.tcool_Rvirs()/sim.tff_Rvirs()
                    pl.plot(1+sim.zs(),t_ratio2,zorder=100,c='r',lw=0.5)                
                ax2.set_xscale('log')
                pl.xlim(*xls)
                ax2.xaxis.set_major_locator(ticker.MultipleLocator())
                ax2.xaxis.set_minor_locator(ticker.NullLocator())
                pl.xlabel(r'redshift')   
                ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '%d'%(x-1)))
                
                
                pl.ylim(0.03,0.03**-1.)
                
                ax2.set_yscale('log')
                ax2.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
                #ax.set_zorder(10)
                #ax.patch.set_visible(False)        
                ax2.yaxis.set_major_formatter(u.arilogformatter)    
                if ax2.is_first_col():                    
                    pl.ylabel(r'$t_{\rm cool}^{(s)}/t_{\rm ff}$',color='r',fontsize=pB.infigfontsize+2)
                    pl.text(0.03,0.04,r'$0.1 R_{\rm vir}$',transform=ax2.transAxes,fontsize=pB.infigfontsize)
                pl.axhline(2.,c='k',lw=0.5,ls=':')
                pl.axvline(sim.z_cross()+1,c='k',lw=0.5,ls=':')
                #ax.set_zorder(ax2.get_zorder()-1) # put ax behind ax2 
                #ax.patch.set_visible(False) # hide the 'canvas'             
        if savefig: 
            pl.savefig(figDir + '%s_profiles%s_nSims%d%s%s_tworows.pdf'%(val_str,('_outer','')[r2Rvir==0.1],len(self.sims),suffix,('','with_tratio')[show_ax2]),
                           bbox_inches='tight')
    def vs_CFindicator5(self,maxR2Rvir=0.05,r2Rvir=0.1,N=25,d=2,t_ratio_factor=0.8,is_one_Myr=True):       
        f = (1,10)[is_one_Myr]
        nCols = len(self.sims)
        pl.figure(figsize=(pB.fig_width_full*1.05,6))
        pl.subplots_adjust(hspace=1e-3,wspace=0.3)
        ind_galaxy   = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(maxR2Rvir))
        ind_innerCGM = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
        label='_nolegend_'
        for isim,sim in enumerate(self.sims):
            good_prof_inds = np.array([prof.isSaved('Mdot_in') for prof in sim.profiles])
            if 'A' in sim.shortname():    xls=(2,8)
            elif 'z5' in sim.shortname(): xls=5.4,9
            else:                         xls=1,3            
            for iRow in range(2):                    
                ax = pl.subplot(3,nCols,iRow*nCols+isim+1)                
                if iRow==0: 
                    y = sim.sfh['sfrs']
                    maxind = len(y)//f*f
                    xs = 1+sim.sfh['zs'][f//2:maxind:f]                
                    vals = y[:maxind].reshape((len(y)//f,f)).mean(axis=1)                     
                    avgs = sim.sfh['means'][f//2:maxind:f]
                    ylabel = r'${\rm SFR}\ [{\rm M}_\odot{\rm yr}^{-1}]$'                    
                    pl.text(0.5,1.1,sim.shortname(),transform=ax.transAxes,ha='center',fontsize=14)
                if iRow==1: 
                    xs = 1+sim.zs()
                    vals = np.array([(prof.gasMassProfile() * prof.profile1D('vrs','MW'))[ind_innerCGM-d:ind_innerCGM+d+1].sum() /
                                     prof.drs_midbins()[ind_innerCGM-d:ind_innerCGM+d+1].sum() 
                                     * (un.km/un.s/un.kpc).to('yr**-1') for prof in sim.profiles])
                    ylabel = r'$\dot{M}(0.1 R_{\rm vir})\ [{\rm M}_\odot{\rm yr}^{-1}]$'                    
                    avgs = np.convolve(vals, np.ones((N,))/N, mode='same')
                    #xs2 = [prof.z for prof in sim.profiles if prof.isSaved('Mdot_in')]
                    #Mdotins = np.array([prof.MdotProfile(suffix='_in')[ind_innerCGM] for prof in sim.profiles if prof.isSaved('Mdot_in')])
                    #pl.plot(xs2,Mdotins,c='k',lw=0.5,label=r'mean')
                pl.plot(xs,vals,c=pB.niceblue,lw=0.5,label='instant')
                pl.plot(xs,avgs,c='k',lw=0.5,label=r'mean')
                if iRow==0:
                    ax.set_yscale('log')
                    ax.yaxis.set_major_formatter(u.arilogformatter)                                                    
                    if isim==1:
                        u.mylegend(loc='lower left',handlelength=1.5,
                                   fontsize=pB.infigfontsize) 
                    if sim.shortname()=='m12i':  pl.ylim(0.3,100)
                    if sim.shortname()=='m12b':  pl.ylim(0.3,100)
                    if sim.shortname() in ('m13A1',): pl.ylim(3,1000)          
                    if sim.shortname() in ('m13A8','m13A4','m13A2'): pl.ylim(0.3,1000)          
                    if sim.shortname()=='m11b': pl.ylim(3e-4,0.1)          
                    if sim.shortname() in ('m11i','m11e'): pl.ylim(0.002,3)
                    if sim.shortname() in ('m11h','m11v'): pl.ylim(0.01,3)
                    if sim.shortname() in ('m11d',): pl.ylim(0.03,10)
                    if sim.shortname() in ('m12z',): pl.ylim(0.1,30)
                    if sim.shortname() in ('m12f','m12m'): pl.ylim(0.3,100)
                    if 'z5' in sim.shortname(): pl.ylim(10,3000)                        
                if iRow==1:
                    pl.axhline(0.,c='.5',lw=0.5)                    
                    if sim.shortname() in ('m12i','m12b'):  pl.ylim(-50,50) 
                    if sim.shortname() in ('m13A1',): 
                        pl.ylim(-300,300)          
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
                    if sim.shortname() in ('m13A8','m13A4'): 
                        pl.ylim(-350,350)          
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(150))
                    if sim.shortname() in ('m13A2',): 
                        pl.ylim(-600,600)          
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(300))
                    if sim.shortname() in ('m11b',): 
                        pl.ylim(-3,3)          
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                    if sim.shortname() in ('m11i','m11e'): 
                        pl.ylim(-13,13)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                    if sim.shortname() in ('m11h','m11v','m11d'): pl.ylim(-25,25)
                    if sim.shortname() in ('m12z','m12f','m12m'): 
                        pl.ylim(-75,75)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
                    if 'z5' in sim.shortname(): pl.ylim(-1500,1500)                        
                    
                    
                    
                ax.set_xscale('log')
                pl.xlim(*xls)
                if ax.is_first_col(): 
                    pl.ylabel(ylabel,color=pB.niceblue,fontsize=pB.infigfontsize+2)
                if sim.z_cross()!=None:
                    pl.axvline(sim.z_cross()+1,c='k',lw=0.5,ls=':')
                if xls==(1,3):
                    ax.xaxis.set_major_locator(ticker.FixedLocator([1,1.5,2,3]))
                else:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator())
                ax.xaxis.set_minor_locator(ticker.NullLocator())                        
                ax.xaxis.set_major_formatter(ticker.NullFormatter())
    
            ax2 = pl.subplot(3,nCols,2*nCols+isim+1)                
            ind_tratio = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
            t_ratio = np.array([(prof.t_cool()/prof.t_ff())[ind_tratio] for prof in sim.profiles])
            pl.plot(1+sim.zs(),t_ratio*t_ratio_factor,zorder=100,c='r',lw=1.)
            ax2.set_xscale('log')
            pl.xlim(*xls)
            if xls==(1,3):
                ax2.xaxis.set_major_locator(ticker.FixedLocator([1,1.5,2,3]))
            else:
                ax2.xaxis.set_major_locator(ticker.MultipleLocator())

            ax2.xaxis.set_minor_locator(ticker.NullLocator())
            pl.xlabel(r'redshift')   
            ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '%s'%u.nSignificantDigits(x-1,1,True)))
            if sim.shortname() in ('m12i','m12b','m13A1','m12z','m12f','m12m'): pl.ylim(0.03,0.03**-1.)                    
            if sim.shortname() in ('m11b','m11i','m11e','m11h','m11v','m11d'): pl.ylim(0.01,6)
            if sim.shortname() in ('m13A8','m13A4','m13A2'): pl.ylim(0.01,100)
            if 'z5' in sim.shortname(): pl.ylim(0.01,100) 
            
            
            ax2.set_yscale('log')
            ax2.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
            ax2.yaxis.set_major_formatter(u.arilogformatter)    
            if ax2.is_first_col():                    
                pl.ylabel(r'$t_{\rm cool}^{(s)}/t_{\rm ff}$ at $0.1 R_{\rm vir}$',color='r',fontsize=pB.infigfontsize+2)
                #pl.text(0.03,0.04,r'$0.1 R_{\rm vir}$',transform=ax2.transAxes,fontsize=pB.infigfontsize)
            pl.axhline(2.,c='k',lw=0.5,ls=':')
            if sim.z_cross()!=None:
                pl.axvline(sim.z_cross()+1,c='k',lw=0.5,ls=':')
        pl.savefig(figDir + 'vs_CFindicator_all_%s.pdf'%sim,bbox_inches='tight')
    def vs_CFindicator6(self,val_str,val_str_show,weight,N=10,normalize=1,max_ys=np.inf,
                        minR2Rvir=1e-5,maxR2Rvirs=(0.05,),f_Tc=1.,useTvir=False,d=5,minSFR=0,is_one_Myr=True):         
        xs = np.array([]); ys = np.array([]); totals = []            
        f = (1,10)[is_one_Myr]
        for isim,sim in enumerate(self.sims):
            maxR2Rvir = maxR2Rvirs[isim%len(maxR2Rvirs)]
            ind        = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(sim.profiles[0].Rcirc2Rvir))
            ind_tratio = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(sim.profiles[0].Rcirc2Rvir))                
            if weight=='SFR':
                profs = [prof for prof in sim.profiles if prof.isSaved('j_vec_weight_SFR') and prof.SFRprofile().cumsum()[200]>=minSFR]
            elif weight=='HI':
                profs = [prof for prof in sim.profiles if prof.isSaved('j_vec_weight_HI')]
            else:
                profs = sim.profiles
            t_ratio = np.array([(prof.t_cool(f_Tc=f_Tc,useTvir=useTvir)/prof.t_ff())[ind_tratio] for prof in profs])
            
            if val_str=='sfhs': 
                y = sim.sfh['sfrs']
                maxind = len(y)//f*f
                t_ratio = 10.**np.interp(log(1+sim.sfh['zs'][f//2:maxind:f]),log(1+sim.zs()),log(t_ratio))
                log_val = y[:maxind].reshape((len(y)//f,f)).mean(axis=1)
            elif val_str=='sfh_std':
                if sim.sfh==None:
                    totals.append(len(xs))
                    continue
                y = sim.sfh['std_log']
                maxind = len(y)//f*f
                y = np.interp(log(1+sim.zs()),log(1+sim.sfh['zs'][f//2:maxind:f]),y)
                log_val = y[:maxind].reshape((len(y)//f,f)).mean(axis=1)
                inds = sim.zs()<5
                t_ratio = t_ratio[inds]
                log_val = log_val[inds]
                
            elif val_str=='diskiness':
                log_val = np.array([0. for prof in profs])
            elif val_str=='circularity':                                
                inds = ((midbins(Snapshot_profiler.log_r2rvir_bins) < log(maxR2Rvir)) & (midbins(Snapshot_profiler.log_r2rvir_bins) > log(minR2Rvir)) )
                log_val = np.array([np.nan_to_num(prof.circularityProfile(weight)) for prof in sim.profiles if prof.isSaved('circularity_MW')])
                if len(log_val)==0: continue
                ms = np.array([prof.gasMassProfile() for prof in sim.profiles if prof.isSaved('circularity_MW')])
                log_val = (log_val * ms)[:,inds].sum(axis=1)/ms[:,inds].sum(axis=1)
                #log_val = np.array([np.nanmax(prof.circularityProfile(weight)) for prof in sim.profiles if prof.isSaved('circularity_MW')])
                t_ratio = [x for ix,x in enumerate(t_ratio) if sim.profiles[ix].isSaved('circularity_MW')]
            elif val_str=='supersonic_fraction':
                log_val = 1-np.array([prof.thickerShells(prof.subsonic_fraction(weight),d,weight)[ind] for prof in profs])                    
            elif val_str=='actual_temperature':
                log_val = np.array([(prof.t_cool(use_actual_temperature=True)/prof.t_cool())[ind] for prof in profs])
            elif val_str=='T_ratio':
                log_val = np.array([prof.thickerShells(
                    prof.profile1D('log_Ts',weight) - log(prof.Tc()),
                    d,weight)[ind] for prof in profs])
            elif val_str=='nH_ratio':
                log_val = np.array([prof.thickerShells(
                    X*prof.rhoProfile()/cons.m_p.to('g').value/prof.nH(),
                    d,weight)[ind] for prof in profs])
            elif val_str=='Tvir2Tc':
                log_val = np.array([prof.Tc()[ind]/prof.Tvir() for prof in profs])
            elif val_str=='Vrot2sigma':
                log_val = np.zeros(len(profs))
                for iprof,prof in enumerate(profs):
                    if not prof.isSaved('HImassProfile'): 
                        log_val[iprof] = -1
                        print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                        continue
                    if weight=='MW': weights = prof.gasMassProfile()
                    else: weights =  prof.get_saved('j_vec_weight_%s'%weight)             
                    weights = weights * ((prof.rs_midbins() / prof.rvir) <maxR2Rvir)
                    v_phis  = prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvir,weight=weight))                    
                    v_phis2 = prof.profile1D('v_phi', weight,power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvir,weight=weight))
                    vphi_inds = ~np.isnan(v_phis) & ~np.isnan(v_phis2)
                    v_phi  = (v_phis*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                    v_phi2 = (v_phis2*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                    sigma = (v_phi2-v_phi**2)**0.5
                    log_val[iprof] = v_phi/sigma
                    if log_val[iprof]>30:
                        return prof
                        #print(v_phi,sigma,vphi_inds[weights!=0],v_phis,prof.central_jvec(maxR2Rvir_disk=maxR2Rvir,weight=weight))
            elif val_str=='sigma2vc':
                log_val = np.zeros(len(profs))
                for iprof,prof in enumerate(profs):
                    if not prof.isSaved('HImassProfile'): 
                        log_val[iprof] = -1
                        print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                        continue
                    if weight=='MW': weights = prof.gasMassProfile()
                    else: weights =  prof.get_saved('j_vec_weight_%s'%weight)             
                    weights = weights * ((prof.rs_midbins() / prof.rvir) <maxR2Rvirs)
                    v_phis  = prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvirs,weight=weight))
                    v_phis2 = prof.profile1D('v_phi', weight,power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvirs,weight=weight))
                    vphi_inds = ~np.isnan(v_phis) & ~np.isnan(v_phis2) & ~np.isnan(prof.vc())
                    v_phi  = (v_phis*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                    v_phi2 = (v_phis2*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                    vc = (prof.vc()*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                    sigma = (v_phi2-v_phi**2)**0.5
                    log_val[iprof] = sigma#/vc
                t_ratio = t_ratio[log_val>0]
                log_val = log_val[log_val>0]                        
                
            elif val_str=='Vrot2vc':
                log_val = np.zeros(len(profs))
                for iprof,prof in enumerate(profs):
                    if not prof.isSaved('HImassProfile'): 
                        log_val[iprof] = -1
                        print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                        continue
                    if weight=='MW': weights = prof.gasMassProfile()
                    else: weights =  prof.get_saved('j_vec_weight_%s'%weight)             
                    weights = weights * ((prof.rs_midbins() / prof.rvir) <maxR2Rvir)
                    v_phis  = prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvir,weight=weight))
                    vphi_inds = ~np.isnan(v_phis)
                    v_phi  = (v_phis*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                    vc = (prof.vc()*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                    log_val[iprof] = v_phi#/vc
                t_ratio = t_ratio[log_val>0]
                log_val = log_val[log_val>0]                        
            elif val_str=='vc':
                log_val = np.zeros(len(profs))
                for iprof,prof in enumerate(profs):
                    if not prof.isSaved('HImassProfile'): 
                        log_val[iprof] = -1
                        print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                        continue
                    if weight=='MW': weights = prof.gasMassProfile()
                    else: weights =  prof.get_saved('j_vec_weight_%s'%weight)             
                    weights = weights * ((prof.rs_midbins() / prof.rvir) <maxR2Rvirs)
                    vphi_inds = ~np.isnan(prof.vc())
                    vc = (prof.vc()*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                    log_val[iprof] = vc
                t_ratio = t_ratio[log_val>0]
                log_val = log_val[log_val>0]                        
                
            else: 
                log_val = np.array([prof.thickerShells(prof.profile1D(val_str,weight),d,weight)[ind] for prof in sim.profiles])                    
                
            # set normalization
            if val_str=='vrs': 
                norm = np.array([prof.vc()[ind] for prof in profs])
            elif val_str=='sfhs': 
                norm = sim.sfh['means'][f//2:maxind:f]
            elif val_str=='log_Ts':
                log_val -= log(np.array([prof.Tvir() for prof in profs]))
                log_val = 10.**log_val
                norm = 1.
            elif val_str in ('circularity','supersonic_fraction','actual_temperature','nH_ratio','T_ratio','Tvir2Tc','Vrot2sigma','sfh_std','Vrot2vc','sigma2vc','vc'):
                norm = 1.
            else: 
                norm = np.convolve(log_val, np.ones((N,))/N, mode='same')   
            xs = np.concatenate([xs,t_ratio])
            ys = np.concatenate([ys,log_val/norm**normalize])                         
            totals.append(len(xs))
        ys[ys>max_ys] = max_ys
        return xs,ys,totals
    def vs_CFindicator7(self,maxR2Rvir=0.05,r2Rvir=0.1,d=2,t_ratio_factor=0.8,weight='HI'):       
        nCols = len(self.sims)
        pl.figure(figsize=(pB.fig_width_half*1.15,3))
        pl.subplots_adjust(hspace=1e-3,wspace=0.3)
        ind_galaxy   = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(maxR2Rvir))
        ind_innerCGM = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
        label='_nolegend_'
        for isim,sim in enumerate(self.sims):
            good_prof_inds = np.array([prof.isSaved('Mdot_in') for prof in sim.profiles])
            xls = cosmo.age(np.array([10,0])).value
            for iRow in range(3):                    
                ax = pl.subplot(3,nCols,iRow*nCols+isim+1)                
                if iRow==1: 
                    y = sim.sfh['sfrs']
                    maxind = len(y)//10*10
                    xs = 1+sim.sfh['zs'][5:maxind:10]                
                    vals = y[:maxind].reshape((len(y)//10,10)).mean(axis=1)                     
                    ylabel = r'${\rm SFR}$'+'\n'+r'$[{\rm M}_\odot{\rm yr}^{-1}]$'                                        
                if iRow==2: 
                    xs = 1+sim.zs()
                    vals = np.array([(prof.gasMassProfile() * prof.profile1D('vrs','MW'))[ind_innerCGM-d:ind_innerCGM+d+1].sum() /
                                     prof.drs_midbins()[ind_innerCGM-d:ind_innerCGM+d+1].sum() 
                                     * (un.km/un.s/un.kpc).to('yr**-1') for prof in sim.profiles])
                    ylabel = r'$\dot{M}(0.1 R_{\rm vir})$'+'\n'+r'$[{\rm M}_\odot{\rm yr}^{-1}]$'                    
                    
                    #xs2 = [prof.z for prof in sim.profiles if prof.isSaved('Mdot_in')]
                    #Mdotins = np.array([prof.MdotProfile(suffix='_in')[ind_innerCGM] for prof in sim.profiles if prof.isSaved('Mdot_in')])
                    #pl.plot(xs2,Mdotins,c='k',lw=0.5,label=r'mean')
                if iRow==0:
                    ylabel = r'$V_{\rm rot} / \sigma_{\rm g}$'
                    xs = 1+sim.zs()
                    vals = np.zeros(len(sim.profiles))
                    for iprof,prof in enumerate(sim.profiles):
                        if not prof.isSaved('HImassProfile'): 
                            vals[iprof] = -1
                            print('no HI mass profile %s z=%.4f'%(prof.galaxyname,prof.z))
                            continue
                        if weight=='HI': weights = prof.HImassProfile()  * (prof.rs_midbins() / prof.rvir <maxR2Rvir) 
                        else:            weights = prof.gasMassProfile() * (prof.rs_midbins() / prof.rvir <maxR2Rvir)
                        v_phis  = prof.profile1D('v_phi', weight,power=1,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvir))
                        v_phis2 = prof.profile1D('v_phi', weight,power=2,z_vec=prof.central_jvec(maxR2Rvir_disk=maxR2Rvir))
                        vphi_inds = ~np.isnan(v_phis) & ~np.isnan(v_phis2)
                        v_phi  = (v_phis*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                        v_phi2 = (v_phis2*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
                        sigma = (v_phi2-v_phi**2)**0.5
                        vals[iprof] = v_phi/sigma
                    
                times = cosmo.age(xs-1)    
                pl.plot(times,vals,c=pB.niceblue,lw=0.5,label='instant')
                N = (100,50)[iRow!=1]                        
                avgs = scipy.ndimage.filters.uniform_filter1d(vals[~np.isnan(vals)], N,mode='nearest')
                pl.plot(times[~np.isnan(vals)],avgs,c='k',lw=0.5,label=r'average')
                if iRow==1:
                    ax.set_yscale('log')
                    if sim.shortname()=='m12i':  pl.ylim(0.5,50)
                    if sim.shortname()=='m12b':  pl.ylim(0.5,80)
                    if sim.shortname() in ('m13A1',): pl.ylim(3,1000)          
                    if sim.shortname() in ('m13A8','m13A4','m13A2'): pl.ylim(1,1000)          
                    if sim.shortname()=='m11b': pl.ylim(3e-4,0.1)          
                    if sim.shortname() in ('m11i','m11e'): pl.ylim(0.002,3)
                    if sim.shortname() in ('m11h','m11v'): pl.ylim(0.01,3)
                    if sim.shortname() in ('m11d',): pl.ylim(0.03,10)
                    if sim.shortname() in ('m12z',): pl.ylim(0.1,30)
                    if sim.shortname() in ('m12f','m12m'): pl.ylim(0.3,100)
                    if 'z5' in sim.shortname(): pl.ylim(10,3000)   
                    if isim==0:
                        u.mylegend(loc='lower right',handlelength=1.5,fontsize=pB.infigfontsize,ncol=2,
                                   handletextpad=0.5,columnspacing=1) 
                    ax.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
                    ax.yaxis.set_minor_locator(ticker.LogLocator(numdecs=5,numticks=5,subs='all'))
                    ax.yaxis.set_major_formatter(u.arilogformatter)                                                    
                if iRow==2:
                    pl.axhline(0.,c='.5',lw=0.5)                    
                    if sim.shortname() in ('m12i','m12b'):  
                        pl.ylim(-65,65) 
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
                    if sim.shortname() in ('m13A1',): 
                        pl.ylim(-300,300)          
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
                    if sim.shortname() in ('m13A8','m13A4'): 
                        pl.ylim(-350,350)          
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(150))
                    if sim.shortname() in ('m13A2',): 
                        pl.ylim(-600,600)          
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(300))
                    if sim.shortname() in ('m11b',): 
                        pl.ylim(-3,3)          
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                    if sim.shortname() in ('m11i','m11e'): 
                        pl.ylim(-13,13)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                    if sim.shortname() in ('m11h','m11v','m11d'): pl.ylim(-25,25)
                    if sim.shortname() in ('m12z','m12f','m12m'): 
                        pl.ylim(-75,75)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
                    if 'z5' in sim.shortname(): pl.ylim(-1500,1500)                      
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: r'$%d$'%x))                                                    
                if iRow==0:
                    pl.text(0.175,0.75,r'$M_{\rm h}^{z=0}=10^{12}{\rm M}_\odot$',transform=ax.transAxes,fontsize=pB.infigfontsize) 
                    pl.ylim(0.,9)
                    pl.axhline(1.,c='.5',lw=0.5)                    
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: r'$%d$'%x))                                                    
                    
                if ax.is_first_col(): 
                    pl.ylabel(ylabel,color=pB.niceblue,fontsize=pB.infigfontsize+2)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
                zs = np.array([0,0.5,1,2,3,5,10])
                ax.set_xticks(cosmo.age(zs).value)
                if ax.is_last_row():
                    ax.set_xticklabels([r'$%s$'%u.nSignificantDigits(z,1,True) for z in zs])
                    pl.xlabel(r'redshift')   
                else:
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
                pl.xlim(*xls)
        pl.savefig(figDir + 'vs_CFindicator_all_%s_ISF.pdf'%sim,bbox_inches='tight')
    def vs_CFindicator8(self,maxR2Rvir=0.05,r2Rvir=0.1,d=2,t_ratio_factor=0.8,weight='HI'):       
        nCols = len(self.sims)
        pl.figure(figsize=(pB.fig_width_half*1.05,2.))
        pl.subplots_adjust(hspace=1e-3,wspace=0.3)
        ind_galaxy   = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(maxR2Rvir))
        ind_innerCGM = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
        label='_nolegend_'
        for isim,sim in enumerate(self.sims):
            ind_tratio = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(sim.profiles[0].Rcirc2Rvir))                
            t_ratio = np.array([(prof.t_cool()/prof.t_ff())[ind_tratio] for prof in sim.profiles])
            
            good_prof_inds = np.array([prof.isSaved('Mdot_in') for prof in sim.profiles])
            xls = cosmo.age(np.array([10,0])).value
            ax = pl.subplot(1,nCols,isim+1)
            xs = 1+sim.zs()
            vals = np.array([(prof.gasMassProfile() * prof.profile1D('vrs','MW'))[ind_innerCGM-d:ind_innerCGM+d+1].sum() /
                             prof.drs_midbins()[ind_innerCGM-d:ind_innerCGM+d+1].sum() 
                             * (un.km/un.s/un.kpc).to('yr**-1') for prof in sim.profiles])
            ylabel = r'$\dot{M}(0.1 R_{\rm vir})\ [{\rm M}_\odot{\rm yr}^{-1}]$'                    
            times = cosmo.age(xs-1)    
            pl.plot(times,vals,c=pB.niceblue,lw=0.5,label='instant')
            N = 50
            avgs = scipy.ndimage.filters.uniform_filter1d(vals[~np.isnan(vals)], N,mode='nearest')
            pl.plot(times[~np.isnan(vals)],avgs,c='k',lw=0.5,label=r'average')
            pl.axhline(0.,c='.5',lw=0.5)                    
            pl.ylim(-65,65) 
            ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: r'$%d$'%x))                                                                        
            if ax.is_first_col(): 
                pl.ylabel(ylabel,color=pB.niceblue,fontsize=pB.infigfontsize+2)
            zs = np.array([0,0.5,1,2,3,5,10])
            ax.set_xticks(cosmo.age(zs).value)
            ax.set_xticklabels([r'$%s$'%u.nSignificantDigits(z,1,True) for z in zs])
            pl.xlabel(r'redshift')   
            pl.xlim(*xls)
            
            ax2 = pl.twinx()
            pl.plot(times,t_ratio*t_ratio_factor,zorder=100,c='r',lw=1.)
            ax2.set_yscale('log')
            ax2.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
            pl.ylim(0.03,0.03**-1.)                    
            pl.axhline(1.,c='k',lw=0.5,ls=':')
            a,b = cosmo.age(0.965).value,cosmo.age(0.775).value
            med = (a+b)/2.
            pl.fill_betweenx(np.array([-65,65]),a,b,edgecolor='none',facecolor='r',alpha=0.2,zorder=-100)
            pl.text(med,50,'inner CGM virializes',color='r',clip_on=False,ha='center',fontsize=pB.infigfontsize)
            pl.annotate('',(med,20),(med,40),arrowprops=pB.slantlinepropsred)
        if ax.is_last_col():            
            ax2.yaxis.set_major_formatter(u.arilogformatter)            
            pl.ylabel(r'$t_{\rm cool}/t_{\rm ff}$ at $0.1 R_{\rm vir}$',color='r',fontsize=pB.infigfontsize+2)        
        pl.savefig(figDir + 'Mdot_vs_tratio_%s_ISF.pdf'%sim,bbox_inches='tight')

    def t_ratio_comparison(self,r2Rvirs=[0.1,0.5]): 
        xs = np.array([]); ys = np.array([]); totals = []
        for isim,sim in enumerate(self.sims):
            for ir,r2Rvir in enumerate(r2Rvirs):
                ind = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
                t_ratio = np.array([(prof.t_cool()/prof.t_ff())[ind] for prof in sim.profiles])
                if ir==0: xs = np.concatenate([xs,t_ratio])
                if ir==1: ys = np.concatenate([ys,t_ratio])
            totals.append(len(xs))
        return xs,ys,totals
    def supersonic_fraction(self,r2Rvirs=[0.1,0.5],_N=20,weight='VW',res=None,d=5): 
        fig = pl.figure(figsize=(pB.fig_width_half,pB.fig_width_half))
        ax = pl.subplot(111)
        if res==None:
            xs = np.array([]); ys = np.array([]); totals = []
            for isim,sim in enumerate(self.sims):
                for ir,r2Rvir in enumerate(r2Rvirs):
                    ind = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
                    log_val = 1-np.array([prof.thickerShells(prof.subsonic_fraction(weight),d,weight)[ind] for prof in sim.profiles])       
                    N = _N//((1,2)['h' in sim.galaxyname])
                    median = np.median(log_val[:log_val.shape[0]//N * N].reshape(log_val.shape[0]//N,N),axis=1)
                    if ir==0: 
                        xs = np.concatenate([xs,log_val])
                        tmp = median
                    if ir==1: 
                        ys = np.concatenate([ys,log_val])
                        if sim.galaxyname in ('m12i','m11d','h206'):                            
                            color={'m11d':'#5D3A9B', 'm12i':'#FFC20A', 'h206':'#D41159'}[sim.galaxyname]                            
                            pl.plot(tmp,median,label=sim.shortname(),zorder=100,lw=1.5,c=color)                                                    
                            indarrows={'m11d':(14,), 'm12i':(13,6), 'h206':(15,12)}[sim.galaxyname]                            
                            for i in indarrows:
                                pl.arrow((tmp[i]+tmp[i-1])/2.,(median[i]+median[i-1])/2., 
                                         (tmp[i-1]-tmp[i])/10.,(median[i-1]-median[i])/10., 
                                         shape='full', length_includes_head=True, 
                                         lw=3.5,head_width=.02,color=color,zorder=101)
                            
                    totals.append(len(xs))
        else:
            xs,ys,totals = res
        ax.scatter(xs,ys, color='grey', edgecolors='k', alpha=0.1, s=1,zorder=0)
        
        
        pl.xlim(0,1)
        pl.xlabel(r'supersonic fraction at $%.1f R_{\rm vir}$'%r2Rvirs[0])
        #u.plotLine(c='.5',lw=0.5,ls='-')
        
        pl.ylim(0,1)
        pl.ylabel(r'supersonic fraction at $%.1f R_{\rm vir}$'%r2Rvirs[1])

        pl.axhline(0.5,c='.5',lw=0.5,ls='-')
        pl.axvline(0.5,c='.5',lw=0.5,ls='-')
        u.mylegend(loc='upper left',labelspacing=0.2,handletextpad=0.5)

        pl.savefig(figDir + 'supersonic_fraction_comparison.pdf',bbox_inches='tight')
        return xs,ys,totals

    def SFR(self):
        pl.figure(figsize=(pB.fig_width_full,5))
        for iPanel in range(2):
            ax = pl.subplot(2,2,iPanel+1)
            for sim in self.sims:
                _sfhs = sim.sfh['std_log'] #sim.sfh['sfrs']/sim.sfh['means']
                sfhs = 10.**np.interp(log(1+sim.zs()),log(1+sim.sfh['zs'][::-1]),log(_sfhs[::-1]))
                if iPanel==0: xs = sim.tcool_Rcircs()/sim.tff_Rcircs()
                if iPanel==1: xs = sim.mvirs()
                pl.plot(xs,sfhs,'.')
            #pl.ylabel()
            pl.semilogx()
            if iPanel==0: 
                pl.axvline(1.,c='k',ls=':')
                pl.xlim(1e-3,1e2)
                
                
                
        
    def vs_profile(self,weight):       
        pl.figure(figsize=(pB.fig_width_full,5))                
        pl.subplots_adjust(hspace=0.2,wspace=0.05)
        iPanel=0; r2Rvir=0.2
        for isim,sim in enumerate(self.sims):
            ax = pl.subplot(2,3,isim+1)
            profs = np.array(sim.profiles) 
            
            log_mvirs = log(np.array([prof.mvir for prof in profs]))
            ind = u.searchsorted(Snapshot_profiler.log_r2rvir_bins,log(r2Rvir))
            vr = np.array([prof.profile1D('vrs',weight)[ind] for prof in profs])
            vc = np.array([prof.vc()[ind] for prof in profs])
            sig_vr = (np.array([prof.profile1D('vrs',weight,2)[ind] for prof in profs]) - vr**2)**0.5
            #smooth_sig_log_nHTs = pl.poly1d(pl.polyfit(log_mvirs,sig_log_nHTs,5))(log_mvirs)            
            pl.fill_between(1+sim.zs(), (vr-sig_vr)/vc,(vr+sig_vr)/vc,facecolor='b',alpha=0.2)
            pl.plot(1+sim.zs(),vr/vc,c='b')
            pl.semilogx()
            if 'm12' in sim.shortname(): pl.xlim(1,4)
            if 'h' in sim.shortname(): pl.xlim(2,8)
            if 'HL' in sim.shortname(): pl.xlim(6,14)
            pl.ylim(-1.5,1.5)  
            #pl.axvline(1+sim.z_cross(),c='k',ls=':')
            
            pl.axhline(1.,c='.5',lw=0.5,zorder=1000)
            pl.axhline(0.,c='.5',lw=0.5,zorder=1000)
            pl.axhline(-1.,c='.5',lw=0.5,zorder=1000)            
            
            pl.text(0.025,0.075,sim.shortname(),fontsize=8,bbox=dict(edgecolor='k',facecolor='w',lw=0.5),transform=ax.transAxes)
        
            ax.xaxis.set_major_locator(ticker.MultipleLocator())            
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            if ax.is_last_row():
                pl.xlabel(r'redshift')   
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '%d'%(x-1)))
            if ax.is_first_col():
                pl.ylabel(r'$v_r/v_{\rm c}$',color='b') 
            else:
                ax.yaxis.set_major_formatter(ticker.NullFormatter())    
                
            ax2=pl.twinx()
            t_ratio = sim.tcool_Rcircs()/sim.tff_Rcircs()
            pl.ylim(1e-2,1e2)
            pl.plot(1+sim.zs(),t_ratio,zorder=100,c='r',lw=0.5)
            ax2.set_yscale('log')
            ax2.yaxis.set_major_locator(ticker.LogLocator(numdecs=5,numticks=5))
            ax.set_zorder(10)
            ax.patch.set_visible(False)        
            
            if ax.is_last_col():
                ax2.yaxis.set_major_formatter(u.arilogformatter)    
                pl.ylabel(r'$t_{\rm cool}/t_{\rm ff}$',color='r',fontsize=pB.infigfontsize)
            else:
                ax2.yaxis.set_major_formatter(ticker.NullFormatter())    
            pl.axhline(1.,c='k',lw=0.5)
            ax.set_zorder(ax2.get_zorder()-1) # put ax behind ax2 
            ax.patch.set_visible(False) # hide the 'canvas'             
        pl.savefig(figDir + 'vs_profiles.pdf',bbox_inches='tight')    
    def quenchedFraction(self):
        fQs_Behroozi19, fQ_Mhalos_Behroozi19, fQ_zs_Behroozi19 = pB2.Behroozi_quenchedFractions()
        log_Mthres = np.array([sim.at_z_cross(log(sim.mvirs())) for sim in self.sims])
        z_crosses = np.array([sim.z_cross() for sim in self.sims])
        
        fig = pl.figure(figsize=(pB.fig_width_half,3))
        ax=pl.subplot(111)
        log_zp1_mesh,logM_mesh= np.arange(0.,1.,0.01),np.arange(9.5,15,.01)
        fQ_mesh = scipy.interpolate.interp2d(log(fQ_zs_Behroozi19+1.)[::-1],
                                                 fQ_Mhalos_Behroozi19,
                                                 fQs_Behroozi19[::-1,:].T)(log_zp1_mesh.flatten(),logM_mesh.flatten())
        C = pl.contour(np.arange(0.,1.,0.01),np.arange(9.5,15.,.01),
                       fQ_mesh,(-0.5,0.1,0.25,0.5,0.75,0.9),zorder=-100,
                       colors='.5',vmin=0.,linewidths=0.5)
        
        
        
        pl.plot(log(1+z_crosses),log_Mthres,'x',c='k',ms=8,label=r'$t_{\rm cool}^{(s)}=t_{\rm ff}\ {\rm at}\ R_{\rm circ}$')   
        pl.ylabel(r'$\log\ M_{\rm halo} / {\rm M}_\odot$')
        pl.xlabel(r'redshift')
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: (r'$%d$'%u.iround(10.**x-1),r'')[False]))
        ax.xaxis.set_major_locator(ticker.FixedLocator(log(np.arange(1,12))))
        pl.ylim(9.75,15)
        pl.xlim(0.,log(9))
        #_zs = np.arange(0,7,.01)
        #pl.plot(log(_zs+1),log(_zs+1)*0.55+11.87,ls='--',c='k')        
        #cax = fig.add_axes([0.775, 0.65, 0.03, 0.15])        
        #cbar = pl.colorbar(mesh,orientation='vertical',cax=cax,ticks=[0.,0.5,1.])        
        #cbar.ax.set_xticklabels([r'$0$', r'$0.5$', r'$1$'])
        #cbar.set_label(r'$f_{\rm SF}$',labelpad=-40)
        pl.clabel(C,fmt=ticker.FuncFormatter(lambda x,pos: (r'$%.1f$'%x,r'$%.2f$'%x)[x in (0.25,0.75)]),inline=True,inline_spacing=0,
                  fontsize=pB.infigfontsize,manual=((0.1,11.2),(0.4,12),(0.05,12.2),(0.4,12.7),(0.1,13.2)))
        pl.text(0.075,0.15,r'contours: quenched fraction',fontsize=pB.infigfontsize,transform=ax.transAxes)
        pl.text(0.7,0.85,r'no halos',fontsize=pB.infigfontsize,transform=ax.transAxes)
        u.mylegend(loc='lower left',borderpad=0,handletextpad=0,markerscale=0.5)
        
        pl.savefig(figDir+'quenched.pdf')
        

    def tratio_Mhalo_redshift(self,isRcirc=True):
        fig = pl.figure(figsize=(pB.fig_width_full*1.05,4))
        pl.subplots_adjust(wspace=0.35)
        
        ax=pl.subplot(121)
        for isim,sim in enumerate(self.sims):
            xs = cosmo.age(sim.zs())
            ys = sim.mvirs()
            pl.plot(xs,ys,lw=0.7,c='k')
        pl.semilogy()
        pl.ylabel(r'$M_{\rm halo}\ [{\rm M}_\odot]$')
        pl.ylim((1e9,1e10)[isRcirc],1e13)                        
        zs = np.array([0,0.5,1,2,3,4,7])
        pl.xlim(cosmo.age(zs[-1]).value,cosmo.age(zs[0]).value)
        ax.set_xticks(cosmo.age(zs).value)
        ax.set_xticklabels([r'$%s$'%u.nSignificantDigits(z,1,True) for z in zs])
        pl.xlabel(r'${\rm redshift}$')   

        ax=pl.subplot(122)        
        Rstr = ('0.5','0.1')[isRcirc]
        zs_to_show = (0.,2.,5.)
        x_zs = {}; y_zs = {}
        for z in zs_to_show:                
            x_zs[z] = []; y_zs[z] = []        
        for isim,sim in enumerate(self.sims):            
            xs = 10.**self.smooth(log(sim.mvirs()))
            ys = 10.**self.smooth(log(sim.quantity_vs_z('t_cool',isRcirc)/sim.quantity_vs_z('t_ff',isRcirc)),N=31)
            for z in zs_to_show:                
                if min(sim.zs())<=z:
                    x_zs[z].append( 10.**np.interp(log(1+z),log(1+sim.zs()),log(xs)) )
                    y_zs[z].append( 10.**np.interp(log(1+z),log(1+sim.zs()),log(ys)) )
        for iz,z in enumerate(zs_to_show):            
            x_zs[z] = np.array(x_zs[z])
            y_zs[z] = np.array(y_zs[z])
            UL = 0.0175
            inds = y_zs[z]>UL
            pl.scatter(x_zs[z][inds],y_zs[z][inds],marker = '*os^v'[iz],s=20,label=r'$z=%d$'%z,edgecolors='.3',facecolors=('.3','none')[z>0])
            pl.scatter(x_zs[z][~inds],[UL*0.9]*len((~inds).nonzero()[0]),marker = u.buildMarker([180]),s=100,edgecolors='.3')
            pl.scatter(x_zs[z][~inds],[UL]*len((~inds).nonzero()[0]),marker = '*os^v'[iz],s=20,edgecolors='.3',facecolors=('.3','none')[z>0])
        pl.loglog()
        pl.xlabel(r'$M_{\rm halo}\ [{\rm M}_\odot]$')
        pl.xlim(0.5e10,1e13)
        ax.yaxis.set_major_formatter(u.arilogformatter)
        pl.ylim(0.01,30)
        pl.ylabel(r'$t_{\rm cool} / t_{\rm ff}\ {\rm at}\ 0.1 R_{\rm vir}$')
        #pl.fill_between([0.5e10,1e13], 10.**-0.5,10**0.5,color='.7',alpha=0.5)
        pl.axhline(10**-0.5,c='grey',lw=0.5,ls='--')
        pl.axhline(10**0.5,c='grey',lw=0.5,ls='--')
        pl.text(0.7e10,5.,r'${\rm virialized}$',va='center',fontsize=pB.infigfontsize,ha='left',backgroundcolor='w')
        pl.text(0.7e10,0.2,r'${\rm not\ virialized}$',va='center',fontsize=pB.infigfontsize,ha='left',backgroundcolor='w')
        pl.legend(frameon=True,ncol=1,columnspacing=0.5,handletextpad=0.0,borderpad=0.3,loc='lower right',fontsize=pB.infigfontsize,labelspacing=0.1)
        
        pl.savefig(figDir + 'tratio_%s_Mhalo_redshift.pdf'%(('vir','circ')[isRcirc]),bbox_inches='tight')    
    def tratio_Mhalo_redshift2(self,isRcirc=True,tratios_to_show=(10**-0.5,10**0.5)):        
        fig = pl.figure(figsize=(pB.fig_width_full*0.65,2.25))
        ax=pl.subplot(111)
        cmap = pl.get_cmap('coolwarm')
        for isim,sim in enumerate(self.sims):
            xs = sim.zs()[::-1]
            ys = sim.mvirs()[::-1]
            pl.plot(xs,ys,lw=0.7,c='k')
            tratios = (sim.quantity_vs_z('t_cool',isRcirc)/sim.quantity_vs_z('t_ff',isRcirc))[::-1]
            inds = [None,None]
            for i,tratio_to_show in enumerate(tratios_to_show):
                larger = (tratios>tratio_to_show).nonzero()[0]
                if len(larger): 
                    inds[i] = larger[0]
            if inds[1]!=None:
                pl.plot(xs[inds[1]-1:],ys[inds[1]-1:],lw=1,c=cmap(0.99),label=(r'$t_{\rm cool}\gg t_{\rm ff}$','_')[isim!=10])
            if inds[0]!=None:
                pl.plot(xs[inds[0]-1:inds[1]],ys[inds[0]-1:inds[1]],lw=1,c=(0.65,0.65,0.65,1),label=(r'$t_{\rm cool}\sim t_{\rm ff}$','_')[isim!=10])
            pl.plot(xs[:inds[0]],ys[:inds[0]],lw=1,c=cmap(0),label=(r'$t_{\rm cool}\ll t_{\rm ff}$','_')[isim!=10])
        pl.legend(frameon=False,loc='upper left',ncol=1,labelspacing=0.1,fontsize=pB.infigfontsize,handlelength=1.5)
        pl.semilogy()
        pl.ylabel(r'$M_{\rm halo}\ [{\rm M}_\odot]$')
        pl.ylim((1e9,2e10)[isRcirc],1.5e13)                        
        pl.xlim(4.,0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        #zs = np.array([0,0.5,1,2,3,4,7])
        #pl.xlim(cosmo.age(zs[-1]).value,cosmo.age(zs[0]).value)
        #ax.set_xticks(cosmo.age(zs).value)
        #ax.set_xticklabels([r'$%s$'%u.nSignificantDigits(z,1,True) for z in zs])
        pl.xlabel(r'${\rm redshift}$')   
        pl.savefig(figDir + 'tratio_%s_Mhalo_redshift2.pdf'%(('vir','circ')[isRcirc]),bbox_inches='tight')    
            
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy.stats as st


def vs_CFindicator_Drummond(tratios,Vrot2sigma,groupInds,layout,bw=None,ax=None,limits=None,
                            show_all=True,labels=('m11','m12','m13'),savefig=True):
    Vrot2sigma_m11 = Vrot2sigma[:groupInds[0]]
    Vrot2sigma_m12 = Vrot2sigma[groupInds[0]:groupInds[1]]
    Vrot2sigma_m13 = Vrot2sigma[groupInds[1]:groupInds[2]]
    tratios_m11 = tratios[:groupInds[0]]
    tratios_m12 = tratios[groupInds[0]:groupInds[1]]
    tratios_m13 = tratios[groupInds[1]:groupInds[2]]

    i_nan_tratios = np.where(np.isnan(tratios))[0]
    tratios = np.delete( tratios, i_nan_tratios)
    Vrot2sigma = np.delete( Vrot2sigma, i_nan_tratios)

    i_nan_Vrot2sigma = np.where(np.isnan(Vrot2sigma))[0]
    tratios = np.delete( tratios, i_nan_Vrot2sigma)
    Vrot2sigma = np.delete( Vrot2sigma, i_nan_Vrot2sigma)

    i_inf_Vrot2sigma = np.where(np.isinf(Vrot2sigma))[0]
    tratios = np.delete( tratios, i_inf_Vrot2sigma)
    Vrot2sigma = np.delete( Vrot2sigma, i_inf_Vrot2sigma)

    i_neg_Vrot2sigma = np.where(Vrot2sigma<0)[0]
    tratios = np.delete( tratios, i_neg_Vrot2sigma)
    Vrot2sigma = np.delete( Vrot2sigma, i_neg_Vrot2sigma)

    i_neg_tratios = np.where(tratios<0)[0]
    tratios = np.delete( tratios, i_neg_tratios)
    Vrot2sigma = np.delete( Vrot2sigma, i_neg_tratios)


    full = np.array([tratios,Vrot2sigma])
    kernel = st.gaussian_kde(np.log10(full),bw_method=bw)


    i_nan_tratios_m11 = np.where(np.isnan(tratios_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_nan_tratios_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_nan_tratios_m11)
    
    i_nan_Vrot2sigma_m11 = np.where(np.isnan(Vrot2sigma_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_nan_Vrot2sigma_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_nan_Vrot2sigma_m11)
    
    i_inf_Vrot2sigma_m11 = np.where(np.isinf(Vrot2sigma_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_inf_Vrot2sigma_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_inf_Vrot2sigma_m11)
    
    i_neg_Vrot2sigma_m11 = np.where(Vrot2sigma_m11<0)[0]
    tratios_m11 = np.delete( tratios_m11, i_neg_Vrot2sigma_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_neg_Vrot2sigma_m11)
    
    i_neg_tratios_m11 = np.where(tratios_m11<0)[0]
    tratios_m11 = np.delete( tratios_m11, i_neg_tratios_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_neg_tratios_m11)
    
    
    full_m11 = np.array([tratios_m11,Vrot2sigma_m11])
    kernel_m11 = st.gaussian_kde(np.log10(full_m11),bw_method=bw)


    
    i_nan_tratios_m12 = np.where(np.isnan(tratios_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_nan_tratios_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_nan_tratios_m12)
    
    i_nan_Vrot2sigma_m12 = np.where(np.isnan(Vrot2sigma_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_nan_Vrot2sigma_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_nan_Vrot2sigma_m12)
    
    i_inf_Vrot2sigma_m12 = np.where(np.isinf(Vrot2sigma_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_inf_Vrot2sigma_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_inf_Vrot2sigma_m12)
    
    i_neg_Vrot2sigma_m12 = np.where(Vrot2sigma_m12<0)[0]
    tratios_m12 = np.delete( tratios_m12, i_neg_Vrot2sigma_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_neg_Vrot2sigma_m12)
    
    i_neg_tratios_m12 = np.where(tratios_m12<0)[0]
    tratios_m12 = np.delete( tratios_m12, i_neg_tratios_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_neg_tratios_m12)
    
    
    full_m12 = np.array([tratios_m12,Vrot2sigma_m12])
    kernel_m12 = st.gaussian_kde(np.log10(full_m12),bw_method=bw)


    
    i_nan_tratios_m13 = np.where(np.isnan(tratios_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_nan_tratios_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_nan_tratios_m13)
    
    i_nan_Vrot2sigma_m13 = np.where(np.isnan(Vrot2sigma_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_nan_Vrot2sigma_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_nan_Vrot2sigma_m13)
    
    i_inf_Vrot2sigma_m13 = np.where(np.isinf(Vrot2sigma_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_inf_Vrot2sigma_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_inf_Vrot2sigma_m13)
    
    i_neg_Vrot2sigma_m13 = np.where(Vrot2sigma_m13<0)[0]
    tratios_m13 = np.delete( tratios_m13, i_neg_Vrot2sigma_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_neg_Vrot2sigma_m13)

    i_neg_tratios_m13 = np.where(tratios_m13<0)[0]
    tratios_m13 = np.delete( tratios_m13, i_neg_tratios_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_neg_tratios_m13)
    
    
    full_m13 = np.array([tratios_m13,Vrot2sigma_m13])
    kernel_m13 = st.gaussian_kde(np.log10(full_m13),bw_method=bw)


    if limits==None:            
        xmin = np.floor (np.log10(np.min(tratios)))
        xmax = np.ceil  (np.log10(np.max(tratios)))
        ymin = np.floor (np.log10(np.min(Vrot2sigma)))
        ymax = np.ceil  (np.log10(np.max(Vrot2sigma)))
        print(xmin,xmax,ymin,ymax)
    else:
        xmin,xmax,ymin,ymax=limits
    

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    f = np.reshape(kernel(positions).T, xx.shape)
    f_m11 = np.reshape(kernel_m11(positions).T, xx.shape)
    f_m12 = np.reshape(kernel_m12(positions).T, xx.shape)
    f_m13 = np.reshape(kernel_m13(positions).T, xx.shape)

    median_m11 = np.array([np.interp(0.5,np.cumsum(f_m11[i])/np.sum(f_m11[i]), 10**yy[i])  for i in range(len(f_m11))])
    median_m12 = np.array([np.interp(0.5,np.cumsum(f_m12[i])/np.sum(f_m12[i]), 10**yy[i])  for i in range(len(f_m12))])
    median_m13 = np.array([np.interp(0.5,np.cumsum(f_m13[i])/np.sum(f_m13[i]), 10**yy[i])  for i in range(len(f_m13))])
    median = np.array([np.interp(0.5,np.cumsum(f[i])/np.sum(f[i]), 10**yy[i])  for i in range(len(f))])
    upper = np.array([np.interp(0.84,np.cumsum(f[i])/np.sum(f[i]), 10**yy[i])  for i in range(len(f))])
    lower = np.array([np.interp(0.16,np.cumsum(f[i])/np.sum(f[i]), 10**yy[i])  for i in range(len(f))])



    if ax==None:
        fig = plt.figure()
        ax = fig.gca()
        fig.set_size_inches(pB.fig_width_half,pB.fig_width_half)
        
    
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(u.arilogformatter)
    # Contourf plot
    #cfset = ax.pcolormesh(10**xx, 10**yy, f, cmap='bone_r', norm=colors.LogNorm(vmin=0.1, vmax=2e1), shading='gouraud')
    ax.scatter(tratios,Vrot2sigma, color='grey', edgecolors='k', alpha=0.1, s=1)

    if show_all:
        i_all = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios)[-10]))
        j_all = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios)[10]))
        ax.plot(10**xx[j_all:i_all,0],median[j_all:i_all], color='k', lw=3.,label='all',zorder=250)
    #ax.plot(10**xx[:,0],upper, color='k', lw=1,zorder=100)
    #ax.plot(10**xx[:,0],lower, color='k', lw=1,zorder=100)
         
    i_11 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m11)[-10]))
    i_12 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m12)[-10]))
    i_13 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m13)[-10]))
    j_11 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m11)[10]))
    j_12 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m12)[10]))
    j_13 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m13)[10]))
    ax.plot(10**xx[j_11:i_11,0],median_m11[j_11:i_11], color='#5D3A9B', lw=1.5, label=labels[0],zorder=200)
    ax.plot(10**xx[j_12:i_12,0],median_m12[j_12:i_12], color='#FFC20A', lw=1.5, label=labels[1],zorder=150)
    ax.plot(10**xx[j_13:i_13,0],median_m13[j_13:i_13], color='#D41159', lw=1.5, label=labels[2],zorder=100)
    
    #ax.plot(tratios,Vrot2sigma,',',color='k')
    if layout!='tratio':
        ax.axvline(t_ratio_virialization, color='grey', lw=0.7, ls='-', dash_capstyle='round')
    ## Or kernel density estimate plot instead of the contourf plot
    #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    # cset = ax.contour(xx, 10**yy, f, colors='k')
    # Label plot
    # ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff} \,\, {\rm at} \,\, 0.1 R_{\rm vir}$')
    if layout=='Vrot2sigma':
        ax.set_ylim(0,8)   
        ax.set_ylabel(r'$\langle V_{\rm rot} \rangle / \sigma_{\rm g}$')
        ax.set_xlim(1e-3, 30)
    if layout=='sigma2vc':
        ax.set_ylim(0,1.2)   
        ax.set_ylabel(r'$\sigma_{\rm g}\ [{\rm km}\ {\rm s}^{-1}]$')
        ax.set_xlim(1e-3, 30)
    if layout=='Vrot2vc':
        ax.set_ylim(0,1.2)   
        ax.set_ylabel(r'$\langle V_{\rm rot} \rangle\ [{\rm km}\ {\rm s}^{-1}]$')
        ax.set_xlim(1e-3, 30)
    if layout=='vc':
        ax.legend(loc='lower right', frameon=False,handlelength=1.5)
        ax.set_ylim(10,300)   
        ax.set_ylabel(r'$v_{\rm c}\ [{\rm km}\ {\rm s}^{-1}]$')
        ax.set_xlim(1e-3, 30)
    if layout=='T2Tvir':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(u.arilogformatter)        
        ax.set_ylim(0.03,3)   
        ax.set_xlim(1e-2, 100)
        ax.set_ylabel(r'$\langle T\rangle(0.1 R_{\rm vir})/T_{\rm vir}$')
        ax.legend(loc='lower right', frameon=False)
        if savefig: plt.savefig(figDir+'log_Ts_vs_t_ratio.png',dpi=400, bbox_inches='tight')
    if layout=='T2Tc':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(u.arilogformatter)        
        ax.set_ylim(0.03,3)   
        ax.set_xlim(1e-2, 100)
        ax.set_ylabel(r'$\langle T\rangle(0.1 R_{\rm vir})/T^{(\rm s)}$')
        ax.legend(loc='lower right', frameon=False)
        if savefig: plt.savefig(figDir+'T2Tc_vs_t_ratio.png',dpi=400, bbox_inches='tight')
    if layout=='nH2nHc':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(u.arilogformatter)        
        ax.set_ylim(0.1,20)   
        ax.set_xlim(1e-3, 70)
        ax.set_ylabel(r'$\langle n_{\rm H}\rangle/n_{\rm H}^{(\rm s)}$ at $0.1 R_{\rm vir}$')
        ax.legend(loc='lower right', frameon=False,ncol=2,handlelength=1.5,columnspacing=1)
        if savefig: plt.savefig(figDir+'nH2nHc_vs_t_ratio.png',dpi=400, bbox_inches='tight')
    if layout=='tratio':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(u.arilogformatter)        
        ax.set_ylim(1e-3,100)   
        ax.set_xlim(1e-3,100)
        ax.set_ylabel(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ \ {\rm at}\ \ 0.5 R_{\rm vir}$')
        ax.legend(loc='lower right', frameon=False)
        u.plotLine(color='grey', lw=0.7, ls='-', dash_capstyle='round')
        if savefig: plt.savefig(figDir+'t_ratio_comparison.png',dpi=400, bbox_inches='tight')
    return ((10**xx[j_11:i_11,0],median_m11[j_11:i_11]),
            (10**xx[j_12:i_12,0],median_m12[j_12:i_12]),
            (10**xx[j_13:i_13,0],median_m13[j_13:i_13]))


def vs_CFindicator_Drummond_nolog(tratios,Vrot2sigma,groupInds,layout,bw=None,ax=None,
                                  limits=None,show_all=True,showMeans=True,
                                  labels=('m11','m12','m13')):
    Vrot2sigma_m11 = Vrot2sigma[:groupInds[0]]
    Vrot2sigma_m12 = Vrot2sigma[groupInds[0]:groupInds[1]]
    Vrot2sigma_m13 = Vrot2sigma[groupInds[1]:groupInds[2]]
    tratios_m11 = tratios[:groupInds[0]]
    tratios_m12 = tratios[groupInds[0]:groupInds[1]]
    tratios_m13 = tratios[groupInds[1]:groupInds[2]]

    i_nan_tratios = np.where(np.isnan(tratios))[0]
    tratios = np.delete( tratios, i_nan_tratios)
    Vrot2sigma = np.delete( Vrot2sigma, i_nan_tratios)

    i_nan_Vrot2sigma = np.where(np.isnan(Vrot2sigma))[0]
    tratios = np.delete( tratios, i_nan_Vrot2sigma)
    Vrot2sigma = np.delete( Vrot2sigma, i_nan_Vrot2sigma)

    i_inf_Vrot2sigma = np.where(np.isinf(Vrot2sigma))[0]
    tratios = np.delete( tratios, i_inf_Vrot2sigma)
    Vrot2sigma = np.delete( Vrot2sigma, i_inf_Vrot2sigma)

    full = np.array([tratios,Vrot2sigma])
    kernel = st.gaussian_kde(full,bw_method=bw)    

    i_nan_tratios_m11 = np.where(np.isnan(tratios_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_nan_tratios_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_nan_tratios_m11)
    
    i_nan_Vrot2sigma_m11 = np.where(np.isnan(Vrot2sigma_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_nan_Vrot2sigma_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_nan_Vrot2sigma_m11)
    
    i_inf_Vrot2sigma_m11 = np.where(np.isinf(Vrot2sigma_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_inf_Vrot2sigma_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_inf_Vrot2sigma_m11)
    
    full_m11 = np.array([tratios_m11,Vrot2sigma_m11])
    kernel_m11 = st.gaussian_kde(full_m11,bw_method=bw)
    

    
    i_nan_tratios_m12 = np.where(np.isnan(tratios_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_nan_tratios_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_nan_tratios_m12)
    
    i_nan_Vrot2sigma_m12 = np.where(np.isnan(Vrot2sigma_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_nan_Vrot2sigma_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_nan_Vrot2sigma_m12)
    
    i_inf_Vrot2sigma_m12 = np.where(np.isinf(Vrot2sigma_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_inf_Vrot2sigma_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_inf_Vrot2sigma_m12)
    
   
    
    full_m12 = np.array([tratios_m12,Vrot2sigma_m12])
    kernel_m12 = st.gaussian_kde(full_m12,bw_method=bw)


    
    i_nan_tratios_m13 = np.where(np.isnan(tratios_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_nan_tratios_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_nan_tratios_m13)
    
    i_nan_Vrot2sigma_m13 = np.where(np.isnan(Vrot2sigma_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_nan_Vrot2sigma_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_nan_Vrot2sigma_m13)
    
    i_inf_Vrot2sigma_m13 = np.where(np.isinf(Vrot2sigma_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_inf_Vrot2sigma_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_inf_Vrot2sigma_m13)
    
    
    full_m13 = np.array([tratios_m13,Vrot2sigma_m13])
    kernel_m13 = st.gaussian_kde(full_m13,bw_method=bw)


    if limits==None:
        xmin = np.floor (np.min(tratios))
        xmax = np.ceil  (np.max(tratios))
        ymin = np.floor (np.min(Vrot2sigma))
        ymax = np.ceil  (np.max(Vrot2sigma))
        print(xmin,xmax,ymin,ymax)
    else:
        xmin,xmax,ymin,ymax = limits
    

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]    
    positions = np.vstack([xx.ravel(), yy.ravel()])

    f     = np.reshape(kernel(positions).T, xx.shape)
    f_m11 = np.reshape(kernel_m11(positions).T, xx.shape)
    f_m12 = np.reshape(kernel_m12(positions).T, xx.shape)
    f_m13 = np.reshape(kernel_m13(positions).T, xx.shape)


    median_m11 = np.array([np.interp(0.5,np.cumsum(f_m11[i])/np.sum(f_m11[i]), yy[i])  for i in range(len(f_m11))])
    median_m12 = np.array([np.interp(0.5,np.cumsum(f_m12[i])/np.sum(f_m12[i]), yy[i])  for i in range(len(f_m12))])
    median_m13 = np.array([np.interp(0.5,np.cumsum(f_m13[i])/np.sum(f_m13[i]), yy[i])  for i in range(len(f_m13))])
    median = np.array([np.interp(0.5,np.cumsum(f[i])/np.sum(f[i]), yy[i])  for i in range(len(f))])
    upper = np.array([np.interp(0.84,np.cumsum(f[i])/np.sum(f[i]), yy[i])  for i in range(len(f))])
    lower = np.array([np.interp(0.16,np.cumsum(f[i])/np.sum(f[i]), yy[i])  for i in range(len(f))])



    if ax==None:
        fig = plt.figure()
        ax = fig.gca()
        fig.set_size_inches(pB.fig_width_half,pB.fig_width_half)
    
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(u.arilogformatter)
    # Contourf plot
    #cfset = ax.pcolormesh(10**xx, yy, f, cmap='bone_r', norm=colors.LogNorm(vmin=0.1, vmax=2e1), shading='gouraud')
    ax.scatter(10.**tratios,Vrot2sigma, color='grey', edgecolors='k', alpha=0.2, s=1,marker='o')
    if show_all:
        i_all = np.argmin(np.abs(xx[:,0]-np.sort(tratios)[-10]))
        j_all = np.argmin(np.abs(xx[:,0]-np.sort(tratios)[10]))
        ax.plot(10**xx[j_all:i_all,0],median[j_all:i_all], color='k', lw=3.,label='all',zorder=250)
    #ax.plot(10**xx[:,0],upper, color='k', lw=1,zorder=100)
    #ax.plot(10**xx[:,0],lower, color='k', lw=1,zorder=100)
    i_11 = np.argmin(np.abs(xx[:,0]-np.sort(tratios_m11)[-10]))
    i_12 = np.argmin(np.abs(xx[:,0]-np.sort(tratios_m12)[-10]))
    i_13 = np.argmin(np.abs(xx[:,0]-np.sort(tratios_m13)[-10]))
    j_11 = np.argmin(np.abs(xx[:,0]-np.sort(tratios_m11)[10]))
    j_12 = np.argmin(np.abs(xx[:,0]-np.sort(tratios_m12)[10]))
    j_13 = np.argmin(np.abs(xx[:,0]-np.sort(tratios_m13)[10]))
    
    if showMeans:
        ax.plot(10**xx[j_11:i_11,0],median_m11[j_11:i_11], color='#5D3A9B', lw=1.5, label=labels[0],zorder=200)
        ax.plot(10**xx[j_12:i_12,0],median_m12[j_12:i_12], color='#FFC20A', lw=1.5, label=labels[1],zorder=150)
        ax.plot(10**xx[j_13:i_13,0],median_m13[j_13:i_13], color='#D41159', lw=1.5, label=labels[2],zorder=100)
    
    #ax.plot(tratios,Vrot2sigma,',',color='k')
    ax.axvline(2, color='.5', lw=0.5, ls='-', dash_capstyle='round')
    ## Or kernel density estimate plot instead of the contourf plot
    #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    # cset = ax.contour(xx, 10**yy, f, colors='k')
    # Label plot
    # ax.clabel(cset, inline=1, fontsize=10)
    
    if layout[:4]=='Mach':
        ax.set_ylim(0,1)   
        ax.set_xlim(1e-2, 100)
        ax.legend(loc='upper right', frameon=False)
        if layout=='Mach_inner':            
            ax.set_xlabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff} \,\, {\rm at} \,\, 0.1 R_{\rm vir}$')
            ax.set_ylabel(r'supersonic fraction at $0.1 R_{\rm vir}$')
            plt.savefig(figDir+'supersonic_fraction_vs_t_ratio.png',dpi=400, bbox_inches='tight')
        if layout=='Mach_outer':
            ax.set_xlabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff} \,\, {\rm at} \,\, 0.5 R_{\rm vir}$')
            ax.set_ylabel(r'supersonic fraction at $0.5 R_{\rm vir}$')
            plt.savefig(figDir+'supersonic_fraction_vs_t_ratio_outer.png',dpi=400, bbox_inches='tight')
    if layout=='logSFR':
        ax.set_ylim(0,0.8)   
        ax.set_xlim(0.01, 50)
        ax.legend(loc='upper right', frameon=False,handlelength=1,ncol=1,columnspacing=0.75,handletextpad=0.5)
        ax.set_xlabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff} \,\, {\rm at} \,\, 0.1 R_{\rm vir}$')
        ax.set_ylabel(r'$\sigma(\log\ {\rm SFR})\ [{\rm dex}]$')
        plt.savefig(figDir+'sfh_std_vs_t_ratio.png',dpi=400, bbox_inches='tight')
    if layout=='sigma2vc':
        ax.set_ylim(0,1.4)   
        ax.set_ylabel(r'$\sigma_{\rm g} / v_{\rm c}$')
        ax.set_xlim(1e-3, 30)        
    if layout=='Vrot2vc':
        ax.set_ylim(0,1.4)   
        ax.set_ylabel(r'$\langle V_{\rm rot}\rangle / v_{\rm c}$')
        ax.set_xlim(1e-3, 30)        
    if layout=='Vrot2sigma':
        ax.set_ylim(0,8)   
        ax.set_ylabel(r'$\langle V_{\rm rot}\rangle / \sigma_{\rm g}$')
        ax.set_xlim(1e-3, 30)
        ax.legend(loc='upper left', frameon=False)
        

def Vrot_and_sigma(ress,suffix='',inds=[5,10,15]):
    pl.figure(figsize=(pB.fig_width_full*1,pB.fig_width_half*1.5))
    pl.subplots_adjust(hspace=0.2,wspace=0.3,left=0.075,right=0.925)
    xmin=-4;xmax=2; ymin=-3; ymax=3
    for iPanel in range(4):
        res = ress[iPanel]
        ax = pl.subplot(2,2,iPanel+1)
        layout = ('vc','Vrot2vc','sigma2vc','Vrot2sigma')[iPanel]
        _tups = vs_CFindicator_Drummond(res[0],
                                      res[1],
                                      np.array(res[2]).take(np.array(inds)),
                                      layout=layout,
                                      ax=ax,
                                      limits=(xmin,xmax,ymin,ymax),
                                      show_all=False)       
        if iPanel==0:
            tups = _tups
            pl.xlabel('')
            pl.semilogy()
            pl.ylim(5,1000)
            ax.yaxis.set_major_formatter(u.arilogformatter)
        if iPanel==1:
            l = pl.plot([1,10],[1,1],lw=1.5,c='k',ls='--')
            pl.legend(l,[r'$v_{\rm c}$'],loc='lower right',frameon=False,handlelength=1.5)
            pl.xlabel('')
            pl.ylim(5,1000)
            pl.semilogy()
            [pl.plot(*tup,ls='--',lw=1.5,c=('#5D3A9B','#FFC20A','#D41159')[itup])
             for itup,tup in enumerate(tups)]
            ax.yaxis.set_major_formatter(u.arilogformatter)
        if iPanel==2:
            l = pl.plot([1,10],[1,1],lw=1.5,c='k',ls='--')
            pl.legend(l,[r'$v_{\rm c}$'],loc='lower right',frameon=False,handlelength=1.5)            
            pl.ylim(5,1000)
            pl.semilogy()
            [pl.plot(*tup,ls='--',lw=1.5,c=('#5D3A9B','#FFC20A','#D41159')[itup])
             for itup,tup in enumerate(tups)]
            ax.yaxis.set_major_formatter(u.arilogformatter)
        if iPanel==3:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        pl.xlim(0.01,30)
    pl.savefig(figDir+'Vrot_and_sigma%s.png'%suffix,dpi=400, bbox_inches='tight')
    

def appendixFig(resMach,resV2s):
    pl.figure(figsize=(pB.fig_width_half,pB.fig_width_half*1.2))
    pl.subplots_adjust(hspace=0.2,wspace=0.3,left=0.075,right=0.925)
    ax = pl.subplot(211)
    vs_CFindicator_Drummond_nolog(log(resMach[0]),resMach[1],resMach[2],
                                         layout='Mach_inner',bw=0.1,show_all=False,
                                        labels=['low res','no md','fiducial'],ax=ax)
    pl.text(0.05,0.075,r'm12i, $0.1 R_{\rm vir}$',transform=ax.transAxes)
    pl.xlim(0.01,100)
    pl.ylabel(r'supersonic fraction')
    ax.legend([],frameon=False)
    ax = pl.subplot(212)
    vs_CFindicator_Drummond(resV2s[0],resV2s[1],resV2s[2],ax=ax,
                            layout='Vrot2sigma',show_all=False,labels=['low res','no md','fiducial'])
    pl.xlim(0.01,100)
    pl.ylim(0,9)
    ax.legend(loc='upper left', frameon=False)

    pl.savefig(figDir+'appendixFig.png',dpi=400, bbox_inches='tight')




def vs_CFindicator_Drummond2(tratios,Vrot2sigma,groupInds,layout,bw=None,ax=None,limits=None,show_all=True):
    Vrot2sigma_m11 = Vrot2sigma[:groupInds[0]]
    Vrot2sigma_m11b = Vrot2sigma[groupInds[0]:groupInds[1]]
    Vrot2sigma_m12 = Vrot2sigma[groupInds[1]:groupInds[2]]
    Vrot2sigma_m13 = Vrot2sigma[groupInds[2]:groupInds[3]]
    tratios_m11 = tratios[:groupInds[0]]
    tratios_m12 = tratios[groupInds[0]:groupInds[1]]
    tratios_m13 = tratios[groupInds[1]:groupInds[2]]

    i_nan_tratios = np.where(np.isnan(tratios))[0]
    tratios = np.delete( tratios, i_nan_tratios)
    Vrot2sigma = np.delete( Vrot2sigma, i_nan_tratios)

    i_nan_Vrot2sigma = np.where(np.isnan(Vrot2sigma))[0]
    tratios = np.delete( tratios, i_nan_Vrot2sigma)
    Vrot2sigma = np.delete( Vrot2sigma, i_nan_Vrot2sigma)

    i_inf_Vrot2sigma = np.where(np.isinf(Vrot2sigma))[0]
    tratios = np.delete( tratios, i_inf_Vrot2sigma)
    Vrot2sigma = np.delete( Vrot2sigma, i_inf_Vrot2sigma)

    i_neg_Vrot2sigma = np.where(Vrot2sigma<0)[0]
    tratios = np.delete( tratios, i_neg_Vrot2sigma)
    Vrot2sigma = np.delete( Vrot2sigma, i_neg_Vrot2sigma)

    i_neg_tratios = np.where(tratios<0)[0]
    tratios = np.delete( tratios, i_neg_tratios)
    Vrot2sigma = np.delete( Vrot2sigma, i_neg_tratios)


    full = np.array([tratios,Vrot2sigma])
    kernel = st.gaussian_kde(np.log10(full),bw_method=bw)


    i_nan_tratios_m11 = np.where(np.isnan(tratios_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_nan_tratios_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_nan_tratios_m11)
    
    i_nan_Vrot2sigma_m11 = np.where(np.isnan(Vrot2sigma_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_nan_Vrot2sigma_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_nan_Vrot2sigma_m11)
    
    i_inf_Vrot2sigma_m11 = np.where(np.isinf(Vrot2sigma_m11))[0]
    tratios_m11 = np.delete( tratios_m11, i_inf_Vrot2sigma_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_inf_Vrot2sigma_m11)
    
    i_neg_Vrot2sigma_m11 = np.where(Vrot2sigma_m11<0)[0]
    tratios_m11 = np.delete( tratios_m11, i_neg_Vrot2sigma_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_neg_Vrot2sigma_m11)
    
    i_neg_tratios_m11 = np.where(tratios_m11<0)[0]
    tratios_m11 = np.delete( tratios_m11, i_neg_tratios_m11)
    Vrot2sigma_m11 = np.delete( Vrot2sigma_m11, i_neg_tratios_m11)
    
    
    full_m11 = np.array([tratios_m11,Vrot2sigma_m11])
    kernel_m11 = st.gaussian_kde(np.log10(full_m11),bw_method=bw)


    
    i_nan_tratios_m12 = np.where(np.isnan(tratios_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_nan_tratios_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_nan_tratios_m12)
    
    i_nan_Vrot2sigma_m12 = np.where(np.isnan(Vrot2sigma_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_nan_Vrot2sigma_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_nan_Vrot2sigma_m12)
    
    i_inf_Vrot2sigma_m12 = np.where(np.isinf(Vrot2sigma_m12))[0]
    tratios_m12 = np.delete( tratios_m12, i_inf_Vrot2sigma_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_inf_Vrot2sigma_m12)
    
    i_neg_Vrot2sigma_m12 = np.where(Vrot2sigma_m12<0)[0]
    tratios_m12 = np.delete( tratios_m12, i_neg_Vrot2sigma_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_neg_Vrot2sigma_m12)
    
    i_neg_tratios_m12 = np.where(tratios_m12<0)[0]
    tratios_m12 = np.delete( tratios_m12, i_neg_tratios_m12)
    Vrot2sigma_m12 = np.delete( Vrot2sigma_m12, i_neg_tratios_m12)
    
    
    full_m12 = np.array([tratios_m12,Vrot2sigma_m12])
    kernel_m12 = st.gaussian_kde(np.log10(full_m12),bw_method=bw)


    
    i_nan_tratios_m13 = np.where(np.isnan(tratios_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_nan_tratios_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_nan_tratios_m13)
    
    i_nan_Vrot2sigma_m13 = np.where(np.isnan(Vrot2sigma_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_nan_Vrot2sigma_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_nan_Vrot2sigma_m13)
    
    i_inf_Vrot2sigma_m13 = np.where(np.isinf(Vrot2sigma_m13))[0]
    tratios_m13 = np.delete( tratios_m13, i_inf_Vrot2sigma_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_inf_Vrot2sigma_m13)
    
    i_neg_Vrot2sigma_m13 = np.where(Vrot2sigma_m13<0)[0]
    tratios_m13 = np.delete( tratios_m13, i_neg_Vrot2sigma_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_neg_Vrot2sigma_m13)

    i_neg_tratios_m13 = np.where(tratios_m13<0)[0]
    tratios_m13 = np.delete( tratios_m13, i_neg_tratios_m13)
    Vrot2sigma_m13 = np.delete( Vrot2sigma_m13, i_neg_tratios_m13)
    
    
    full_m13 = np.array([tratios_m13,Vrot2sigma_m13])
    kernel_m13 = st.gaussian_kde(np.log10(full_m13),bw_method=bw)


    if limits==None:            
        xmin = np.floor (np.log10(np.min(tratios)))
        xmax = np.ceil  (np.log10(np.max(tratios)))
        ymin = np.floor (np.log10(np.min(Vrot2sigma)))
        ymax = np.ceil  (np.log10(np.max(Vrot2sigma)))
        print(xmin,xmax,ymin,ymax)
    else:
        xmin,xmax,ymin,ymax=limits
    

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    f = np.reshape(kernel(positions).T, xx.shape)
    f_m11 = np.reshape(kernel_m11(positions).T, xx.shape)
    f_m12 = np.reshape(kernel_m12(positions).T, xx.shape)
    f_m13 = np.reshape(kernel_m13(positions).T, xx.shape)

    median_m11 = np.array([np.interp(0.5,np.cumsum(f_m11[i])/np.sum(f_m11[i]), 10**yy[i])  for i in range(len(f_m11))])
    median_m12 = np.array([np.interp(0.5,np.cumsum(f_m12[i])/np.sum(f_m12[i]), 10**yy[i])  for i in range(len(f_m12))])
    median_m13 = np.array([np.interp(0.5,np.cumsum(f_m13[i])/np.sum(f_m13[i]), 10**yy[i])  for i in range(len(f_m13))])
    median = np.array([np.interp(0.5,np.cumsum(f[i])/np.sum(f[i]), 10**yy[i])  for i in range(len(f))])
    upper = np.array([np.interp(0.84,np.cumsum(f[i])/np.sum(f[i]), 10**yy[i])  for i in range(len(f))])
    lower = np.array([np.interp(0.16,np.cumsum(f[i])/np.sum(f[i]), 10**yy[i])  for i in range(len(f))])



    if ax==None:
        fig = plt.figure()
        ax = fig.gca()
        fig.set_size_inches(pB.fig_width_half,pB.fig_width_half)
        
    
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(u.arilogformatter)
    # Contourf plot
    #cfset = ax.pcolormesh(10**xx, 10**yy, f, cmap='bone_r', norm=colors.LogNorm(vmin=0.1, vmax=2e1), shading='gouraud')
    ax.scatter(tratios,Vrot2sigma, color='grey', edgecolors='k', alpha=0.1, s=1)

    if show_all:
        i_all = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios)[-10]))
        j_all = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios)[10]))
        ax.plot(10**xx[j_all:i_all,0],median[j_all:i_all], color='k', lw=3.,label='all',zorder=250)
    #ax.plot(10**xx[:,0],upper, color='k', lw=1,zorder=100)
    #ax.plot(10**xx[:,0],lower, color='k', lw=1,zorder=100)
         
    i_11 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m11)[-10]))
    i_12 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m12)[-10]))
    i_13 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m13)[-10]))
    j_11 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m11)[10]))
    j_12 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m12)[10]))
    j_13 = np.argmin(np.abs(10**xx[:,0]-np.sort(tratios_m13)[10]))
    ax.plot(10**xx[j_11:i_11,0],median_m11[j_11:i_11], color='#5D3A9B', lw=1.5, label='m11',zorder=200)
    ax.plot(10**xx[j_12:i_12,0],median_m12[j_12:i_12], color='#FFC20A', lw=1.5, label='m12',zorder=150)
    ax.plot(10**xx[j_13:i_13,0],median_m13[j_13:i_13], color='#D41159', lw=1.5, label='m13',zorder=100)
    
    #ax.plot(tratios,Vrot2sigma,',',color='k')
    if layout!='tratio':
        ax.axvline(t_ratio_virialization, color='grey', lw=0.7, ls='-', dash_capstyle='round')
    ## Or kernel density estimate plot instead of the contourf plot
    #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    # cset = ax.contour(xx, 10**yy, f, colors='k')
    # Label plot
    # ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff} \,\, {\rm at} \,\, 0.1 R_{\rm vir}$')
    if layout=='Vrot2sigma':
        ax.set_ylim(0,8)   
        ax.set_ylabel(r'$V_{\rm rot} / \sigma_{\rm g}$')
        ax.set_xlim(1e-3, 30)
    if layout=='sigma2vc':
        ax.set_ylim(0,1.2)   
        ax.set_ylabel(r'$\sigma_{\rm g}\ [{\rm km}\ {\rm s}^{-1}]$')
        ax.set_xlim(1e-3, 30)
    if layout=='Vrot2vc':
        ax.set_ylim(0,1.2)   
        ax.set_ylabel(r'$V_{\rm rot}\ [{\rm km}\ {\rm s}^{-1}]$')
        ax.set_xlim(1e-3, 30)
    if layout=='vc':
        ax.legend(loc='lower right', frameon=False,handlelength=1.5)
        ax.set_ylim(10,300)   
        ax.set_ylabel(r'$v_{\rm c}\ [{\rm km}\ {\rm s}^{-1}]$')
        ax.set_xlim(1e-3, 30)
    if layout=='T2Tvir':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(u.arilogformatter)        
        ax.set_ylim(0.03,3)   
        ax.set_xlim(1e-2, 100)
        ax.set_ylabel(r'$\langle T\rangle(0.1 R_{\rm vir})/T_{\rm vir}$')
        ax.legend(loc='lower right', frameon=False)
        plt.savefig(figDir+'log_Ts_vs_t_ratio.png',dpi=400, bbox_inches='tight')
    if layout=='tratio':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(u.arilogformatter)        
        ax.set_ylim(1e-3,100)   
        ax.set_xlim(1e-3,100)
        ax.set_ylabel(r'$t_{\rm cool}^{(s)}/t_{\rm ff}\ \ {\rm at}\ \ 0.5 R_{\rm vir}$')
        ax.legend(loc='lower right', frameon=False)
        u.plotLine(color='grey', lw=0.7, ls='-', dash_capstyle='round')
        plt.savefig(figDir+'t_ratio_comparison.png',dpi=400, bbox_inches='tight')
    return ((10**xx[j_11:i_11,0],median_m11[j_11:i_11]),
            (10**xx[j_12:i_12,0],median_m12[j_12:i_12]),
            (10**xx[j_13:i_13,0],median_m13[j_13:i_13]))

