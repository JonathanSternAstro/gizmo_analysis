import sys, os, subprocess, multiprocessing, traceback
from workdirs import *
import FIRE_files as ff
from FIRE_files import cosmo, u
import time, importlib
import pylab as pl, numpy as np, glob, pdb
from numpy import log10 as log
from projectPlotBasics import *
from astropy import units as un, constants as cons
import h5py
import scipy, scipy.stats
from matplotlib import ticker

sys.path.append(basedir+'FIRE_studio/')
import abg_python
import abg_python.snapshot_utils
import firestudio
from firestudio.studios.gas_studio import GasStudio

Z_solar = 0.0129
gamma = 5/3.
mu = 0.62
ne2nH = 1.2
X = 0.7; He2H = 0.1; Y = 4.*He2H * X

foldernames = ['m11_ad_25_lr_2_cooling_1',
            'm11_ad_25_lr_light_2_cooling_1',
            'm11_extralight_def_r30r',
            'm11_ad_25_lr_extralight_2_cooling_1',
            'm11_def_r30r',
            'm11_light_def_r30r']
snapshotdirs = [simdir+foldername+'/output/' for foldername in foldernames]
simnames = ['cooling_normal',
           'cooling_light',
           'feedback_extralight',
           'cooling_extralight',
           'feedback_normal',
           'feedback_light']

snapshotdirsdic = dict([(simnames[i],snapshotdirs[i]) for i in range(len(simnames))])

snapshotdirsdic['feedback_normalmine'] = '/projects/b1026/jonathan/gizmo_split_new_mmr2/output/'
def lsFunc(sim):
    if sim.galaxyname.split('_')[0]=='cooling': return '--'
    if sim.galaxyname.split('_')[0]=='feedback': return '-'
def cFunc(sim):
    if sim.galaxyname.split('_')[1]=='normal': return 'b'
    if sim.galaxyname.split('_')[1]=='light': return 'm'
    if sim.galaxyname.split('_')[1]=='extralight': return 'r'
def labelFunc(sim):
    if sim.galaxyname.split('_')[1]=='normal': return r'$f_{\rm CGM}=0.75$'
    if sim.galaxyname.split('_')[1]=='light': return r'$f_{\rm CGM}=0.25$'
    if sim.galaxyname.split('_')[1]=='extralight': return r'$f_{\rm CGM}=0.08$'

    
    

def grid_particle_data_2D(Xs,Ys,vals,bins):
    m,x,y,_ = scipy.stats.binned_statistic_2d(Xs,
                                              Ys,
                                              vals,
                                              statistic='median',
                                              bins=bins)
    xx,yy = np.meshgrid((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2)
    interp = np.array(m)
    naninds = np.isnan(m)
    interp[np.isnan(interp)] = scipy.interpolate.griddata(
         (xx[~naninds], yy[~naninds]), # points we know
         m[~naninds],                  # values we know
         (xx[naninds], yy[naninds]))   # points to interpolate
    return xx,yy,interp
def grid_particle_data_3D(Xs,Ys,Zs,vals,bins):
    m,x,y,_ = scipy.stats.binned_statistic_2d(np.array([Xs,Ys,Zs]),
                                              vals,
                                              statistic='median',
                                              bins=bins)
    xx,yy = np.meshgrid((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2)
    interp = np.array(m)
    naninds = np.isnan(m)
    interp[np.isnan(interp)] = scipy.interpolate.griddata(
         (xx[~naninds], yy[~naninds]), # points we know
         m[~naninds],                  # values we know
         (xx[naninds], yy[naninds]))   # points to interpolate
    return xx,yy,interp
class SnapshotProjection:
    def __init__(self, snapshot, arr, edge_on,r_max, width,fn,weight='mass',pixels=1200,overwrite=True):
        self.r_max = r_max
        self.studiodic = {}        
        if edge_on:
            coords = np.array([snapshot.coords()[:,i] for i in (1,2,0)]).T
        else:
            coords = snapshot.coords()
        self.studiodic['Coordinates'] = coords
        if weight=='mass': self.studiodic['Masses'] = snapshot.masses()
        if weight=='vol':  self.studiodic['Masses'] = snapshot.volume()
        self.studiodic['SmoothingLength'] = snapshot.dic['SmoothingLength']
        self.studiodic['BoxSize']= snapshot.f['Header'].attrs['BoxSize']
        self.studiodic['quantity'] = arr

        self.mystudio = GasStudio(
            snapdir = None, snapnum = snapshot.iSnapshot,
            snapdict = self.studiodic,datadir = projectionsdir,
             frame_half_width = r_max,frame_depth = width,
             quantity_name = 'quantity',take_log_of_quantity = False, 
             galaxy_extractor = False,pixels=pixels,
             single_image='Density',overwrite = overwrite,savefig=False,      
             use_hsml=True,intermediate_file_name=fn)
    def project(self):
        columnMap, quantityMap = self.mystudio.projectImage([])
        Xs = np.linspace(-self.r_max,self.r_max,columnMap.shape[0])
        return Xs,columnMap,quantityMap

class PowerLawPotential:
    def __init__(self,m,vc_Rvir,Rvir):
        self.m = m
        self.vc_Rvir = vc_Rvir
        self.Rvir = Rvir
    def vc(self, r):
        return self.vc_Rvir * (r/self.Rvir)**self.m

    
class KY_snapshot(ff.Snapshot):    
    zvec = np.array([0,0,1.])
    def __init__(self,fn,sim,center,pr=True):
        self.sim = sim
        self.f = h5py.File(fn,'r')
        self.dic = ff.h5py_dic([self.f],pr=pr)    
        self.iSnapshot = int(fn[-8:-5])
        
        if center is None:
            if self.sim.centerOnBlackHole:
                self.center = self.peakDensity(5) 
            else:
                self.center = self.peakDensity(0) 
        else:
            self.center = center
    def centerOfMass(self):
        DMmasses = self.dic[('PartType1','Masses')]
        return (self.dic[('PartType1','Coordinates')].T*DMmasses).sum(axis=1) / DMmasses.sum()                    
    def peakDensity(self,iPartType=0):
        if iPartType==5: return np.array([0,0,0]) #self.absCoords(5)[0]
             
        o = self.sim.origin
        inds = ((o[0]-50 < self.absCoords(iPartType)[:,0]) & (self.absCoords(iPartType)[:,0]<o[0]+50) &
                (o[1]-50 < self.absCoords(iPartType)[:,1]) & (self.absCoords(iPartType)[:,1]<o[1]+50) &
                (o[2]-50 < self.absCoords(iPartType)[:,2]) & (self.absCoords(iPartType)[:,2]<o[2]+50))
        hist,(x,y,z),_ = scipy.stats.binned_statistic_dd(self.absCoords(iPartType)[inds,:],
                                                     values=self.masses(iPartType)[inds],
                                                     statistic='sum',bins=200)
        ix,iy,iz = np.unravel_index(hist.argmax(),hist.shape)
        return np.mean(np.array([x[ix:ix+2],y[iy:iy+2],z[iz:iz+2]]),axis=1)
        
    def rho(self): # in g/cm^3
        return ((un.Msun/un.kpc**3).to('g/cm**3') *
                self.dic[('PartType0','Density')] * 1e10)
    def coords(self,iPartType=0): #in kpc
        absCoords = self.absCoords(iPartType=iPartType)
        if len(absCoords):
            return self.dic[('PartType%d'%iPartType,'Coordinates')] - self.center
        return np.array([])
    def absCoords(self,iPartType=0): #in kpc
        return self.dic[('PartType%d'%iPartType,'Coordinates')]

    def time(self): # in Gyr
        return self.f['Header'].attrs['Time']
    def number_of_particles(self):
        return self.f['Header'].attrs['NumPart_ThisFile']
    def vs(self,iPartType=0): # in km/s
        return self.dic[('PartType%d'%iPartType,'Velocities')]
    def v_phi(self,zvec=None):
        if zvec is None: zvec=self.zvec
        return ff.Snapshot.v_phi(self,zvec)
    def v_theta(self,zvec=None):
        if zvec is None: zvec=self.zvec
        return ff.Snapshot.v_theta(self,zvec)
    def cos_theta(self,zvec=None):
        if zvec is None: zvec=self.zvec
        return ff.Snapshot.cos_theta(self,self.zvec)
    def phi(self,iPartType=0):
        return np.arctan2(self.coords(iPartType=iPartType)[:,1],self.coords(iPartType=iPartType)[:,0])
def calc_properties_for_time_series(loadvals,iSnapshot,rMdot,rVrot):
    try:        
        print('starting snapshot #%d,   process id: %d'%(iSnapshot, os.getpid()),flush=True)
        sim = KY_sim(*loadvals)
        snapshot = sim.getSnapshot(iSnapshot)
        
        time = snapshot.time() 
    
        prof = sim.getProfiler(iSnapshot)
        SFR = prof.SFRprofile().sum() 
        stellar_ages = prof.stellar_ages()        
    
        ind = np.searchsorted(prof.rs_midbins(),rMdot.to('kpc').value)
        Mdot = prof.MdotProfile()[ind]
    
        weight='HI'; weights =  prof.HImassProfile()
        weights = weights * (prof.rs_midbins() < rVrot.to('kpc').value)
        vc_profile = prof.vc()
        v_phis = prof.profile1D('v_phi', weight,power=1) 
        v_phis2 = prof.profile1D('v_phi', weight,power=2)
        vphi_inds = ~np.isnan(v_phis) & ~np.isnan(v_phis2) & ~np.isnan(vc_profile)
        
        vc = (vc_profile*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
        v_phi  = (v_phis*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
        v_phi2 = (v_phis2*weights)[vphi_inds].sum() / weights[vphi_inds].sum()
        sigma = (v_phi2-v_phi**2)**0.5
        
        tcool = prof.t_cool(useP=False)[ind] 
        tcoolB = prof.t_cool(useP=True)[ind] 
        tff = prof.t_ff()[ind] 
        vcRcirc = vc_profile[ind]
        
        nH = X * prof.rhoProfile()[ind] / cons.m_p.to('g').value                            
        nHB = X*mu * (prof.P2k_CGM(maxR2Rvir=1) / prof.Tc())[ind]
        Z = prof.Z2Zsun()[ind]        
        
        T = 10.**prof.profile1D('log_Ts','VW')[ind]        
        Tc = prof.Tc()[ind]
        
        prof.tofile()
        print('ending snapshot #%d,   process id: %d'%(iSnapshot, os.getpid()),flush=True)
    except:
        traceback.print_exc() 
        raise # optional    
    return time,SFR,Mdot,vc,v_phi,sigma,tcool,tcoolB,tff,nH,nHB,Z,vcRcirc, T, Tc, stellar_ages
        
class KY_profiler(ff.Snapshot_profiler):
    z = 0 #for cooling function and critical density calculation
    sub_halo_centers = sub_halo_rvirs = sub_halo_gasMasses = None
    stellar_ages_time_bins = np.arange(0,30,0.01)
    def __init__(self,snapshot,Rcirc2Rvir,recalc=False):        
        self.Rcirc2Rvir = Rcirc2Rvir
        if not isinstance(snapshot,tuple):
            self.snapshot = snapshot
            self.galaxyname, self.time, self.rvir = self.snapshot.sim.galaxyname,self.snapshot.time(),self.snapshot.sim.rvir.value
        else:
            self.galaxyname, self.time, self.rvir = snapshot
        self._saveDic = {}        
        self.recalc = recalc
        self.load()
    def vc(self):
        vc = (((cons.G*un.Msun / un.kpc)**0.5).to('km/s').value * 
                (self.massProfile().cumsum() /self.rs_midbins())**0.5 )
        if self.snapshot.sim.analyticGravity!=None:
            analytic_vcs = self.snapshot.sim.analyticGravity.vc(self.rs_midbins()).to('km/s').value
            vc = (vc**2 + analytic_vcs**2)**0.5
        return vc
    def stellar_ages(self,max_r=10):
        if not self.isSaved('stellar_ages'):            
            masses = self.snapshot.masses(4)
            if len(masses)==0: 
                hist = np.zeros(self.stellar_ages_time_bins.shape[0]-1)
            else:
                times = self.snapshot.StarFormationTimes()
                inds = self.snapshot.rs(4) < max_r
                # TODO: fix for mass loss
                hist,x,_ = scipy.stats.binned_statistic(times[inds],masses[inds],statistic='sum',bins=self.stellar_ages_time_bins)   
            self.save('stellar_ages', hist)            
        return self.get_saved('stellar_ages')
    
    def filename(self):
        return profiledir + '%s/profiler_%.0fMyr.npz'%(self.galaxyname,self.time*1000)
    def isSaved(self,save_name):
        if save_name in self._saveDic: return True
        if (not self.recalc) and (os.path.exists(self.filename())) and (save_name in np.load(self.filename()).files): return True
        return False
    def tofile(self,overwrite=False,maxsize=None):
        simprofiledir = profiledir+'%s'%(self.galaxyname)
        if not os.path.exists(simprofiledir): os.mkdir(simprofiledir)
        super().tofile(overwrite,maxsize)
         
    
    

class KY_sim: 
    a = 1
    h = 1
    sub_halo_centers = sub_halo_rvirs = sub_halo_gasMasses = None
    def __init__(self,simname,snapshots_dir,rvir=100*un.kpc,dynamicCentering=False,recalc=False,
                 centerOnBlackHole=False,origin=([1500,1500,1500]),Nsnapshots=None,analyticGravity=None,
                 Rcirc=None,snapshot_dt_Myr=25,pr=True):              
        self.rvir = rvir #meaningless, used for defining bins in units of rvir
        self.Rcirc = Rcirc
        self.origin = origin
        fns = sorted(glob.glob(snapshots_dir+'*snapshot_*.hdf5'),key=lambda fn: int(fn.split('_')[-1][:-5]))
        fns = fns[:Nsnapshots]
        iSnapshots = [int(fn.split('_')[-1][:-5]) for fn in fns]
        self.snapshots = [None]*(max(iSnapshots)+1)
        self.profiles = [None]*(max(iSnapshots)+1)
        self.fns_dic = dict([(iSnapshots[i],fns[i]) for i in range(len(fns))])
        self.galaxyname=simname  
        self.recalc=recalc
        self.dynamicCentering = dynamicCentering
        self.centerOnBlackHole = centerOnBlackHole
        self.analyticGravity = analyticGravity
        self.pr = pr
        self.snapshots[0] = KY_snapshot(self.fns_dic[0],self,center=None,pr=self.pr) #for calculating center
        self.loadvals = (simname,snapshots_dir,rvir,dynamicCentering,recalc,
                         centerOnBlackHole,origin,Nsnapshots,analyticGravity,Rcirc)
        self.snapshot_dt_Myr = snapshot_dt_Myr
        
        #print([('PartType%d'%iPartType, len(self.snapshots[0].masses(iPartType))) for iPartType in range(6)])
        
    def __str__(self):
        return '%s'%(self.galaxyname)
    def __repr__(self):
        return self.__str__()
    def Nsnapshots(self):
        return len(self.snapshots)
    def getSnapshot(self,iSnapshot):
        if self.snapshots[iSnapshot]==None:
            if self.dynamicCentering:
                self.snapshots[iSnapshot] = KY_snapshot(self.fns_dic[iSnapshot],self,center=None,pr=self.pr)
            else:
                self.snapshots[iSnapshot] = KY_snapshot(self.fns_dic[iSnapshot],self,center=self.snapshots[0].center,pr=self.pr)
        return self.snapshots[iSnapshot]
    def getProfiler(self,iSnapshot):
        if self.profiles[iSnapshot]==None:
            self.profiles[iSnapshot] = KY_profiler(self.getSnapshot(iSnapshot),Rcirc2Rvir=self.Rcirc/self.rvir,recalc=self.recalc)
        return self.profiles[iSnapshot]
    def loadAllSnapshots(self):
        [self.getSnapshot(iSnapshot) for iSnapshot in self.fns_dic.keys()]
    def times(self):
        return [snapshot.time() for snapshot in self.snapshots]
    def timeSeries(self,rMdot,rVrot,multipleProcs=1,justLoad=False,SFRwindow_Myr=300):        
        if not justLoad:            
            pool = multiprocessing.Pool(processes=multipleProcs,maxtasksperchild=1)
            for iSnapshot in range(self.Nsnapshots()):
                pool.apply_async(calc_properties_for_time_series, (self.loadvals,iSnapshot,rMdot,rVrot))
            pool.close()
            pool.join()
            
        times  = np.zeros(self.Nsnapshots())
        SFRs   = np.zeros(self.Nsnapshots())
        Mdots  = np.zeros(self.Nsnapshots())
        vcs    = np.zeros(self.Nsnapshots())
        Vrots  = np.zeros(self.Nsnapshots())
        sigmas = np.zeros(self.Nsnapshots())
        tcools = np.zeros(self.Nsnapshots())
        tcoolBs = np.zeros(self.Nsnapshots())
        tffs = np.zeros(self.Nsnapshots())
        nHs = np.zeros(self.Nsnapshots())
        nHsB = np.zeros(self.Nsnapshots())
        Zs = np.zeros(self.Nsnapshots())
        vcRcircs = np.zeros(self.Nsnapshots())
        Ts = np.zeros(self.Nsnapshots())
        Tcs = np.zeros(self.Nsnapshots())
        stellar_ages = np.zeros(self.Nsnapshots())
        
        for iSnapshot in u.Progress(range(self.Nsnapshots())):
            res = calc_properties_for_time_series(self.loadvals,iSnapshot,rMdot,rVrot)
            (times[iSnapshot],SFRs[iSnapshot],Mdots[iSnapshot],
             vcs[iSnapshot],Vrots[iSnapshot],sigmas[iSnapshot],
             tcools[iSnapshot],tcoolBs[iSnapshot],tffs[iSnapshot],
             nHs[iSnapshot],nHsB[iSnapshot],Zs[iSnapshot],vcRcircs[iSnapshot],
             Ts[iSnapshot],Tcs[iSnapshot],_)= res
        SFRwindow = SFRwindow_Myr // self.snapshot_dt_Myr
        SFR_means = np.convolve(SFRs,np.ones(SFRwindow)/SFRwindow,mode='same')
        
        np.savez(profiledir + 'timeSeries_%s.npz'%self,
                 times=times,SFRs=SFRs,SFRmeans = SFR_means,Mdots=Mdots,
                 Vrots=Vrots,sigmas = sigmas,vcs=vcs,
                 tcools=tcools,tffs=tffs,tcoolBs=tcoolBs,
                 nHs = nHs,nHsB=nHsB,Zs=Zs,vcRcircs=vcRcircs, Ts=Ts,Tcs=Tcs)
        return times, SFRs, Mdots, vcs, Vrots, sigmas, tcools, tcoolBs, tffs, nHs, nHsB, Zs, Ts,Tcs, stellar_ages
    def movie(self,frameFunc,multipleProcs=1,start=None,end=None,**kwargs):        
        pool = multiprocessing.Pool(processes=multipleProcs,maxtasksperchild=1)
        for iSnapshot in range(self.Nsnapshots())[start:end]:
            pool.apply_async(frameFunc, (self.loadvals,iSnapshot),kwargs)
        pool.close()
        pool.join()
    def quantities_at_Rcirc(self):        
        ## load time series data
        f = np.load(profiledir + 'timeSeries_%s.npz'%self)
        timeSeries = (f['times'],f['SFRs'],f['SFRmeans'],f['Mdots'], f['Vrots'], f['sigmas'], 
                      f['vcs'], f['tcools'], f['tffs'], f['nHs'], f['Zs'], f['vcRcircs'], 
                      f['Ts'],f['Tcs'])
        (times, SFRs, SFRs_means, Mdots, Vrots, sigmas, vcs, tcools, 
         tffs, nHs, Zs, vcRcircs, Ts, Tcs) = timeSeries
        
    
        fig = pl.figure(figsize=(ff.pB.fig_width_full,6.5))
        plotsSpec  = matplotlib.gridspec.GridSpec(ncols=2, nrows=4, figure=fig,hspace=0.2,wspace=0.4,
                                                  left=0.15,right=0.97,bottom=0.1,top=0.95)
    
        #plots
        for iPanel in range(8):
            ax = fig.add_subplot(plotsSpec[iPanel%4,iPanel//4])
            pl.xlim(0,14)        
            fs = 10
            if iPanel==0: 
                ys = SFRs
                #pl.plot(times,SFRs_means,c='k')
                pl.ylabel(r'${\rm SFR}\ [{\rm M}_\odot{\rm yr}^{-1}]$',fontsize=fs)
            if iPanel==1: 
                ys= Mdots
                pl.ylabel(r'${\dot M}(30\ {\rm kpc})\ [{\rm M}_\odot{\rm yr}^{-1}]$',fontsize=fs)
                pl.axhline(0,c='k',lw=0.7,ls=':')
            if iPanel==2: 
                ys = Vrots
                pl.plot(times,sigmas,c='m')            
                pl.plot(times,vcs,ls='--',c='k',lw=1,zorder=-100)    
                pl.ylabel(r'$V_{\rm rot}$ or $\sigma_{\rm g}$ $[{\rm km}\ {\rm s}^{-1}]$',fontsize=fs)
                pl.ylim(0.,250)
                pl.text(0.2,175,r'$V_{\rm rot}$',color='b')
                pl.text(0.2,60,r'$\sigma_{\rm g}$',color='m')
            if iPanel==3:
                ys = tcools/tffs
                pl.semilogy()
                pl.ylabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff}\ \ {\rm at}\ \ 30\ {\rm kpc}$',fontsize=fs)
                pl.axhline(1,c='k',lw=0.7,ls='dashed')
                ax.yaxis.set_major_formatter(u.arilogformatter)                
            if iPanel==4:
                ys = nHs
                pl.semilogy()
                pl.ylabel(r'$n_{\rm H}\ {\rm at}\ 30\ {\rm kpc}\ [{\rm cm}^{-3}]$',fontsize=fs)
                ax.yaxis.set_major_formatter(u.arilogformatter)
            if iPanel==5:
                ys = Zs
                pl.semilogy()
                pl.ylabel(r'$Z\ {\rm at}\ \ 30\ {\rm kpc}\ [{\rm Z}_\odot]$',fontsize=fs)
                ax.yaxis.set_major_formatter(u.arilogformatter)
                pl.ylim(0.1,5)
            if iPanel==6:
                ys = vcRcircs
                pl.ylabel(r'$v_{\rm c}\ {\rm at}\ \ 30\ {\rm kpc}\ [{\rm km}\ {\rm s}^{-1}]$',fontsize=fs)
                pl.ylim(0.,250)
            if iPanel==7:
                ys = Ts
                pl.semilogy()
                pl.ylabel(r'$T\ {\rm at}\ \ 30\ {\rm kpc}\ [[{\rm K}]$',fontsize=fs)
                ax.yaxis.set_major_formatter(u.arilogformatter)
                pl.plot(times,Tcs,ls='--',c='k')
                pl.ylim(1e4,3e6)

            if ax.is_last_row():
                pl.xlabel(r'${\rm time}\ [{\rm Gyr}]$',fontsize=fs)
            pl.plot(times,ys,c='b',lw=0.7)    
        fig.savefig(figdir+'/quantities_at_Rcirc_%s.png'%(self),dpi=300)        
       
    
def temperature_and_pressure_movieFrame_async(loadvals,iSnapshot, calculateProjections,edge_on=True,lazy=True):
    edge_str = ('face_on','edge_on')[edge_on]
    try:        
        print('starting snapshot #%d,   process id: %d'%(iSnapshot, os.getpid()),flush=True)
        sim = KY_sim(*loadvals)
        snapshot = sim.getSnapshot(iSnapshot)
        prof = sim.getProfiler(iSnapshot)
        simprojectionsdir = projectionsdir + '%s'%sim        
        projections_fn = simprojectionsdir + '/my_projections_%s_%d.npz'%(edge_str,iSnapshot)
        if calculateProjections and (not os.path.exists(projections_fn) or not lazy):    
            logT_projected = SnapshotProjection(snapshot, log(snapshot.Ts()),
                                                edge_on=edge_on,r_max=100,width=100,
                                                fn='%s_%d_T_%s'%(sim,iSnapshot,edge_str),
                                                overwrite=True,pixels=400)
            Tprojection_result = logT_projected.project()
            
            inds = np.searchsorted(ff.Snapshot_profiler.log_r2rvir_bins,log(snapshot.r2rvirs()))
            mean_lognHTs = np.concatenate([[1e-10], prof.profile1D('log_nHTs','VW'),[1e-10]])[inds]
            logPs_normalized = snapshot.log_nHTs() - mean_lognHTs
            P_projected = SnapshotProjection(snapshot, logPs_normalized,weight='vol',
                                                edge_on=edge_on,r_max=100,width=100,
                                                fn='%s_%d_P_%s'%(sim,iSnapshot,edge_str),
                                                overwrite=True,pixels=400)
            Pprojection_result = P_projected.project() 
            
            if not os.path.exists(simprojectionsdir): os.mkdir(simprojectionsdir)
            np.savez(projections_fn,
                    time        = snapshot.time(),
                    Xs          = Tprojection_result[0],
                    massColumn  = Tprojection_result[1],
                    volColumn   = Pprojection_result[1],
                    Tprojection = Tprojection_result[2],
                    Pprojection = Pprojection_result[2])
            prof.tofile()
        else: ## load projections         
            print('loading projection')
            f = np.load(projections_fn)
            Tprojection_result = f['Xs'],f['massColumn'],f['Tprojection']
            Pprojection_result = f['Xs'],f['volColumn'],f['Pprojection']
        
        ## load time series data
        f = np.load(profiledir + 'timeSeries_%s.npz'%sim)
        timeSeries = f['times'],f['SFRs'],f['SFRmeans'],f['Mdots'], f['Vrots'], f['sigmas'], f['vcs'], f['tcools'], f['tffs']
        times, SFRs, SFRs_means, Mdots, Vrots, sigmas, vcs, tcools, tffs = timeSeries
        
    
        fig = pl.figure(figsize=(ff.pB.fig_width_full,5.5))
        imagesSpec = matplotlib.gridspec.GridSpec(ncols=2, nrows=2, figure=fig,hspace=0.1,wspace=0.2,left=0.1,right=0.93,bottom=0.15,top=0.85)
        plotsSpec  = matplotlib.gridspec.GridSpec(ncols=2, nrows=4, figure=fig,hspace=0.2,wspace=0.2,left=0.1,right=0.93,bottom=0.1,top=0.9)
    
        #images
        for iRow in range(2):
            ax = fig.add_subplot(imagesSpec[iRow,1])
            if iRow==0: 
                res = Tprojection_result; label = r'temperature $[{\rm K}]$'
                cmap = 'RdBu_r'; cbar_range = 2e3,5e6; ticks = [1e3,1e4,1e5,1e6]; ticklabels = [r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$']
            if iRow==1:
                res = Pprojection_result; label = r'$P/\langle P(r)\rangle$'
                cmap='viridis'; cbar_range = 10**-0.5,10**0.5; ticks = [0.1,1,10]; ticklabels = [r'$0.1$', r'$1$', r'$10$']; 
        
            pl.pcolormesh(res[0],res[0],10.**res[2].T,
                          vmin=cbar_range[0],vmax=cbar_range[1],cmap=cmap,norm=matplotlib.colors.LogNorm(*cbar_range))
            if ax.is_first_row():
                pl.title(r'$%.2f\ {\rm Gyr}$'%snapshot.time())
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())            
            cbar = pl.colorbar(orientation='vertical',ax=ax,fraction=0.1,shrink=0.8,pad=0.02,ticks=ticks)
            cbar.set_label(label)
            cbar.ax.set_yticklabels(ticklabels,fontsize=10)
            if iRow==1:
                cbar.ax.set_yticklabels([r'$0.4$',r'',r'$0.6$',r'',r'',r'',r'$2$',r'$3$'],fontsize=10,minor=True)
            if edge_on:
                pl.plot([-75,-25],[-85,-85],c='k',lw=0.7)
                pl.text(-50,-78,r'$50\ {\rm kpc}$',fontsize=7,ha='center')
            else:
                pl.xlim(-60,60)
                pl.ylim(-60,60)
                pl.plot([-45,-15],[-50,-50],c='k',lw=0.7)
                pl.text(-30,-45,r'$30\ {\rm kpc}$',fontsize=7,ha='center')
        #plots
        for iRow in range(4):
            ax = fig.add_subplot(plotsSpec[iRow,0])            
            pl.xlim(0,14)        
            fs = 10
            if iRow==0: 
                ys = SFRs
                #pl.plot(times,SFRs_means,c='k')
                pl.ylabel(r'${\rm SFR}\ [{\rm M}_\odot{\rm yr}^{-1}]$',fontsize=fs)
            if iRow==1: 
                ys= Mdots
                pl.ylabel(r'${\dot M}(30\ {\rm kpc})\ [{\rm M}_\odot{\rm yr}^{-1}]$',fontsize=fs)
                pl.axhline(0,c='k',lw=0.7,ls=':')
            if iRow==2: 
                ys = Vrots
                pl.plot(times,sigmas,c='m')            
                pl.plot(times,vcs,ls='--',c='k',lw=1,zorder=-100)    
                pl.ylabel(r'$V_{\rm rot}$ or $\sigma_{\rm g}$ $[{\rm km}\ {\rm s}^{-1}]$',fontsize=fs)
                pl.ylim(0.,250)
                pl.text(0.2,175,r'$V_{\rm rot}$',color='b')
                pl.text(0.2,60,r'$\sigma_{\rm g}$',color='m')
                pl.plot([times[iSnapshot]],[sigmas[iSnapshot]],'x',c='k')    
                circleCenter = fig.transFigure.inverted().transform(ax.transData.transform((times[iSnapshot], sigmas[iSnapshot])))
                circle = pl.Circle(circleCenter , 0.02, edgecolor='k',facecolor='m',ls=None,alpha=0.2,transform=fig.transFigure)
                ax.add_artist(circle)
            if iRow==3:
                ys = tcools/tffs
                pl.semilogy()
                pl.ylabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff}\ \ {\rm at}\ \ 30\ {\rm kpc}$',fontsize=fs)
                pl.axhline(1,c='k',lw=0.7,ls='dashed')
                ax.yaxis.set_major_formatter(u.arilogformatter)
                pl.xlabel(r'${\rm time}\ [{\rm Gyr}]$',fontsize=fs)
                
            pl.plot(times,ys,c='b',lw=0.7)    
            pl.plot([times[iSnapshot]],[ys[iSnapshot]],'x',c='k')    
            circleCenter = fig.transFigure.inverted().transform(ax.transData.transform((times[iSnapshot], ys[iSnapshot])))
            circle = pl.Circle(circleCenter , 0.02, edgecolor='k',facecolor='b',ls=None,alpha=0.2,transform=fig.transFigure)
            ax.add_artist(circle)
        simmoviedir = moviedir+'%s'%sim
        if not os.path.exists(simmoviedir): os.mkdir(simmoviedir)
        fig.savefig(simmoviedir+'/temperature_and_pressure_movieFrame_%s_%d.png'%(edge_str,iSnapshot),dpi=300)        
    except:
        traceback.print_exc() 
        raise # optional    
    
def T_projection_example(sims):    
    res = []
    for i,sim in enumerate(sims):
        iSnapshot = min(424, max(sim.fns_dic.keys()))
        snapshot = sim.getSnapshot(iSnapshot)
        logT_projected = SnapshotProjection(snapshot, 
                                        log(snapshot.Ts()),
                                        True,100,100,
                                        'T_projection_example')
        res.append( logT_projected.project() )
    return res
def T_projection_plot(res):
    pl.figure(figsize=(ff.pB.fig_width_full,4.3))
    axs = [pl.subplot(1,2,i+1) for i in range(2)]
    for iPanel in range(2):
        ax = axs[iPanel]; pl.sca(ax)
        q = res[iPanel]
        pl.pcolormesh(q[0],q[0],q[2].T,vmin=4,vmax=6,cmap='RdBu_r')
        pl.title(r'$t_{\rm cool}%s t_{\rm ff}$'%(('\gg','\ll')[iPanel]))
        pl.xlabel(r'$[{\rm kpc}]$')
        pl.ylabel(r'$[{\rm kpc}]$')
        if iPanel==1: 
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_ticks_position('both')
    cbar = pl.colorbar(orientation='horizontal',ax=axs,
                       aspect=50,shrink=0.7,pad=0.2)
    cbar.set_label(r'$\langle \log\ T\rangle_\rho\ [{\rm K}]$')
    cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    pl.savefig(figdir+'comparison.png',dpi=600,bbox_inches='tight')    
def sim_properties(sim, iSnapshots):
    fig = pl.figure(figsize=(6,10))
    pl.subplots_adjust(wspace=0.2)
    for iPanel in range(10):
        ax = pl.subplot(5,2,iPanel+1)
        for ii,iSnapshot in enumerate(iSnapshots):
            prof = sim.getProfiler(iSnapshot)
            label = ff.u.nSignificantDigits((prof.snapshot.time()*1000),2)
            if iPanel==0: ys = vcs = prof.vc()
            if iPanel==1: ys = prof.jzProfile()
            if iPanel==2: ys = prof.profile1D('Z2Zsuns','MW')
            if iPanel==3: ys = prof.nH(useP=False)                        
            if iPanel==4: ys = 10.**prof.profile1D('log_Ts','VW')
            if iPanel==5: ys = 1 - prof.profile1D('isSubsonic','VW')
            if iPanel==6: ys = prof.t_cool(useP=False)
            if iPanel==7: ys = prof.t_cool(useP=False) / prof.t_ff()                
            if iPanel==8: 
                v_phis = prof.profile1D('v_phi', 'MW',power=1) 
                ys = v_phis/vcs
            if iPanel==9: 
                v_phis2 = prof.profile1D('v_phi','MW',power=2)
                sigma = (v_phis2-v_phis**2)**0.5
                ys = v_phis/sigma
            
            pl.plot(prof.rs_midbins(), ys,label=r'$%d\ {\rm Myr}$'%label,c=cmap(iSnapshot/sim.Nsnapshots()))
        if iPanel==0:
            pl.ylim(0,200)
            pl.ylabel(r'$v_{\rm c}=\sqrt{GM(<r)/r}\ [{\rm km}\ {\rm s}^{-1}]$')
            pl.legend(fontsize=8,handlelength=1)
        if iPanel==1:
            pl.plot(prof.rs_midbins(), prof.vc()*prof.rs_midbins(),label=r'$v_c \times r$',c='k',ls=':')
            pl.ylim(1e2,3e4); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            ax.set_yscale('log')
            pl.ylabel(r'$j_z = v_{\phi}r\ [{\rm kpc}\ {\rm km}\ {\rm s}^{-1}]$')
        if iPanel==2:
            pl.ylim(0.05,2)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$Z / Z_\odot$')
        if iPanel==3:
            pl.ylim(1e-7,0.01)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$n_{\rm H}\ [{\rm cm}^{-3}]$')
        if iPanel==4:
            pl.ylim(1e4,3e6)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$T\ [{\rm K}]$')
            pl.plot(prof.rs_midbins(), prof.Tc(),label=r'$T_c$',c='k',ls=':')
        if iPanel==5:
            pl.ylim(-0.1,1.1)
            pl.ylabel(r'supersonic fraction')
        if iPanel==6:
            pl.ylim(1,1e5)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$t_{\rm cool}^{(s)}\ [{\rm Gyr}]$')
            pl.axhline(1e4,c='k',ls=':')
        if iPanel==7:
            pl.ylim(0.01,100)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff}$')
            pl.axhline(1,c='k',ls=':')
        if iPanel==8:
            pl.ylim(0.,1.5)
            pl.ylabel(r'$V_{\rm rot} / v_{\rm c}$') 
        if iPanel==9:
            pl.ylim(0,10)
            pl.ylabel(r'$V_{\rm rot} / \sigma$') 
        
        pl.xlim(1,1000)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ff.u.arilogformatter)
        if ax.is_last_row():
            pl.xlabel(r'$r\ [{\rm kpc}]$')    
        if ax.is_last_col():
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_ticks_position('both')
    pl.text(0.5,0.95,sim.galaxyname,ha='center',transform=fig.transFigure)
    pl.savefig(figdir+'sim_properties_%s.pdf'%sim.galaxyname)    
def CGM_properties(sim, iSnapshots,Rcirc):
    pl.ioff()

    fig = pl.figure(figsize=(10,10))
    pl.subplots_adjust(wspace=0.3)
    for iPanel in range(9):
        ax = pl.subplot(3,3,iPanel+1)
        for ii,iSnapshot in enumerate(iSnapshots):
            prof = sim.getProfiler(iSnapshot)
            label = ff.u.nSignificantDigits((prof.snapshot.time()*1000),2)
            if iPanel==0: ys = vcs = prof.vc()
            if iPanel==1: ys = prof.jzProfile()
            if iPanel==2: ys = prof.profile1D('Z2Zsuns','MW')
            if iPanel==3: ys = prof.nH(useP=False)                                    
            if iPanel==4: ys = 10.**prof.profile1D('log_Ts','VW')
            if iPanel==5: ys = 1 - prof.profile1D('isSubsonic','VW')
            if iPanel==6: ys = prof.t_cool(useP=False)
            if iPanel==7: ys = prof.t_cool(useP=False) / prof.t_ff()      
            if iPanel==8: ys = prof.MdotProfile()
            
            pl.plot(prof.rs_midbins(), ys,label=r'$%d\ {\rm Myr}$'%label,c=cmap(iSnapshot/iSnapshots.max()))
            pl.axvline(Rcirc,c='.5',ls=':',lw=0.3)
        if iPanel==0:
            pl.ylim(0,200)
            pl.ylabel(r'$v_{\rm c}=\sqrt{GM(<r)/r}\ [{\rm km}\ {\rm s}^{-1}]$')
            pl.legend(fontsize=8,handlelength=1)
        if iPanel==1:
            pl.plot(prof.rs_midbins(), prof.vc()*prof.rs_midbins(),label=r'$v_c \times r$',c='k',ls=':')
            pl.ylim(1e2,3e4); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            ax.set_yscale('log')
            pl.ylabel(r'$j_z = v_{\phi}r\ [{\rm kpc}\ {\rm km}\ {\rm s}^{-1}]$')
        if iPanel==2:
            pl.ylim(0.01,2)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$Z / Z_\odot$')
        if iPanel==3:
            pl.ylim(1e-7,0.01)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$n_{\rm H}\ [{\rm cm}^{-3}]$')
        if iPanel==4:
            pl.ylim(1e4,3e6)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$T\ [{\rm K}]$')
            pl.plot(prof.rs_midbins(), prof.Tc(),label=r'$T_c$',c='k',ls=':')
        if iPanel==5:
            pl.ylim(-0.1,1.1)
            pl.ylabel(r'supersonic fraction')
        if iPanel==6:
            pl.ylim(1,1e5)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$t_{\rm cool}^{(s)}\ [{\rm Gyr}]$')
            pl.axhline(1e4,c='k',ls=':')
        if iPanel==7:
            pl.ylim(0.01,100)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff}$')
            pl.axhline(1,c='k',ls=':')
        if iPanel==8:
            pl.ylim(-20.,20)
            pl.ylabel(r'$\dot{M}\ [{\rm M}_\odot\ {\rm yr}^{-1}]$') 
            pl.axhline(0,c='k',ls=':')
        pl.xlim(3,300)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ff.u.arilogformatter)
        if ax.is_last_row():
            pl.xlabel(r'$r\ [{\rm kpc}]$')    
        if ax.is_last_col():
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_ticks_position('both')
    pl.text(0.5,0.95,sim.galaxyname,ha='center',transform=fig.transFigure)
    pl.savefig(figdir+'CGM_properties_%s.pdf'%sim.galaxyname)    
def compare_sims(sims, iSnapshot):
    fig = pl.figure(figsize=(6,10))
    pl.subplots_adjust(wspace=0.2)
    for iPanel in range(10):
        ax = pl.subplot(5,2,iPanel+1)
        for isim,sim in enumerate(sims):
            prof = sim.getProfiler(iSnapshot)
            label = sim.galaxyname
            if iPanel==0: ys = vcs = prof.vc()
            if iPanel==1: ys = prof.jzProfile()
            if iPanel==2: ys = prof.profile1D('Z2Zsuns','MW')
            if iPanel==3: ys = prof.nH(useP=False)                        
            if iPanel==4: ys = 10.**prof.profile1D('log_Ts','VW')
            if iPanel==5: ys = 1 - prof.profile1D('isSubsonic','VW')
            if iPanel==6: ys = prof.t_cool(useP=False)
            if iPanel==7: ys = prof.t_cool(useP=False) / prof.t_ff()                
            if iPanel==8: 
                v_phis = prof.thickerShells(prof.profile1D('v_phi', 'MW',power=1),5,'MW')
                ys = v_phis/vcs
            if iPanel==9: 
                v_phis2 = prof.thickerShells(prof.profile1D('v_phi','MW',power=2),5,'MW')
                sigma = (v_phis2-v_phis**2)**0.5
                ys = v_phis/sigma
            pl.plot(prof.rs_midbins(), ys,label='%s'%label)
        if iPanel==0:
            pl.ylim(0,200)
            pl.ylabel(r'$v_{\rm c}=\sqrt{GM(<r)/r}\ [{\rm km}\ {\rm s}^{-1}]$')
            pl.legend(fontsize=8,handlelength=1)
        if iPanel==1:
            pl.plot(prof.rs_midbins(), prof.vc()*prof.rs_midbins(),label=r'$v_c \times r$',c='k',ls=':')
            pl.ylim(1e2,3e4); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            ax.set_yscale('log')
            pl.ylabel(r'$j_z = v_{\phi}r\ [{\rm kpc}\ {\rm km}\ {\rm s}^{-1}]$')
        if iPanel==2:
            pl.ylim(0.05,2)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$Z / Z_\odot$')
        if iPanel==3:
            pl.ylim(1e-7,0.01)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$n_{\rm H}\ [{\rm cm}^{-3}]$')
        if iPanel==4:
            pl.ylim(1e4,3e6)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$T\ [{\rm K}]$')
            pl.plot(prof.rs_midbins(), prof.Tc(),label=r'$T_c$',c='k',ls=':')
        if iPanel==5:
            pl.ylim(-0.1,1.1)
            pl.ylabel(r'supersonic fraction')
        if iPanel==6:
            pl.ylim(1,1e5)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$t_{\rm cool}^{(s)}\ [{\rm Gyr}]$')
            pl.axhline(1e4,c='k',ls=':')
        if iPanel==7:
            pl.ylim(0.01,100)
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ff.u.arilogformatter)
            pl.ylabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff}$')
            pl.axhline(1,c='k',ls=':')
        if iPanel==8:
            pl.ylim(0.,1.5)
            pl.ylabel(r'$V_{\rm rot} / v_{\rm c}$') 
            pl.axhline(1,c='k',ls=':')
        if iPanel==9:
            pl.ylim(0,10)
            pl.ylabel(r'$V_{\rm rot} / \sigma$') 
            pl.axhline(1,c='k',ls=':')
        
        pl.xlim(1,1000)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ff.u.arilogformatter)
        if ax.is_last_row():
            pl.xlabel(r'$r\ [{\rm kpc}]$')    
        if ax.is_last_col():
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_ticks_position('both')
    pl.savefig(figdir+'compare_sims_%s.pdf'%sim.galaxyname.split('_')[0])    

def param_figs_by_time_withMvirRcirc(sims,isRcirc,showOther=False,max_log_t_ratio=1.5,t_ratio_cross=2,savefig=True):
    fig = pl.figure(figsize=(pB.fig_width_full*1.2,7.5))
    pl.subplots_adjust(left=0.2,wspace=0.35,hspace=0.1,right=0.85,bottom=0.06)
    log_t_ff_zs_p1 = np.arange(0,1,.01)
    t_ffs = np.zeros((len(sims),log_t_ff_zs_p1.shape[0]))
    nHs_all = np.zeros((len(sims),log_t_ff_zs_p1.shape[0]))
    axs = []
    norm = pl.Normalize(-max_log_t_ratio, max_log_t_ratio)
    for iPanel in range(6):
        ax=pl.subplot(3,2,iPanel+1)
        axs.append(ax)
        for isim,sim in enumerate(sims):
            xs = sim.times()
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

def temperature_and_pressure(sim,iSnapshots, Tprojection_results, Pprojection_results):
    pl.figure(figsize=(ff.pB.fig_width_full,3.))
    pl.subplots_adjust(wspace=0.05,hspace=0.1)
    for iRow in range(2):
        axs = [pl.subplot(2,len(iSnapshots),iRow*len(iSnapshots)+iPanel+1) for iPanel in range(len(iSnapshots))]   
        if iRow==0: 
            res = Tprojection_results; label = r'temperature $[{\rm K}]$'
            cmap = 'RdBu_r'; cbar_range = 1e3,1e6; ticks = [1e3,1e4,1e5,1e6]; ticklabels = [r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$']
        if iRow==1:
            res = Pprojection_results; label = r'$P/\langle P(r)\rangle$'
            cmap='viridis'; cbar_range = 0.1,10; ticks = [0.1,1,10]; ticklabels = [r'$0.1$', r'$1$', r'$10$']
        for iPanel,iSnapshot in enumerate(iSnapshots):
            snapshot = sim.getSnapshot(iSnapshot)
            ax = axs[iPanel]; pl.sca(ax)
            pl.pcolormesh(res[iPanel][0],res[iPanel][0],10.**res[iPanel][2].T,
                          vmin=cbar_range[0],vmax=cbar_range[1],cmap=cmap,norm=matplotlib.colors.LogNorm(*cbar_range))
            if ax.is_first_row():
                pl.title(r'$%.1f\ {\rm Gyr}$'%snapshot.time())
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())            
            if ax.is_first_col():
                pl.plot([-75,-25],[-85,-85],c='k',lw=0.7)
                pl.text(-50,-78,r'$50\ {\rm kpc}$',fontsize=7,ha='center')
        cbar = pl.colorbar(orientation='vertical',ax=axs,fraction=0.1,shrink=0.8,pad=0.02,ticks=ticks)
        cbar.set_label(label)
        cbar.ax.set_yticklabels(ticklabels,fontsize=10)
    pl.savefig(figdir+'temperature_and_pressure_%s.png'%sim,dpi=600,bbox_inches='tight')    
    
def temperature_and_pressure_movieFrame(sim,iSnapshot, Tprojection_results, Pprojection_results, timeSeries):
    fig = pl.figure(figsize=(ff.pB.fig_width_full,5.5))
    imagesSpec = matplotlib.gridspec.GridSpec(ncols=2, nrows=2, figure=fig,hspace=0.1,wspace=0.2,left=0.025,right=0.93,bottom=0.15,top=0.85)
    plotsSpec  = matplotlib.gridspec.GridSpec(ncols=2, nrows=4, figure=fig,hspace=0.2,wspace=0.2,left=0.025,right=0.93,bottom=0.1,top=0.9)
    
    #images
    for iRow in range(2):
        ax = fig.add_subplot(imagesSpec[iRow,1])
        if iRow==0: 
            res = Tprojection_results; label = r'temperature $[{\rm K}]$'
            cmap = 'RdBu_r'; cbar_range = 1e3,1e6; ticks = [1e3,1e4,1e5,1e6]; ticklabels = [r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$']
        if iRow==1:
            res = Pprojection_results; label = r'$P/\langle P(r)\rangle$'
            cmap='viridis'; cbar_range = 10**-0.5,10**0.5; ticks = [0.1,1,10]; ticklabels = [r'$0.1$', r'$1$', r'$10$']; 
        snapshot = sim.getSnapshot(iSnapshot)
        pl.pcolormesh(res[0],res[0],10.**res[2].T,
                      vmin=cbar_range[0],vmax=cbar_range[1],cmap=cmap,norm=matplotlib.colors.LogNorm(*cbar_range))
        if ax.is_first_row():
            pl.title(r'$%.2f\ {\rm Gyr}$'%snapshot.time())
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())            
        pl.plot([-75,-25],[-85,-85],c='k',lw=0.7)
        pl.text(-50,-78,r'$50\ {\rm kpc}$',fontsize=7,ha='center')
        cbar = pl.colorbar(orientation='vertical',ax=ax,fraction=0.1,shrink=0.8,pad=0.02,ticks=ticks)
        cbar.set_label(label)
        cbar.ax.set_yticklabels(ticklabels,fontsize=10)
        if iRow==1:
            cbar.ax.set_yticklabels([r'$0.4$',r'',r'$0.6$',r'',r'',r'',r'$2$',r'$3$'],fontsize=10,minor=True)
    #plots
    for iRow in range(4):
        ax = fig.add_subplot(plotsSpec[iRow,0])
        times, SFRs, SFRs_means, Mdots, Vrots, sigmas, vcs, tcools, tffs = timeSeries
        if sim.galaxyname=='feedback_normal': pl.xlim(0,8)        
        if sim.galaxyname=='feedback_light':  pl.xlim(0,6)
        fs = 10
        if iRow==0: 
            ys = SFRs
            #pl.plot(times,SFRs_means,c='k')
            pl.ylabel(r'${\rm SFR}\ [{\rm M}_\odot{\rm yr}^{-1}]$',fontsize=fs)
            if sim.galaxyname=='feedback_normal': pl.ylim(0,30)
            if sim.galaxyname=='feedback_light':  pl.ylim(0,6)
        if iRow==1: 
            ys= Mdots
            pl.ylabel(r'${\dot M}(25\ {\rm kpc})\ [{\rm M}_\odot{\rm yr}^{-1}]$',fontsize=fs)
            pl.axhline(0,c='k',lw=0.7,ls=':')
            if sim.galaxyname=='feedback_normal': pl.ylim(-30,30)
            if sim.galaxyname=='feedback_light':  pl.ylim(-6,6)
        if iRow==2: 
            ys = Vrots
            pl.plot(times,sigmas,c='m')            
            pl.plot(times,vcs,ls='--',c='k',lw=1,zorder=-100)    
            pl.ylabel(r'$V_{\rm rot}$ or $\sigma_{\rm g}$ $[{\rm km}\ {\rm s}^{-1}]$',fontsize=fs)
            pl.ylim(0.,250)
            pl.text(0.2,150,r'$V_{\rm rot}$',color='b')
            pl.text(0.2,60,r'$\sigma_{\rm g}$',color='m')
            pl.plot([times[iSnapshot]],[sigmas[iSnapshot]],'x',c='k')    
            circleCenter = fig.transFigure.inverted().transform(ax.transData.transform((times[iSnapshot], sigmas[iSnapshot])))
            circle = pl.Circle(circleCenter , 0.02, edgecolor='k',facecolor='m',ls=None,alpha=0.2,transform=fig.transFigure)
            ax.add_artist(circle)
        if iRow==3:
            ys = tcools/tffs
            pl.semilogy()
            pl.ylabel(r'$t_{\rm cool}^{(s)} / t_{\rm ff}\ \ {\rm at}\ \ 25\ {\rm kpc}$',fontsize=fs)
            pl.axhline(1,c='k',lw=0.7,ls='dashed')
            pl.ylim(0.3,30)    
            ax.yaxis.set_major_formatter(u.arilogformatter)
            pl.xlabel(r'${\rm time}\ [{\rm Gyr}]$',fontsize=fs)
            
        pl.plot(times,ys,c='b',lw=0.7)    
        pl.plot([times[iSnapshot]],[ys[iSnapshot]],'x',c='k')    
        circleCenter = fig.transFigure.inverted().transform(ax.transData.transform((times[iSnapshot], ys[iSnapshot])))
        circle = pl.Circle(circleCenter , 0.02, edgecolor='k',facecolor='b',ls=None,alpha=0.2,transform=fig.transFigure)
        ax.add_artist(circle)
    pl.savefig(moviedir+'temperature_and_pressure_movieFrame_%s_%d.png'%(sim,iSnapshot),dpi=600,bbox_inches='tight')    
