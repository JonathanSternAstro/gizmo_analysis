
import sys,string,os

## load input parameter
print(sys.argv)
simname = sys.argv[1] 
iSnapshot= int(sys.argv[2]) #e.g. '1' -- recieved by automatic script which runs over iSnapshot

homedir = os.getenv("HOME")+'/'
projectdir = homedir+'jonathanmain/CGM/KY_sims/'
figdir = projectdir+'figures/'
project_workdir = '/projects/b1026/jonathan/KY_sims/'
simdir = project_workdir+'sim_outputs/'
projectionsdir = project_workdir+'projections/'

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
import first_pass as l


t_start = time.time()

r = 30*un.kpc
sim = l.KY_sim(simname,l.snapshotdirsdic[simname],2.1e11*un.Msun,128.585*un.kpc,dynamicCentering=True,recalc=True)
snapshot = sim.getSnapshot(iSnapshot)
prof = sim.getProfiler(iSnapshot,Rcirc2Rvir=r/sim.rvir)

calculateProfiles = True
if calculateProfiles:
    prof.SFRprofile()
    prof.MdotProfile()
    prof.profile1D('log_nHTs','VW')
    prof.HImassProfile()
    prof.vc()
    prof.profile1D('v_phi', 'HI',power=1) 
    prof.profile1D('v_phi', 'HI',power=2)
    prof.jzProfile()
    prof.profile1D('Z2Zsuns','MW')
    prof.nH(useP=False)                        
    prof.profile1D('log_Ts','VW')
    prof.profile1D('isSubsonic','VW')
    prof.t_cool(useP=False)
    prof.t_ff()                
    prof.tofile()

movieFrame = False
if movieFrame:    
    calculateProjections = True
    if calculateProjections:    
        logT_projected = l.SnapshotProjection(snapshot, log(snapshot.Ts()),
                                            edge_on=True,r_max=100,width=100,
                                            fn='%s_%d_T_edge_on'%(sim,iSnapshot))
        Tprojection_result = logT_projected.project()
        
        inds = np.searchsorted(ff.Snapshot_profiler.log_r2rvir_bins,log(snapshot.r2rvirs()))
        mean_lognHTs = np.concatenate([[1e-10], prof.profile1D('log_nHTs','VW'),[1e-10]])[inds]
        logPs_normalized = snapshot.log_nHTs() - mean_lognHTs
        P_projected = l.SnapshotProjection(snapshot, logPs_normalized,weight='vol',
                                            edge_on=True,r_max=100,width=100,
                                            fn='%s_%d_P_edge_on'%(sim,iSnapshot))
        Pprojection_result = P_projected.project() 
        
        np.savez(projectionsdir + 'my_projections_%s_%d.npz'%(sim,iSnapshot),
                time        = snapshot.time(),
                Xs          = Tprojection_result[0],
                massColumn  = Tprojection_result[1],
                volColumn   = Pprojection_result[1],
                Tprojection = Tprojection_result[2],
                Pprojection = Pprojection_result[2])
    else: ## load projections         
        f = np.load(projectionsdir + 'my_projections_%s_%d.npz'%(sim,iSnapshot))
        Tprojection_result = f['Xs'],f['massColumn'],f['Tprojection']
        Pprojection_result = f['Xs'],f['volColumn'],f['Pprojection']
    
    ## load time series data
    f2 = np.load(l.profiledir + 'SFRs_%s.npz'%sim)
    f3 = np.load(l.profiledir + 'Vrots_%s.npz'%sim)
    #print(f2['times'][0], f3['times'][0])
    timeSeries = f2['times'][:],f2['SFRs'][:],f2['SFRmeans'][:],f2['Mdots'][:], f3['Vrots'], f3['sigmas'], f3['vcs'], f3['tcools'], f3['tffs']
    
    l.temperature_and_pressure_movieFrame(sim,iSnapshot, Tprojection_result, Pprojection_result,timeSeries)


dt = time.time()-t_start
print('Done in %d seconds'%dt)
