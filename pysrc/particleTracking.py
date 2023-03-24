#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, h5py, pdb,time, glob,os , string
from importlib import reload
import pylab as pl, numpy as np
from numpy import log10 as log
from astropy import units as un, constants as cons
import scipy, scipy.stats
from matplotlib import ticker, patches
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('/mnt/home/jstern/gizmo_analysis/pysrc/')



# In[2]:


import workdirs as wd
import FIRE_files as ff
from FIRE_files import cosmo, u
from projectPlotBasics import *
import first_pass as l
cmap = pl.get_cmap('viridis')
figdir = wd.basedir+'figures/'


# In[3]:
## load input parameter
sys.path = [x for x in sys.path if 'python2' not in x]
print(sys.argv)
simname = sys.argv[1] #e.g. 'vc200_Rs0_Mdot4847_Rcirc10_fgas02_res1e4_n10_NoLowCool_tracking'
track_length,dt,rmax,vc,Rcirc,subsample = map(int, sys.argv[2].split('_')) # e.g. '2250_10_40_200_15_100'
lastSnapshot = int(sys.argv[3]) #e.g. '325' -- recieved by automatic script which runs over last snapshots number
Nsnapshots = track_length // dt
if lastSnapshot<Nsnapshots-1:
    print('not enough snapshots',file=sys.stderr)
    exit()
CF_path = '/mnt/home/jstern/cooling_flow/pysrc'
sys.path.append(CF_path)
import cooling_flow as CF, HaloPotential as Halo


# In[4]:


vc = vc*un.km/un.s
Rcirc = Rcirc*un.kpc
simdir = wd.simdir+'/%s/output/'%simname
outputdir = wd.tracksdir+simname
if not os.path.isdir(outputdir): os.mkdir(outputdir) 
npz_fn = outputdir+'/particle_tracks_%s_%s.npz'%(sys.argv[2],sys.argv[3])
if os.path.exists(npz_fn):
    print('file already calculated!',file=sys.stderr)
    exit()

# In[5]:

snapNumbers = range(lastSnapshot-Nsnapshots,lastSnapshot,1)
ts = lastSnapshot*dt-np.arange(len(snapNumbers))*dt


# In[6]:


sim = l.KY_sim(simname,simdir,200*un.kpc,origin=np.zeros(3),Rcirc = Rcirc,snapshot_dt_Myr=dt,pr=False,
              analyticGravity= Halo.PowerLaw(m=0.,vc_Rvir=vc,Rvir=200*un.kpc))
sim.z = 0 #for cooling function
print(sim.galaxyname, sim.Nsnapshots())


# In[7]:

snapNumbersReverse = snapNumbers[::-1]
snapshot0 = sim.getSnapshot(snapNumbersReverse[0])
snapshot_last = sim.getSnapshot(snapNumbersReverse[-1])

q_last = snapshot0.dic[('PartType0','ParticleIDs')] 
q_first = snapshot_last.dic[('PartType0','ParticleIDs')] 
p_last = snapshot0.dic[('PartType4','ParticleIDs')] 
tmp = np.concatenate([q_last,p_last]) 
_,inds1,inds2 = np.intersect1d(tmp,q_first,assume_unique=True,return_indices=True)
accreted_inds = ((snapshot_last.rs()<(rmax+1)) & (snapshot_last.rs()>rmax))[inds2]
accreted_IDs = tmp[inds1][accreted_inds][::subsample]

N_particles = len(accreted_IDs.nonzero()[0])
print("number of tracked particles: %d"%N_particles)

coords = np.zeros((len(snapNumbers),N_particles,3))
vs = np.zeros((len(snapNumbers),N_particles,3))
Ts = np.zeros((len(snapNumbers),N_particles))
nHs = np.zeros((len(snapNumbers),N_particles))
tcools = np.zeros((len(snapNumbers),N_particles))

for iq in range(len(snapNumbersReverse)):
    print(iq)
    snapshot = sim.getSnapshot(snapNumbersReverse[iq])
    q = snapshot.dic[('PartType0','ParticleIDs')] 
    p = snapshot.dic[('PartType4','ParticleIDs')] 
    tmp = np.concatenate([q,p])
    _,_,indices = np.intersect1d(accreted_IDs,tmp,assume_unique=True,return_indices=True)
    coords[iq,:,:] = np.concatenate([snapshot.coords(0),snapshot.coords(4)])[indices,:]
    vs[iq,:,:]     = np.concatenate([snapshot.vs(0),snapshot.vs(4)])[indices,:]
    if len(p)==0:
        Ts[iq,:] = snapshot.Ts()[indices]
        nHs[iq,:] = snapshot.nHs()[indices]
        tcools[iq,:] = snapshot.t_cool()[indices]
    else:
        l = len(p)
        Ts[iq,:] = np.concatenate([snapshot.Ts(), np.nan*np.ones(l)])[indices]
        nHs[iq,:] = np.concatenate([snapshot.nHs(), np.nan*np.ones(l)])[indices]
        tcools[iq,:] = np.concatenate([snapshot.t_cool(), np.nan*np.ones(l)])[indices]
    sim.delSnapshot(snapNumbersReverse[iq])
        

np.savez(npz_fn,coords=coords,vs=vs,Ts=Ts,nHs=nHs,tcools=tcools)
print("saved %s"%npz_fn)

