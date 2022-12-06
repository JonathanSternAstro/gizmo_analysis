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
track_length_in_10Myr,dt,rmax = map(int, sys.argv[2].split('_')) # e.g. '225_10_40'
lastSnapshot = int(sys.argv[3]) #e.g. '325' -- recieved by automatic script which runs over last snapshots number
Nsnapshots = track_length_in_10Myr // (dt//10)
if lastSnapshot<Nsnapshots:
    print('not enough snapshots',file=sys.stderr)
    exit()
CF_path = '/mnt/home/jstern/cooling_flow/pysrc'
sys.path.append(CF_path)
import cooling_flow as CF, HaloPotential as Halo


# In[4]:


vc = 200. *un.km/un.s
Rcirc = 10.*un.kpc
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


snapshots = [sim.getSnapshot(i) for i in snapNumbers][::-1]


# In[8]:


qs = [snapshot.dic[('PartType0','ParticleIDs')] for snapshot in snapshots]


# In[9]:


ps = [snapshot.dic[('PartType4','ParticleIDs')] for snapshot in snapshots]



# In[10]:


# select by location in CGM
tmp = np.concatenate([qs[0],ps[0]]) 
_,inds1,inds2 = np.intersect1d(tmp,qs[-1],assume_unique=True,return_indices=True)
accreted_inds = ((snapshots[-1].rs()<(rmax+1)) & (snapshots[-1].rs()>rmax))[inds2]
accreted_IDs = tmp[inds1][accreted_inds]


# In[11]:


print("number of accreted particles: %d"%len(accreted_IDs.nonzero()[0]))
accreted_inds_dic = [None] * len(qs)
for iq,q in enumerate(qs):
    tmp = np.concatenate([q,ps[iq]])
    _,_,indices = np.intersect1d(accreted_IDs,tmp,assume_unique=True,return_indices=True)
    accreted_inds_dic[iq] = indices


# In[12]:


concat  = lambda snap,attr: np.concatenate([getattr(snap,attr)(a) for a in (0,4)])
def concat2(snap,attr,l): 
    vals = getattr(snap,attr)()
    if l == 0: return vals
    return np.concatenate([vals, np.nan*np.ones(l)])


# In[25]:


coords = np.array([concat(snapshots[i],'coords')[accreted_inds_dic[i],:]       for i in range(len(snapshots))])
print("finished coords")
vs     = np.array([concat(snapshots[i],'vs')[accreted_inds_dic[i],:]           for i in range(len(snapshots))])
print("finished vs")
Ts     = np.array([concat2(snapshots[i],'Ts',len(ps[i]))[accreted_inds_dic[i]]  for i in range(len(snapshots))])
print("finished Ts")
nHs    = np.array([concat2(snapshots[i],'nHs',len(ps[i]))[accreted_inds_dic[i]] for i in range(len(snapshots))])
print("finished nHs")
tcools = np.array([concat2(snapshots[i],'t_cool',len(ps[i]))[accreted_inds_dic[i]]       for i in range(len(snapshots))])
print("finished tcools")


# In[20]:


np.savez(npz_fn,coords=coords,vs=vs,Ts=Ts,nHs=nHs,tcools=tcools)
print("saved %s"%npz_fn)

