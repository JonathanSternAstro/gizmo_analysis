import sys, os
homedir = os.getenv("HOME")+'/'
if homedir=='/home/jonathan/': homedir+='Dropbox/'
sys.path.append(homedir+'jonathanmain/CGM/pysrc')
sys.path.append(homedir+'other_repositories')
sys.path.append(homedir+'other_repositories/FIRE_studio')
sys.path.append('../pysrc')


CF_path = homedir+'jonathanmain/CGM/rapidCoolingCGM/published_pysrc/'
sys.path.append(CF_path)
import cooling_flow as CF, HaloPotential as Halo

projectdir = homedir+'jonathanmain/CGM/KY_sims/'
figdir = projectdir+'figures/'
project_workdir = '/projects/b1026/jonathan/KY_sims/'
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
from importlib import reload

matplotlib.use('Agg')
pl.ioff()


import first_pass as l
cmap = pl.get_cmap('viridis')


print(sys.argv)
simname = sys.argv[1] 
vc = float(sys.argv[2])

simdir = '/projects/b1026/jonathan/my_gizmo/%s/output/'%simname
sim = l.KY_sim(simname,simdir,200*un.kpc,origin=np.zeros(3),Rcirc = 10*un.kpc,snapshot_dt_Myr=10,
              analyticGravity= Halo.PowerLaw(m=0.,vc_Rvir=vc*un.km/un.s,Rvir=200*un.kpc))

print(sim.galaxyname, sim.Nsnapshots())


iSnapshots = np.linspace(1,sim.Nsnapshots()-1,4).astype(int)
#iSnapshots = np.linspace(1,49,4).astype(int)

l.CGM_properties(sim, iSnapshots,Rcirc=10)

pl.savefig(figdir+'radial_profiles_%s.pdf'%str(sim))

#exit()

_ = sim.timeSeries(rMdot = 30*un.kpc,rVrot=5*un.kpc,multipleProcs=10,justLoad=False)

sim.quantities_at_Rcirc()

for edge_on in (True,False):
    sim.movie(l.temperature_and_pressure_movieFrame_async,
              multipleProcs=12,
              calculateProjections=True,start=None,
              edge_on=edge_on,
              lazy=True)

