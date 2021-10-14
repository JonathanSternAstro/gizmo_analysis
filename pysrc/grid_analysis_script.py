import sys, h5py, pdb,time, glob
from importlib import reload
import pylab as pl, numpy as np
from numpy import log10 as log
from astropy import units as un, constants as cons
import scipy, scipy.stats
from matplotlib import ticker
sys.path.append('../pysrc')

import FIRE_files as ff
from FIRE_files import cosmo, u
from projectPlotBasics import *
import first_pass as l
cmap = pl.get_cmap('viridis')

simnames = [x.split('/')[-1] for x in glob.glob('../../data/vc*')]

sims = []
for simname in simnames:
    vc = float(simname.split('_')[0][2:])
    PL_potential = l.PowerLawPotential(m=0.,vc_Rvir=vc*un.km/un.s,Rvir=200*un.kpc)
    Rcirc = float(simname.split('_')[3][5:])
    simdir = '../../data/%s/output/'%simname
    sim = l.KY_sim(simname,simdir,
                   dynamicCentering=True,recalc=False,Nsnapshots=None,
                   origin=np.zeros(3),Rcirc = Rcirc*un.kpc,
                   analyticGravity=PL_potential)
    print(sim.galaxyname, sim.Nsnapshots())
    sims.append(sim)
    
    
for sim in sims:
    iSnapshots = np.linspace(1,sim.Nsnapshots()-1,4).astype(int)
    l.CGM_properties(sim, iSnapshots,Rcirc=10)    
    
for sim in sims:
    _ = sim.timeSeries(rMdot = 30*un.kpc,rVrot=5*un.kpc,multipleProcs=10,justLoad=False)
    sim.quantities_at_Rcirc()
    
for sim in sims:
    for edge_on in (True,False):
        for iSnapshot in range(sim.Nsnapshots()):
            l.temperature_and_pressure_movieFrame_async(sim.loadvals,iSnapshot,
            calculateProjections=True,edge_on=edge_on,lazy=True)
