# general python libraries
import time, importlib, sys
import pylab as pl, numpy as np, glob, pdb, scipy, scipy.stats
from numpy import log10 as log
from astropy.cosmology import Planck15 as cosmo
from astropy import units as un, constants as cons

# FIRE studio libraries
import abg_python
import abg_python.snapshot_utils
import firestudio
from firestudio.studios.gas_studio import GasStudio
from firestudio.studios.star_studio import StarStudio

import FIRE_files as ff
import simulationFiles

###parameters to set
#simname (e.g. m12i)
#resolution (e.g. 7100)
#simgroup (e.g. md)
#iSnapshot (e.g. 102)


###load snapshot
simFilesDic = simulationFiles.simulationFiles_dic[(simname,resolution,simgroup)]
snapdir_name = ('','snapdir')[simFilesDic['snapshot_in_directory']] 
meta = ff.Snapshot_meta(simname,simgroup,resolution,iSnapshot=iSnapshot,**simulationFiles.simulationFiles_dic[(simname,resolution,simgroup)])        
snapshot = ff.Snapshot(meta,pr=False,loadAll=False)
snapdict = abg_python.snapshot_utils.openSnapshot(            
    simulationFiles.simulationFiles_dic[(simname,resolution,simgroup)]['snapshotDir'],
    snapnum = iSnapshot,ptype=0,cosmological=1,snapdir_name=snapdir_name,
    header_only=False,keys_to_extract=['SmoothingLength'])

#creat input dictionary for FIRE studio
studiodic = {}
studiodic['Coordinates'] =  snapshot.coords()
studiodic['Masses'] =  snapshot.HImasses() 
studiodic['BoxSize'] = snapdict['BoxSize']
studiodic['SmoothingLength'] = snapdict['SmoothingLength']

# create NHI projection 
mystudio = GasStudio(
 snapdir = None, 
 snapnum = iSnapshot,
 snapdict = studiodic,
 datadir = filedir,
 frame_half_width = r_max,
 frame_depth = z_width,
 quantity_name = 'Masses',
 take_log_of_quantity = False, 
 galaxy_extractor = False,
 pixels=1200,
 single_image='Density',
 overwrite = True,
 savefig=False,      
 use_hsml=True,
 intermediate_file_name=projection_output_filename,
 )
NHImap, _ = mystudio.projectImage([])
NHImap += log( (0.7*un.Msun*un.pc**-2/cons.m_p/1e10*snapshot.sim.h)).to('cm**-2').value ) #fix units
Xs = np.linspace(-r_max,r_max,NHImap.shape[0])
Ys = np.linspace(-r_max,r_max,NHImap.shape[1])
        

