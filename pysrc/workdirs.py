import sys, os
homedir = os.getenv("HOME")+'/'

if 'ysz5546' in homedir: #quest
    basedir = homedir+'github_repositories/gizmo_analysis/'
    projectdir = '/projects/b1026/jonathan/KY_sims/'
    simdir = project_workdir = projectdir+'sim_outputs/'
    profiledir = projectdir + 'radialProfiles/'
    pyobjDir = '/projects/b1026/jonathan/analysis_pyobjs/no_yt/'
elif homedir=='/mnt/home/jstern/': #rusty
    basedir = homedir+'gizmo_analysis/'
    projectdir = homedir+'ceph/'
    profiledir = projectdir+'sim_analysis/radial_profiles/'
    tracksdir = projectdir+'sim_analysis/particle_tracks/'
    simdir = project_workdir = homedir+'Drummond_data/'
    pyobjDir = homedir+'ceph/radial_profiles/'
elif 'jovyan' in homedir: # binder
    basedir = homedir+'fire_analysis/'
    projectdir = basedir
    simdir = project_workdir = homedir+'/data/'
    profiledir = projectdir + 'radialProfiles/'
elif homedir=='/Users/jonathanstern/':
    basedir = homedir+'Dropbox/github_repositories/gizmo_analysis/'
    projectdir = basedir+'ipynb/subsonic_solution/pyobjs/'
    profiledir = projectdir
    tracksdir = projectdir
    simdir = '' 
    #pyobjDir = homedir+'ceph/radial_profiles/'
    

tables_dir=basedir+'CoolingTables/'
figDir = figdir = projectdir+'figures/'
moviedir = projectdir+'figures/movieFrames/'
projectionsdir = projectdir+'projections/'
pyobjDir2 = basedir+'pyobjs/'
Mstardir = basedir+'data/Mstars/'

if 'tg839127' in homedir:
    pyobjDir = '/work/04613/tg839127/simulation_data/FIRE/no_yt/'
elif not( 'ysz5546' in homedir):
    pyobjDir = basedir+'pyobj/no_yt_analysis/'


