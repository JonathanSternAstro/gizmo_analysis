import pylab as pl, numpy as np, scipy, scipy.stats
import FIRE_files as ff
from FIRE_files import cosmo, u
from projectPlotBasics import *
import first_pass as l
from numpy import log10 as log, log as ln

def parsename(sim):
    vc, Rsonic, Mdot, Rcirc = str(sim).split('_')[:4]
    return [int(x) for x in (vc[2:],Rsonic[2:],Mdot[4:],Rcirc[5:])]

def ratio_comparison_scatter_plot(sims,x1,x2,y1,y2):
    pl.figure(figsize=(8,8))
    ax=pl.subplot(111)
    for sim in sims:
        vc, Rsonic, Mdot, Rcirc = parsename(sim)
        f = np.load(l.profiledir + 'timeSeries_%s.npz'%sim)
        inds = f['times']>1
        xs, ys = f[x1]/f[x2], f[y1]/f[y2]
        if y2=='Mdots': ys *=-1
        pl.plot( xs[inds], ys[inds],ls='none',label=sim,marker='.')
    pl.loglog()
    pl.axhline(1.,c='k',lw=0.7)
    pl.axvline(1.,c='k',lw=0.7)
    if y1=='Ts'     and y2=='Tcs':  pl.ylabel(r'$T/T_{\rm g}$ at 30 kpc')
    if x1=='tcools' and x2=='tffs': pl.xlabel(r'$t_{\rm cool}^{(s)}/t_{\rm ff}$ at 30 kpc')
    ax.xaxis.set_major_formatter(u.arilogformatter)
    ax.yaxis.set_major_formatter(u.arilogformatter)
#     pl.legend(fontsize=7,ncol=1,loc='lower right')    
def compare_to_initial_Mdot(sims,x1,x2):
    pl.figure(figsize=(8,8))
    ax=pl.subplot(111)
    for sim in sims:
        vc, Rsonic, Mdot, Rcirc = parsename(sim)
        f = np.load(l.profiledir + 'timeSeries_%s.npz'%sim)
        inds = f['times']>1
        xs, ys = f[x1]/f[x2], -f['Mdots']/Mdot*1000
#         print(ys[::10])
        pl.plot( xs[inds], ys[inds],ls='none',label=sim,marker='.')
    ax.set_xscale('log')
    ax.set_yscale('symlog',linthreshy=0.1,linscaley=0.1)
    pl.axhline(1.,c='k',lw=0.7)
    pl.axvline(1.,c='k',lw=0.7)
    pl.ylabel(r'$\dot{M}/\dot{M}_0$ at 30 kpc')
    if x1=='tcools' and x2=='tffs': pl.xlabel(r'$t_{\rm cool}^{(s)}/t_{\rm ff}$ at 30 kpc')
    ax.xaxis.set_major_formatter(u.arilogformatter)
    ax.yaxis.set_major_formatter(u.arilogformatter)
    pl.ylim(-10,10)
#     pl.legend(fontsize=7,ncol=1,loc='lower right')    

def timeSeries_additions(sim,window = 0.3,minSFR=0.02):
    f = np.load(l.profiledir + 'timeSeries_%s.npz'%sim)
    timeSeriesDic = dict([(k,f[k]) for k in f.files])
    SFRs_means  = np.zeros(f['times'].shape[0])
    logSFRs_std = np.zeros(f['times'].shape[0]) 
    timeSeriesDic['ones'] = np.ones(f['times'].shape[0]) 
    SFRs = f['SFRs']*(f['SFRs']>minSFR) + minSFR*(f['SFRs']<minSFR)
    for it,t in enumerate(f['times']):
        inds = (f['times']<t+window/2.) & (f['times'] > t-window/2.)
        SFRs_means[it]   = SFRs[inds].mean()
        logSFRs_std[it] = log(SFRs[inds]).std()    
    timeSeriesDic['SFRs_means'],timeSeriesDic['logSFRs_std'] = SFRs_means, logSFRs_std
    timeSeriesDic['P2ks'] = timeSeriesDic['nHs']*timeSeriesDic['Ts']
    timeSeriesDic['P2ks_expected'] = timeSeriesDic['nHsB']*timeSeriesDic['Tcs']
    return timeSeriesDic
    

def ratio_comparison_group_plot(sims,timeSeriesDic,x1,x2,y1,y2,groupfunc,groupdic,ax=None,isLog=True,bins=10.**np.arange(-1.5,1.5,.1)):
    """
    groupfunc converts simname to group identifier
    groupdic gives plotting kwargs for each group identifier
    """
    if ax==None:
        pl.figure(figsize=(4,4))
        ax=pl.subplot(111)
    tups = {}
    for sim in sims:
        vc, Rsonic, Mdot, Rcirc = parsename(sim)
        group = groupfunc(sim)
        f = timeSeriesDic[sim]
        inds = f['times']>1
        if group not in tups: tups[group] = [np.array([]),np.array([])]
        tups[group][0] = np.concatenate([tups[group][0], f[x1][inds]/f[x2][inds]])
        tups[group][1] = np.concatenate([tups[group][1], f[y1][inds]/f[y2][inds]])        
    for group in tups:       
        pl.plot( tups[group][0], tups[group][1],ls='none',marker=',',alpha=0.25,c='.3')
        m,x,_ = scipy.stats.binned_statistic(tups[group][0],tups[group][1],statistic='median', bins=bins)
        c,x,_ = scipy.stats.binned_statistic(tups[group][0],tups[group][1],statistic='count', bins=bins)   
        pl.plot((x[1:]+x[:-1])[c>10]/2,m[c>10],**groupdic[group],lw=3)
    pl.axhline(1.,c='k',lw=0.3)
    pl.axvline(1.,c='k',lw=0.3)
    if y1=='Ts'     and y2=='Tcs':  pl.ylabel(r'$T/T_{\rm g}$ at 30 kpc')
    if y1=='sigmas' and y2=='vcs':  pl.ylabel(r'$\sigma_{\rm g} / v_{\rm c}$')
    if x1=='tcools' and x2=='tffs': pl.xlabel(r'$t_{\rm cool}^{(s)}/t_{\rm ff}$ at 30 kpc')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(u.arilogformatter)
    if isLog:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(u.arilogformatter)    
    if y1=='logSFRs_std':
        pl.ylim(0,0.75)
        pl.legend(fontsize=7,ncol=1,loc='upper left')    
        pl.ylabel(r'$\sigma(\log {\rm SFR})$ dex')
    elif y1=='nHs':
        pl.ylim(1e-5,0.003)
        pl.ylabel(r'hydrogen density at 30 kpc')
        pl.legend(fontsize=7,ncol=1,loc='lower left')    
    elif y1[:4]=='P2ks':
        pl.ylim(1,3e3)
        pl.ylabel(r'%s pressure at 30 kpc [K cm$^{-3}$]'%y1[4:])
        pl.legend(fontsize=7,ncol=1,loc='lower left')    
    else: 
        pl.ylim(0.05,3)
        pl.legend(fontsize=7,ncol=1,loc='lower right')    
    if x1=='P2ks':
        pl.xlim(1,3e3)
        pl.xlabel(r'pressure at 30 kpc [K cm$^{-3}$]')
    else:
        pl.xlim(0.1,50)      
    