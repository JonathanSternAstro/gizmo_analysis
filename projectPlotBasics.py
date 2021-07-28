import matplotlib
import pylab as pl
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
#matplotlib.use('agg')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('font', family='serif', size=10)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
fig_width_full_pt = 513.11743  # Get this from LaTeX using \showthe\columnwidth
fig_width_half_pt = 245.26653
inches_per_pt = 1.0/72.27               # Convert pt to inch
fig_width_full = fig_width_full_pt*inches_per_pt  # width in inches
fig_width_half = fig_width_half_pt*inches_per_pt 
golden_mean = (5**0.5-1.0)/2.0         # Aesthetic ratio
fig_size_full =  [fig_width_full,fig_width_full*golden_mean]
fig_size_half =  [fig_width_half,fig_width_half*golden_mean]

cmap=pl.get_cmap('viridis')
cmap2=pl.get_cmap('plasma')
infigfontsize = 8
other_args = dict(c='k',ls='--',lw=0.5)
niceblue = (0, 0.45, 0.74)

layout_2x2 = dict(hspace=0.2,wspace=0.3,top=0.95,bottom=0.075,left=0.1,right=0.98)
size_2x2 = fig_width_full,fig_width_full*golden_mean*0.7
layout_3x1 = dict(hspace=0.2,wspace=0.45,top=0.95,bottom=0.075,left=0.1,right=0.98)
size_3x1 = fig_width_full,fig_width_full*golden_mean*0.4
layout_3x2 = dict(hspace=0.2,wspace=0.4,top=0.95,bottom=0.075,left=0.1,right=0.98)
size_3x2 = fig_width_full,fig_width_full*golden_mean*0.7
layout_3x3 = dict(hspace=0.2,wspace=0.45,top=0.95,bottom=0.075,left=0.1,right=0.98)
size_3x3 = fig_width_full,fig_width_full*golden_mean

slantlineprops = slantlinepropsgray = {'width':0.1,'headwidth':1.5,'headlength':2,'color':'0.5'}
slantlinepropsblue  = {'width':0.3,'headwidth':1.5,'headlength':2,'color':'b'}
slantlinepropsgreen  = {'width':0.3,'headwidth':1.5,'headlength':2,'color':'g'}
slantlinepropsmagenta  = {'width':0.3,'headwidth':1.5,'headlength':2,'color':'m'}
slantlinepropsred   = {'width':0.3,'headwidth':1.5,'headlength':2,'color':'r'}
slantlinepropsblack = {'width':0.1,'headwidth':1.5,'headlength':2,'color':'k'}
slantlinepropswhite = {'width':0.3,'headwidth':1.5,'headlength':2,'color':'w'}

