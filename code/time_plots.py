import os
import sys
import pickle
import numpy  as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from pprint     import pprint as pp
from sklearn.externals import joblib as jl
from itertools  import product,combinations


# Run main function
fn = "turn16.pickle"
folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
folder  = folder1 if os.path.exists(folder1) else folder2
d_load  = jl.load(folder + fn)

print(d_load['a_cost'])

################
### FUNCTION ###
################

def get_a_obj_from_a_a_obj(a_a_obj, ind):
    ret = [a_obj[ind] for a_obj in a_a_obj]
    return ret

def plot_line_from_pd(pd_data, ax=None):
    plt.plot


##############
### SCRIPT ###
##############

########
# DATA #
########

# Data from d_load
##########################################################

gamma      		= d_load['gamma'][0]
endowment  		= d_load['endowment'][0]
num_sellers     	= d_load['num_sellers'][0]
num_buyers 		= d_load['num_buyers'][0]
a_buyer_pos             = d_load['a_buyer_loc'][0]
a_seller_pos            = d_load['a_seller_loc'][0]
m_tax                   = d_load['m_tax'][0]
a_m_quantity_bought     = d_load['m_quantity_bought']
pd_quantity             = pd.DataFrame(d_load['a_quantity_nash'])
pd_quantity_sold        = pd.DataFrame(d_load['a_quantity_sold'])
pd_price         	= pd.DataFrame(d_load['a_price_nash'])
pd_cost       		= pd.DataFrame(d_load['a_cost'])
pd_profit 		= pd.DataFrame(d_load['a_profit'])

# Data from calculations
##########################################################
pd_num_buyers_per_seller = pd.DataFrame([np.count_nonzero(m_quant, axis=1) for
    m_quant in a_m_quantity_bought])
pd_quantity_unsold      = pd_quantity - pd_quantity_sold

num_timesteps           = len(d_load['a_cost']) - 1
#check for if the stuff ended early
if np.isnan(d_load['a_profit'][num_timesteps][0]):
    num_timesteps = num_timesteps - 1


########
# PLOT #
########

sb.set_style("darkgrid")

# set figure size
##########################################################
nrow 			= 3
ncol 			= 2
width_scale		= 12
height_scale    	= 4
figsize 		= (width_scale*ncol,height_scale*nrow)

# create the plot window
##########################################################
fig = plt.figure(figsize=figsize)
fig.subplots_adjust(hspace=.2, wspace=0.2)
plt.rc('text', usetex=True)

# set all axes instances
##########################################################
ax_circle           = plt.subplot2grid((nrow,ncol), (0,0))
ax_profit           = plt.subplot2grid((nrow,ncol), (0,1))
ax_cost_and_price   = plt.subplot2grid((nrow,ncol), (1,0))
ax_quantity_sold    = plt.subplot2grid((nrow,ncol), (1,1), sharex=ax_profit)
ax_quantity_unsold  = plt.subplot2grid((nrow,ncol), (2,0), sharex=ax_cost_and_price)
ax_num_buyers       = plt.subplot2grid((nrow,ncol), (2,1), sharex=ax_profit)


# create plot title 
############################################################
fn    = 'gamma=%s_endowment=%s_sellers=%s_buyers=%s'%(gamma, endowment,
        num_sellers, num_buyers)
title = fn.replace('_',', ')
fig.suptitle(title, fontsize=14) 

# plot the profit for each seller 
############################################################
ax = ax_profit
pd_data = pd_profit
ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])
ax.set_xlabel('timestep')
ax.set_ylabel('profit')
ax.set_xlim(0, num_timesteps)

# plot the cost for each seller 
############################################################
ax = ax_quantity_unsold
pd_data = pd_quantity_unsold
print(pd_cost)
ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])
ax.set_xlabel('timestep')
ax.set_ylabel('cost')
ax.set_xlim(0, num_timesteps)

# plot the price for each seller 
############################################################
ax = ax_quantity_sold
pd_data = pd_quantity_sold
ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])
ax.set_xlabel('timestep')
ax.set_ylabel('price')
ax.set_xlim(0, num_timesteps)

# plot the sold and unsold quantity
############################################################
ax = ax_cost_and_price
ax.set_xlim(0, num_timesteps)

# for the left y axis of axis Y:
pd_data = pd_price
color = 'Red'

ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])
ax.spines['left'].set_color(color)
ax.tick_params(axis='y', color=color)
[i.set_color(color) for i in ax.get_yticklabels()]
ax.yaxis.set_label_position("left")
ax.set_xlabel('timestep')
ax.set_ylabel('quantity sold', color=color)

# for the 'right' axis it is Similar
axt   = ax.twinx()
pd_data = pd_cost
color = 'blue'

axt.plot(range(len(pd_data)), pd_data[0], pd_data[1], color=color)
axt.set_xlim(0, num_timesteps)
axt.spines['right'].set_color(color)
axt.tick_params(axis='y', color=color)
[i.set_color(color) for i in axt.get_yticklabels()]
axt.yaxis.set_label_position("right")
axt.set_xlabel('timestep')
axt.set_ylabel('quantity unsold', color=color)

# plot the number of buyers for each seller
############################################################
ax = ax_num_buyers
pd_data = pd_num_buyers_per_seller
ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])
ax.set_xlabel('timestep')
ax.set_ylabel('Number of buyers')
ax.set_xlim(0, num_timesteps)

# Circle of buyers and sellers
############################################################
ax = ax_circle

# plot a grey circle 
r 		= 1 		
angles = np.linspace(0, 2*np.pi, 500)
xs     = r * np.sin(angles)  									
ys     = r * np.cos(angles) 
ax.plot(xs, ys, color='grey', linestyle='--', zorder=1)
colors = sb.color_palette("deep",len(a_seller_pos))

for i, s in enumerate(a_seller_pos):  						
    xcoord = [ r * np.sin( 2*np.pi*s ) ]
    ycoord = [ r * np.cos( 2*np.pi*s ) ]
    ax.scatter(xcoord, ycoord, marker='o', color=colors[i], s=200, zorder=3,
            label='buyer %d'%(i+1))

# for each buyer, index of his seller
seller_ind    = np.argmin(m_tax, axis=0)
for b,s in zip( a_buyer_pos, seller_ind ):
    xcoord = [ r * np.sin( 2*np.pi*b ) ]
    ycoord = [ r * np.cos( 2*np.pi*b ) ]
    ax.scatter(xcoord, ycoord, marker='s', color=colors[s], s=80, zorder=2)
    
ax.set_xlim(-1.2*r, 1.2*r)
ax.set_ylim(-1.2*r, 1.5*r)
ax.legend(loc='best', ncol=6, frameon=False, fontsize=8)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# write figure to the output 
############################################################
plt.show()
# write figure to the output 
############################################################
out_folder = './output/plots/'
if not os.path.exists(out_folder): os.makedirs(out_folder)
plt.savefig(out_folder + fn + '.pdf', bbox_inches='tight')


