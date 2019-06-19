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
from plot_util import customaxis


fn = "turn_gamma=0.6.pickle"
# Run main function
folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
folder  = folder1 if os.path.exists(folder1) else folder2
d_load  = jl.load(folder + fn)
bug_check = True


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
print(pd_cost)

# Data from calculations
##########################################################
pd_num_buyers_per_seller = pd.DataFrame([np.count_nonzero(m_quant, axis=1) for
    m_quant in a_m_quantity_bought])
pd_quantity_unsold      = pd_quantity - pd_quantity_sold
# Cournot price at Nash quantity (A + n\bar c)/(n+1)
pd_cournot              = (endowment + pd_cost.sum(1))/(num_sellers+1)

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
nrow 			= 2
if bug_check: nrow += 1
ncol 			= 2
width_scale		= 12
height_scale    	= 4
figsize 		= (width_scale*ncol,height_scale*nrow)
a_marker = ['s', 'x', '*', 'o', 'D']
fontsize = 14

# create the plot window
##########################################################
fig = plt.figure(figsize=figsize)
fig.subplots_adjust(hspace=.3, wspace=0.3)
plt.rc('text', usetex=True)

# set all axes instances
##########################################################
ax_circle           = plt.subplot2grid((nrow,ncol), (0,0), rowspan=2)
ax_profit           = plt.subplot2grid((nrow,ncol), (0,1))
ax_cost_and_price   = plt.subplot2grid((nrow,ncol), (1,1), sharex=ax_profit)
if bug_check:
    ax_quantity_unsold  = plt.subplot2grid((nrow,ncol), (2,0), sharex=ax_cost_and_price)
    ax_num_buyers       = plt.subplot2grid((nrow,ncol), (2,1), sharex=ax_profit)

#annotate
a_ax = [ax_circle, ax_profit, ax_cost_and_price]
a_annote = ['(a)','(b)','(c)','(d)']
for (ax, annote) in zip(a_ax, a_annote[0:len(a_ax)]):
    ax.annotate(annote,
            xy          =  (-0.12, 0.96),
            xycoords    = 'axes fraction',
            fontsize    =  12,
            ha          = 'left',
            va          = 'top' )

# plot the profit and quantity sold for each seller 
############################################################
a_color = sb.color_palette("deep", num_sellers)

# Profit
ax = ax_profit
color = a_color[0]
pd_data = pd_profit
for i in range(num_sellers):
    ax.plot(range(len(pd_data)), pd_data[i], '-', marker=a_marker[i],
            color=color, label = 'Firm %d'%(i+1))
ax.set_xlim(0, num_timesteps)
customaxis(ax = ax, position = 'left', color = color, label = 'Profit',
        scale = 'linear', size = fontsize, full_nrs = False, location = 0.0)
ax.legend()
leg = ax.get_legend()
(leg.legendHandles[i].set_color('black') for i in range(num_sellers))

# Quantity
axt = ax.twinx()
color = a_color[1]
pd_data = pd_quantity_sold
for i in range(num_sellers):
    axt.plot(range(len(pd_data)), pd_data[i], '-', marker=a_marker[i],
            color=color)
axt.grid('False')
customaxis(ax = axt, position = 'right', color = color, label = 'Quantity',
        scale = 'linear', size = fontsize, full_nrs = False, location = 1.0)

if bug_check:

# plot the unsold quantity for each seller 
############################################################
    ax = ax_quantity_unsold
    pd_data = pd_quantity_unsold
    print(pd_cost)
    ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])
    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('Unsold Capacity', fontsize=14)
    ax.set_xlim(0, num_timesteps)

# plot the number of buyers for each seller
############################################################
    ax = ax_num_buyers
    pd_data = pd_num_buyers_per_seller
    ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])
    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('Number of Buyers', fontsize=14)
    ax.set_xlim(0, num_timesteps)

# plot the cost and price
############################################################
ax = ax_cost_and_price
ax.set_xlim(0, num_timesteps)

# for the left y axis of axis Y:
pd_data = pd_price
color = 'Red'

for i in range(num_sellers):
    ax.plot(range(len(pd_data)), pd_data[i], color=color, marker = a_marker[i],
            label='Price of Firm %d'%(i+1))
if gamma == 0:
    ax.plot(range(len(pd_data)), pd_cournot, color=color, linestyle='dashed',
            label='Theoretical Cournot Price')
ax.spines['left'].set_color(color)
ax.tick_params(axis='y', color=color)
[i.set_color(color) for i in ax.get_yticklabels()]
ax.yaxis.set_label_position("left")
ax.set_ylabel('Price', color=color, fontsize=14)
ax.set_xlim(0, num_timesteps)
ax.legend()

# for the 'right' axis it is similar
axt   = ax.twinx()
pd_data = pd_cost
color = 'blue'

axt.plot(range(len(pd_data)), pd_data, color=color)
axt.set_xlim(0, num_timesteps)
axt.spines['right'].set_color(color)
axt.tick_params(axis='y', color=color)
[i.set_color(color) for i in axt.get_yticklabels()]
axt.yaxis.set_label_position("right")
axt.set_ylabel('Cost', color=color, fontsize=14)
axt.set_ylim(20, 100)
axt.grid(False)

# Circle of buyers and sellers
############################################################
ax = ax_circle

# plot a grey circle 
r       = 1
angles  = np.linspace(0, 2*np.pi, 500)
xs      = r * np.sin(angles)  									
ys      = r * np.cos(angles) 
ax.plot(xs, ys, color='grey', linestyle='--', zorder=1)
colors  = sb.color_palette("deep",len(a_seller_pos))

for i, s in enumerate(a_seller_pos):  						
    xcoord = [ r * np.sin( 2*np.pi*s ) ]
    ycoord = [ r * np.cos( 2*np.pi*s ) ]
    ax.scatter(xcoord, ycoord, marker='o', color=colors[i], s=400, zorder=3,
            label='Firm %d'%(i+1))

# for each buyer, find index of his seller
seller_ind    = np.argmax(a_m_quantity_bought[0], axis=0)
for s in range(len(a_seller_pos)):
    a_pos_subset = a_buyer_pos[seller_ind == s]
    a_xcoord = [r * np.sin(2 * np.pi * a_pos_subset)]
    a_ycoord = [r * np.cos(2 * np.pi * a_pos_subset)]
    ax.scatter(a_xcoord, a_ycoord, marker='x', color=colors[s], s=80, zorder=2,
            label='Buyer who bought from Firm %d'%(s+1)) 
ax.set_xlim(-1.2*r, 1.2*r)
ax.set_ylim(-1.2*r, 1.5*r)
ax.legend(loc='center', ncol=2, frameon=False, fontsize=8, labelspacing=2)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.axis('equal')

# write figure to the output 
############################################################
fn    = 'gamma=%s_endowment=%s_sellers=%s_buyers=%s'%(gamma, endowment,
        num_sellers, num_buyers)
out_folder = './output/plots/'
if not os.path.exists(out_folder): os.makedirs(out_folder)
plt.savefig(out_folder + fn + '.pdf', bbox_inches='tight')

# write figure to the screen (must go after file creation or doesn't write
# correctly) 
############################################################
plt.show()
