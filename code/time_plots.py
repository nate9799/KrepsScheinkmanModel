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


# load file
############################################################
num_sellers     = 2
num_buyers      = 12
gamma           = 0.0
a_cost          = np.array([80, 120])
endowment       = 200.
randomize       = False
fn              = 'S=%s_B=%s_gamma=%s_a_cost=%s_endow=%s_randomize=%s.pkl'%(num_sellers,
                num_buyers, gamma, a_cost, endowment, randomize)
print(fn)
fn = "turn0.pickle"
# Run main function
print('executing num_sell=%s, num_buy=%s, a_cost=%s, gamma=%s, endowment = %s, randomize=%s'%(num_sellers,
    num_buyers, gamma, a_cost, endowment, randomize))

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

# DATA
############################################################

gamma      		= d_load['gamma']
endowment  		= d_load['endowment']
num_sellers     	= d_load['num_sellers']
num_buyers 		= d_load['num_buyers']
a_quantity              = d_load['a_quantity_nash']
a_quantity_sold         = d_load['a_quantity_sold']
a_buyer_pos             = d_load['a_buyer_loc']
a_seller_pos            = d_load['a_seller_loc']
m_tax                   = d_load['m_tax']
m_quantity_bought       = d_load['m_quantity_bought']
pd_price         	= pd.DataFrame(d_load['a_price_nash'])
pd_cost       		= pd.DataFrame(d_load['a_cost'])
pd_profit 		= pd.DataFrame(d_load['a_profit'])


# positions where the buyers are located


# PLOT
##########################################################

sb.set_style("darkgrid")

# set figure size
##########################################################
nrow 			= 3
ncol 			= 1
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
ax_profit = plt.subplot2grid((nrow,ncol), (0,0))
ax_cost	  = plt.subplot2grid((nrow,ncol), (1,0))
ax_price  = plt.subplot2grid((nrow,ncol), (2,0))


# create plot title 
############################################################
fn    = 'gamma=%s_a_cost=%s_endowment=%s_sellers=%s_buyers=%s'%(gamma, a_cost,
        endowment, num_sellers, num_buyers)
title = fn.replace('_',', ')
fig.suptitle(title, fontsize=14) 

# plot the price for each buyer 
############################################################
ax = ax_profit
pd_data = pd_profit
ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])

# plot the cost for each buyer 
############################################################
ax = ax_cost
pd_data = pd_cost
print(pd_cost)
ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])

# plot the price for each buyer 
############################################################
ax = ax_price
pd_data = pd_price
ax.plot(range(len(pd_data)), pd_data[0], pd_data[1])

# write figure to the output 
############################################################
plt.show()
out_folder = './output/plots/'
if not os.path.exists(out_folder): os.makedirs(out_folder)
plt.savefig(out_folder + fn + '.pdf', bbox_inches='tight')


