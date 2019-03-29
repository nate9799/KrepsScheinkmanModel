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


def circle_dist(a, b):
    '''
    Finds distance if 1==0.
    '''
    return .5-abs(abs(a-b)-.5)

# load file
############################################################
num_sellers     = 2
num_buyers      = 12
gamma           = 0.5
cost            = 100.
endowment       = 200.
randomize       = True
fn              = 'S=%s_B=%s_gamma=%s_cost=%s_endow=%s_randomize=%s.pkl'%(num_sellers,
                num_buyers, gamma, cost, endowment, randomize)
# Run main function
print('executing num_sell=%s, num_buy=%s, cost=%s, gamma=%s, endowment = %s, randomize=%s'%(num_sellers,
    num_buyers, gamma, cost, endowment, randomize))


folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
folder  = folder1 if os.path.exists(folder1) else folder2
d_load  = jl.load(folder + fn)

##############
### SCRIPT ###
##############

# import data
############################################################

cost       		= d_load['cost']
gamma      		= d_load['gamma']
endowment  		= d_load['endowment']
num_sellers     	= d_load['num_sellers']
num_buyers 		= d_load['num_buyers']
a_price         	= d_load['a_price_nash']
a_quantity              = d_load['a_quantity_nash']
a_quantity_sold         = d_load['a_quantity_sold']
a_profit 		= d_load['a_profit']
a_buyer_pos             = d_load['a_buyer_loc']
a_seller_pos            = d_load['a_seller_loc']
m_tax                   = d_load['m_tax']
m_quantity_bought       = d_load['m_quantity_bought']

# positions where the buers are located

sb.set_style("darkgrid")

# set figure size
##########################################################
nrow 			= 2
ncol 			= 2
width_scale		= 6
height_scale    	= 4
figsize 		= (width_scale*ncol,height_scale*nrow)

# create the plot window
##########################################################
fig = plt.figure(figsize=figsize)
fig.subplots_adjust(hspace=.2, wspace=0.2)
plt.rc('text', usetex=True)


# set all axes instances
##########################################################
circle_axis   	= plt.subplot2grid( (nrow,ncol), (0,0) )
price_axis    	= plt.subplot2grid( (nrow,ncol), (0,1) )
profit_axis	= plt.subplot2grid( (nrow,ncol), (1,0) )
who_buys_axis	= plt.subplot2grid( (nrow,ncol), (1,1) )


# create plot title 
############################################################
fn    = 'gamma=%s_cost=%s_endowment=%s_sellers=%s_buyers=%s'%(gamma, cost,
        endowment, num_sellers, num_buyers)
title = fn.replace('_',', ')
fig.suptitle(title, fontsize=14) 


# plot the location of buyers and sellers
############################################################
ax = circle_axis

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
    
    ax.scatter(xcoord, ycoord, marker='o', color=colors[i], s=200, zorder=3, label='buyer %s'%i)

# for each buyer, index of his seller
seller_ind    = np.argmax(m_quantity_bought, axis=0)

for b,s in zip( a_buyer_pos, seller_ind ):
    xcoord = [ r * np.sin( 2*np.pi*b ) ]
    ycoord = [ r * np.cos( 2*np.pi*b ) ]
    
    ax.scatter(xcoord, ycoord, marker='s', color=colors[s], s=80, zorder=2)
    

ax.set_xlim(-1.2*r, 1.2*r)
ax.set_ylim(-1.2*r, 1.5*r)
ax.legend(loc='best', ncol=6, frameon=False, fontsize=8)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)


# plot the price for each buyer 
############################################################
ax  	= price_axis

N       = len(a_price) # number of sellers
x       = np.arange(1,N+1)
colors  = sb.color_palette("deep",3)
w       = 0.2


ax.bar( 	x-w, a_price,          width=w,  color=colors[0],  alpha=1.0,  align='center' )
ax.bar( 	x-w, cost,             width=w,  color='black',    alpha=0.2,  align='center' )


ax.set_xticks(x)
ax.set_xlabel('sellers',                        fontsize=13)
ax.set_ylabel('prices',      color = colors[0], fontsize=13)
ax.autoscale(tight=True)

# plot the volume for each seller, on same x axis as price plot, but on separate y axis 
############################################################
ax = price_axis.twinx()
ax.grid(False)

ax.bar(x+w, a_quantity,      width=w,  color=colors[1], alpha=1.0, align='center')
ax.bar(x+w, a_quantity_sold, width=w, color='black',    alpha=0.2, align='center')

ax.set_xticks(x)
ax.set_ylabel('quantities', color = colors[1], fontsize=13)


# plot the profit for each seller
############################################################
ax = profit_axis

ax.bar(x, 	 a_profit,         width=w, color=colors[2], alpha=1.0, align='center')

ax.set_xticks(x)
ax.set_xlabel('sellers',                        fontsize=13)
ax.set_ylabel('profit', 	color = colors[2], fontsize=13)
ax.autoscale(tight=True)

# plot nr of sellers for each buyer 
############################################################
ax = who_buys_axis
ax.set_visible(False)


# write figure to the output 
############################################################
plt.show()
out_folder = './output/plots/'
if not os.path.exists(out_folder): os.makedirs(out_folder)
plt.savefig(out_folder + fn + '.pdf', bbox_inches='tight')


