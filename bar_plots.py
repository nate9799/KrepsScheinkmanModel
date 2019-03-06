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

sb.set_style("darkgrid")

# import data
########################################################################################################################
file_path 		= "/home/nate/Desktop/"
file_name 		= file_path + "here.pickle"
d_load 			= jl.load(file_name)

cost       		= d_load['cost']
gamma      		= d_load['gamma']
endowment  		= d_load['endowment']
num_sellers 	= d_load['num_sellers']
num_buyers 		= d_load['num_buyers']
a_price      	= d_load['a_price_nash']
a_quantity      = d_load['a_quantity_nash']
a_quantity_sold = d_load['a_quantity_sold']
a_profit 		= d_load['a_profit']


# set figure size
####################################################################################################################
nrow 			= 2
ncol 			= 2
width_scale		= 6
height_scale	= 4
figsize 		= (width_scale*ncol,height_scale*nrow)

# create the plot window
####################################################################################################################
plt.clf()
plt.rc('text', usetex=True)
fig = plt.figure(figsize=figsize)
fig.subplots_adjust(hspace=.12, wspace=0.4)

# set all axes instances
####################################################################################################################
circle_axis   	= plt.subplot2grid( (nrow,ncol), (0,0) )
price_axis    	= plt.subplot2grid( (nrow,ncol), (1,0) )
profit_axis		= plt.subplot2grid( (nrow,ncol), (2,0) )
who_buys_axis	= plt.subplot2grid( (nrow,ncol), (3,0) )


# create plot title 
########################################################################################################################
fig = plt.figure(figsize=(14,5))

title = 'Gamma=%s, Cost/unit=%s, endowment=%s, #Sellers=%s, #Buyers=%s'(gamma, cost, endowment, num_sellers, num_buyers)
fig.suptitle(title, fontsize=14) 


# plot the location of buyers and sellers
########################################################################################################################
ax = circle_axis

# plot a grey circle 
r 		= 1 		
angles = np.linspace(0, 2*np.pi, 500)
xs     = r * np.sin(angles)  									
ys     = r * np.cos(angles) 

ax.plot(xs, ys, color='grey', linestyle='--', zorder=1)


seller_pos    =  [ 0.1, 0.3, 0.5, 0.8 ] 					# positions where the sellers are located 
colors        = sb.color_palette("deep",len(seller_pos))

for i, s in enumerate(seller_pos):  						
    
    xcoord = [ r * np.sin( 2*np.pi*s ) ]
    ycoord = [ r * np.cos( 2*np.pi*s ) ]
    
    ax.scatter(xcoord, ycoord, marker='o', color=colors[i], s=400, zorder=3, label='buyer %s'%i)


buyer_pos     = [ 0.14, 0.21, 0.33, 0.54, 0.93 ] 			# positions where the buers are located
seller_ind    = [ 0,      1,     1,    2,    3 ] 			# for each buyer, index of his seller

for b,s in zip( buyer_pos, seller_ind ):
    
    xcoord = [ r * np.sin( 2*np.pi*b ) ]
    ycoord = [ r * np.cos( 2*np.pi*b ) ]
    
    ax.scatter(xcoord, ycoord, marker='s', color=colors[s], s=100, zorder=2)
    
ax.legend(loc='best', ncol=6, frameon=False)


# plot the price for each buyer 
########################################################################################################################
ax  	= price_axis

N       = len(a_price) # number of sellers
x       = np.arange(1,N+1)
colors  = sb.color_palette("deep",3)
w       = 0.2


ax.bar( 	x-w, a_price,          width=w,  color=colors[0],  alpha=1.0,  align='center' )
ax.bar( 	x-w, cost,             width=w,  color='black',    alpha=0.2,  align='center' )

ax.set_xlabel('sellers',                        fontsize=13)
ax.set_ylabel('prices',      color = colors[0], fontsize=13)
ax.autoscale(tight=True)

# plot the volume for each seller, on same x axis as price plot, but on separate y axis 
########################################################################################################################
ax = price_axis.twinx()
ax.grid(False)

ax.bar(	x+w, a_quantity,       width=w,  color=colors[2],  alpha=1.0,  align='center' )
ax.bar(	x+w, a_quantity_sold,  width=w,  color='black',    alpha=0.2,  align='center' )

ax.set_ylabel('quantities', color = colors[2], fontsize=13)


# plot the profit for each seller
########################################################################################################################
ax = profit_axis
ax.grid(False)

ax.bar(	x, 	 a_profit,         width=w,  color=colors[2],  alpha=1.0,  align='center' )

ax.set_xlabel('sellers',                        fontsize=13)
ax3.set_ylabel('quantities', color = colors[2], fontsize=13)
ax.autoscale(tight=True)

plt.show()


