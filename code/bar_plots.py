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


################
### FUNCTION ###
################

def plot_1timestep_from_data(a_cost, gamma, endowment, num_sellers, num_buyers,
        a_price, a_quantity, a_quantity_sold, a_profit, a_buyer_pos,
        a_seller_pos, m_tax, m_quantity_bought):

# set figure size
##########################################################
    nrow            = 1
    ncol            = 2
    width_scale     = 9
    height_scale    = 6
    figsize         = (width_scale*ncol,height_scale*nrow)

# create the plot window
##########################################################
    sb.set_style("darkgrid")
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=.2, wspace=0.2)
    plt.rc('text', usetex=True)

# set all axes instances
##########################################################
    circle_axis   	= plt.subplot2grid( (nrow,ncol), (0,0) )
    price_axis    	= plt.subplot2grid( (nrow,ncol), (0,1) )

#annotate
    a_ax = [circle_axis, price_axis]
    a_annote = ['(a)','(b)','(c)','(d)']
    for (ax, annote) in zip(a_ax, a_annote[0:len(a_ax)]):
        ax.annotate(annote,
                xy          =  (-0.12, 0.96),
                xycoords    = 'axes fraction',
                fontsize    =  12,
                ha          = 'left',
                va          = 'top' )

# Circle of buyers and sellers
############################################################
    ax = circle_axis

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
    seller_ind    = np.argmax(m_quantity_bought, axis=0)
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

# plot the price for each buyer 
############################################################
    a_xvalues = np.arange(1, num_sellers + 1)
    w = 0.2
    fontsize = 13
    colors = sb.color_palette("deep",3)

    ax = price_axis
    color = colors[0]

    ax.bar(a_xvalues-w, a_price, width=.8*w, color=color, alpha=1.0, align='center')
    ax.bar(a_xvalues-w, a_cost,  width=.8*w, color='black',   alpha=0.2, align='center')
    ax.set_xlabel('')
    ax.set_xticks(a_xvalues)
    ax.set_xticklabels(['Firm %d'%i for i in a_xvalues])
    customaxis(ax = ax, position = 'left', color = color, label = 'Prices',
            scale = 'linear', size = fontsize, full_nrs = False, location = 0.0)

# plot the volume for each seller, on same x axis as price plot, but on separate y axis 
############################################################
    axt = price_axis.twinx()
    color = colors[1]

    axt.grid(False)
    axt.bar(a_xvalues, a_quantity_sold, width=.8*w, color=color, align='center')
    axt.set_xticks(a_xvalues)
    axt.set_ylabel('Quantities', color = color, fontsize=fontsize)
    customaxis(ax = axt, position = 'right', color = color, label = 'Quantities',
            scale = 'linear', size = fontsize, full_nrs = False, location = 1.0)

# plot the profit for each seller
############################################################
    axt2 = price_axis.twinx()
    color = colors[2]

    axt2.grid(False)
    axt2.bar(a_xvalues+w, a_profit, width=.8*w, color=color, align='center')
    axt2.set_xticks(a_xvalues)
    axt2.set_ylabel('Profit', color = color, fontsize=fontsize)
    axt2.autoscale(tight=True)
    customaxis(ax = axt2, position = 'right', color = color, label = 'Profit',
            scale = 'linear', size= fontsize, full_nrs = False, location = 1.1)


def plot_1timestep_from_a_dic(d_load, i=0, show=False):
    a_cost       	= d_load['a_cost'][i]
    gamma      		= d_load['gamma'][i]
    endowment  		= d_load['endowment'][i]
    num_sellers     	= d_load['num_sellers'][i]
    num_buyers 		= d_load['num_buyers'][i]
    a_price         	= d_load['a_price_nash'][i]
    a_quantity          = d_load['a_quantity_nash'][i]
    a_quantity_sold     = d_load['a_quantity_sold'][i]
    a_profit 		= d_load['a_profit'][i]
    a_buyer_pos         = d_load['a_buyer_loc'][i]
    a_seller_pos        = d_load['a_seller_loc'][i]
    m_tax               = d_load['m_tax'][i]
    m_quantity_bought   = d_load['m_quantity_bought'][i]
    plot_1timestep_from_data(a_cost, gamma, endowment, num_sellers,
            num_buyers, a_price, a_quantity, a_quantity_sold, a_profit,
            a_buyer_pos, a_seller_pos, m_tax, m_quantity_bought)
# write figure to the output 
############################################################
    fn    = 'gamma=%s_a_cost=%s_endowment=%s_sellers=%s_buyers=%s_timestep=%s'%(gamma,
            a_cost, endowment, num_sellers, num_buyers, i)
    out_folder = './output/plots/'
    if not os.path.exists(out_folder): os.makedirs(out_folder)
    plt.savefig(out_folder + fn + '.pdf', bbox_inches='tight')
    if show: plt.show()

def plot_1timestep_from_dic(d_load, show=False):
    a_cost       	= d_load['a_cost']
    gamma      		= d_load['gamma']
    endowment  		= d_load['endowment']
    num_sellers     	= d_load['num_sellers']
    num_buyers 		= d_load['num_buyers']
    a_price         	= d_load['a_price_nash']
    a_quantity          = d_load['a_quantity_nash']
    a_quantity_sold     = d_load['a_quantity_sold']
    a_profit 		= d_load['a_profit']
    a_buyer_pos         = d_load['a_buyer_loc']
    a_seller_pos        = d_load['a_seller_loc']
    m_tax               = d_load['m_tax']
    m_quantity_bought   = d_load['m_quantity_bought']
    plot_1timestep_from_data(a_cost, gamma, endowment, num_sellers,
            num_buyers, a_price, a_quantity, a_quantity_sold, a_profit,
            a_buyer_pos, a_seller_pos, m_tax, m_quantity_bought)
# write figure to the output 
############################################################
    fn    = 'gamma=%s_a_cost=%s_endowment=%s_sellers=%s_buyers=%s'%(gamma, a_cost,
            endowment, num_sellers, num_buyers)
    out_folder = './output/plots/'
    if not os.path.exists(out_folder): os.makedirs(out_folder)
    plt.savefig(out_folder + fn + '.pdf', bbox_inches='tight')
    if show: plt.show()

##############
### SCRIPT ###
##############

fn = "turn_gamma=0.6.pickle"
folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
folder  = folder1 if os.path.exists(folder1) else folder2
d_load  = jl.load(folder + fn)

plot_1timestep_from_a_dic(d_load, 13, show=True)
