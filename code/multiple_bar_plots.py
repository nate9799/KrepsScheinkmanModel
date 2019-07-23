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

fontsize = 15

################
### FUNCTION ###
################

# Circle of buyers and sellers
############################################################
def ax_circle_from_data(a_seller_pos, a_buyer_pos, m_quantity_bought, a_color,
        gamma, a_cost, ax = None):
# Setup
    if ax is None:
        _, ax = plt.subplots()
    ax.set_title(label='$\gamma$ = {}'.format(gamma))
    num_sellers = len(a_seller_pos)
# plot a grey circle 
    r       = 1
    angles  = np.linspace(0, 2*np.pi, 500)
    xs      = r * np.sin(angles)
    ys      = r * np.cos(angles)
    ax.plot(xs, ys, color='grey', linestyle='--', zorder=1)
# plot sellers
    for i, s in enumerate(a_seller_pos):
        xcoord = [ r * np.sin( 2*np.pi*s ) ]
        ycoord = [ r * np.cos( 2*np.pi*s ) ]
        ax.scatter(xcoord, ycoord, marker='o', facecolor='none',
                color=a_color[i], s=400, zorder=3, label='Firm %d'%(i+1))
# plot buyers
# for each buyer, find index of his seller
    seller_ind = np.argmax(m_quantity_bought, axis=0)
    for s in range(num_sellers):
        a_pos_subset = a_buyer_pos[seller_ind == s]
        a_xcoord = [r * np.sin(2 * np.pi * a_pos_subset)]
        a_ycoord = [r * np.cos(2 * np.pi * a_pos_subset)]
        ax.scatter(a_xcoord, a_ycoord, marker='x', color=a_color[s], s=80,
                zorder=2, label='Buyer closest to from Firm %d'%(s+1)) 
    ax.set_xlim(-1.2*r, 1.2*r)
    ax.set_ylim(-1.2*r, 1.5*r)
    ax.legend(loc='center', ncol=2, frameon=False, fontsize=2*fontsize/3,
            labelspacing=2)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axis('equal')

# Barplots price, cost, profit, and quantity
############################################################
def ax_bars_from_data(a_price, a_cost, a_quantity_sold, a_profit, a_color,
        ax = None):
# Setup
    if ax is None:
        _, ax = plt.subplots()
    num_sellers = len(a_price)
    a_color_alt = sb.color_palette("deep", num_sellers)
    a_xvalues = np.arange(1, num_sellers + 1)
    w = 0.2
# plot the price for each buyer 
    ax.set_title('Firm 1\'s cost is {}\% of firm 2\'s and makes {}\% more profit'.format(a_cost[0], int(round(a_profit[0]/a_profit[1] -1, 2)*100)))
    color = a_color[0]
    ax.bar(a_xvalues-w, a_price, width=.8*w, color=color,   alpha=1.0, align='center')
    ax.bar(a_xvalues-w, a_cost,  width=.8*w, color='black', alpha=0.2, align='center')
    ax.set_xlabel('')
    ax.set_xticks(a_xvalues)
    ax.set_xticklabels(['Firm %d'%i for i in a_xvalues])
    [t.set_color(i) for (i,t) in zip(a_color_alt[0:num_sellers], ax.xaxis.get_ticklabels())]
    customaxis(ax = ax, position = 'left', color = color, label = 'Prices',
            scale = 'linear', size = fontsize, full_nrs = False, location = 0.0)
# plot the quantity for each seller, on same x axis as price plot, but on separate y axis 
    axt = ax.twinx()
    color = a_color[1]
    axt.grid(False)
    axt.bar(a_xvalues, a_quantity_sold, width=.8*w, color=color, align='center')
    axt.set_xticks(a_xvalues)
    axt.set_ylabel('Quantities', color = color, fontsize=fontsize)
    customaxis(ax = axt, position = 'right', color = color, label = 'Quantities',
            scale = 'linear', size = fontsize, full_nrs = False, location = 1.0)
# plot the profit for each seller
    axt2 = ax.twinx()
    color = a_color[2]
    axt2.grid(False)
    axt2.bar(a_xvalues+w, a_profit, width=.8*w, color=color, align='center')
    axt2.set_xticks(a_xvalues)
    axt2.set_ylabel('Profit', color = color, fontsize=fontsize)
    axt2.autoscale(tight=True)
    customaxis(ax = axt2, position = 'right', color = color, label = 'Profit',
            scale = 'linear', size= fontsize, full_nrs = False, location = 1.12)
    ax.set_xlim([min(a_xvalues)-1.5*w, max(a_xvalues)+1.5*w])

def draw_ax2_from_a_dic(d_load, i, ax_circle, ax_bar):
# get data
    if i >= 0:
        a_cost              = d_load['a_cost'][i]
        gamma               = d_load['gamma'][i]
        endowment           = d_load['endowment'][i]
        num_sellers         = d_load['num_sellers'][i]
        num_buyers          = d_load['num_buyers'][i]
        a_price             = d_load['a_price_nash'][i]
        a_quantity          = d_load['a_quantity_nash'][i]
        a_quantity_sold     = d_load['a_quantity_sold'][i]
        a_profit            = d_load['a_profit'][i]
        a_buyer_pos         = d_load['a_buyer_loc'][i]
        a_seller_pos        = d_load['a_seller_loc'][i]
        m_tax               = d_load['m_tax'][i]
        m_quantity_bought   = d_load['m_quantity_bought'][i]
    else:
        a_cost              = d_load['a_cost']
        gamma               = d_load['gamma']
        endowment           = d_load['endowment']
        num_sellers         = d_load['num_sellers']
        num_buyers          = d_load['num_buyers']
        a_price             = d_load['a_price_nash']
        a_quantity          = d_load['a_quantity_nash']
        a_quantity_sold     = d_load['a_quantity_sold']
        a_profit            = d_load['a_profit']
        a_buyer_pos         = d_load['a_buyer_loc']
        a_seller_pos        = d_load['a_seller_loc']
        m_tax               = d_load['m_tax']
        m_quantity_bought   = d_load['m_quantity_bought']

    print('cost')
    print(a_cost)
    print('gamma')
    print(gamma)
    print('a_price')
    print(a_price)
    print('a_profit')
    print(a_profit)
    print('a_quantity_sold')
    print(a_quantity_sold)
# Plots
    a_color = sb.color_palette("deep", (3 + num_sellers))
    ax_circle_from_data(a_seller_pos, a_buyer_pos, m_quantity_bought,
            a_color=a_color, gamma=gamma, a_cost=a_cost, ax=ax_circle)
    ax_bars_from_data(a_price, a_cost, a_quantity_sold, a_profit,
            a_color=a_color[num_sellers:], ax=ax_bar)

def draw_grid_of_timesteps(a_fn, a_index, folder=''):
# get data
    a_d_load = [jl.load(folder + fn) for fn in a_fn]
# set figure size
    nrow            = len(a_fn)
    ncol            = 2
    width_scale     = 9
    height_scale    = 6
    figsize         = (width_scale*ncol,height_scale*nrow)
# create the plot window
    sb.set_style("darkgrid")
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=.2, wspace=0.2)
    plt.rc('text', usetex=True)
# set all axes instances
    a_ax_circle = [plt.subplot2grid((nrow,ncol), (row,0)) for row in range(nrow)]
    a_ax_bar = [plt.subplot2grid((nrow,ncol), (row,1)) for row in range(nrow)]
#annotate
    a_ax = a_ax_circle + a_ax_bar
    a_ax[::2] = a_ax_circle
    a_ax[1::2] = a_ax_bar
    #a_annote = ['Row {}'.format(i) for i in np.arange(1, len(a_fn) + 1)]
    for (ax, annote) in zip(a_ax_circle, a_annote):
        ax.annotate(annote,
                xy          =  (-0.17, 0.5),
                xycoords    = 'axes fraction',
                fontsize    =  fontsize,
                ha          = 'left',
                va          = 'top' )
    [draw_ax2_from_a_dic(d_load, index, ax_circle, ax_bar) for
            (d_load,index,ax_circle,ax_bar) in zip(a_d_load, a_index,
                a_ax_circle, a_ax_bar)]
    return fig

def write_plot(fig):
    fn_out = '1x2plot1'
    out_folder = './output/plots/'
    if not os.path.exists(out_folder): os.makedirs(out_folder)
    plt.savefig(out_folder + fn_out + '.png', bbox_inches='tight')
    plt.show()

##############
### SCRIPT ###
##############

if __name__ == "__main__":
    a_fn = ["S=2_B=12_gamma=0.0_scalar_tax=0.05_mean_cost=100_cost_ratio=1.0_endow=120.0_randomize=True.pkl"]
            #"S=2_B=12_gamma=0.0_scalar_tax=0.05_mean_cost=100_cost_ratio=1.01_endow=120.0_randomize=True.pkl",
            #"S=2_B=12_gamma=0.5_scalar_tax=1.0_mean_cost=100_cost_ratio=1.0_endow=120.0_randomize=True.pkl",
            #"S=2_B=12_gamma=0.5_scalar_tax=1.0_mean_cost=100_cost_ratio=1.01_endow=120.0_randomize=True.pkl",
            #"S=2_B=12_gamma=0.5_scalar_tax=1.0_mean_cost=90_cost_ratio=1.25_endow=120.0_randomize=True.pkl"]
    folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
    folder  = folder1 if os.path.exists(folder1) else folder2
    folder  = "C:/Users/CAREBEARSTARE3_USER/Documents/WORK/MITInternship/ModelWithSandro/abmcournotmodel/code/output/data/"

    a_select_indices = [-1,-1,-1,-1,-1]

    fig = draw_grid_of_timesteps(a_fn, a_select_indices, folder=folder)
    write_plot(fig)

