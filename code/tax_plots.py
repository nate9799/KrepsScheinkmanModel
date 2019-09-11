import os
import sys
import pickle
import numpy  as np
import pandas as pd
import seaborn as sb
import glob
from matplotlib import pyplot as plt
from matplotlib import ticker as tk
from pprint     import pprint as pp
from sklearn.externals import joblib as jl
from itertools  import product,combinations
from plot_util  import customaxis
from time_plots import dic_from_a_dic


# plot the profit and quantity sold for each seller 
############################################################
def make_ax_profit_quantity(pd_profit, pd_quantity_sold, a_marker, a_color,
        num_timesteps, num_sellers, fontsize=20, ax=None):
    color = a_color[0]
    pd_data = pd_profit
    for i in range(num_sellers):
        ax.plot(range(len(pd_data)), pd_data[i], '-', marker=a_marker[i],
                markerfacecolor='none', color=color, label = 'Firm %d'%(i+1))
    ax.set_xlim(0, num_timesteps)
    ax.set_xticks(np.arange(0, num_timesteps+1))
    customaxis(ax = ax, position = 'left', color = color, label = 'Profit',
            scale = 'linear', size = fontsize, full_nrs = False, location = 0.0)
    ax.get_xaxis().set_major_formatter(tk.FuncFormatter(lambda x, p:
        format(int(x),',')))
    ax.legend()
    leg = ax.get_legend()
    (leg.legendHandles[i].set_color('black') for i in range(num_sellers))
# FIXME: title
    #ax.set_title("Firm 1's initial cost is 99\% of Firm 2's".format(gamma),
            fontsize=fontsize +5)
# Quantity
    axt = ax.twinx()
    color = a_color[1]
    pd_data = pd_quantity_sold
    for i in range(num_sellers):
        axt.plot(range(len(pd_data)), pd_data[i], '-', marker=a_marker[i],
                markerfacecolor='none', color=color)
    axt.grid('False')
    customaxis(ax = axt, position = 'right', color = color, label = 'Quantity',
            scale = 'linear', size = fontsize, full_nrs = False, location = 1.0)

# plot the cost and price
############################################################
def make_ax_cost_price(pd_cost, pd_price, pd_cournot, a_marker, a_color,
        num_timesteps, num_sellers, fontsize = 20, ax = None):
    ax.set_xlim(0, num_timesteps)
# for the left y axis of axis:
    pd_data = pd_price
    color = a_color[0]
    for i in range(num_sellers):
        ax.plot(range(len(pd_data)), pd_data[i], color=color, marker = a_marker[i],
                markerfacecolor='none', label='')
    ax.spines['left'].set_color(color)
    ax.tick_params(axis='y', color=color)
    [i.set_color(color) for i in ax.get_yticklabels()]
    ax.yaxis.set_label_position("left")
    ax.set_ylabel('Price', color=color, fontsize=fontsize)
    ax.set_xlim(0, num_timesteps)
# for the 'right' axis it is similar
    axt   = ax.twinx()
    pd_data = pd_cost
    color = a_color[1]
    for i in range(num_sellers):
        axt.plot(range(len(pd_data)), pd_data[i], marker= a_marker[i],
                markerfacecolor='none', color=color)
    axt.spines['right'].set_color(color)
    axt.tick_params(axis='y', color=color)
    [i.set_color(color) for i in axt.get_yticklabels()]
    axt.yaxis.set_label_position("right")
    axt.set_ylabel('Cost', color=color, fontsize=fontsize)
    axt.set_ylim(0, 110)
    axt.grid(False)
    return ax

def make_column(fn, ax_profit_quantity, ax_cost_price):
    print(fn)
    d_load = jl.load(fn)
# Data from d_load
##########################################################
    endowment           = d_load['endowment'][0]
    num_sellers         = d_load['num_sellers'][0]
    num_buyers          = d_load['num_buyers'][0]
    a_buyer_pos         = d_load['a_buyer_loc'][0]
    a_seller_pos        = d_load['a_seller_loc'][0]
    m_tax               = d_load['m_tax'][0]
    a_m_quantity_bought = d_load['m_quantity_bought']
    pd_scalar_tax       = pd.DataFrame(d_load['scalar_tax'])
    pd_gamma            = pd.DataFrame(d_load['gamma'])
    pd_quantity         = pd.DataFrame(d_load['a_quantity_nash'])
    pd_quantity_sold    = pd.DataFrame(d_load['a_quantity_sold'])
    pd_price            = pd.DataFrame(d_load['a_price_nash'])
    pd_cost             = pd.DataFrame(np.array(d_load['a_cost']))
    pd_profit           = pd.DataFrame(d_load['a_profit'])
# Data from calculations
    pd_num_buyers_per_seller = pd.DataFrame([np.count_nonzero(m_quant, axis=1) for
        m_quant in a_m_quantity_bought])
    pd_quantity_unsold      = pd_quantity - pd_quantity_sold
# Cournot price at Nash quantity (A + n\bar c)/(n+1)
    pd_cournot              = (endowment + pd_cost.sum(1))/(num_sellers+1)
    num_timesteps           = len(d_load['a_cost']) - 1
#check for if the stuff ended early
    if np.isnan(d_load['a_profit'][num_timesteps][0]):
        num_timesteps = num_timesteps - 1
# Stuff for plotting
    a_marker = ['s', 'x', '*', 'o', 'D']
    a_color = sb.color_palette("deep", num_sellers+4)
# Plots Here
    make_ax_profit_quantity(pd_profit, pd_quantity_sold, a_marker,
            a_color[0:2], num_timesteps, num_sellers, gamma,
            ax=ax_profit_quantity)
    make_ax_cost_price(pd_cost, pd_price, pd_cournot, a_marker, a_color[2:4],
            num_timesteps, num_sellers, gamma, ax=ax_cost_price)

def write_time_plot_from_file(a_fn, folder = None):
    sb.set_style("darkgrid")
# set figure size
##########################################################
    nrow            = 2
    ncol            = 2
    width_scale     = 12
    height_scale    = 4
    figsize         = (width_scale*ncol,height_scale*nrow)
    fontsize        = 14
# create the plot window
##########################################################
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=.3, wspace=0.3)
    plt.rc('text', usetex=True)
# set all axes instances
##########################################################
    ax_profit_quantity1 = plt.subplot2grid((nrow,ncol), (0,0))
    ax_profit_quantity2 = plt.subplot2grid((nrow,ncol), (0,1))
    ax_cost_price1      = plt.subplot2grid((nrow,ncol), (1,0), sharex=ax_profit_quantity1)
    ax_cost_price2      = plt.subplot2grid((nrow,ncol), (1,1), sharex=ax_profit_quantity2)
# annotate
##########################################################
    a_ax = fig.get_axes()
    a_annote = ['(a)','(b)','(c)','(d)']
    for (ax, annote) in zip(a_ax, a_annote[:len(a_ax)]):
        ax.annotate(annote,
                xy          = (-0.12, 0.96),
                xycoords    = 'axes fraction',
                fontsize    =  12,
                ha          = 'left',
                va          = 'top' )
    make_column(a_fn[0], ax_profit_quantity1, ax_cost_price1)
    make_column(a_fn[1], ax_profit_quantity2, ax_cost_price2)

# write figure to the output 
############################################################
    fn_out = "timestep"
    out_folder = './output/plots/'
    if not os.path.exists(out_folder): os.makedirs(out_folder)
    plt.savefig(out_folder + fn_out + '.png', bbox_inches='tight')

# write figure to the screen (must go after file creation or doesn't write
# correctly) 
############################################################
    #plt.show()

##############
### SCRIPT ###
##############

if __name__ == "__main__":
    a_fn = np.array(['alt_adv_fast_turn_gamma=0.0_scalar_tax=1.0_endow=200.pickle'.format(i)
        for i in np.arange(125,205,5)])
    folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
    folder  = "C:/Users/CAREBEARSTARE3_USER/Documents/WORK/MITInternship/ModelWithSandro/abmcournotmodel/code/output/data/results/"
    a_fn = glob.glob(folder + "turn*")
    a_dic = [jl.load(fn) for fn in a_fn]
    dic = dic_from_a_dic(a_dic)
    write_time_plot_from_file(dic)

