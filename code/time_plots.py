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
from plot_util import customaxis


# plot the profit and quantity sold for each seller 
############################################################
def make_ax_profit_quantity(pd_profit, pd_quantity_sold, a_marker, a_color,
        num_sellers, gamma, fontsize=20, x_ticks=None, ax=None):
    color = a_color[0]
    pd_data = pd_profit
    if x_ticks is None:
        x_ticks = range(len(pd_data))
    x_ticks = np.round(x_ticks, 2)
    for i in range(num_sellers):
        ax.plot(x_ticks, pd_data[i], '-', marker=a_marker[i],
                markerfacecolor='none', color=color, label = 'Firm %d'%(i+1))
    ax.set_xlim(min(x_ticks), max(x_ticks))
    ax.set_xticks(x_ticks)
    customaxis(ax = ax, position = 'left', color = color, label = 'Profit',
            scale = 'linear', size = fontsize, full_nrs = False, location = 0.0)
    ax.get_xaxis().set_major_formatter(tk.FuncFormatter(lambda x, p:
        format(x,',')))
    ax.legend()
    leg = ax.get_legend()
    (leg.legendHandles[i].set_color('black') for i in range(num_sellers))
    ax.set_title("$\gamma$ = {} and Firm 1's initial cost is 99\% of Firm 2's".format(gamma),
            fontsize=fontsize +5)
# Quantity
    axt = ax.twinx()
    color = a_color[1]
    pd_data = pd_quantity_sold
    for i in range(num_sellers):
        axt.plot(x_ticks, pd_data[i], '-', marker=a_marker[i],
                markerfacecolor='none', color=color)
    axt.grid(False)
    customaxis(ax = axt, position = 'right', color = color, label = 'Quantity',
            scale = 'linear', size = fontsize, full_nrs = False, location = 1.0)

# plot the profit and quantity sold for each seller 
############################################################
def make_ax_profit_quantity_share(pd_profit, pd_quantity_sold, a_marker,
        a_color, num_sellers, gamma, fontsize=20, x_ticks=None, ax=None):
    # FIXME: x_ticks
    color = a_color[0]
    pd_data = pd_profit.div(pd_profit.sum(1), 0)
    print('HIHIHI')
    print(pd_profit)
    print(pd_data)
    if x_ticks is None:
        x_ticks = range(len(pd_data))
    x_ticks = np.round(x_ticks, 2)
    for i in range(num_sellers):
        ax.plot(x_ticks, pd_data[i], '-', marker=a_marker[i],
                markerfacecolor='none', color=color, label = 'Firm %d'%(i+1))
    ax.set_xlim(min(x_ticks), max(x_ticks))
    ax.set_xticks(x_ticks)
    customaxis(ax = ax, position = 'left', color = color, label = 'Profit Share',
            scale = 'linear', size = fontsize, full_nrs = False, location = 0.0)
    ax.get_xaxis().set_major_formatter(tk.FuncFormatter(lambda x, p:
        format(x,',')))
    ax.legend()
    leg = ax.get_legend()
    (leg.legendHandles[i].set_color('black') for i in range(num_sellers))
# Quantity
    axt = ax.twinx()
    color = a_color[1]
    pd_data = pd_quantity_sold.div(pd_quantity_sold.sum(1), 0)
    print('BYBYBY')
    print(pd_quantity_sold)
    print(pd_data)
    for i in range(num_sellers):
        axt.plot(x_ticks, pd_data[i], '-', marker=a_marker[i],
                markerfacecolor='none', color=color)
    axt.grid(False)
    customaxis(ax = axt, position = 'right', color = color, label = 'Quantity Share',
            scale = 'linear', size = fontsize, full_nrs = False, location = 1.0)

# plot the cost and price
############################################################
def make_ax_cost_price(pd_cost, pd_price, pd_cournot, a_marker, a_color,
        num_sellers, gamma, fontsize = 20, x_ticks=None, ax = None):
# for the left y axis of axis:
    pd_data = pd_price
    color = a_color[0]
    if x_ticks is None:
        x_ticks = range(len(pd_data))
    for i in range(num_sellers):
        ax.plot(x_ticks, pd_data[i], color=color, marker = a_marker[i],
                markerfacecolor='none', label='')
    if gamma == 0:
        ax.plot(x_ticks, pd_cournot, color=color, linestyle='dashed',
                label='Theoretical Cournot Price')
        ax.legend(loc='lower left')
    ax.spines['left'].set_color(color)
    ax.tick_params(axis='y', color=color)
    [i.set_color(color) for i in ax.get_yticklabels()]
    ax.yaxis.set_label_position("left")
    ax.set_ylabel('Price', color=color, fontsize=fontsize)
    ax.set_xlim(min(x_ticks), max(x_ticks))
# for the 'right' axis it is similar
    axt   = ax.twinx()
    pd_data = pd_cost
    color = a_color[1]
    for i in range(num_sellers):
        axt.plot(x_ticks, pd_data[i], marker= a_marker[i],
                markerfacecolor='none', color=color)
    axt.spines['right'].set_color(color)
    axt.tick_params(axis='y', color=color)
    [i.set_color(color) for i in axt.get_yticklabels()]
    axt.yaxis.set_label_position("right")
    axt.set_ylabel('Cost', color=color, fontsize=fontsize)
    axt.set_ylim(0, 110)
    axt.grid(False)
    return ax

def make_column(fn, ax_profit_quantity, ax_profit_quantity_share,
        ax_cost_price, x_ticks_name=None):
    print(fn)
    d_load = jl.load(fn)
# Data from d_load
##########################################################
    print(d_load)
    scalar_tax          = d_load['scalar_tax'][0]
    gamma               = d_load['gamma'][0]
    endowment           = d_load['endowment'][0]
    num_sellers         = int(d_load['num_sellers'][0])
    num_buyers          = int(d_load['num_buyers'][0])
    m_tax               = d_load['m_tax'][0]
    pd_quantity         = pd.DataFrame(d_load['a_quantity_nash'])
    pd_quantity_sold    = pd.DataFrame(d_load['a_quantity_sold'])
    pd_price            = pd.DataFrame(d_load['a_price_nash'])
    pd_cost             = pd.DataFrame(np.array(d_load['a_cost']))
    pd_profit           = pd.DataFrame(d_load['a_profit'])
# x_ticks stuff
    x_ticks = range(len(pd_profit))
    if x_ticks_name is not None:
        print(x_ticks_name)
        x_ticks = d_load[x_ticks_name]
    x_ticks = np.round(x_ticks, 3)
    print(x_ticks)
# Data from calculations
    pd_quantity_unsold      = pd_quantity - pd_quantity_sold
# Cournot price at Nash quantity (A + n\bar c)/(n+1)
    pd_cournot              = (endowment + pd_cost.sum(1))/(num_sellers+1)
    num_timesteps           = len(d_load['a_cost']) - 1
#check for if the stuff ended early
    if np.isnan(d_load['a_profit'][num_timesteps][0]):
        num_timesteps = num_timesteps - 1
# Stuff for plotting
    a_marker = ['s', 'x', '*', 'o', 'D']
    a_color = sb.color_palette("deep", 6)
# Plots Here
    make_ax_profit_quantity(pd_profit, pd_quantity_sold, a_marker,
            a_color[0:2], num_sellers, gamma, x_ticks=x_ticks,
            ax=ax_profit_quantity)
    make_ax_profit_quantity_share(pd_profit, pd_quantity_sold, a_marker,
            a_color[2:4], num_sellers, gamma, x_ticks=x_ticks,
            ax=ax_profit_quantity_share)
    print(5)
    make_ax_cost_price(pd_cost, pd_price, pd_cournot, a_marker, a_color[4:6],
            num_sellers, gamma, x_ticks=x_ticks, ax=ax_cost_price)

def write_time_plot_from_file(fn, fn_out, folder = None):
    print(fn)
    sb.set_style("darkgrid")
# set figure size
##########################################################
    nrow            = 3
    ncol            = 1
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
    ax_profit_quantity          = plt.subplot2grid((nrow,ncol), (0,0))
    ax_profit_quantity_share    = plt.subplot2grid((nrow,ncol), (1,0), sharex=ax_profit_quantity)
    ax_cost_price               = plt.subplot2grid((nrow,ncol), (2,0), sharex=ax_profit_quantity)
# annotate
##########################################################
    a_ax = fig.get_axes()
    a_annote = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    for (ax, annote) in zip(a_ax, a_annote[:len(a_ax)]):
        ax.annotate(annote,
                xy          =  (-0.12, 0.96),
                xycoords    = 'axes fraction',
                fontsize    =  12,
                ha          = 'left',
                va          = 'top' )
    make_column(fn, ax_profit_quantity, ax_profit_quantity_share, ax_cost_price, 'scalar_tax')
# write figure to the output 
############################################################
    fn_out = 'plot_' + fn_out
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
    a_fn_out = np.array(['turn_gamma=1.0_endow=120.0_taxmethod=cardinal_seed={}.pickle'.format(i)
        for i in range(1,100)])
    folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
    folder3 = "C:/Users/CAREBEARSTARE3_USER/Documents/WORK/MITInternship/ModelWithSandro/abmcournotmodel/code/output/data/"
    folder = folder3
    a_fn = glob.glob(folder + "turn*")
    write_time_plot_from_file(folder + 'mean_turn1.pkl', 'mean_turn2')
    print('asdfasdfasfdas')
    [write_time_plot_from_file(a_fn[i], a_fn_out[i]) for i in range(len(a_fn))]
