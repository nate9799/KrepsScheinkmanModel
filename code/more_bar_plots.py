import os
import numpy as np
from sklearn.externals import joblib as jl
from matplotlib import pyplot as plt
from itertools  import product,combinations


# Create combinations
##########################################################
num_sellers     = 2
num_buyers      = 12
cost            = 100.
endowment       = 200.
randomize       = False
a_gamma         = np.round(np.linspace(0.,2.,41), 3)

folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
folder  = folder1 if os.path.exists(folder1) else folder2
a_fn = ['S=%s_B=%s_gamma=%s_cost=%s_endow=%s_randomize=%s.pkl'%(num_sellers,
    num_buyers, gamma, cost, endowment, randomize) for gamma in a_gamma]
a_dict = [jl.load(folder + fn) for fn in a_fn]

# create the plot window
##########################################################

# set all axes instances
##########################################################
min_profit_axis = plt.subplot2grid( (1,3), (0,0) )
max_profit_axis = plt.subplot2grid( (1,3), (0,1) )
tot_profit_axis = plt.subplot2grid( (1,3), (0,2) )

str_target = 'a_profit'
# min profit
##########################################################
ax = min_profit_axis
a_min_profit = [min(dic[str_target]) for dic in a_dict]
ax.bar(a_gamma, a_min_profit, width = .03)
ax.set_xlabel('gamma',                        fontsize=13)
ax.set_ylabel('minimum profit of single seller',  fontsize=13)

# max profit
##########################################################
ax = max_profit_axis
a_max_profit = [max(dic[str_target]) for dic in a_dict]
ax.bar(a_gamma, a_max_profit, width = .03)
ax.set_xlabel('gamma',                        fontsize=13)
ax.set_ylabel('maximum profit of single seller',  fontsize=13)

# tot profit
##########################################################
ax = tot_profit_axis
a_tot_profit = [sum(dic[str_target]) for dic in a_dict]
ax.bar(a_gamma, a_tot_profit, width = .03)
ax.set_xlabel('gamma',                        fontsize=13)
ax.set_ylabel('sum of both sellers\' profit',  fontsize=13)

# show
##########################################################
plt.show()
