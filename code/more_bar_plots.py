import os
import numpy as np
from sklearn.externals import joblib as jl
from matplotlib import pyplot as plt
from itertools  import product,combinations


# Create combinations
##########################################################
num_sellers     = 2
num_buyers      = 12
a_cost          = np.array([80., 120.])
endowment       = 200.
randomize       = False
a_gamma         = np.round(np.linspace(0.,.3,11), 3)

folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
folder  = folder1 if os.path.exists(folder1) else folder2
a_fn = ['S=%s_B=%s_gamma=%s_a_cost=%s_endow=%s_randomize=%s.pkl'%(num_sellers,
    num_buyers, gamma, a_cost, endowment, randomize) for gamma in a_gamma]
a_dict = [jl.load(folder + fn) for fn in a_fn]

# create the plot window
##########################################################

# set all axes instances
##########################################################
eff_comp_axis   = plt.subplot2grid( (1,4), (0,0) )
min_profit_axis = plt.subplot2grid( (1,4), (0,1) )
max_profit_axis = plt.subplot2grid( (1,4), (0,2) )
tot_profit_axis = plt.subplot2grid( (1,4), (0,3) )

str_target = 'a_profit'
# Effective competition
##########################################################
ax = eff_comp_axis
a_min_price = np.array([min(dic['a_price_nash']) for dic in a_dict])
endowment   = a_dict[0]['endowment']
a_cost      = a_dict[0]['a_cost']
avg_cost    = sum(a_cost)/float(len(a_cost))
print(a_min_price)
a_eff_comp  = (endowment - a_min_price)/(a_min_price - avg_cost)
ax.bar(a_gamma, a_eff_comp, width = .01)
ax.set_xlabel('gamma',                          fontsize=13)
ax.set_ylabel('percieved number of firms',      fontsize=13)


# min profit
##########################################################
ax = min_profit_axis
a_min_profit = [min(dic[str_target]) for dic in a_dict]
ax.bar(a_gamma, a_min_profit, width = .01)
ax.set_xlabel('gamma',                        fontsize=13)
ax.set_ylabel('minimum profit of single seller',  fontsize=13)

# max profit
##########################################################
ax = max_profit_axis
a_max_profit = [max(dic[str_target]) for dic in a_dict]
ax.bar(a_gamma, a_max_profit, width = .01)
ax.set_xlabel('gamma',                        fontsize=13)
ax.set_ylabel('maximum profit of single seller',  fontsize=13)
# tot profit
##########################################################
ax = tot_profit_axis
a_tot_profit = [sum(dic[str_target]) for dic in a_dict]
ax.bar(a_gamma, a_tot_profit, width = .01)
ax.set_xlabel('gamma',                          fontsize=13)
ax.set_ylabel('sum of both sellers\' profit',   fontsize=13)

# show
##########################################################
plt.show()
