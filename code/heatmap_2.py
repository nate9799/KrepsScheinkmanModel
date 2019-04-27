import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.externals import joblib as jl
from matplotlib import pyplot as plt


def get_dict(cost_ratio, gamma):
    fn      = 'S=%s_B=%s_gamma=%s_mean_cost=%s_cost_ratio=%s_endow=%s_randomize=%s.pkl'%(num_sellers,
            num_buyers, gamma, mean_cost, cost_ratio, endowment, randomize)
    ret = jl.load(folder + fn)
    return ret

def get_pd_from_m_dict2(m_dict, str_target1, str_target2, func=None):
    if func is None:
        func=lambda x, y : x
    m_out = [[func(dict[str_target1], dict[str_target2]) for dict in a_dict]
            for a_dict in m_dict]
    ret = pd.DataFrame(m_out)
    return ret

def get_pd_from_m_dict(m_dict, str_target, func=None, reorient=True):
    if func is None:
        func=lambda x : x
    i = 1
    if reorient:
        i = -1
    m_targ = [[func(dict[str_target]) for dict in a_dict] for a_dict in m_dict[::i]]
    ret = pd.DataFrame(m_targ)
    ret.columns = a_cost_ratio
    ret.index = a_gamma[::i]
    return ret

def heat_ax_from_pd(pd, title, ax):
    sns.heatmap(pd_profit, annot=True, fmt = '.4g', ax = ax)
    ax.set_title(title)
    ax.set_xlabel('cost_ratio')
    ax.set_ylabel('gamma')
    return ax

# Create combinations
##########################################################
num_sellers     = 2
num_buyers      = 12
mean_cost       = 100.
a_cost_ratio    = np.round(np.linspace(1.0, 2.0, 11), 3)
endowment       = 200.
randomize       = False
a_gamma         = np.round(np.linspace(0.00, .3, 11), 3)

folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
folder  = folder1 if os.path.exists(folder1) else folder2

m_dict = [[get_dict(cost_ratio, gamma) for cost_ratio in a_cost_ratio]
        for gamma in a_gamma]

# Make plots
pd_gamma = get_pd_from_m_dict(m_dict, 'gamma')
pd_cost_ratio = get_pd_from_m_dict(m_dict, 'cost_ratio')
pd_price = get_pd_from_m_dict(m_dict, 'a_price_nash', lambda x : max(x))
pd_price = get_pd_from_m_dict(m_dict, 'a_price_nash', lambda x : min(x))
print(pd_price)
print(pd_gamma)
pd_profit = get_pd_from_m_dict(m_dict, 'a_quantity_nash', lambda x : min(x))
ax = plt.axes()

plt.show()
pd_eff_comp = get_pd_from_m_dict(m_dict, 'a_price_nash',
        lambda a_x:(endowment - max(a_x))/(max(a_x) - 100.))
print(pd_eff_comp)
sns.heatmap(pd_eff_comp)
plt.show()


