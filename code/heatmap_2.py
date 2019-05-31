import os
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.externals import joblib as jl
from matplotlib import pyplot as plt


def get_dict(cost_ratio, gamma):
    fn      = 'S=%s_B=%s_gamma=%s_mean_cost=%s_cost_ratio=%s_endow=%s_randomize=%s.pkl'%(num_sellers,
            num_buyers, gamma, mean_cost, cost_ratio, endowment, randomize)
    ret = jl.load(folder + fn)
    return ret

def get_pd_from_m_dict3(m_dict, str_target1, str_target2, str_target3,
        func=None, reorient=False):
    if func is None:
        func=lambda x, y, z : x
    i = 1
    if reorient:
        i = -1
    m_out = [[func(dict[str_target1], dict[str_target2], dict[str_target3]) for
        dict in a_dict] for a_dict in m_dict]
    ret = pd.DataFrame(m_out)
    print(ret)
    ret = ret.T
    ret = ret.interpolate()
    ret.columns = [str(gamma) for gamma in a_gamma]
    ret['cost_ratio'] = a_cost_ratio[::i]
    print(ret)
    return ret

def get_pd_from_m_dict2(m_dict, str_target1, str_target2, func=None, reorient=False):
    if func is None:
        func=lambda x, y : x
    i = 1
    if reorient:
        i = -1
    m_out = [[func(dict[str_target1], dict[str_target2]) for dict in a_dict]
            for a_dict in m_dict]
    ret = pd.DataFrame(m_out)
    ret = ret.T
    ret = ret.interpolate()
    ret.columns = [str(gamma) for gamma in a_gamma]
    ret['cost_ratio'] = a_cost_ratio[::i]
    print(ret)
    return ret

def get_pd_from_m_dict(m_dict, str_target, func=None, reorient=False):
    if func is None:
        func=lambda x : x
    i = 1
    if reorient:
        i = -1
    m_targ = [[func(dict[str_target]) for dict in a_dict] for a_dict in m_dict[::i]]
    ret = pd.DataFrame(m_targ)
    ret = ret.T
    ret = ret.interpolate()
    ret.columns = [str(gamma) for gamma in a_gamma]
    ret['cost_ratio'] = a_cost_ratio[::i]
    print(ret)
    return ret

def heat_ax_from_pd(pd_tmp, title, ax):
    sb.heatmap(pd_tmp, annot=True, fmt = '.4g', ax = ax)
    ax.set_title(title)
    ax.set_xlabel('cost_ratio')
    ax.set_ylabel('gamma')
    return ax

def plot_line_from_pd(pd_tmp, gamma, color, is_min=True):
    if is_min:
        linestyle='--'
        label = 'Seller 1, gamma = %s'%(gamma)
    else:
        linestyle=':'
        label = 'Seller 0, gamma = %s'%(gamma)
    ret = plt.plot('cost_ratio', str(gamma), data=pd_tmp, marker='', color=color,
            linewidth=2, linestyle=linestyle, label=label)
    return ret

def plot_multi_line_from_pd(pd_0, pd_1, a_gamma, ylab, title=''):
    a_color = sb.color_palette("deep", len(a_gamma))
    for gamma, color in zip(a_gamma, a_color):
        plot_line_from_pd(pd_1, gamma, color, False)
        plot_line_from_pd(pd_0, gamma, color, True)
    plt.xlabel('Ratio between the two sellers\' cost per unit, with seller 1 having the lower cost')
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
 

def plot_multi_line_from_pd1(pd_0, a_gamma, ylab, title=''):
    a_color = sb.color_palette("deep", len(a_gamma))
    for gamma, color in zip(a_gamma, a_color):
        plot_line_from_pd(pd_0, gamma, color, True)
    plt.xlabel('Ratio between the two sellers\' cost per unit, with seller 1 having the lower cost')
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
 

# Create combinations
##########################################################
num_sellers     = 2
num_buyers      = 12
mean_cost       = 100.
a_cost_ratio    = np.round(np.linspace(1.0, 1.9, 10), 3)
endowment       = 200.
randomize       = False
a_gamma         = np.round(np.linspace(0.00, .3, 11), 3)

folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
folder  = folder1 if os.path.exists(folder1) else folder2

m_dict = [[get_dict(cost_ratio, gamma) for cost_ratio in a_cost_ratio]
        for gamma in a_gamma]

tmp = m_dict[0][0]
print(tmp['m_quantity_bought'] * np.reshape(tmp['a_price_nash'], (2,1)))
a_gamma_chosen = [0.0, 0.18]

# Make plots
pd_price_0 = get_pd_from_m_dict(m_dict, 'a_price_nash', lambda x : x[0])
pd_price_1 = get_pd_from_m_dict(m_dict, 'a_price_nash', lambda x : x[1])
y_lab_price = 'Price offered, not including tax'
plot_multi_line_from_pd(pd_price_0, pd_price_1, a_gamma_chosen, y_lab_price)


# Make plots
pd_tax_revenue = get_pd_from_m_dict3(m_dict, 'm_tax', 'm_quantity_bought',
        'a_price_nash', lambda x,y,z : sum(sum((x-1.) * y * np.reshape(z, (2,1)))))
y_lab_tax = 'Total Tax Revenue'
print(pd_tax_revenue)
plot_multi_line_from_pd1(pd_tax_revenue, a_gamma_chosen, y_lab_tax)

pd_price_0 = get_pd_from_m_dict(m_dict, 'a_profit', lambda x : x[0])
pd_price_1 = get_pd_from_m_dict(m_dict, 'a_profit', lambda x : x[1])
y_lab_profit = 'Profit'
plot_multi_line_from_pd(pd_price_0, pd_price_1, a_gamma_chosen, y_lab_profit)

