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

# Import data
file_path = "/home/nate/Desktop/"
file_name = file_path + "here.pickle"
d_load = jl.load(file_name)

sb.set_style("darkgrid")

cost       = d_load['cost']
gamma      = d_load['gamma']
endowment  = d_load['endowment']
num_sellers= d_load['num_sellers']
num_buyers = d_load['num_buyers']
a_price      = d_load['a_price_nash']
a_quantity      = d_load['a_quantity_nash']
a_quantity_sold = d_load['a_quantity_sold']
a_profit = d_load['a_profit']

N       = len(a_price) # number of sellers
x       = np.arange(1,N+1)
colors  = sb.color_palette("deep",3)
w       = 0.2

fig = plt.figure(figsize=(14,5))
fig.suptitle('Gamma={}, Cost/unit={}, endowment={}, #Sellers={}, #Buyers={}'.format(gamma, cost, endowment, num_sellers, num_buyers))
ax  = plt.gca()
ax2 = ax.twinx()
ax3 = ax.twinx()
ax2.grid(False)
ax3.grid(False)


ax.bar( x-w, a_price,          width=w,  color=colors[0],  alpha=1.0,  align='center' )
ax.bar( x-w, cost,             width=w,  color='black',    alpha=0.2,  align='center' )
ax2.bar(x, a_profit,         width=w,  color=colors[1],  alpha=1.0,  align='center' )
ax3.bar(x+w, a_quantity,       width=w,  color=colors[2],  alpha=1.0,  align='center' )
ax3.bar(x+w, a_quantity_sold,  width=w,  color='black',    alpha=0.2,  align='center' )

#ax.axhline(y=cost, color=colors[0], linestyle=':')



ax.set_xlabel('sellers',                        fontsize=13)
ax.set_ylabel('prices',      color = colors[0], fontsize=13)
ax2.set_ylabel('profit',     color = colors[1], fontsize=13)
ax3.set_ylabel('quantities', color = colors[2], fontsize=13)
ax.autoscale(tight=True)

plt.show()
