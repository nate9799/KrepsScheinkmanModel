import numpy as np
import pandas as pd
from sklearn.externals import joblib as jl
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

# CONSTANTS 
NUM_BUYERS = 100
STARTUP_COST = 1
COST_PER_TIMESTEP = 3
GAMMA = 1 + (3.5/16)
WEALTH_TAX = .0
RATE_NEW_SELLER = .0
PLOT_TITLE = 'Gamma = {0:.2f}, Buyers = {1:}, Cost = {2:}, Tax = {3:2f}, Rate = {4:2f}'.format(GAMMA,
        NUM_BUYERS, COST_PER_TIMESTEP, WEALTH_TAX, RATE_NEW_SELLER)

file_path = "C:/Users/CAREBEARSTARE3_USER/Documents/MITInternship/ModelWithSandro/Pickle/"
FILE_NAME_FORMATER = file_path + 'gamma{0:.2f}_buyers{1:}_cost{2:}_tax{3:.2f}_rate{4:2f}.pickle'
file_name_read = FILE_NAME_FORMATER.format(GAMMA, NUM_BUYERS, COST_PER_TIMESTEP, WEALTH_TAX, RATE_NEW_SELLER)
print(file_name_read)

d_load = jl.load(file_name_read)

df_price = d_load['price']
df_profit = d_load['profit']
df_wealth = d_load['wealth']
a_num_sellers = df_price.notna().sum(axis=1)
df_num_sales_by_seller = d_load['num_sales_by_seller']
df_num_sales_by_seller.fillna(0, inplace=True)
df_num_sales_by_dist = d_load['num_sales_by_dist']
df_num_closest_buyer = d_load['num_closest_buyer']
df_num_closest_buyer.fillna(0, inplace=True)
print(df_num_closest_buyer)

MAX_TIMESTEP = len(a_num_sellers)-1

print(type(a_num_sellers))

############################################
# set figure size
############################################
nrow            = 7 
ncol            = 1
width_scale     = 15
height_scale    = 6
figsize         = ( width_scale*ncol, height_scale*nrow )

############################################
# create the figure
############################################
fig = plt.figure(figsize=figsize)
fig.suptitle(PLOT_TITLE)
fig.subplots_adjust(wspace=0.4, hspace=.4)

############################################
# Create axes.
############################################
ax_price = plt.subplot2grid((nrow,ncol), (0,0))
ax_profit = plt.subplot2grid((nrow,ncol), (1,0), sharex=ax_price)
ax_wealth = plt.subplot2grid((nrow,ncol), (2,0), sharex=ax_price)
ax_num_sellers = plt.subplot2grid((nrow,ncol), (3,0), sharex=ax_price)
ax_num_sales_by_seller = plt.subplot2grid((nrow,ncol), (4,0), sharex=ax_price)
ax_num_sales_by_dist = plt.subplot2grid((nrow,ncol), (5,0), sharex=ax_price) #ax_price.twinx() 
ax_num_closest_buyer = plt.subplot2grid((nrow,ncol), (6,0), sharex=ax_price)

############################################
# Add data to axes
############################################
colors = sns.color_palette("dark", 5)
ax_price.plot(range(len(df_price)), df_price)
ax_profit.plot(range(len(df_profit)), df_profit)
ax_wealth.plot(range(len(df_wealth)), df_wealth)
ax_num_sellers.plot(range(len(a_num_sellers)), a_num_sellers)
ax_num_sales_by_seller.stackplot(range(len(df_num_sales_by_seller)), df_num_sales_by_seller.T)
ax_num_sales_by_dist.stackplot(range(len(df_num_sales_by_dist)), np.array(df_num_sales_by_dist).T)
ax_num_closest_buyer.stackplot(range(len(df_num_closest_buyer)), np.array(df_num_closest_buyer).T)

############################################
# Add y axis labels to axes
############################################
ax_price.set(ylabel = 'price', xlim = (-10, MAX_TIMESTEP), yscale = 'linear')
ax_profit.set(ylabel = 'profit', xlim = (-10, MAX_TIMESTEP), yscale = 'linear')
ax_wealth.set(ylabel = 'wealth', xlim = (-10, MAX_TIMESTEP), yscale = 'linear')
ax_num_sellers.set(ylabel = 'num_sellers', xlim = (-10, MAX_TIMESTEP))
ax_num_sales_by_seller.set(ylabel = 'sales_by_seller', xlim = (-10, MAX_TIMESTEP))
ax_num_sales_by_dist.set(ylabel = 'sales_by_dist', xlim = (-10, MAX_TIMESTEP))
ax_num_closest_buyer.set(ylabel = 'closest_buyers', xlim = (-10, MAX_TIMESTEP))

############################################
# print
############################################
plt.show()

#rn = np.cumsum(np.random.rand(40))
#sandros_axis.plot( range(len(rn)), rn, color=colors[1] )
#sandros_axis.set_yscale('log')

