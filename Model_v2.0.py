# Technical debt from pricing model. can speed up by factor of n
# also, occasional errors in negative prices
# improved pricing model in 1.2
import numpy as np
import pandas as pd
from sklearn.externals import joblib as jl



#####################
#####################
##### FUNCTIONS #####
#####################
#####################

#####################
### CALCULATE TAX ###
#####################

def circle_dist(a, b):
    return .5-abs(abs(a-b)-.5)


def get_m_tax():
    # Matrix with absolute dist
    m_dist = [[circle_dist(buyer_loc, seller_loc)
        for buyer_loc in A_BUYER_LOC] for seller_loc in a_seller_loc]
    # Matrix with rel dist with list of prices. '+ 1' because min dist is 1.
    m_rel_dist = np.argsort(m_dist, axis=0) + 1.0
    # Lot of potential speed up here, only need to do exponent once.
    m_tax = np.array(m_rel_dist**GAMMA + 0.0)
    return m_tax, m_rel_dist


#######################
### FIND NEXT PRICE ###
#######################

def update_all_prices(m_rel_price, m_tax):
    m_price_perc_other = find_all_price_perc(m_rel_price)
    m_price_to_beat = m_price_perc_other/m_tax

    m_price_to_beat[:, ::-1].sort()
    m_profit = m_price_to_beat * np.arange(1, 1 + m_price_to_beat.shape[1])
    a_best_ind = m_profit.argmax(1)
    a_prices = m_price_to_beat[range(len(a_best_ind)), a_best_ind]
    a_prices = [np.nextafter(price, 0) for price in a_prices]
    return a_prices

def find_all_price_perc(m_rel_price):
    a_a_ret = []
    m_tmp = m_rel_price.T
    for col in m_tmp:
        minVal, minVal2 = get2minFromArray(col)
        a_tmp = [minVal if value != minVal else minVal2 for value in col]
        a_a_ret.append(a_tmp)
    return np.array(a_a_ret).T

def get2minFromArray(array):
    minVal = minVal2 = np.inf
    for value in array:
        if value < minVal:
            minVal2 = minVal
            minVal = value
        elif value < minVal2:
            minVal2 = value
    return minVal, minVal2


########################
### CALCULATE PROFIT ###
########################

def get_profit_from_price(a_seller_price, m_rel_price, m_tax, m_rel_dist):
    # Get vector with each buyer's best rel price
    a_buyers_chosen_seller = m_rel_price.argmin(axis=0)

    # Calculate number of sales for each seller 
    a_profit = [-COST_PER_TIMESTEP] * len(a_seller_price)
    a_num_sales_by_seller = [0] * len(a_seller_price)
    a_num_sales_by_dist = [0] * len(a_seller_price)


    for ind_seller, ind_buyer in zip(a_buyers_chosen_seller, range(len(a_buyers_chosen_seller))):
        a_profit[ind_seller] += a_seller_price[ind_seller]
        a_num_sales_by_seller[ind_seller] += 1
        a_num_sales_by_dist[int(m_rel_dist[ind_seller, ind_buyer]) - 1] += 1
    return a_profit, a_num_sales_by_seller, a_num_sales_by_dist


#####################
### REMOVE SELLER ###
#####################

# has side effects
def remove_seller(ind):
    pass


#############
### WRITE ###
#############

def create_dataframe_from_arrays(a_a_data, a_a_col):
    # preallocate size?, esp. columns
    # loop for adding rows 
    ret = pd.DataFrame()
    for a_data, a_col in zip(a_a_data, a_a_col):
        df_tmp = pd.DataFrame(data = [a_data], columns = a_col)
        ret = pd.concat([ret, df_tmp], ignore_index=True)
    return ret


##################
##################
##### SCRIPT #####
##################
##################

#############
### SETUP ###
#############

# CONSTANTS 
NUM_BUYERS = 100
NUM_SELLERS_START = 20
GAMMA = 1 + (5/16) #get this from command line
A_BUYER_LOC = np.random.rand(NUM_BUYERS) #get this from command line
A_BUYER_LOC.sort()

# Initialize 
a_seller_loc = np.random.rand(NUM_SELLERS_START) #get this from command line
A_SELLER_LOC_START = a_seller_loc.copy()
m_tax, m_rel_dist = get_m_tax()
m_rel_price = np.array(m_tax.copy())
a_seller_price = np.array([1.] * NUM_SELLERS_START)
a_seller_wealth = np.array([00.*COST_PER_TIMESTEP] * NUM_SELLERS_START)
a_map_ind_to_lbl = list(range(NUM_SELLERS_START))
a_a_profit = []
a_a_price = []
a_a_wealth = []
a_a_num_sales_by_seller = []
a_a_num_sales_by_dist = []
a_a_map_ind_to_lbl = []
a_a_num_closest_buyer = []
i_next_label = NUM_SELLERS_START
incrementer = .05
print('start')


############
### LOOP ###
############

for timestep in range(0, NUM_TIMESTEPS):
    print(timestep)

# Calculate seller behavior.
    if (m_tax.min() < 0):
        print(m_tax)
        print(fred)
# Tecnical Debt is here
    for ind_seller in range(0, len(a_seller_price)):
        a_seller_price_tmp = update_all_prices(m_rel_price, m_tax)
        a_seller_price[ind_seller] = a_seller_price_tmp[ind_seller]
    if (a_seller_price.min() <0):
        print(a_seller_price)
        print(frex)
    m_rel_price = m_tax * a_seller_price[:, np.newaxis]
    a_seller_profit, a_num_sales_by_seller, a_num_sales_by_dist = get_profit_from_price(a_seller_price, m_rel_price, m_tax, m_rel_dist)
    a_seller_wealth = a_seller_wealth + a_seller_profit 

# Record seller behavior.
    a_a_price.append(a_seller_price.copy())
    a_a_profit.append(a_seller_profit.copy())
    a_a_wealth.append(a_seller_wealth.copy())
    a_a_num_sales_by_seller.append(a_num_sales_by_seller.copy())
    a_a_num_sales_by_dist.append(a_num_sales_by_dist.copy())
    a_a_num_closest_buyer.append(np.count_nonzero(m_rel_dist == 1, axis = 1))
    a_a_map_ind_to_lbl.append(a_map_ind_to_lbl.copy())

# Add sellers.
    incrementer += RATE_NEW_SELLER
    while incrementer >= 1:
        print(hi)
        incrementer += -1
        loc = find_profitable_loc(a_seller_loc, A_BUYER_LOC)
        exp_profit, best_price = get_exp_profit_and_price(loc, A_BUYER_LOC, a_seller_loc, a_seller_price)
        a_seller_price = np.append(a_seller_price, best_price)
        a_seller_loc = np.append(a_seller_loc, loc)
        a_seller_wealth = np.append(a_seller_wealth, 0.)
        a_map_ind_to_lbl.append(i_next_label)

        i_next_label += 1
        m_tax, m_rel_dist = get_m_tax()
        m_rel_price = m_tax * a_seller_price[:, np.newaxis]

# Remove all sellers with wealth < 0, unless there would be less than two left,
# in which case end simulation
    if (True and any(a_seller_wealth < 0)):
        if len(a_seller_wealth) - np.count_nonzero(a_seller_wealth < 0) < 2:
            print('BRAKE')
            break
        mask_keep = a_seller_wealth >= 0
        # Now remove.
        a_seller_price = a_seller_price[mask_keep]
        a_seller_wealth = a_seller_wealth[mask_keep]
        a_seller_loc = a_seller_loc[mask_keep]
        a_map_ind_to_lbl = [label for label, isKeep in zip(a_map_ind_to_lbl, mask_keep) if isKeep]
        m_tax, m_rel_dist = get_m_tax()
        m_rel_price = m_tax * a_seller_price[:, np.newaxis]


##############
### OUTPUT ###
##############

if not all(NUM_BUYERS == sum(num_sales) for num_sales in a_a_num_sales_by_seller):
    print('num_sales_by_seller is borked')
if not all(NUM_BUYERS == sum(num_sales) for num_sales in a_a_num_sales_by_dist):
    print('num_sales_by_dist is borked')

# Get output objects.
df_price = create_dataframe_from_arrays(a_a_price, a_a_map_ind_to_lbl)
df_profit = create_dataframe_from_arrays(a_a_profit, a_a_map_ind_to_lbl)
df_wealth = create_dataframe_from_arrays(a_a_wealth, a_a_map_ind_to_lbl)
df_num_sales_by_seller = create_dataframe_from_arrays(a_a_num_sales_by_seller, a_a_map_ind_to_lbl)
df_num_closest_buyer = create_dataframe_from_arrays(a_a_num_closest_buyer, a_a_map_ind_to_lbl)
num_col = max(len(arr) for arr in a_a_num_sales_by_dist)
df_num_sales_by_dist = pd.DataFrame(data = a_a_num_sales_by_dist, columns = list(range(1, num_col + 1)))

# Turn into dictionary.
d_write = {'price' : df_price, 'profit' : df_profit, 'wealth' : df_wealth,
'num_sales_by_dist' : df_num_sales_by_dist, 'num_sales_by_seller' : df_num_sales_by_seller,
'num_closest_buyer' : df_num_closest_buyer, 'seller_start_loc' : A_SELLER_LOC_START,
'buyer_start_loc' : A_BUYER_LOC}

# Get output destination filename.
file_path = "C:/Users/CAREBEARSTARE3_USER/Documents/MITInternship/ModelWithSandro/Pickle/"
FILE_NAME_FORMATER = file_path + 'gamma{0:.2f}_buyers{1:}_cost{2:}_tax{3:.2f}_rate{4:2f}.pickle'
file_name_write = FILE_NAME_FORMATER.format(GAMMA, NUM_BUYERS, COST_PER_TIMESTEP, RATE_NEW_SELLER)
print(file_name_write)

# Send output object to destination file.
jl.dump(d_write, file_name_write)

