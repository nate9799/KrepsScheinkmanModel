from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import gambit as gmb


# CONSTANTS
## Global Constants
NUM_BUYERS = 100
NUM_SELLERS = 2
COST = 10
GAMMA = 0
SLOPE = 1
ENDOWMENT = 20

## Setup Constants
a_seller_loc = [.2, .7]
a_buyer_loc = [2, 2, 2, 2, 2, 7, 7, 7, 7, 7]
SELLER_PRICE = 1
SELLER_QUANTITY = 5

A_SELLER_PRICES = np.arange(10,21,1)
A_SELLER_QUANTITIES = np.arange(0,80,5)
#A_BUYER_LOC = np.random.rand(NUM_BUYERS)
#A_SELLER_LOC = np.random.rand(NUM_SELLERS)


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

def get_m_tax(a_buyer_loc, a_seller_loc, gamma=GAMMA):
    # Matrix with absolute dist
    m_dist = [[circle_dist(buyer_loc, seller_loc)
        for buyer_loc in a_buyer_loc] for seller_loc in a_seller_loc]
    # Matrix with rel dist with list of prices. '+ 1' because min dist is 1.
    m_rel_dist = np.argsort(m_dist, axis=0) + 1
    # Lot of potential speed up here, only need to do exponent once.
    m_tax = np.array(m_rel_dist**gamma, dtype=int)
    return m_tax


###################
### FIND BUNDLE ###
###################

def find_bundle(a_quantity_unsold, a_price_rel, endowment = ENDOWMENT):
    a_quantity_order = np.argsort(a_price_rel)
    tot_quantity = 0
    a_quantity_bought = [0.] * len(a_quantity_unsold)
    for ind in a_quantity_order:
        tot_quantity = sum(a_quantity_bought)
        demand_remaining = endowment - tot_quantity - a_price_rel[ind]
        if demand_remaining <= 0:
            return a_quantity_bought
        elif demand_remaining <= a_quantity_unsold[ind]:
            a_quantity_bought[ind] = demand_remaining
            return a_quantity_bought
        else:
            a_quantity_bought[ind] = a_quantity_unsold[ind]
    return a_quantity_bought


##########################
### FIND QUANTITY SOLD ###
##########################

def find_quantity_sold(a_quantity, m_price_rel):
    num_buyers = np.size(m_price_rel, 1)
    num_sellers = np.size(m_price_rel, 0)
    ret = np.array([[0]*num_buyers] * num_sellers)
    a_quantity_unsold = a_quantity
    for col in m_price_rel.T:
        a_quantity_bought = find_bundle(a_quantity_unsold, col)
        a_quantity_unsold = a_quantity_unsold - a_quantity_bought
        if sum(a_quantity_unsold) <= 0:
            return a_quantity
    a_quantity_sold = a_quantity - a_quantity_unsold
    return a_quantity_sold


###################
### FIND PROFIT ###
###################

def find_profit(a_quantity, a_price, m_tax, cost):
    m_price_rel = m_tax * a_price[:, None]
    a_quantity_sold = find_quantity_sold(a_quantity, m_price_rel)
    a_revenue = a_price * a_quantity_sold
    a_cost = cost * a_quantity
    a_profit = a_revenue - a_cost
    return a_profit

def find_profit_2(seller0_price, seller0_quantity, seller1_price, seller1_quantity, m_tax, cost):
    a_profit = find_profit(np.array([seller0_quantity, seller1_quantity]),
            np.array([seller0_price, seller1_price]), m_tax, cost)
    return a_profit[1]

def heatmap(a_price_range, a_quantity_range, seller0_price, seller0_quantity, m_tax, cost):
    heat = [[find_profit_2(seller0_price, seller0_quantity, price_tmp,
        quantity_tmp, m_tax, cost) for quantity_tmp in a_quantity_range]
        for price_tmp in reversed(a_price_range)]
    return(sns.heatmap(heat, annot=True, fmt = '.3g'))


#########################
### CREATE GAME TABLE ###
#########################

def arrayInd_to_array2Ind(a_ind, num_cols, num_rows):
    a_ind_col = a_ind % num_cols
    a_ind_row = a_ind // num_cols
    return a_ind_col, a_ind_row

def make_game_table_value(a_ind_game_table, a_strat_quantity, a_strat_price, m_tax, cost):
    a_ind_quantity, a_ind_price = arrayInd_to_array2Ind(np.array(a_ind_game_table),
            len(a_strat_quantity), len(a_strat_price))
    a_quantity = np.array([a_strat_quantity[ind] for ind in a_ind_quantity])
    a_price = np.array([a_strat_price[ind] for ind in a_ind_price])
    ret = find_profit(a_quantity, a_price, m_tax, cost)
    return ret

def make_game_table(a_strat_quantity, a_strat_price, m_tax, cost, num_sellers):
    print(num_sellers)
    a_num_strats = [len(a_strat_quantity) * len(a_strat_price)] * num_sellers
    ret = gmb.Game.new_table(a_num_strats)
    for profile in ret.contingencies:
        a_profit = make_game_table_value(profile, a_strat_quantity, a_strat_price, m_tax, cost)
        for ind in range(num_sellers):
            ret[profile][ind] = int(a_profit[ind])
    return ret


##################
##################
##### SCRIPT #####
##################
##################

plt.figure(figsize=(20, 20))
m_tax = get_m_tax(a_buyer_loc, a_seller_loc, GAMMA)

ax_heat = heatmap(A_SELLER_PRICES, A_SELLER_QUANTITIES, SELLER_PRICE, SELLER_QUANTITY, m_tax, COST)

ax_heat.set(ylabel = 'price', xlabel = 'quantity')
ax_heat.set_yticklabels(reversed(A_SELLER_PRICES), rotation=0)
ax_heat.set_xticklabels(A_SELLER_QUANTITIES)
#plt.show()

make_game_table(A_SELLER_QUANTITIES, A_SELLER_PRICES, m_tax, COST, 2)

