from matplotlib import pyplot as plt
from sklearn.externals import joblib as jl
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
a_buyer_loc = [.2, .2, .2, .2, .2, .7, .7, .7, .7, .7]
SELLER_PRICE = 1
SELLER_QUANTITY = 5

A_SELLER_PRICES = np.array([15,14])
A_SELLER_QUANTITIES = np.arange(40,60,10)
FILE_OUT = '/home/nate/Desktop/here.pickle'
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

def get_m_tax_and_m_dist(a_buyer_loc, a_seller_loc, gamma=GAMMA):
    # Matrix with absolute dist
    m_dist = [[circle_dist(buyer_loc, seller_loc)
        for buyer_loc in a_buyer_loc] for seller_loc in a_seller_loc]
    # Matrix with rel dist with list of prices. '+ 1' because min dist is 1.
    m_rel_dist = np.argsort(m_dist, axis=0) + 1
    # Lot of potential speed up here, only need to do exponent once.
    m_tax = np.array(m_rel_dist**gamma, dtype=int)
    return m_tax, np.array(m_dist)


###################
### FIND BUNDLE ###
###################

def find_bundle(a_quantity_unsold, a_price_rel, a_dist, endowment = ENDOWMENT):
# lexsort uses second array for primary sort, then uses first column for
# secondary, etc.
    a_quantity_order = np.lexsort((a_dist, a_price_rel))
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

def find_quantity_sold(a_quantity, m_price_rel, m_dist):
    num_buyers = np.size(m_price_rel, 1)
    num_sellers = np.size(m_price_rel, 0)
    a_quantity_unsold = a_quantity.copy()
    for a_price_rel, a_dist in zip(m_price_rel.T, m_dist.T):
        a_quantity_bought = find_bundle(a_quantity_unsold, a_price_rel, a_dist)
        a_quantity_unsold = a_quantity_unsold - a_quantity_bought
        if sum(a_quantity_unsold) <= 0:
            return a_quantity
    a_quantity_sold = a_quantity - a_quantity_unsold
    return a_quantity_sold


###################
### FIND PROFIT ###
###################

def find_profit(a_quantity, a_price, m_tax, m_dist, cost):
    m_price_rel = m_tax * a_price[:, None]
    a_quantity_sold = find_quantity_sold(a_quantity, m_price_rel, m_dist)
    a_revenue = a_price * a_quantity_sold
    a_cost = cost * a_quantity
    a_profit = a_revenue - a_cost
    return a_profit, a_quantity_sold

def find_profit_2(seller0_price, seller0_quantity, seller1_price, seller1_quantity,
        **kwargs):
    a_profit = find_profit(np.array([seller0_quantity, seller1_quantity]),
            np.array([seller0_price, seller1_price]), **kwargs)
    return a_profit[1]

def heatmap(a_price_range, a_quantity_range, seller0_price, seller0_quantity, **kwargs):
    heat = [[find_profit_2(seller0_price, seller0_quantity, price_tmp,
        quantity_tmp, **kwargs) for quantity_tmp in a_quantity_range]
        for price_tmp in reversed(a_price_range)]
    return(sns.heatmap(heat, annot=True, fmt = '.3g'))


#########################
### CREATE GAME TABLE ###
#########################

def arrayInd_to_array2Ind(a_ind, num_cols, num_rows):
    a_ind_col = a_ind % num_cols
    a_ind_row = a_ind // num_cols
    return a_ind_col, a_ind_row

def find_profit_handler(a_ind_game_table, a_strat_quantity, a_strat_price,
        **kwargs):
    a_ind_quantity, a_ind_price = arrayInd_to_array2Ind(np.array(a_ind_game_table),
            len(a_strat_quantity), len(a_strat_price))
    a_quantity = np.array([a_strat_quantity[ind] for ind in a_ind_quantity])
    a_price = np.array([a_strat_price[ind] for ind in a_ind_price])
    ret, ignore = find_profit(a_quantity, a_price, **kwargs)
    return ret

def make_game_table(a_strat_quantity, a_strat_price, num_sellers, **kwargs):
    print(num_sellers)
    a_num_strats = [len(a_strat_quantity) * len(a_strat_price)] * num_sellers
    print(a_num_strats)
    ret = gmb.Game.new_table(a_num_strats)
    for profile in ret.contingencies:
        a_profit = find_profit_handler(profile, a_strat_quantity,
                a_strat_price, **kwargs)
        for ind in range(num_sellers):
            ret[profile][ind] = int(a_profit[ind])
    return ret


#####################
### CREATE PICLKE ###
#####################

def make_dic_of_pure_nash(a_strat_quantity, a_strat_price, a_buyer_loc,
        a_seller_loc, cost, gamma):
# Setup
    m_tax, m_dist = get_m_tax_and_m_dist(a_buyer_loc, a_seller_loc, gamma)
    num_sellers = len(a_seller_loc)
    num_strategies = len(a_strat_quantity) * len(a_strat_price)
    kwargs = {'m_tax' : m_tax, 'm_dist' : m_dist, 'cost' : cost}
# Create and Solve Game.
    game = make_game_table(a_strat_quantity, a_strat_price, num_sellers, **kwargs)
    solver = gmb.nash.ExternalEnumPureSolver()
    nash = solver.solve(game)
    m_strat_nash = np.array(nash[0]).reshape((num_sellers, num_strategies))
# Check that there is only one nash equilibrium, and it is pure.  
    a_a_tmp = np.nonzero(m_strat_nash == 1)
    if len(a_a_tmp[0]) != num_sellers:
        raise Exception('Problem with  Nash Equilibrium : No pure Nash Equilibrium found or more than one found. Nash Data = {}'.format(nash))
    if any(a_a_tmp[0] != range(num_sellers)):
        raise Exception('Problem with Nash Equilibrium : No pure Nash Equilibrium found or more than one found. Nash Data = {}'.format(nash))
# Find the strategies chosen.  
    a_ind_strat_nash = a_a_tmp[1]
# Find output values
    a_ind_quantity, a_ind_price = arrayInd_to_array2Ind(np.array(a_ind_strat_nash),
            len(a_strat_quantity), len(a_strat_price))
    a_quantity = np.array([a_strat_quantity[ind] for ind in a_ind_quantity])
    a_price = np.array([a_strat_price[ind] for ind in a_ind_price])
    a_profit, a_quantity_sold = find_profit(a_quantity, a_price, **kwargs)
# Create dic
    ret = {'a_profit' : a_profit,
           'gamma' : gamma,
           'a_quantity_nash' : a_quantity,
           'a_price_nash' : a_price,
           'a_quantity_sold' : a_quantity_sold,
           'a_buyer_loc' : a_buyer_loc,
           'a_seller_loc' : a_seller_loc}
    ret.update(kwargs)
    print(ret)
    return ret

##################
##################
##### SCRIPT #####
##################
##################


#plt.figure(figsize=(20, 20))
#ax_heat = heatmap(A_SELLER_PRICES, A_SELLER_QUANTITIES, SELLER_PRICE, SELLER_QUANTITY,
#                  m_tax = m_tax, m_dist = m_dist, cost = COST)
#ax_heat.set(ylabel = 'price', xlabel = 'quantity')
#ax_heat.set_yticklabels(reversed(A_SELLER_PRICES), rotation=0)
#ax_heat.set_xticklabels(A_SELLER_QUANTITIES)
#plt.show()

d_write = make_dic_of_pure_nash(A_SELLER_QUANTITIES, A_SELLER_PRICES,
        a_buyer_loc, a_seller_loc, COST, GAMMA)

jl.dump(d_write, FILE_OUT)

