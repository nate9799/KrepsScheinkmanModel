from matplotlib import pyplot as plt
from sklearn.externals import joblib as jl
import numpy as np
import seaborn as sns
import gambit as gmb


#####################
#####################
##### FUNCTIONS #####
#####################
#####################

#######################
### INNER FUNCTIONS ###
#######################

def arrayInd_to_array2Ind(a_ind, num_cols, num_rows):
    '''
    Sort of deflattens an array.
    '''
    a_ind_col = a_ind % num_cols
    a_ind_row = a_ind // num_cols
    return a_ind_col, a_ind_row

def get_rand_strats(lower_bound, upper_bound, num_strats, frac=.05):
    a_strat_base, step = np.linspace(lower_bound, upper_bound, num_strats, retstep=True)
    scale = frac * step
    a_rand = np.random.uniform(low=-scale, high=+scale, size=len(a_strat_base))
    a_strat_rand = a_strat_base + a_rand
    return(a_strat_rand)


#####################
### CALCULATE TAX ###
#####################

def circle_dist(a, b):
    '''
    Finds distance if 1==0.
    '''
    return .5-abs(abs(a-b)-.5)

def get_m_tax_and_m_dist(a_buyer_loc, a_seller_loc, gamma):
    '''
    Calculates tax matrix and distance matrix.
    '''
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

def find_bundle(a_quantity_unsold, a_price_rel, a_dist, endowment):
    '''
    Finds bundle bought for a specific buyer.
    '''
# lexsort uses second array for primary sort, then uses first column for
# secondary, etc.
    a_quantity_bought = [0.] * len(a_quantity_unsold)
    if sum(a_quantity_unsold) <= 0:
        return a_quantity_bought
    a_quantity_order = np.lexsort((a_dist, a_price_rel))
    tot_quantity = 0
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

def find_quantity_sold(a_quantity, m_price_rel, m_dist, endowment):
    '''
    Finds quantity sold for each seller, given strategies, taxation,
    and basic settings.
    '''
# Setup
    num_buyers = np.size(m_price_rel, 1)
    num_sellers = np.size(m_price_rel, 0)
    a_quantity_unsold = a_quantity.copy()
    m_quantity_bought = []

# Loop
    for a_price_rel, a_dist in zip(m_price_rel.T, m_dist.T):
        a_quantity_bought = find_bundle(a_quantity_unsold, a_price_rel, a_dist, endowment)
        a_quantity_unsold = a_quantity_unsold - a_quantity_bought
        m_quantity_bought.append(a_quantity_bought)

# Return
    m_quantity_bought = np.array(m_quantity_bought).T
    a_quantity_sold = a_quantity - a_quantity_unsold
    return a_quantity_sold, m_quantity_bought


###################
### FIND PROFIT ###
###################

def find_profit(a_quantity, a_price, m_tax, m_dist, cost, endowment, just_profit = False):
    '''
    Finds profit and quantity sold for each seller, given strategies, taxation,
    and basic settings.
    '''
    m_price_rel = m_tax * a_price[:, None]
    a_quantity_sold, m_quantity_bought = find_quantity_sold(a_quantity,
            m_price_rel, m_dist, endowment)
    a_revenue = a_price * a_quantity_sold
    a_cost = cost * a_quantity
    print(a_quantity)
    a_profit = a_revenue - a_cost
    if just_profit: return a_profit
    return a_profit, a_quantity_sold, m_quantity_bought


#####################
### HEATMAP MAKER ###
#####################

def heatmap(a_price_range, a_quantity_range, seller0_price, seller0_quantity, **kwargs):
    '''
    Creates a heat map of strategies a player can take agains another player in
    a 2 player game.
    '''
    heat = [[find_profit(np.array([seller0_quantity, quantity_tmp]),
        np.array([seller0_price, price_tmp]), just_profit=True, **kwargs)[0]
        for quantity_tmp in a_quantity_range] for price_tmp in reversed(a_price_range)]
    print(heat)
    return(sns.heatmap(heat, annot=True, fmt = '.3g'))


###############################
### HANDLER FOR FIND PROFIT ###
###############################

def find_profit_handler_ind(a_ind_game_table, a_strat_quantity, a_strat_price,
        **kwargs):
    '''
    Runs find_profit over a game, and finds the nash equilibrium.
    '''
    a_ind_quantity, a_ind_price = arrayInd_to_array2Ind(np.array(a_ind_game_table),
            len(a_strat_quantity), len(a_strat_price))
    a_quantity = np.array([a_strat_quantity[ind] for ind in a_ind_quantity])
    a_price = np.array([a_strat_price[ind] for ind in a_ind_price])
    a_profit, a_quantity_sold, m_quantity_bought = find_profit(a_quantity, a_price, **kwargs)
    ret = {'a_profit' : a_profit,
           'a_quantity_nash' : a_quantity,
           'm_quantity_bought' : m_quantity_bought,
           'a_price_nash' : a_price,
           'a_quantity_sold' : a_quantity_sold}
    return ret

def find_profit_handler_nash(nash, a_strat_quantity, a_strat_price,
        num_sellers, **kwargs):
    '''
    Runs find_profit over a game, and finds the nash equilibrium.
    '''
    print(nash)
    if len(nash) == 0:
        raise Exception('Problem with  Nash Equilibrium : No pure Nash Equilibrium found.  Nash Data = {}'.format(nash))
    num_strategies = len(a_strat_quantity) * len(a_strat_price)
    a_m_strat_nash = np.array(nash[0]).reshape((num_sellers, num_strategies))
    if  len(nash) == 1:
        a_a_tmp = np.nonzero(a_m_strat_nash == 1)
        a_ind_strat_nash = a_a_tmp[1]
        ret = find_profit_handler_ind(a_ind_strat_nash, a_strat_quantity,
                a_strat_price, **kwargs)
        return ret
# Setup for loop
    ret = {}
    tot_profit = np.NINF
    for m_strat_nash in a_m_strat_nash:
# Find the strategies chosen.  
        a_a_tmp = np.nonzero(m_strat_nash == 1)
        a_ind_strat_nash = a_a_tmp[1]
# Find output values
        d_tmp = find_profit_handler_ind(a_ind_strat_nash, a_strat_quantity,
                a_strat_price, **kwargs)
        profit_tmp = sum(d_tmp['a_profit'])
# Check if output is optimal
        if profit_tmp > profit:
            tot_profit = tot_profit_tmp
            ret = d_tmp
    return ret


#########################
### CREATE GAME TABLE ###
#########################

def make_game_table(a_strat_quantity, a_strat_price, num_sellers, **kwargs):
    '''
    Creates gambit game table using a list of strategies generated from a list of
    possible quantities and possible prices.
    '''
    print(num_sellers)
    a_num_strats = [len(a_strat_quantity) * len(a_strat_price)] * num_sellers
    print(a_num_strats)
    ret = gmb.Game.new_table(a_num_strats)
    for profile in ret.contingencies:
        d_tmp = find_profit_handler_ind(profile, a_strat_quantity,
                a_strat_price, **kwargs)
        a_profit = d_tmp['a_profit']
        for ind in range(num_sellers):
            ret[profile][ind] = int(a_profit[ind])
    return ret


#####################
### CREATE PICLKE ###
#####################

def make_dic_of_pure_nash(a_strat_quantity, a_strat_price, a_buyer_loc,
        a_seller_loc, cost, gamma, endowment):
    '''
    Creates tax matrix, then the  game table for gambit, then finds pure nash
    solution(s), then makes a dictionary for pickle
    '''
# Setup
    m_tax, m_dist = get_m_tax_and_m_dist(a_buyer_loc, a_seller_loc, gamma)
    num_sellers = len(a_seller_loc)
    num_buyers = len(a_buyer_loc)
    kwargs = {'m_tax' : m_tax, 'm_dist' : m_dist, 'cost' : cost, 'endowment' : endowment}

# Create and Solve Game.
    game = make_game_table(a_strat_quantity, a_strat_price, num_sellers, **kwargs)
    solver = gmb.nash.ExternalEnumPureSolver()
    nash = solver.solve(game)
    d_nash = find_profit_handler_nash(nash, a_strat_quantity, a_strat_price,
            num_sellers, **kwargs)

# Create dic to return
    ret = {'gamma' : gamma,
           'a_buyer_loc' : a_buyer_loc,
           'a_seller_loc' : a_seller_loc,
           'num_buyers' : num_buyers,
           'num_sellers' : num_sellers}
    ret.update(kwargs)
    ret.update(d_nash)
    print(ret)
    return ret


##################
##################
##### SCRIPT #####
##################
##################

def main():
# CONSTANTS
## Global Constants
    COST = 10
    GAMMA = 0
    ENDOWMENT = 20

## Setup Constants
    a_seller_loc = [.2, .7]
    a_buyer_loc = [.2, .2, .2, .2, .2, .7, .7, .7, .7, .7]
    SELLER_PRICE = 1
    SELLER_QUANTITY = 5
    A_SELLER_PRICES = get_rand_strats(COST, 15, 16)
    A_SELLER_QUANTITIES = get_rand_strats(30, 35, 16)
################## TEST ###################
    m_tax, m_dist = get_m_tax_and_m_dist(a_buyer_loc, a_seller_loc, GAMMA)
    num_sellers = len(a_seller_loc)
    num_buyers = len(a_buyer_loc)
    kwargs = {'m_tax' : m_tax, 'm_dist' : m_dist, 'cost' : COST, 'endowment' : ENDOWMENT}
    heatmap(A_SELLER_PRICES, A_SELLER_PRICES, 13.33333, 33.33333, **kwargs)
    print(A_SELLER_PRICES)
    print(A_SELLER_QUANTITIES)
    plt.show()
################## TEST ###################

    #d_write = make_dic_of_pure_nash(A_SELLER_QUANTITIES, A_SELLER_PRICES,
    #        a_buyer_loc, a_seller_loc, COST, GAMMA, ENDOWMENT)

    # './' doesn't seem to work for some reason.
    out_folder2 = '/home/nate/Documents/abmcournotmodel/code/'
    out_folder = out_folder2 + 'output/data/'
    file_name = 'here.pickle'
    # Need to install os
    #if not os.path.exists(out_folder): os.makedirs(out_folder)
    file_out = out_folder + file_name

    #jl.dump(d_write, file_out)

if __name__ == "__main__":
    main()

