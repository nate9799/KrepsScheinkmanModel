from matplotlib import pyplot as plt
from sklearn.externals import joblib as jl
import numpy as np
import seaborn as sns
import gambit as gmb
import pandas as pd
from itertools import combinations_with_replacement, product
import sys
import os


#####################
#####################
##### FUNCTIONS #####
#####################
#####################

###############################
### STACKEXCHANGE FUNCTIONS ###
###############################

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


#######################
### SETUP FUNCTIONS ###
#######################

def get_a_strat_quantity(lower_bound, upper_bound, num_strats,
        is_randomized=False):
    if is_randomized:
        return get_rand_strats(lower_bound, upper_bound, num_strats, frac=.05)
    return np.linspace(lower_bound, upper_bound, num_strats)
    

def get_rand_strats(lower_bound, upper_bound, num_strats, frac=.05):
    '''
    Slightly randomizes an array.
    '''
    a_strat_base, step = np.linspace(lower_bound, upper_bound, num_strats, retstep=True)
    scale = frac * step
    a_rand = np.random.uniform(low=-scale, high=+scale, size=len(a_strat_base))
    a_strat_rand = a_strat_base + a_rand
    return(a_strat_rand)

def find_prices_from_quantities(a_quantity, endowment, num_buyers, num_sellers,
        discretization_factor=1):
    if discretization_factor == 1:
        a_tot_quant = np.unique([sum(combo) for combo in
                combinations_with_replacement(a_quantity, num_sellers)])
    else:
# This assumes that a_quantity forms an arithmetic sequence.  
        num_prices = (len(a_quantity) -1) * num_sellers * discretization_factor + 1
        lower = min(a_quantity) * num_sellers
        upper = max(a_quantity) * num_sellers
        a_tot_quant = np.linspace(lower, upper, num_prices)
    ret = np.sort(endowment - a_tot_quant/num_buyers)
    return ret


#####################
### CALCULATE TAX ###
#####################

def circle_dist(a, b):
    '''
    Finds distance if 1==0.
    '''
    return .5-abs(abs(a-b)-.5)

def get_m_tax(a_buyer_loc, a_seller_loc, gamma):
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
    return m_tax


def theoretical_Cournot( S, B, cost, endowment ):
    """
    @Nate, comment here what this function does.
    """

    Q = 0
    P = 0

    return Q, P



###################
### FIND BUNDLE ###
###################

def find_bundle(a_quantity_unsold, a_price_rel, endowment):
    '''
    Finds bundle bought for a specific buyer.
    '''
# lexsort uses second array for primary sort, then uses first column for
# secondary, etc.
    a_quantity_bought = [0.] * len(a_quantity_unsold)
    if sum(a_quantity_unsold) <= 0:
        return a_quantity_bought
    a_quantity_order = np.lexsort((-a_quantity_unsold, a_price_rel))
    tot_quantity = 0.
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

def find_quantity_sold(a_quantity, m_price_rel, endowment):
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
    for a_price_rel in m_price_rel.T:
        a_quantity_bought = find_bundle(a_quantity_unsold, a_price_rel, endowment)
        a_quantity_unsold = a_quantity_unsold - a_quantity_bought
        m_quantity_bought.append(a_quantity_bought)

# Return
    m_quantity_bought = np.array(m_quantity_bought).T
    a_quantity_sold = a_quantity - a_quantity_unsold
    return a_quantity_sold, m_quantity_bought


###################
### FIND PROFIT ###
###################

def find_profit_cournot(a_quantity, cost, endowment, num_buyers=10 ):
    '''
    Finds profit for each seller based on theoretical cournot model, given
    quantities, cost, endowment, and number of buyers.
    '''
    tot_quant = sum(a_quantity)
    price = endowment - (tot_quant/num_buyers)
    a_profit = a_quantity * (price - cost)
    return a_profit, price

def find_profit(a_quantity, a_price, m_tax, cost, endowment, just_profit = True):
    '''
    Finds profit and quantity sold for each seller, given strategies, taxation,
    and basic settings.
    '''
    a_quantity = np.array(a_quantity)
    a_price = np.array(a_price)
    m_price_rel = m_tax * a_price[:, None]#FIXME:
    a_quantity_sold, m_quantity_bought = find_quantity_sold(a_quantity,
            m_price_rel, endowment)
    a_revenue = a_price * a_quantity_sold
    a_cost = cost * a_quantity
    a_profit = a_revenue - a_cost
    if just_profit: return a_profit
    ret = {'a_profit' : a_profit,
           'a_quantity_nash' : a_quantity,
           'm_quantity_bought' : m_quantity_bought,
           'a_price_nash' : a_price,
           'a_quantity_sold' : a_quantity_sold}
    return ret


#####################
### HEATMAP MAKER ###
#####################

def heatmap(a_price_range, a_quantity_range, seller0_price, seller0_quantity, **kwargs):
    '''
    Creates a heat map of strategies a player can take agains another player in
    a 2 player game.
    '''
    heat = [[find_profit(np.array([seller0_quantity, quantity_tmp]),
        np.array([seller0_price, price_tmp]), just_profit=True, **kwargs)[1]
        for quantity_tmp in a_quantity_range] for price_tmp in reversed(a_price_range)]
    print(heat)
    return sns.heatmap(heat, annot=True, fmt = '.3g')

def heatmap_quantity(a_strat_quantity, a_strat_price, **kwargs):
    '''
    Creates a heat map of strategies a player can take agains another player in
    a 2 player game.
    '''
    num_sellers = 2
    a_num_strats = [len(a_strat_quantity)] * num_sellers
    m_payoff0 = np.ones(a_num_strats)
    m_payoff1 = np.ones(a_num_strats)
    m_payoff2 = np.ones(a_num_strats)
    for x, y in np.ndindex(m_payoff0.shape):
        a_quantity = a_strat_quantity[[x,y]]
        a_profit = find_profit_handler(a_quantity, a_strat_price,
                num_sellers=num_sellers, just_profit=True, inner_game=True,
                **kwargs)
        m_payoff0[x,y] = a_profit[0]
        m_payoff1[x,y] = a_profit[1]
        COST              = 100
        ENDOWMENT         = 200
        m_payoff2[x,y], _ = find_profit_cournot(a_quantity, COST, ENDOWMENT)[0]
    #plt = lib.figure()
    #circle_axis = plt.subplot2grid( (1,2), (0,0) )
    #price_axis = plt.subplot2grid( (1,2), (0,1) )
    a_strat_quantity_rnd = np.round(a_strat_quantity, 3)
    pd_payoff0 = pd.DataFrame(data=m_payoff0, index=a_strat_quantity_rnd,
            columns=a_strat_quantity_rnd)
    pd_payoff1 = pd.DataFrame(data=m_payoff1.T, index=a_strat_quantity_rnd,
            columns=a_strat_quantity_rnd)
    ax0 = plt.subplot2grid((1,2), (0,0))
    ax1 = plt.subplot2grid((1,2), (0,1))
    pd_payoff0[pd_payoff0 < -2e-10] = np.NAN
    pd_payoff1[pd_payoff1 < -2e-10] = np.NAN
    sns.heatmap(pd_payoff0, annot=True, fmt = '.3g', ax=ax0)
    sns.heatmap(pd_payoff1, annot=True, fmt = '.3g', ax=ax1)
    plt.show()
    return pd_payoff0

def heatmap_price(a_quantity, a_strat_price, **kwargs):
    '''
    Creates a heat map of strategies a player can take agains another player in
    a 2 player game.
    '''
    num_sellers = 2
    a_num_strats = [len(a_strat_price)] * num_sellers
    m_payoff0 = np.ones(a_num_strats)
    m_payoff1 = np.ones(a_num_strats)
    print(type(m_payoff0))
    for x, y in np.ndindex(m_payoff0.shape):
        a_price = a_strat_price[[x,y]]
        a_profit = find_profit(a_quantity, a_price, just_profit=True, **kwargs)
        m_payoff0[x,y] = a_profit[0]
        m_payoff1[x,y] = a_profit[1]
    #plt = lib.figure()
    #circle_axis = plt.subplot2grid( (1,2), (0,0) )
    #price_axis = plt.subplot2grid( (1,2), (0,1) )
    a_strat_price_rnd = np.round(a_strat_price, 3)
    pd_payoff0 = pd.DataFrame(data=m_payoff0, index=a_strat_price_rnd,
            columns=a_strat_price_rnd)
    pd_payoff1 = pd.DataFrame(data=m_payoff1, index=a_strat_price_rnd,
            columns=a_strat_price_rnd)
    ax0 = plt.subplot2grid((1,2), (0,0) )
    ax1 = plt.subplot2grid((1,2), (0,1) )
    sns.heatmap(pd_payoff0, annot=True, fmt = '.3g', ax=ax0)
    sns.heatmap(pd_payoff1.T, annot=True, fmt = '.3g', ax=ax1)
    plt.show()
    return pd_payoff0


###############################
### HANDLER FOR FIND PROFIT ###
###############################

def find_profit_handler( num_sellers, num_buyers, a_tmp_quant, just_profit=False, inner_game=True, **kwargs):
    '''
    Creates game with strategies from a_strat_quantity, then runs
    find_profit_handler_from_quantity repeatedly to find payoffs. Finally,
    calculates nash for quantity, then returns the payoff. 
    '''

# Gambit stuff
    game = make_game_table(  num_sellers, num_buyers, a_tmp_quant, inner_game=inner_game, **kwargs)
    profile = find_profile_best_nash_from_game(game, num_sellers)

# Handle no soutions
    if len(profile) == 0 & just_profit:
        return np.ones(num_sellers) * -sys.maxint
    if len(profile) == 0:
        raise Exception("No pure Nash found: {}".format(profile))
    if just_profit:
        return get_a_payoff_from_profile(game, profile, num_sellers)
    if inner_game:
        a_quantity = a_tmp_quant
        a_strat_price = make_a_strat_price(num_buyers, a_quantity, **kwargs)
        a_price = a_strat_price[profile]
        return find_profit(a_quantity, a_price, just_profit = False, **kwargs)

    a_strat_quant = a_tmp_quant
    a_quantity = a_strat_quant[profile]
    ret =  find_profit_handler( num_sellers, num_buyers, a_quantity, just_profit=False, inner_game=True, **kwargs )

    return ret


#########################
### CREATE GAME TABLE ###
#########################

def make_a_strat_price( num_buyers, a_quantity, cost, endowment, m_tax ):
    """
    bla bla bla
    """

    _, price           = find_profit_cournot(a_quantity, cost, endowment, num_buyers=num_buyers )
    n                  = 10 # number of price-strategies
    dist               = 20 # deviation from cournot that we search for
    a_strat_price      = np.linspace( price - dist, price + dist, n )

    return a_strat_price


def make_game_table( num_sellers, num_buyers, a_tmp_quant, inner_game=True, **kwargs ):
    '''
    Creates gambit game table for predetermined quantities and a list of prices for strategies.
    @Nate, explain here for instance that a_tmp_quant can be a fixed nr or a list of quantities etc.
    '''

    if inner_game:
        a_quantity         = a_tmp_quant
        a_strat_price      = make_a_strat_price( num_buyers, a_quantity, **kwargs )
        a_num_strats       = [len(a_strat_price)] * num_sellers
    else:
        a_strat_quantity = a_tmp_quant
        a_num_strats = [len(a_strat_quantity)] * num_sellers
    ret = gmb.Game.new_table(a_num_strats)
    for profile in ret.contingencies:
        if inner_game:
            a_price = a_strat_price[profile]
            a_profit = find_profit(a_quantity, a_price, just_profit=True,
                    **kwargs)
        else:
            a_quantity = a_strat_quantity[profile]
            a_profit = find_profit_handler(num_sellers, num_buyers, a_quantity,
                    just_profit=True, inner_game=True,
                    **kwargs)
        for ind in range(num_sellers):
            float(a_profit[ind])
            ret[profile][ind] = int(a_profit[ind])
    return ret


###########################
### GAME TABLE ANALYSIS ###
###########################

def find_a_profile_nash_strat_from_game(game, num_players):
    '''
    Finds array of pure nash profiles from a game where all players have an
    EQUAL NUMBER OF STRATEGIES.  A profile is a np.array where the ith element
    is the index of the strategy chosen by the ith player.
    '''
    solver = gmb.nash.ExternalEnumPureSolver()
    a_nash = solver.solve(game)
# Handle Exceptions
    if len(a_nash) == 0:
        #raise Exception("No pure Nash found: {}".format(a_nash))
        return np.array([])
    num_nash = len(a_nash)
    a_nash = np.array(a_nash)
    a_nash = a_nash.reshape((num_nash, num_players, a_nash.size/(num_nash * num_players)))
    ret = np.array([np.nonzero(nash)[1] for nash in a_nash])
    return ret

def find_profile_best_nash_from_game(game, num_players):
    '''
    Finds array of pure nash profiles from a game where all players have an
    EQUAL NUMBER OF STRATEGIES.  A profile is a np.array where the ith element
    is the index of the strategy chosen by the ith player.
    '''
    a_profile = find_a_profile_nash_strat_from_game(game, num_players)
    num_nash = len(a_profile)
    if num_nash == 0: 
        #raise Exception("No pure Nash found: {}".format(a_profile))
        return np.array([])
    if num_nash == 1: return a_profile[0]
    profit = -sys.maxint
    for profile in a_profile:
        profit_tmp = sum(get_a_payoff_from_profile(game, profile, num_players))
        if profit_tmp > profit:
            ret = profile
            profit = profit_tmp
    return ret

def get_a_payoff_from_profile(game, profile, num_players):
    '''
    Finds array of payoffs for a given profile from a game.  Does NOT require
    all players have same number of strategies.
    '''
    return [game[profile][player] for player in range(num_players)]


#####################
### CREATE PICLKE ###
#####################

def make_dic_of_pure_nash( num_sellers, num_buyers, a_strat_quantity, a_seller_loc, a_buyer_loc, cost, gamma, endowment ):
    '''
    Creates tax matrix, then the  game table for gambit, then finds pure nash
    solution(s), then makes a dictionary for pickle
    '''

# Setup
    m_tax = get_m_tax(a_buyer_loc, a_seller_loc, gamma)
    kwargs = {'m_tax' : m_tax, 'cost' : cost, 'endowment' : endowment}

# Create and Solve Game.
    d_nash = find_profit_handler( num_sellers, num_buyers, a_strat_quantity, just_profit=False, inner_game=False, **kwargs)

# Create dic to return. Note Gambit forces the use of Python 2, hence 'update'.
    ret = {'gamma' :        gamma,
           'a_buyer_loc' :  a_buyer_loc,
           'a_seller_loc' : a_seller_loc,
           'num_buyers' :   num_buyers,
           'num_sellers' :  num_sellers }
    ret.update(kwargs)
    ret.update(d_nash)

    return ret


##################
##################
##### SCRIPT #####
##################
##################

def main( num_sellers=2, num_buyers=6, gamma=0, cost=100, endowment=None, randomize=False ):
    """ Add documentation here """

# check that input is correct
    if endowment is None: endowment = 2*cost
    assert num_buyers%num_sellers == 0, "number of sellers does not divide number of buyers"


# setup buyer and seller locations
    if randomize:

        a_seller_loc = np.random.uniform( low=0, high=1, size=num_sellers )
        a_buyer_loc  = np.random.uniform( low=0, high=1, size=num_buyers )

    else:

        a_seller_loc = np.linspace( start=0, stop=1, num=num_sellers, endpoint=False )
        a_buyer_loc  = np.linspace( start=0, stop=1, num=num_buyers,  endpoint=False )


# set the quantity discretization and calculate Nash
    n                   = 11 # number of quantity-strategies
    dist                = 20 # deviation from cournot that we search for
    q, p                = theoretical_Cournot( num_sellers, num_buyers, cost, endowment )
    a_strat_quantity    = np.linspace( q-dist, q+dist, n )

    d_write = make_dic_of_pure_nash( num_sellers, num_buyers, a_strat_quantity, a_seller_loc, a_buyer_loc, cost, gamma, endowment )

# write to the output
    folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
    folder  = folder1 if os.path.exists(folder1) else folder2
    fn      = 'S=%s_B=%s_gamma=%s_cost=%s_endow=%s=randomize=%s.pkl'%(num_sellers, num_buyers, gamma, cost, endowment, randomize)
    jl.dump(d_write, folder + fn )


def parameter_combination( i ):
    """
    Execute the i-th parameter combination.
    """

    num_sellers     = [ 2, 4  ]
    num_buyers      = [ 4, 12 ]
    cost            = [ 100 ]
    gamma           = [ 0, 0.5 ]
    endowment       = [ 200 ]
    randomize       = [ True, False ]

    combs           = product(num_sellers, num_buyers, cost, gamma, endowment, randomize)
    comb            = combs[i]

    num_sellers, num_buyers, cost, gamma, endowment, randomize = comb

    main( num_sellers, num_buyers, gamma, cost, endowment, randomize )



if __name__ == "__main__":

    i = int(sys.argv[1]) - 1

    parameter_combination( i )

