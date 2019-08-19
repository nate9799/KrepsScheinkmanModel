from sklearn.externals import joblib as jl
import numpy as np
import gambit as gmb
from itertools import combinations_with_replacement, product
import sys
import os
import warnings

from pdb import set_trace


#####################
#####################
##### FUNCTIONS #####
#####################
#####################


#######################
### SETUP FUNCTIONS ###
#######################


def get_a_cost_from_ratio(mean, ratio, num_sellers, is_arithmetic=True):
    '''
    Creates a sequence with a given mean using a ratio. Sequence can be
    arithmetic or geometric.
    '''
    a_ratio = [ratio ** i for i in range(num_sellers)]
    if is_arithmetic:
        a_ratio = [(ratio-1) * i + 1 for i in range(num_sellers)]
    return get_a_cost_from_a_ratio(mean, a_ratio)

def get_a_cost_from_a_ratio(mean, a_ratio):
    assert a_ratio[0] == 1, "a_ratio[0] should be 1, is {}".format(a_ratio[0])
    ret = np.array(a_ratio)
    avg_ratio = sum(a_ratio)/float(len(a_ratio))
    ret = ret * mean / avg_ratio
    return ret

def get_a_strat_from_center(n, center, jump, shift = 0):
    dist = float(n * jump)/2
    ret = np.linspace(center - dist, center + dist, n) + shift
    return(ret)


#####################
### CALCULATE TAX ###
#####################

def circle_dist(a, b):
    '''
    Finds distance if 1==0.
    '''
    return .5-abs(abs(a-b)-.5)

def get_m_tax_dist(a_buyer_loc, a_seller_loc, gamma, scalar_tax):
    '''
    Calculates tax matrix and distance matrix.
    '''
    # Matrix with absolute dist
    m_dist = [[circle_dist(buyer_loc, seller_loc)
        for buyer_loc in a_buyer_loc] for seller_loc in a_seller_loc]
    # Matrix with rel dist with list of prices. '+ 1' because min dist is 1.
    # Lot of potential speed up here, only need to do exponent once.
    m_tax = scalar_tax * (np.array(m_dist)**gamma) + 1.
    print(m_tax)
    return m_tax

def get_m_tax_ordinal(a_buyer_loc, a_seller_loc, gamma, scalar_tax):
    '''
    Calculates tax matrix and distance matrix.
    '''
    # Matrix with absolute dist
    m_dist = [[circle_dist(buyer_loc, seller_loc)
        for buyer_loc in a_buyer_loc] for seller_loc in a_seller_loc]
    # Matrix with rel dist with list of prices. '+ 1' because min dist is 1.
    m_rel_dist = np.argsort(m_dist, axis=0) + 1.
    # Lot of potential speed up here, only need to do exponent once.
    m_tax = scalar_tax * (np.array(m_rel_dist)**gamma - 1.) + 1.
    print(m_tax)
    return m_tax

# FIXME: handle a_cost
def theoretical_Cournot(num_sellers, num_buyers, cost, endowment):
    """
    Calculates theoretical Cournot for S sellers and B buyers at given price and endowment/buyer.
    """
    num_sellers = float(num_sellers)
    num_buyers = float(num_buyers)
    Q = float(num_buyers * (endowment - cost)/(num_sellers + 1))
    P = float(endowment - (Q/num_buyers))
    return Q, P

def theoretical_Cournots_from_a_cost(num_sellers, num_buyers, a_cost, endowment):
    """
    Calculates theoretical Cournot for S sellers and B buyers at given price and endowment/buyer.
    """
    assert len(a_cost) == num_sellers, "len(a_cost) != num_sellers. is {}, should be {}".format(len(a_cost), num_sellers)
    num_sellers = float(num_sellers)
    num_buyers = float(num_buyers)
    avg_cost = sum(a_cost)/num_sellers
    a_quantities = (num_buyers/(num_sellers + 1)) * (endowment + num_sellers * (avg_cost - a_cost) - a_cost)
    price = float(endowment - (sum(a_quantities)/num_buyers))
    return a_quantities, price

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

def find_profit_cournot(num_buyers, a_quantity, m_tax, a_cost, endowment):
    '''
    Finds profit for each seller based on theoretical cournot model, given
    quantities, cost, endowment, and number of buyers.
    '''
    tot_quant = sum(a_quantity)
    price = endowment - (tot_quant/num_buyers)
    a_profit = a_quantity * (price - a_cost)
    return a_profit, price

def find_profit(a_quantity, a_price, m_tax, a_cost, endowment, just_profit = True):
    '''
    Finds profit and quantity sold for each seller, given strategies, taxation,
    and basic settings.
    '''
    a_quantity = np.array(a_quantity)
    a_price = np.array(a_price)
    m_price_rel = m_tax * a_price[:, None]
    a_quantity_sold, m_quantity_bought = find_quantity_sold(a_quantity,
            m_price_rel, endowment)
    a_revenue = a_price * a_quantity_sold
    a_cost_tot = a_cost * a_quantity
    a_profit = a_revenue - a_cost_tot
    if just_profit: return a_profit
    ret = {'a_profit' : a_profit,
           'a_quantity_nash' : a_quantity,
           'm_quantity_bought' : m_quantity_bought,
           'a_price_nash' : a_price,
           'a_quantity_sold' : a_quantity_sold}
    return ret

def find_payoff_of_mixed_strategy(game, a_nash):
# Get a_strat and probability of a_strat
    num_nash = len(a_nash)
    if num_nash == 0:
        raise Exception
    num_players = len(a_nash[0])
    a_payoff = np.zero(num_players)
    for profile in game:
        prob = 1
# Check how enumerate works
        for strat_ind, player_ind in enumerate(profile):
            prob_player = a_nash[player_ind]
            if prob_player == 0:
                prob = 0
                break
            prob = prob * prob_seller
        if prob == 0:
            continue
        for seller_ind in range(num_players):
            a_payoff[player_ind] = prob * game[profile][player_ind]
    return a_payoff


###########################
### GAME TABLE ANALYSIS ###
###########################

def find_a_profile_nash_strat_from_game(game, num_players, a_num_strats):
    '''
    Finds array of pure nash profiles from a game.  A profile is a np.array
    where the ith element is the index of the strategy chosen by the ith
    player.
    '''
    solver = gmb.nash.ExternalEnumPureSolver()
    a_nash = solver.solve(game)
# Handle Exceptions
    if len(a_nash) == 0:
        return np.array([])
# Parse a_nash into an actual array
    a_split = np.cumsum(a_num_strats)
    a_split = a_split[0:(len(a_split)-1)]
    a_split = np.insert(a_split, 0, 0)
    a_nash = np.array([np.nonzero(nash)[0] for nash in a_nash])
    ret = a_nash - a_split
    return ret

def find_profile_best_nash_from_game(game, num_players, a_num_strats):
    '''
    Finds pure nash profile with highest total payoff from a game.  A profile
    is a np.array where the ith element is the index of the strategy chosen by
    the ith player.
    '''
    a_profile = find_a_profile_nash_strat_from_game(game, num_players, a_num_strats)
    num_nash = len(a_profile)
    if num_nash == 0: 
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

def find_a_mixed_profile_nash_strat_from_game(game, num_players, a_num_strats):
    '''
    Finds array of pure nash profiles from a game.  A profile is a np.array
    where the ith element is the index of the strategy chosen by the ith
    player.
    '''
    solver = gmb.nash.ExternalEnumPureSolver()
    a_nash = solver.solve(game)
# Handle Exceptions
    if len(a_nash) == 0:
        return np.array([])
# Parse a_nash into an actual array
    a_split = np.cumsum(a_num_strats)
    a_split = a_split[0:(len(a_split)-1)]
    a_split = np.insert(a_split, 0, 0)
    a_nash = np.array([np.nonzero(nash)[0] for nash in a_nash])
    ret = a_nash - a_split
    return ret

############################
### SEARCH ON GAME TABLE ###
############################

def refine_m_strat(m_strat, func_payoff_handler, scale_factor,
        no_zoom_if_edge_nash=True, **kwargs):
    '''
    Given matrix of strategies find a new matrix of strategies. Use nash as new
    center of m_strat, and zoom in by scale_factor. If nash is on edge, and
    no_zoom_if_edge_nash=True, then don't zoom in if nash is on edge.
    '''
    assert scale_factor <= 1, 'scale_factor must be <= 1. Scale factor is {}'.format(scale_factor)
# Find nash
    game, a_num_strats = make_game_table_from_m_strat(m_strat,
            find_profit_handler_tmp, **kwargs)
    num_players = len(a_num_strats)
    profile = find_profile_best_nash_from_game(game, num_players)
    if len(profile) == 0:
        raise Exception("No pure Nash found in outer game, fix the code: {}".format(profile))
# Find m_strat locations
    a_strat_nash = np.array([a_strat[profile[i]] for i, a_strat in
        enumerate(m_strat)])
    a_jump = abs(m_strat[:, 0] - m_strat[:,1])
# Handle zooming
    if no_zoom_if_edge_nash:
        if any(profile == 0) or any(profile == a_num_strats):
            scale_factor = 1
            is_zoomed = False
    a_jump = a_jump * scale_factor
# Create m_strat around nash.
    ret = np.array([get_a_strat_from_center(num_strats, strat_nash, jump) for
            strat_nash, jump, num_strats in zip(a_strat_nash, a_jump,
                a_num_strats)])
    return ret

def find_psuedocontinuous_nash(a_strat, num_sellers, func_payoff_handler,
        scale_factor, **kwargs):
    '''
    Basically a handler for refine_m_strat.
    '''
    m_strat = np.array([a_strat] * num_sellers)
    num_iter = 5
    for i in range(num_iter):
        m_strat = refine_m_strat(m_strat, func_payoff_handler, scale_factor, **kwargs)
    return m_strat


#########################
### CREATE GAME TABLE ###
#########################

def make_game_table_from_a_a_strat(a_a_strat, func_payoff, **kwargs):
    '''
    Creates game table on given prices.
    '''
    a_num_strats = [len(a_strat) for a_strat in a_a_strat]
    num_players = len(a_num_strats)
    ret = gmb.Game.new_table(a_num_strats)
    for profile in ret.contingencies:
        a_strat_nash = np.array([a_strat[profile[i]] for i, a_strat in
            enumerate(a_a_strat)])
        a_payoff = func_payoff(a_strat_nash, **kwargs)
        for ind in range(num_players):
            try: ret[profile][ind] = int(a_payoff[ind])
            except: set_trace()
    return ret, a_num_strats

def get_a_cost_from_kwargs(m_tax, a_cost, endowment):
    return a_cost

def make_a_strat_price(num_buyers, a_quantity, num_strats=21,
        is_randomized=False, is_shifted = False, **kwargs):
    """
    Makes array of possilbe price strategies.  Don't allow even num_strats.
    """
    #assert num_strats/2. != num_strats/2, "price discretization number num_strats must be odd."
    _, price = find_profit_cournot(num_buyers, a_quantity, **kwargs)
    a_cost = get_a_cost_from_kwargs(**kwargs)
    return np.linspace(min(a_cost), price, num_strats)

def helper_func_price(a_price, a_quantity, m_tax, a_cost, endowment):
    return find_profit(a_quantity, a_price, m_tax, a_cost, endowment,
            just_profit = True)

def helper_func_quant(a_quantity, num_sellers, num_buyers, **kwargs):
    return find_nash_price(num_sellers, num_buyers, a_quantity,
            just_profit=True, **kwargs)

def make_game_table_price(num_sellers, num_buyers, a_quantity, **kwargs):
    a_strat_price = make_a_strat_price(num_buyers, a_quantity, **kwargs)
    a_a_strat_price = [a_strat_price] * num_sellers
    kwargs_new = dict(a_quantity=a_quantity, **kwargs)
    return make_game_table_from_a_a_strat(a_a_strat_price, helper_func_price,
            **kwargs_new)

def make_game_table_quant(num_sellers, num_buyers, a_a_strat_quant, **kwargs):
    kwargs_new = dict(num_sellers=num_sellers, num_buyers=num_buyers, **kwargs)
    return make_game_table_from_a_a_strat(a_a_strat_quant, helper_func_quant,
            **kwargs_new)


#################
### FIND NASH ###
#################

def find_nash_price(num_sellers, num_buyers, a_quantity, just_profit=False,
        **kwargs):
    '''
    Finds nash price given quantity. Can also output other information as a
    dictionary. Price options are automatically generated.
    '''
# Gambit stuff
    game, a_num_strats = make_game_table_price(num_sellers, num_buyers,
            a_quantity, **kwargs)
    profile = find_profile_best_nash_from_game(game, num_sellers, a_num_strats)
# Handle no soutions
    if len(profile) == 0:
        if not just_profit:
            raise Exception("No pure Nash found: {}".format(profile))
        warnings.warn('No pure Nash found in inner game'.format(profile))
        return np.ones(num_sellers) * (-sys.maxint/4)
# Boundary Warnings
    if any(profile == 0) or any (profile == np.array(a_num_strats)-1):
        warnings.warn('Boundary hit in game table. profile: {}'.format(profile))
# Output
    if just_profit:
        return get_a_payoff_from_profile(game, profile, num_sellers)
    a_strat_price = make_a_strat_price(num_buyers, a_quantity, **kwargs)
    a_price = a_strat_price[profile]
    ret = find_profit(a_quantity, a_price, just_profit = False, **kwargs)
    return ret

def empty_ret(num_sellers, num_buyers):
    a_NA = np.empty(num_sellers)
    a_NA[:] = np.nan
    m_NA = np.empty((num_sellers, num_buyers))
    m_NA[:] = np.nan
    ret = {'a_profit' : a_NA,
           'a_quantity_nash' : a_NA,
           'm_quantity_bought' : m_NA,
           'a_price_nash' : a_NA,
           'a_quantity_sold' : a_NA}
    return ret

def find_nash_quant(num_sellers, num_buyers, a_a_strat_quant, is_zoomed=False,
        **kwargs):
    '''
    Finds nash quantity given range of quantities, then outputs all the
    information as a dictionary. This is where the zooming in happens.
    '''
# filter negatives
    a_a_strat_quant = [a_strat[a_strat >= 0.] for a_strat in a_a_strat_quant]
# Gambit stuff
    game, a_num_strats = make_game_table_quant(num_sellers, num_buyers,
            a_a_strat_quant, **kwargs)
    profile = find_profile_best_nash_from_game(game, num_sellers, a_num_strats)
# Handle no soutions
    if len(profile) == 0:
        warnings.warn("No pure Nash found in outer game, fix the code: {}".format(profile))
        return empty_ret(num_sellers, num_buyers)
# Boundary Warnings
    if any(profile == 0) or any (profile == np.array(a_num_strats)-1):
        warnings.warn('Boundary hit in game table. profile: {}'.format(profile))
# Output
    a_quantity = np.array([a_strat[profile[i]] for i, a_strat in
        enumerate(a_a_strat_quant)])
    ret = find_nash_price(num_sellers, num_buyers, a_quantity,
            just_profit=False, **kwargs)
# Zoom in
    if not is_zoomed:
        a_a_strat_quant = [get_a_strat_from_center(17, quant, 5) for quant in a_quantity]
        ret = find_nash_quant(num_sellers, num_buyers, a_a_strat_quant,
                is_zoomed=True, **kwargs)
    return ret


#####################
### CREATE PICLKE ###
#####################

def make_dic_of_pure_nash(num_sellers, num_buyers, a_strat_quantity,
        a_seller_loc, a_buyer_loc, a_cost, gamma, scalar_tax, endowment,
        mean_cost, cost_ratio, m_tax):
    '''
    Creates tax matrix, then the  game table for gambit, then finds pure nash
    solution(s), then makes a dictionary for pickle
    '''
# Setup
    kwargs = {'m_tax' : m_tax, 'a_cost' : a_cost, 'endowment' : endowment}
    a_a_strat_quant = [a_strat_quantity] * num_sellers
# Create and Solve Game.
    d_nash = find_nash_quant(num_sellers, num_buyers, a_a_strat_quant,
            **kwargs)
# Create dic to return. Note Gambit forces the use of Python 2, hence 'update'.
    ret = {'gamma'        : gamma,
           'scalar_tax'   : scalar_tax,
           'mean_cost'    : mean_cost,
           'cost_ratio'   : cost_ratio,
           'a_buyer_loc'  : a_buyer_loc,
           'a_seller_loc' : a_seller_loc,
           'num_buyers'   : num_buyers,
           'num_sellers'  : num_sellers }
    ret.update(kwargs)
    ret.update(d_nash)
    return ret


#############
### WRITE ###
#############

def write_output(d_out, fn):
# write to the output
    folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
    folder  = folder1 if os.path.exists(folder1) else folder2
    jl.dump(d_write, folder + fn)


############
### MAIN ###
############

def main(num_sellers=2, num_buyers=6, gamma=0, scalar_tax=1., mean_cost=100, cost_ratio=1.0,
        endowment=200, randomize=False, random_seed=17, tax_model='ordinal'):
    """ Add documentation here """
# check that input is correct
    #assert num_buyers%num_sellers == 0, "number of sellers does not divide number of buyers"
# setup buyer and seller locations
    if randomize:
        np.random.seed(random_seed)
        a_seller_loc = np.random.uniform(low=0, high=1, size=num_sellers)
        a_buyer_loc  = np.random.uniform(low=0, high=1, size=num_buyers)
    else:
        a_seller_loc = np.linspace(start=0, stop=1, num=num_sellers, endpoint=False)
        a_buyer_loc  = np.linspace(start=0, stop=.999, num=num_buyers,  endpoint=False)
# set the quantity discretization and calculate Nash
    num_strats = 21 # number of quantity-strategies
    dist = 40 # deviation from cournot that we search for
# Determine a_cost
    a_cost = get_a_cost_from_ratio(mean_cost, cost_ratio, num_sellers)
    if cost_ratio == 1.01:
        a_cost = np.array([99., 100.])
    if cost_ratio == 1.0:
        a_cost = np.array([100., 100.])
# need to fix theoretical Cournot.
    q_min, _ = theoretical_Cournot(1, num_buyers/float(num_sellers),
            min(a_cost), endowment)
    q_max, _ = theoretical_Cournot(num_sellers, num_buyers, max(a_cost),
            endowment)
    # Not the best estimate
    a_strat_quantity = np.arange(0, num_buyers*50, 40)
# Calculate m_tax
    if tax_model == 'ordinal':
        m_tax = get_m_tax_ordinal(a_buyer_loc, a_seller_loc, gamma, scalar_tax)
    elif tax_model == 'cardinal':
        m_tax = get_m_tax_dist(a_buyer_loc, a_seller_loc, gamma, scalar_tax)
    d_write = make_dic_of_pure_nash(num_sellers, num_buyers, a_strat_quantity,
            a_seller_loc, a_buyer_loc, a_cost, gamma, scalar_tax, endowment,
            mean_cost, cost_ratio, m_tax)
    print(d_write)
    fn = 'S=%s_B=%s_gamma=%s_scalar_tax=%s_mean_cost=%s_cost_ratio=%s_endow=%s_randomize=%s_tax_model=%s.pkl'%(num_sellers,
            num_buyers, gamma, scalar_tax, mean_cost, cost_ratio, endowment, randomize)
    write_output(d_out, fn)


def parameter_combination(i):
    """
    Execute the i-th parameter combination.
    """
# Create combinations
    num_sellers     = [2]
    num_buyers      = [12]
    gamma           = [1.]
    scalar_tax      = np.round(np.linspace(0.0, 1.0, 11), 3)
    mean_cost       = [100]
    cost_ratio      = [1.01]#np.round(np.linspace(1.0, 2.0, 11), 3)
    endowment       = [120.]
    random_seed     = [17]
    randomize       = [True]
    tax_model       = ['cardinal']
    combs           = product(num_sellers, num_buyers, gamma, scalar_tax, mean_cost, cost_ratio, endowment, randomize, random_seed, tax_model)
    comb            = list(combs)[i]
    num_sellers, num_buyers, gamma, scalar_tax, mean_cost, cost_ratio, endowment, randomize, random_seed, tax_model = comb
# Run main function
    print('executing num_sell=%s, num_buy=%s, gamma=%s, scalar_tax=%s, mean_cost=%s, cost_ratio=%s, endowment = %s, randomize=%s, random_seed=%s, tax_model=%s'%comb)
    main(num_sellers, num_buyers, gamma, scalar_tax, mean_cost, cost_ratio, endowment, randomize, random_seed, tax_model)


if __name__ == "__main__":
    i = int(sys.argv[1]) - 1
    parameter_combination(i) 

