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

def get_a_strat_from_center(n, center, jump, shift = 0):
    dist = float(n * jump)/2
    ret = np.linspace(center - dist, center + dist, n) + shift
    return(ret)


#################
### RANDOMIZE ###
#################

def make_even_spaced_array_noisy(arr, random_seed=None):
    dist = abs(arr[1] - arr[0])
    clip = dist/2.
    if not random_seed is None:
        np.random.seed(random_seed)
    a_rand = np.random.normal(loc=0, scale = dist/8, size=len(arr))
    a_rand_clip = np.clip(a_rand, -clip, clip)
    ret = arr + a_rand_clip
    return ret


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

#################
### BUY ORDER ###
#################

def buy_order(m_price_rel):
    ret = m_price_rel.T[m_price_rel.min(0).argsort()].T
    return ret


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


def find_purchase_distribution(quantity_unsold, a_demand_remaining):
    '''
    Finds how much each buyer purchases at a given price for a particular seller
    '''
    if quantity_unsold == 0:
        return np.zeros(len(a_demand_remaining))
    if any (a_demand_remaining < 0):
        raise ValueError('An element of a_demand_remaining is negative when it should be nonnegative.')
    if a_demand_remaining.sum() <= quantity_unsold:
        return a_demand_remaining
# Double argsort is intentional and for later unsorting.
    a_unsort = a_demand_remaining.argsort().argsort()
    a_demand_sorted = np.sort(a_demand_remaining)
# Finding proper distribution for sorted array:
# Say a_demand_sorted looks like    [0, 0, 1, 1, 1,  3,  4,  5]
    a_back_count = np.arange(len(a_demand_sorted) - 1, -1., -1)
    a_purchase_cutoffs = a_demand_sorted.cumsum() + a_demand_sorted * a_back_count
# a_purchase cutoffs would be       [0, 0, 6, 6, 6, 12, 14, 15]
    a_mask_is_cutoff = a_purchase_cutoffs > quantity_unsold
# if quantity_unsold is 13, this is [0, 0, 0, 0, 0,  0,  1,  1] (actually as trues and falses)
    ind_cutoff = a_mask_is_cutoff.argmax()
    if ind_cutoff == 0:
        return np.ones_like(a_demand_sorted) * (quantity_unsold/len(a_demand_sorted))
    a_demand_sorted[a_mask_is_cutoff] = a_demand_sorted[ind_cutoff - 1]
# We transform a_demand_sorted into [0, 0, 1, 1, 1,  3,  3,  3]
    quant_remaining_initial = quantity_unsold - a_purchase_cutoffs[ind_cutoff - 1]
    a_demand_sorted += a_mask_is_cutoff * (quantity_remaining_initial/a_mask_is_cutoff.sum())
# Finally, we have distribution     [0, 0, 1, 1, 1,  3, 3.5, 3.5]
    a_quantity_bought = a_demand_sorted[a_unsort]
    return a_quantity_bought


def find_next_purchase_recursive(a_quantity_unsold, a_endow_remaining, m_price_rel):
    '''
    Calculates who buys what from whom. Effectively it iterates over
    m_price_rel from least price to highest. At each entry in m_price_rel it
    iterates over, the function conducts a sale, then records it.
    The iteration is recorded by masking elements of m_price_rel with np.inf
    when they have been iterated over or should be skipped.
    @m_price_rel is a numpy matrix with sellers as rows and buyers as columns
    '''
# Check that there are sellers with remaining quantity
    if all (a_quantity_unsold == 0):
        return np.zeros_like(m_price_rel)
# Find min price
    price_min = m_price_rel.min()
# See if all prices have been done before.
    if price_min == np.inf:
        return np.zeros_like(m_price_rel)
# Mask all buyers that can no longer purchase, because their endowment is less
# than the lowest price remaining.
    a_buyer_finished_mask = (a_endow_remaining <= price_min)
    m_price_rel[:, a_buyer_finished_mask] = np.inf
# Check if buyers still exist for this price, after masking. If not, then we
# are done for this price.
    if price_min < m_price_rel.min():
        return find_next_purchase_recursive(a_quantity_unsold, a_endow_remaining, m_price_rel)
# For first, and only the first, seller offering price, conduct sale:
# Find first index of seller offering minimum price:
    ind_seller = m_price_rel.min(1).argmin()
# Find buyers that can purchase from seller at min price
    a_mask = (m_price_rel[ind_seller] == price_min)
# Find demand remaining at price point
    a_demand_remaining = (a_endow_remaining - price_min) * a_mask
# Find purchase
    a_quantity_bought = find_purchase_distribution(a_quantity_unsold[ind_seller], a_demand_remaining)
# Record purchase
    a_endow_remaining -= a_quantity_bought
    a_quantity_unsold[ind_seller] -= a_quantity_bought.sum()
    m_price_rel[ind_seller, a_mask] = np.inf
# Recurse
    m_quantity_bought = find_next_purchase_recursive(a_quantity_unsold, a_endow_remaining, m_price_rel)
    m_quantity_bought[ind_seller] += a_quantity_bought
    return m_quantity_bought


##########################
### FIND QUANTITY SOLD ###
##########################

def find_quantity_sold_alt(a_quantity, m_price_rel, endowment):
    '''
    Finds quantity sold for each seller, given prices, quantity, and endowment.
    It works by iterating over m_price_rel, from smallest price to greatest.
    '''
    a_endow_remaining = np.ones_like(m_price_rel[0]) * endowment
    m_price_rel_tmp = m_price_rel.copy()
    a_quantity_unsold = a_quantity.copy()
    m_quantity_bought = find_next_purchase_recursive(a_quantity_unsold, a_endow_remaining, m_price_rel_tmp)
    a_quantity_sold = m_quantity_bought.sum(1)
    return a_quantity_sold, m_quantity_bought

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
    m_price_rel_tmp = buy_order(m_price_rel)
# Loop
    for a_price_rel in m_price_rel_tmp.T:
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

def find_profit_cournot(num_buyers, a_quantity, a_cost, m_tax, endowment, **kwargs):
    '''
    Finds profit for each seller based on theoretical cournot model, given
    quantities, cost, endowment, and number of buyers.
    '''
    tot_quant = sum(a_quantity)
    price = endowment - (tot_quant/num_buyers)
    a_profit = a_quantity * (price - a_cost)
    return a_profit, price

def find_profit(a_quantity, a_price, m_tax, a_cost, endowment, just_profit = True, **kwargs):
    '''
    Finds profit and quantity sold for each seller, given strategies, taxation,
    and basic settings.
    '''
    a_quantity = np.array(a_quantity)
    a_price = np.array(a_price)
    m_price_rel = m_tax * a_price[:, None]
    a_quantity_sold, m_quantity_bought = find_quantity_sold_alt(a_quantity,
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

# NOT USED RIGHT NOW
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

def make_a_strat_price(num_buyers, a_quantity, a_cost, num_strats=21,
        randomize_price=False, is_shifted = False, random_seed_price=-1, **kwargs):
    """
    Makes array of possilbe price strategies.  Don't allow even num_strats.
    """
    #assert num_strats/2. != num_strats/2, "price discretization number num_strats must be odd."
    _, price = find_profit_cournot(num_buyers, a_quantity, a_cost, **kwargs)
    ret = np.linspace(min(a_cost), price, num_strats)
    if randomize_price == True:
        ret = make_even_spaced_array_noisy(ret, random_seed_price)
    return ret

def helper_func_price(a_price, a_quantity, m_tax, a_cost, endowment, randomize_price, **kwargs):
    return find_profit(a_quantity, a_price, m_tax, a_cost, endowment,
            just_profit = True, **kwargs)

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

def make_dic_of_pure_nash(num_sellers, num_buyers, a_strat_quantity, a_cost,
        endowment, m_tax, randomize_price=False, random_seed_price=-1):
    '''
    Creates tax matrix, then the  game table for gambit, then finds pure nash
    solution(s), then makes a dictionary for pickle
    '''
# Setup
    kwargs = {'m_tax' : m_tax, 'a_cost' : a_cost, 'endowment' : endowment,
            'randomize_price' : randomize_price, 'random_seed_price' : random_seed_price}
    a_a_strat_quant = [a_strat_quantity] * num_sellers
# Create and Solve Game.
    ret = find_nash_quant(num_sellers, num_buyers, a_a_strat_quant,
            **kwargs)
# Create dic to return. Note Gambit forces the use of Python 2, hence 'update'.
    ret.update(kwargs)
    return ret


#############
### WRITE ###
#############

def write_output(d_write, fn):
# write to the output
    folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
    folder  = folder1 if os.path.exists(folder1) else folder2
    jl.dump(d_write, folder + fn)


############
### MAIN ###
############

def main(num_sellers=2, num_buyers=6, gamma=0, scalar_tax=1., a_cost=[100, 100],
        endowment=200, randomize_quant=False, random_seed_quant=17,
        randomize_price=False, random_seed_price=-1, randomize_loc=False,
        random_seed_loc=17, tax_model='ordinal', m_tax=None):
    """ Add documentation here """
# check that input is correct
    #assert num_buyers%num_sellers == 0, "number of sellers does not divide number of buyers"
# setup buyer and seller locations
    if randomize_loc:
        np.random.seed(random_seed_loc)
        a_seller_loc = np.random.uniform(low=0, high=1, size=num_sellers)
        a_buyer_loc  = np.random.uniform(low=0, high=1, size=num_buyers)
    else:
        a_seller_loc = np.linspace(start=0, stop=1, num=num_sellers, endpoint=False)
        a_buyer_loc  = np.linspace(start=0, stop=.999, num=num_buyers,  endpoint=False)
# set the quantity discretization and calculate Nash
    num_strats = 41 # number of quantity-strategies
    dist = 50 # deviation from cournot that we search for
    if dist <= 1:
        raise ValueError('dist too small, should be at least 1, is {}'.format(dist))
# need to fix theoretical Cournot.
    a_strat_quantity = np.arange(0, num_buyers*dist, num_strats)
    if randomize_quant == True:
        a_strat_quantity = make_even_spaced_array_noisy(a_strat_quantity, random_seed_quant)
# Normalize a_cot inot numpy array
    a_cost = np.array(a_cost)
# Calculate m_tax
    if tax_model == 'ordinal':
        m_tax = get_m_tax_ordinal(a_buyer_loc, a_seller_loc, gamma, scalar_tax)
    elif tax_model == 'cardinal':
        m_tax = get_m_tax_dist(a_buyer_loc, a_seller_loc, gamma, scalar_tax)
# Special for when used by turn_model.py
    elif tax_model == 'given':
        m_tax = m_tax
    d_settings = {'gamma'       : gamma,
           'scalar_tax'         : scalar_tax,
           'a_cost'             : a_cost,
           'a_buyer_loc'        : a_buyer_loc,
           'a_seller_loc'       : a_seller_loc,
           'num_buyers'         : num_buyers,
           'num_sellers'        : num_sellers,
           'randomize_quant'    : randomize_quant,
           'random_seed_quant'  : random_seed_quant,
           'randomize_price'    : randomize_price,
           'random_seed_price'  : random_seed_price,
           'randomize_loc'      : randomize_loc,
           'random_seed_loc'    : random_seed_loc,
           'tax_model'          : tax_model}
    d_write = make_dic_of_pure_nash(num_sellers, num_buyers, a_strat_quantity,
        a_cost, endowment, m_tax, randomize_price, random_seed_price)
    d_write.update(d_settings)
    print(d_write)
    return d_write


def parameter_combination(i):
    """
    Execute the i-th parameter combination.
    """
# Create combinations
    num_sellers         = [2]
    num_buyers          = [12]
    gamma               = [1.]
    scalar_tax          = np.round(np.linspace(0.0, 1.0, 11), 3)
    a_cost              = [[99, 100], [100, 100]]
    endowment           = [120.]
    random_seed_quant   = [17, 34, 51]
    randomize_quant     = [False]
    random_seed_price   = [17, 34, 51]
    randomize_price     = [False]
    random_seed_loc     = [17, 34, 51]
    randomize_loc       = [True]
    tax_model           = ['cardinal']
    combs = product(num_sellers, num_buyers, gamma, scalar_tax, a_cost,
            endowment, randomize_quant, random_seed_quant, randomize_price,
            random_seed_price, randomize_loc, random_seed_loc, tax_model)
    comb = list(combs)[i]
    num_sellers, num_buyers, gamma, scalar_tax, a_cost, endowment, randomize_quant, random_seed_quant, randomize_price, random_seed_price, randomize_loc, random_seed_loc, tax_model = comb
# Run main function
    print('executing num_sell=%s, num_buy=%s, gamma=%s, scalar_tax=%s, a_cost=%s, endowment = %s, randomize_quant=%s, random_seed_quant=%s, randomize_price=%s, random_seed_price=%s, randomize_loc=%s, random_seed_loc=%s, tax_model=%s'%comb)
    d_write = main(num_sellers, num_buyers, gamma, scalar_tax, a_cost,
            endowment, randomize_quant, random_seed_quant, randomize_price,
            random_seed_price, randomize_loc, random_seed_loc, tax_model)
    fn = 'S=%s_B=%s_gamma=%s_scalar_tax=%s_a_cost=%s_endow=%s_tax_model=%s_rand_price=%s_seed_price=%s_rand_quant=%s_seed_quant=%s_rand_loc=%s_seed_loc=%s.pkl'%(num_sellers,
            num_buyers, gamma, scalar_tax, a_cost, endowment, tax_model,
            randomize_price, random_seed_price, randomize_quant,
            random_seed_quant, randomize_loc, random_seed_loc)
    write_output(d_write, fn)


if __name__ == "__main__":
    i = int(sys.argv[1]) - 1
    parameter_combination(i) 
