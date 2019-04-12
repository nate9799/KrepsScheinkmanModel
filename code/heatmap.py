from matplotlib import pyplot as plt
from sklearn.externals import joblib as jl
import numpy as np
import seaborn as sns
import gambit as gmb
import pandas as pd
from itertools import combinations_with_replacement, product
import sys
import os
import model


#####################
### HEATMAP MAKER ###
#####################

def heatmap(a_price_range, a_quantity_range, seller0_price, seller0_quantity,
        **kwargs):
    '''
    Creates a heat map of strategies a player can take agains another player in
    a 2 player game.
    '''
    heat = [[model.find_profit(np.array([seller0_quantity, quantity_tmp]),
        np.array([seller0_price, price_tmp]), just_profit=True, **kwargs)[1]
        for quantity_tmp in a_quantity_range]
        for price_tmp in reversed(a_price_range)]
    print(heat)
    return sns.heatmap(heat, annot=True, fmt = '.3g')

def heatmap_quantity(num_sellers, num_buyers, a_strat_quantity,
        view_cournot=True, **kwargs):
    '''
    Creates a heat map of strategies a player can take agains another player in
    a 2 player game.
    '''
    assert num_sellers == 2, 'too many sellers for check to work'
    game = model.make_game_table(num_sellers, num_buyers, a_strat_quantity,
            inner_game=False, **kwargs)
    n = 9 # number of price discretizations
    a_num_strats = [len(a_strat_quantity)] * num_sellers
    m_payoff0 = np.ones(a_num_strats)
    m_payoff1 = np.ones(a_num_strats)
    m_payoff2 = np.ones(a_num_strats)
    for x, y in np.ndindex(m_payoff0.shape):
        a_quantity = a_strat_quantity[[x,y]]
        m_payoff0[x,y] = game[x,y][0]
        m_payoff1[y,x] = game[x,y][1]
        tmp, _ = model.find_profit_cournot(num_buyers, a_quantity, **kwargs)
        m_payoff2[x,y] = tmp[0]
    a_strat_quantity_rnd = np.round(a_strat_quantity, 4)
    pd_payoff0 = pd.DataFrame(data=m_payoff0, index=a_strat_quantity_rnd,
            columns=a_strat_quantity_rnd)
    pd_payoff1 = pd.DataFrame(data=m_payoff2, index=a_strat_quantity_rnd,
            columns=a_strat_quantity_rnd)
    ax0 = plt.subplot2grid((1,2), (0,0))
    ax1 = plt.subplot2grid((1,2), (0,1))
    pd_payoff0[pd_payoff0 < -2e-10] = np.NAN
    pd_payoff1[pd_payoff1 < -2e-10] = np.NAN
    print(pd_payoff0)
    sns.heatmap(pd_payoff0/100, annot=True, fmt = '.4g', ax=ax0)
    sns.heatmap(pd_payoff1/100, annot=True, fmt = '.4g', ax=ax1)
    plt.show()
    return pd_payoff0

def heatmap_price(num_sellers, num_buyers, a_quantity, **kwargs):
    '''
    Creates a heat map of strategies a player can take agains another player in
    a 2 player game.
    '''
    assert num_sellers == 2, 'too many sellers for check to work'
    a_strat_price = model.make_a_strat_price(num_buyers, a_quantity, **kwargs)
    a_num_strats = [len(a_strat_price)] * num_sellers
    game = model.make_game_table(num_sellers, num_buyers, a_quantity,
            inner_game=True, **kwargs)
    m_payoff0 = np.ones(a_num_strats)
    m_payoff1 = np.ones(a_num_strats)
    for x, y in np.ndindex(m_payoff0.shape):
        m_payoff0[x,y] = game[x,y][0]
        m_payoff1[y,x] = game[x,y][1]
    #plt = lib.figure()
    a_strat_price_rnd = np.round(a_strat_price, 3)
    pd_payoff0 = pd.DataFrame(data=m_payoff0, index=a_strat_price_rnd,
            columns=a_strat_price_rnd)
    pd_payoff1 = pd.DataFrame(data=m_payoff1, index=a_strat_price_rnd,
            columns=a_strat_price_rnd)
    ax0 = plt.subplot2grid((1,2), (0,0))
    ax1 = plt.subplot2grid((1,2), (0,1))
    sns.heatmap(pd_payoff0, annot=True, fmt = '.3g', ax=ax0)
    sns.heatmap(pd_payoff1.T, annot=True, fmt = '.3g', ax=ax1)
    plt.show()
    return pd_payoff0


####################
### CALL HEATMAP ###
####################

def make_heatmap(num_sellers, num_buyers, a_strat_quantity, a_seller_loc, a_buyer_loc, a_cost, gamma, endowment):
    '''
    Creates tax matrix, then the  game table for gambit, then finds pure nash
    solution(s), then makes a dictionary for pickle
    '''
# Setup
    m_tax = model.get_m_tax(a_buyer_loc, a_seller_loc, gamma)
    kwargs = {'m_tax' : m_tax, 'a_cost' : a_cost, 'endowment' : endowment}
# Create and Solve Game.
    a_quantity = a_strat_quantity[[3,0]]
    ret1 = heatmap_quantity(num_sellers, num_buyers, a_strat_quantity,
            view_cournot=True, **kwargs)
    ret2 = heatmap_price(num_sellers, num_buyers, a_quantity, **kwargs)
    return ret1, ret2


############
### MAIN ###
############

def main(num_sellers=2, num_buyers=6, gamma=0, a_cost=np.array([100, 100]), endowment=200, randomize=False):
    """ Add documentation here """
# check that input is correct
    assert num_buyers%num_sellers == 0, "number of sellers does not divide number of buyers"
# setup buyer and seller locations
    if randomize:
        a_seller_loc = np.random.uniform(low=0, high=1, size=num_sellers)
        a_buyer_loc  = np.random.uniform(low=0, high=1, size=num_buyers)
    else:
        a_seller_loc = np.linspace(start=0, stop=1, num=num_sellers, endpoint=False)
        a_buyer_loc  = np.linspace(start=0, stop=1, num=num_buyers,  endpoint=False)
# set the quantity discretization and calculate Nash
    n                   = 11 # number of quantity-strategies
    dist                = 20 # deviation from cournot that we search for
    num_strats          = 20 # number of quantity-strategies
    q_min, _ = model.theoretical_Cournot(1, num_buyers, min(a_cost),
            endowment)
    q_max, _ = model.theoretical_Cournot(num_sellers, num_buyers, max(a_cost),
            endowment)
    a_strat_quantity = np.linspace(q_min-dist, q_max+dist, num_strats)
    pd_quant, pd_price = make_heatmap(num_sellers, num_buyers, a_strat_quantity, a_seller_loc, a_buyer_loc, a_cost, gamma, endowment)


def parameter_combination(i):
    """
    Execute the i-th parameter combination.
    """
    num_sellers     = [2]
    num_buyers      = [10, 12]
    gamma           = [0.]
    endowment       = [200.]
    randomize       = [True, False]
    combs           = product(num_sellers, num_buyers, gamma, endowment, randomize)
    comb            = list(combs)[i]
    num_sellers, num_buyers, gamma, endowment, randomize = comb
    print('executing num_sell=%s, num_buy=%s, gamma=%s, endowment = %s, randomize=%s'%comb)
    a_cost = np.array([100, 100])
    main(num_sellers, num_buyers, gamma, a_cost, endowment, randomize)


if __name__ == "__main__":
    #i = int(sys.argv[1]) - 1
    parameter_combination(1)

