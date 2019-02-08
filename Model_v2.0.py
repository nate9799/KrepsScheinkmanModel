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

def get_m_tax(a_buyer_loc, a_seller_loc, gamma=GAMMA):
    # Matrix with absolute dist
    m_dist = [[circle_dist(buyer_loc, seller_loc)
        for buyer_loc in a_buyer_loc] for seller_loc in a_seller_loc]
    # Matrix with rel dist with list of prices. '+ 1' because min dist is 1.
    m_rel_dist = np.argsort(m_dist, axis=0) + 1.0
    # Lot of potential speed up here, only need to do exponent once.
    m_tax = np.array(m_rel_dist**gamma, dtype=float)
    return m_tax


###################
### FIND BUNDLE ###
###################

def find_bundle(a_quantity_unsold, a_price_rel, endowment = ENDOWMENT):
    a_quantity_order = np.argsort(a_quantity_unsold)
    tot_quantity = 0
    a_quantity_bought = [0.] * len(a_quantity_unsold)
    for ind in a_quantity_order:
        tot_quanity = sum(a_quantity_bought)
        demand_remaining = endowment - tot_quantity - a_price_rel[ind]
        if demand_remaining <= 0:
            return a_quantity_bought
        elif demand_remaining <= a_quantity_unsold[ind]:
            a_quantity_bought[ind] = demand_remaining
            return a_quantity_bought
        else:
            a_quantity_bought[ind] = a_quantity_unsold[ind]





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
    a_quantity_unsold = a_quantity - a_quantity_sold
    a_profit = a_revenue - a_cost
    return a_profit


##################
##################
##### SCRIPT #####
##################
##################

# CONSTANTS 
NUM_BUYERS = 100
NUM_SELLERS = 20
A_BUYER_LOC = np.random.rand(NUM_BUYERS)
A_SELLER_LOC = np.random.rand(NUM_SELLERS_START)
GAMMA = 0
SLOPE = 1
ENDOWMENT = 2
