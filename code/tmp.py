import numpy as np

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
    a_mask_cutoff = a_purchase_cutoffs > quantity_unsold
# if quantity_unsold is 13, this is [0, 0, 0, 0, 0,  0,  1,  1] (actually as trues and falses)
    ind_cutoff = a_mask_cutoff.argmax()
    if ind_cutoff == 0:
        return np.ones_like(a_demand_sorted) * (quantity_unsold/len(a_demand_sorted))
    a_demand_sorted[a_mask_cutoff] = a_demand_sorted[ind_cutoff - 1]
# We transform a_demand_sorted into [0, 0, 1, 1, 1,  3,  3,  3]
    quant_remaining_initial = quantity_unsold - a_purchase_cutoff[ind_cutoff - 1]
    a_demand_sorted += a_mask_cutoff * (quantity_remaining_initial/a_mask_cutoff.sum())
# Finally, we have distribution     [0, 0, 1, 1, 1,  3, 3.5, 3.5]
    a_quantity_bought = a_demand_sorted[a_unsort]
    return a_quantity_bought


def find_next_purchase_recursive(a_quantity_unsold, a_endow_remaining, m_price_rel):
    '''
    Calculates who buys what from whom. Effectively it iterates over
    m_price_rel from least price to highest. At each price it conducts a sale,
    then records it.
    @m_price_rel is a numpy matrix with sellers as rows and buyers as columns
    '''
# Find min price
# Clear out sellers without quantity.
# Make numpy array so that 
    if all (a_quantity_unsold == 0):
        return np.zeros_like(m_price_rel)
    price = m_price_rel.min()
    if price == np.inf:
        return np.zeros_like(m_price_rel)
    a_buyer_finished = a_endow_remaining <= price
    m_price_rel[:, a_buyer_finished] = np.inf
# Check if buyers still exist
    if price < m_price_rel.min():
        return find_next_purchase_recursive(a_quantity_unsold, a_endow_remaining, m_price_rel)
# For first seller offering price, conduct sale:
# Find first index of seller offering minimum price:
    ind_seller = m_price_rel.min(1).argmin()
# Find buyers that can purchase from seller at min price
    a_mask = (m_price_rel[ind_seller] == price)
# Find demand remaining
    a_demand_remaining = (a_endow_remaining * a_mask) - price
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

##############
### Script ###
##############

if __name__ == "__main__":
    a_quantity = np.array([91.125,72.875])
    a_price = np.array([106.33333333333333333333, 106.33333333333333333333])
    m_price_rel = np.ones((2, 3)) * a_price[:, None]
    endowment = 200
    a_quantity_sold_old, m_quantity_bought_old = find_quantity_sold(a_quantity, m_price_rel, endowment)
    print('OLD a_qauntity_sold')
    print(a_quantity_sold_old)
    print('OLD m_quantity_bought')
    print(m_quantity_bought_old)
    a_quantity_sold_old, m_quantity_bought_old = find_quantity_sold_alt(a_quantity, m_price_rel, endowment)
    print('NEW a_qauntity_sold')
    print(a_quantity_sold_new)
    print('NEW m_quantity_bought')
    print(m_quantity_bought_new)

