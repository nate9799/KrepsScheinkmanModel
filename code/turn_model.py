from sklearn.externals import joblib as jl
import model
import pandas as pd
import numpy as np

def new_a_cost(a_cost, a_profit):
    ind = np.argmin(a_profit)
    a_cost[ind] = .95*a_cost[ind]
    return a_cost

def get_settings_from_dic_ret(dic_ret):
    '''
    Gets relevent information from dic.
    '''
    ret = { 'num_sellers' : dic_ret['num_sellers'],
            'num_buyers' : dic_ret['num_buyers'],
            'a_strat_quantity' : np.linspace(0, 800, 21),
            'a_seller_loc' : dic_ret['a_seller_loc'],
            'a_buyer_loc' : dic_ret['a_buyer_loc'],
            'a_cost' : dic_ret['a_cost'],
            'gamma' : dic_ret['gamma'],
            'scalar_tax' : 1,
            'endowment' : dic_ret['endowment'],
            'mean_cost' : dic_ret['mean_cost'],
            'cost_ratio' : dic_ret['cost_ratio']} 
    return ret

def turn_dic_from_dic(dic):
    '''
    Computes a single turn
    '''
    a_cost = dic['a_cost']
    a_profit = dic['a_profit']
    a_cost = new_a_cost(a_cost, a_profit)
    d_settings = get_settings_from_dic_ret(dic)
    d_settings['a_cost'] = a_cost
    ret = model.make_dic_of_pure_nash(**d_settings)
    return ret

def loop(d_settings, num_loops):
    dic_ret = model.make_dic_of_pure_nash(**d_settings)
    a_dic_ret = [dic_ret]
    for i in range(num_loops-1):
        print(i)
        dic_ret = turn_dic_from_dic(dic_ret)
        a_dic_ret.append(dic_ret)
    return a_dic_ret

def dic_from_a_dic(a_dic):
    '''
    Takes an array of dictionaries with the same keys, then turns it into a
    single dictionary with the same keys, except now the values are arrays of
    values of the dictionary.

    For example, an array of dictionaries which all have key 'key' and value
    'value' would become a single dictionary with the key 'key' and value
    ['value', 'value', ..., 'value']. 
    '''
    a_a_value = []
    a_key = a_dic[0].keys()
    for key  in a_key:
        a_value = []
        for dic in a_dic:
            a_value.append(dic[key])
        a_a_value.append(a_value)
    ret = dict(zip(a_key, a_a_value))
    return ret

def process(d_settings, num_loops):
    a_d_turns = loop(d_settings, num_loops)
    d_turns = dic_from_a_dic(a_d_turns)
    folder = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    fn = folder + 'out.pickle'
    jl.dump(d_turns, fn)

if __name__ == "__main__":
    num_sellers = 2
    num_buyers = 13
    d_settings = { 'num_sellers' : num_sellers,
            'num_buyers' : num_buyers,
            'a_strat_quantity' : np.linspace(0, 800, 21),
            'a_seller_loc' : np.linspace(start=0, stop=1, num=num_sellers, endpoint=False),
            'a_buyer_loc'  : np.linspace(start=0, stop=.999, num=num_buyers,  endpoint=False),
            'a_cost' : [100, 100],
            'gamma' : 0,
            'scalar_tax' : 1,
            'endowment' : 200,
            'mean_cost' : 100,
            'cost_ratio' : 1} 
    process(d_settings, 20)
