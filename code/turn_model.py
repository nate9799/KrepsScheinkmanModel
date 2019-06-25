from sklearn.externals import joblib as jl
import sys
import os
import model
import pandas as pd
import numpy as np

def new_a_cost(a_cost, a_profit):
    a_profit_ratio = a_profit/np.sum(a_profit)
    a_cost_decrease_ratio = 1 - .05*a_profit_ratio
    ret = a_cost_decrease_ratio*a_cost
    print(ret)
    return ret

def get_settings_from_dic_ret(dic_ret):
    '''
    Gets relevent information from dic.
    '''
    num_buyers = dic_ret['num_buyers']
    ret = { 'num_sellers' : dic_ret['num_sellers'],
            'num_buyers' : num_buyers,
            'a_strat_quantity' : np.arange(0, 100*num_buyers, 21),
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
        if np.isnan(dic_ret['a_profit'][0]):
            print('ended with NAN')
            break
        dic_ret = turn_dic_from_dic(dic_ret)
        a_dic_ret.append(dic_ret)
        print(a_dic_ret)
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

def process(d_settings, num_loops, price_diff):
    a_d_turns = loop(d_settings, num_loops)
    d_turns = dic_from_a_dic(a_d_turns)
    folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
    folder  = folder1 if os.path.exists(folder1) else folder2
    fn = folder + 'alt_fast_adv={}_turn_gamma={}_scalar_tax={}_endow={}.pickle'.format(price_diff, round(d_settings['gamma'],3), d_settings['scalar_tax'], d_settings['endowment'])
    jl.dump(d_turns, fn)

if __name__ == "__main__":
    i = int(sys.argv[1]) - 1
    gamma = i/10.
    scalar_tax = 1.0
    price_diff = 10
    endowment = 200
    num_sellers = 2
    num_buyers = 12
    np.random.seed(17)
    a_seller_loc = np.random.uniform(low=0, high=1, size=num_sellers)
    a_seller_loc = np.random.uniform(low=0, high=1, size=num_sellers)
    d_settings = { 'num_sellers' : num_sellers,
            'num_buyers' : num_buyers,
            'a_strat_quantity' : np.linspace(0, num_buyers*100, 21),
            'a_seller_loc' : np.linspace(start=0, stop=1, num=num_sellers, endpoint=False),
            'a_buyer_loc'  : np.linspace(start=0, stop=.999, num=num_buyers,  endpoint=False),
            'a_cost' : np.array([100.-price_diff, 100.]),
            'gamma' : gamma,
            'scalar_tax' : scalar_tax,
            'endowment' : endowment,
            'mean_cost' : 100,
            'cost_ratio' : 1} 
    process(d_settings, 20, price_diff)

