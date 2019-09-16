from sklearn.externals import joblib as jl
import sys
import os
import model
import pandas as pd
import numpy as np


def new_m_tax(d_settings):
    d_settings['m_tax'] = .9 * (d_settings['m_tax'] - 1) + 1
    d_settings['tax_model'] = 'given'
    return d_settings

def new_scalar_tax(d_settings):
    d_settings['scalar_tax'] = d_settings['scalar_tax'] - .1
    return d_settings

def new_a_cost(d_settings):
    a_cost = d_settings['a_cost']
    a_profit = d_settings['a_profit']
    a_profit_ratio = a_profit/np.sum(a_profit)
    a_cost_decrease_ratio = 1 - .05*a_profit_ratio
    d_setings['a_cost'] = a_cost_decrease_ratio*a_cost
    return d_settings


def get_settings_from_dic_ret(dic_ret):
    '''
    Gets relevent information from dic.
    '''
    num_buyers = dic_ret['num_buyers']
    ret = { 'num_sellers' : dic_ret['num_sellers'],
            'num_buyers' : dic_ret['num_buyers'],
            'gamma' : dic_ret['gamma'],
            'scalar_tax' : dic_ret['scalar_tax'],
            'a_cost' : dic_ret['a_cost'],
            'endowment' : dic_ret['endowment'],
            'randomize_price' : dic_ret['randomize_price'],
            'random_seed_price' : dic_ret['random_seed_price'],
            'randomize_loc' : dic_ret['randomize_loc'],
            'random_seed_loc' : dic_ret['random_seed_loc'],
            'tax_model' : dic_ret['tax_model'],
            'm_tax' : dic_ret['m_tax']} 
    return ret

def turn_dic_from_dic(dic, a_func):
    '''
    Computes a single turn
    '''
    d_settings = get_settings_from_dic_ret(dic)
    for func in a_func:
        d_settings = func(d_settings)
    ret = model.main(**d_settings)
    return ret

def loop(d_settings, num_loops, a_func):
    dic_ret = model.main(**d_settings)
    a_dic_ret = [dic_ret]
    for i in range(num_loops-1):
        print(i)
        if np.isnan(dic_ret['a_profit'][0]):
            print('ended with NAN')
            break
        dic_ret = turn_dic_from_dic(dic_ret, a_func)
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

def process(d_settings, num_loops):
    a_func = [new_scalar_tax]
    a_d_turns = loop(d_settings, num_loops, a_func)
    d_turns = dic_from_a_dic(a_d_turns)
    folder1 = '/home/nate/Documents/abmcournotmodel/code/output/data/'
    folder2 = '/cluster/home/slera//abmcournotmodel/code/output/data/'
    folder  = folder1 if os.path.exists(folder1) else folder2
    fn = folder + 'turn_fast_gamma={}_endow={}_taxmethod={}_seed_quant={}.pickle'.format(round(d_settings['gamma'],3), d_settings['endowment'], d_settings['tax_model'], d_settings['random_seed_quant'])
    print(fn)
    jl.dump(d_turns, fn)

if __name__ == "__main__":
    i = int(sys.argv[1]) - 1
    d_settings = { 'num_sellers' : 2,
            'num_buyers' : 12,
            'gamma' : 1.,
            'scalar_tax' : 1.0,
            'a_cost' : [99., 100.],
            'endowment' : 120.,
            'randomize_quant' : True,
            'random_seed_quant' : i,
            'randomize_loc' : True,
            'random_seed_loc' : 17,
            'tax_model' : 'cardinal'} 
    process(d_settings, 11)

