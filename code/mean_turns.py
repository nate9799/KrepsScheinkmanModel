import pandas as pd
import numpy as np
import glob
from sklearn.externals import joblib as jl


def mean_dic_from_a_dic_base(a_dic, a_ignore_keys = []):
    '''
    Helper Function
    Takes an array of dictionaries with the same keys, then turns it into a
    single dictionary with the same keys, except now the values are the means
    of the values of the dictionary. If the key is an array, then it calculates
    the mean on the entries of the array.
    '''
    a_a_value_mean = []
    a_key = a_dic[0].keys()
    a_key = [key for key in a_key if key not in a_ignore_keys]
    for key in a_key:
        a_value = []
        for dic in a_dic:
            a_value.append(dic[key])
        a_value_mean = np.nanmean(a_value, axis=0)
        a_a_value_mean.append(a_value_mean)
    ret = dict(zip(a_key, a_a_value_mean))
    return ret

def mean_dic_from_a_dic(a_dic, is_remove_nan=True):
    '''
    Takes an array of dictionaries with the same keys, then turns it into a
    single dictionary with the same keys, except now the values are the means
    of the values of the dictionary. If the key is an array, then it calculates
    the mean on the entries of the array.
    Note: This will screw up a_buyer_loc and a_seller_loc
    '''
    if is_remove_nan:
        a_dic = [d_data for d_data in a_dic if not np.isnan(np.sum(d_data['a_profit']))]
        if len(a_dic) == 0:
            raise ValueError('All runs had NaN\'s')
    print('number of dics')
    print(len(a_dic))
    d_base = a_dic[0]
    a_random_seed_quant = [d_data['random_seed_quant'] for d_data in a_dic]
    a_random_seed_loc = [d_data['random_seed_loc'] for d_data in a_dic]
    ret = {'a_random_seed_quant' : a_random_seed_quant,
           'a_random_seed_loc' : a_random_seed_loc,
           'num_buyers' : d_base['num_buyers'],
           'gamma' : d_base['gamma'],
           'scalar_tax' : d_base['scalar_tax'],
           'a_cost' : d_base['a_cost'],
           'endowment' : d_base['endowment'],
           'randomize_loc' : d_base['randomize_loc'],
           'randomize_quant' : d_base['randomize_quant'],
           'tax_model' : d_base['tax_model'],
           'm_tax' : d_base['m_tax']} 
    a_ignore_keys = ret.keys()
    d_data = mean_dic_from_a_dic_base(a_dic, a_ignore_keys)
    ret.update(d_data)
    return ret


if __name__ == "__main__":
    folder = './output/data/'
    a_fn = glob.glob(folder + 'turn*')
    if len(a_fn) == 0:
        raise ValueError('No Files Found')
    a_dic = [jl.load(fn) for fn in a_fn]
    mean_dic = mean_dic_from_a_dic(a_dic)
    jl.dump(mean_dic, folder + 'mean_turn1.pkl')

