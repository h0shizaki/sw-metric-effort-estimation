import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def load_data(path, effort_label, to_drop):
    if path.endswith('csv'):
        dataset = pd.read_csv(path).drop(to_drop, axis=1).replace(-1, np.nan).dropna()
    elif path.endswith('xlsx'):
        dataset = pd.read_excel(path).drop(to_drop, axis=1).replace(-1, np.nan).dropna()
    else:
        return None

    train = dataset.iloc[:-1]
    test = dataset.iloc[-1]
    train_x = train.drop(effort_label, axis=1)
    test_x = test.drop(effort_label)
    train_y = train[effort_label]
    test_y = test[effort_label]

    return (train_x , train_y, test_x, test_y)

def interval01(train_x, test_x):
    max_x = np.max(train_x, axis=0)
    min_x = np.min(train_x, axis=0)

    train_x_adj = (train_x - min_x) / (max_x - min_x)
    test_x_adj = (test_x - min_x) / (max_x - min_x)

    return (train_x_adj, test_x_adj) 

def calculate_nn(train_x, test_x, categorial_label):
    train_x_adj, test_x_adj = interval01(train_x, test_x)

    numerical_distance = (train_x_adj.drop(categorial_label, axis=1) - test_x_adj.drop(categorial_label)) **2
    categorical_distance = (1*(train_x_adj[categorial_label] == test_x_adj[categorial_label]))
    euc_distance = np.sqrt(np.sum(pd.concat([numerical_distance, categorical_distance], axis=1) ,axis=1)/np.shape(train_x)[0] )
    rank = np.argsort(euc_distance).values
    return rank

def uavg(rank, train_y, k) :
    estimate_effort = np.mean(train_y[rank[:k]])
    return estimate_effort

def irwm(rank, train_y , k) :
    estimate_effort = np.sum((list(range(k,0,-1)) * train_y[rank[:k]])/np.sum(range(k+1)))
    return estimate_effort

def lsa(rank, train_y, k, train_x, test_x, size_label):
    software_size_train =  train_x[size_label]
    software_size_test = test_x[size_label]
    estimate_effort = np.mean(train_y[rank[:k]]/software_size_train[rank[:k]]) * software_size_test
    return estimate_effort

def rtm(rank, train_y, k, train_x, test_x, categorical_label, size_label, group_label):
    software_size_train = train_x[size_label]
    productivity_train = train_y / software_size_train

    software_size_test = test_x[size_label]
    group_test = test_x[group_label].iloc[0]

    M = productivity_train.loc[(train_x[group_label] == group_test).values].mean()

    all_analogues_productivity = productivity_train * 0
    for i in train_x.index:
        analogues_train = train_x.drop([i])
        analogues_test = train_x.loc[i]
        all_analogues_productivity.loc[i] = productivity_train.iloc[calculate_nn(analogues_train, analogues_test, categorical_label)[0]]

    r, _ = pearsonr(productivity_train.loc[(train_x[group_label] == group_test).values], all_analogues_productivity.loc[(train_x[group_label] == group_test).values],)
    initial_predict = np.mean(train_y[rank[:k]])
    nearest_size = np.mean(software_size_train[rank[:k]])
    nearest_prod = initial_predict / nearest_size

    estimate_effort = software_size_test * (nearest_prod + (M - nearest_prod) * (1 - np.abs(r)))

    return estimate_effort


