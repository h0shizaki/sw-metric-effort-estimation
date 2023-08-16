import pandas as pd
import numpy as np
from analogy import load_data, calculate_nn, uavg, irwm, lsa, rtm


if __name__ == '__main__':
    k=3
    # path='resource/02.desharnais.csv'
    # effort_label='Effort'
    # size_label='PointsAjust'
    # categorical_label = ['Language']
    # group_label = ['Language']
    # to_drop = ['id', 'PointsNonAdjust', 'Adjustment', 'YearEnd', 'Project']
    path ='./resource/albrecht.csv'
    effort_label='Effort'
    size_label='AdjFP'
    categorical_label = []
    group_label = ['Inquiry']
    to_drop = ['FPAdj', 'RawFP']

    train_x , train_y , test_x , test_y = load_data(path, effort_label, to_drop)
    
    rank = calculate_nn(train_x,test_x,categorical_label)

    print('With K=3')
    print('estimate effort with UAVG: ' + uavg(rank, train_y , k).astype(str))
    print('estimate effort with IRWM: ' +irwm(rank, train_y,k).astype(str))
    print('estimate effort with LSA: ' +lsa(rank, train_y, k, train_x, test_x, size_label).astype(str))
    print('estimate effort with RTM: ' +rtm(rank, train_y, k, train_x, test_x, categorical_label, size_label, group_label).astype(str))

    k= 5
    print('With K=5')
    print('estimate effort with UAVG: ' + uavg(rank, train_y , k).astype(str))
    print('estimate effort with IRWM: ' +irwm(rank, train_y,k).astype(str))
    print('estimate effort with LSA: ' +lsa(rank, train_y, k, train_x, test_x, size_label).astype(str))
    print('estimate effort with RTM: ' +rtm(rank, train_y, k, train_x, test_x, categorical_label, size_label, group_label).astype(str))
