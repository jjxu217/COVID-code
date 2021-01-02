import os
import numpy as np
import pandas as pd
import math
from pyomo.environ import *
from collections import defaultdict
from pyomo.opt import SolverFactory
from datetime import timedelta
from datetime import datetime
import subprocess
import random

import shutil

def sto_file_generator(date, sto):
    #open the cor file
    prob_name = 'SR_{}'.format(date.strftime("%m%d"))
    sto_file = open(sto, 'w')
    
    sto_file.write('STOCH ' + prob_name + ' \n')
    sto_file.write('BLOCKS DISCRETE REPLACE\n')
    probability = 1 / 3
    
    for j in range(S):
        demand_state = pd.read_csv('../../IHME/' + date.strftime("%m%d") + '/' + data.loc[j, 'State'] + '.csv', header=0, index_col=0)
        paths = [demand_state['InvVen_mean'], demand_state['InvVen_mean'] + (demand_state['InvVen_upper'] - demand_state['InvVen_mean']) / 2, demand_state['InvVen_mean'] + (demand_state['InvVen_lower'] - demand_state['InvVen_mean']) / 2] 
        for k in range(3):
            sto_file.write(' BL BLOCK{} PERIOD2 {}\n'.format(j, probability))

            path = {}
            for time in range(1, t + 1):
                n = 'c_e_dummy_d({}_{})_'.format(j, time)
                path[n] =  math.ceil(paths[k][time - 1])
            path_sorted = sorted(path.items())
            for i in path_sorted:
                sto_file.write('\t RHS {} {}\n'.format(i[0], i[1]))
  
    sto_file.write('ENDATA \n')
    sto_file.close()

def analysis_results(now, date):
    prev_date = date - timedelta(7)
    data = pd.read_csv('{}/{}.csv'.format(now, prev_date.strftime("%m%d")), header=0, index_col=0)
    sol = pd.read_csv('{}/res/SR_{}/sol.csv'.format(now, date.strftime("%m%d")), header=None, index_col=None)
    new_stock_self = [0 for _ in range(52)]
    new_flow = defaultdict(int)
    new_stock_sns = [0 for _ in range(51)]
    for i in range(sol.shape[0]):
        name = sol.iloc[i, 0].split('(')
        if name[0] == 's0':
            new_stock_self[int(name[1].split(')')[0])] = int(round(sol.iloc[i, 1]))
        if name[0] == 's0_nb':
            cnt = int(name[1].split(')')[0])
            f, t = flow_mapping_rev[cnt] 
            if f == 51:
                new_stock_sns[t] = int(round(sol.iloc[i, 1]))    
            else:
                new_flow[flow_mapping_rev[cnt]] = int(round(sol.iloc[i, 1]))
            
            
    data['stock_self'] = new_stock_self
    for i in range(51):
        data.loc[i, 'stock_sns'] += new_stock_sns[i]

    for i in range(51):
        l1 = eval(data.loc[i, 'nb_idx'])
        l2 = [new_flow[j, i] for j in l1]
        data.loc[i, 'stock_nb'] = str(l2)
        
    data.to_csv('{}/{}.csv'.format(now, date.strftime("%m%d")))
    
#generate SPMS model
#solve with SD
#generate new initial data file
def solve_model(now, date):
    #now: string format
    #date: datetime format
    #generate SMPS
    os.system("python -m pyomo.pysp.convert.smps -m SR.py --symbolic-solver-labels --basename SR_{} --output-directory {}/SR_{}".format(date.strftime("%m%d"), now, date.strftime("%m%d")))
    sto = "{}/SR_{name}/SR_{name}.sto".format(now, name=date.strftime("%m%d"))
    if os.path.exists(sto):
        os.remove(sto)
    sto_file_generator(date, sto)
    
    #move sd input file
    source = "{}/SR_{name}".format(now, name=date.strftime("%m%d"))
    destination = "../../SD/spInput/"
    input_file = "../../SD/spInput/SR_{}".format(date.strftime("%m%d"))
    if os.path.exists(input_file):
        os.remove(input_file)
    shutil.move(source, destination)
    
    #run SD
    subprocess.run(['./SD_xcode', 'SR_{}'.format(date.strftime("%m%d")),'./spInput/', './spOutput'], cwd='/Users/jjxu/Desktop/covid_system/SD/')
    source = "../../SD/spOutputtwoSD/SR_{}".format(date.strftime("%m%d"))
    destination = "{}/res/SR_{}".format(now, date.strftime("%m%d"))
    shutil.move(source, destination)
    
    #analysis current solution and generate new init data
    analysis_results(now, date)

def SR_all():
    ##########
    
    for i in range(P):
        date = pd.to_datetime(first_date) + timedelta(i * t)
        with open('temp.txt', 'w') as f:
            f.write('now=\'{}\'\ndate=\'{}\''.format(now, date.strftime('%Y/%m/%d')))
            
        solve_model(now, date)  
        print('Finish {}'.format(date.strftime('%m%d')))
        
    new_first_date = '2020/11/19'
    new_P = 4
    prev_date = pd.to_datetime(new_first_date) - timedelta(t)

    old_date = pd.to_datetime(first_date) + timedelta((P-1) * t)
    df = pd.read_csv('{}/{}.csv'.format(now, old_date.strftime('%m%d')), header=0, index_col=0)
    df.to_csv('{}/{}.csv'.format(now, prev_date.strftime('%m%d')))

    for i in range(new_P):
        date = pd.to_datetime(new_first_date) + timedelta(i * t)
        with open('temp.txt', 'w') as f:
            f.write('now=\'{}\'\ndate=\'{}\''.format(now, date.strftime('%Y/%m/%d')))
            
        solve_model(now, date)  
        print('Finish {}'.format(date.strftime('%m%d')))

    param = 'flow{}_stock{}_ini{}_lbd{}'.format(flow_bound_ratio, stock_bound_ratio, init_ratio, lbd)
    if not os.path.exists('../res_all/' + param):
            os.makedirs('../res_all/' + param)
    source = "{}".format(now)
    destination = '../res_all/' + param 
    shutil.move(source, destination)
    os.rename('../res_all/{}/{}'.format(param, now), '../res_all/{}/{}'.format(param, 'SR'))

###############



t = 7
T = 14
P = 10
S = 51
P = 10
SNS_stock = 12000


flow_bound_ratio = 0.2
stock_bound_ratio = 0.5

lbd = 0.5
for init_ratio in [0.7, 0.8]:
    now = datetime.now()
    now = now.strftime("%m%d_%H_%M")
    os.mkdir(now)

    raw = pd.read_csv('../../init_data.csv', header=0, index_col=0)

    #general param
    states_name = raw['State']
    first_date = '2020/03/25'
    #write to file
    with open('temp.txt', 'w') as f:
        f.write('now=\'{}\'\ndate=\'{}\''.format(now, first_date))
        
    with open(now + '/config.txt', 'w') as f:
        f.write('lbd={} \nSNS_stock={} \ninit_ratio={} \nflow_bound_ratio={} \nstock_bound_ratio={}\n'.format(lbd, SNS_stock, init_ratio, flow_bound_ratio,stock_bound_ratio))

    ###########
    prev_date = pd.to_datetime(first_date) - timedelta(t)
    data = raw.rename(columns={'Available capacity': "stock_self"})
    data['stock_self'] = round(init_ratio * data['stock_self'])
    data['stock_nb'] = None
    for i in range(S):
        data.loc[i, 'stock_nb'] = str([0 for _ in eval(data.loc[i, 'nb'])])
    data['stock_sns'] = [0 for _ in range(S+1)]
    data.to_csv(now + '/' + prev_date.strftime('%m%d') + '.csv')

    ########
    idx = 0
    flow_mapping = {}
    flow_mapping_rev = {}
    out_flow = defaultdict(list)
    in_flow = defaultdict(list)

    for i in range(51):
        nbs = eval(data.iloc[i, 3])
        out_flow[i] = nbs
        for n in nbs:
            flow_mapping[(i, n)] = idx
            flow_mapping_rev[idx] = (i, n)
            in_flow[n].append(i)
            idx += 1

    idx_nb = idx
    in_flow[51] = []
    out_flow[51] = list(range(51))
    for n in range(51):
        flow_mapping[(51, n)] = idx
        flow_mapping_rev[idx] = (51, n)
        idx += 1
    ###########

    SR_all()
