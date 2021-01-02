import os
import numpy as np
import pandas as pd
import math
from pyomo.environ import *
from collections import defaultdict
from pyomo.opt import SolverFactory
from datetime import timedelta
from datetime import datetime
import shutil
import time


now = datetime.now()
now = now.strftime("%m%d_%H_%M")
os.mkdir(now)

raw = pd.read_csv('../../init_data.csv', header=0, index_col=0)


#general param
states_name = raw['State']
first_date = '2020/03/25'
t = 7
T = 14
P = 10
S = 51
SNS_stock = 0

###
lbd = 0.5
init_ratio = 0.8
flow_bound_ratio = 0.2
stock_bound_ratio = 0.5
###

#write to file
with open(now + '/config.txt', 'w') as f:
    f.write('lbd={}\n SNS_stock={}\n init_ratio={}\n flow_bound_ratio={}\n stock_bound_ratio={}\n'.format(lbd, SNS_stock, init_ratio, flow_bound_ratio,stock_bound_ratio))


#flow set
idx = 0
flow_mapping = {}
flow_mapping_rev = {}
out_flow = defaultdict(list)
in_flow = defaultdict(list)

for i in range(S):
    nbs = eval(raw.loc[i, 'nb_idx'])
    out_flow[i] = nbs
    for n in nbs:
        flow_mapping[(i, n)] = idx
        flow_mapping_rev[idx] = (i, n)
        in_flow[n].append(i)
        idx += 1

in_flow[51] = []
out_flow[51] = list(range(51))
for n in range(S):
    flow_mapping[(51, n)] = idx
    flow_mapping_rev[idx] = (51, n)
    idx += 1


def solve_model(date):
    ##########
    ## original_data:DataFram, contains initial info
    ## date:datatime format
    ##########

    prev_date = date - timedelta(t)
    data = pd.read_csv('{}/{}.csv'.format(now, prev_date.strftime('%m%d')), header=0, index_col=0)
    model = ConcreteModel()

    #Sets
    F = idx
    model.time_set = list(range(1, t + 1))
    model.state_set = list(range(S))
    model.state_p_set = list(range(S + 1))
    model.flow_set = list(range(F))
    model.flow_no_sns_set = list(range(F - S))

    # instantiate Param
    ini_self = [round(i) for i in data['stock_self']]
    ini_nb = defaultdict(list)
    for i in range(S):
        ini_nb[i] = eval(data.loc[i, 'stock_nb'])

    # Parameters
    model.demand = Param(model.state_set, model.time_set, within=NonNegativeReals, mutable=True)

    model.U = [round(flow_bound_ratio * ini_self[flow_mapping_rev[j][0]]) for j in model.flow_set] 
    model.G = [round(stock_bound_ratio * i) for i in ini_self]

    for i in range(S):
        state = data.loc[i, 'State']
        df = pd.read_csv('../../IHME/{}/{}.csv'.format(date.strftime('%m%d'), state), header=0, index_col=0)
        for tt in range(t):
            model.demand[i, tt + 1].value = df.iloc[tt, 1]

    # Var
    model.s0 = Var(model.state_p_set, within=NonNegativeReals)
    model.s0_nb = Var(model.flow_set, within=NonNegativeReals)
    model.x = Var(model.flow_no_sns_set, within=Reals)
    model.delta = Var(model.state_set, model.time_set, within=NonNegativeReals)

    #Constraints
    model.stock_self = Constraint(model.state_set, rule=lambda model, j: model.s0[j] + sum(model.x[flow_mapping[(j, i)]] for i in out_flow[j]) - model.s0_nb[flow_mapping[(51, j)]]  == ini_self[j])
    model.stock_SNS_self = Constraint(rule=lambda model: model.s0[51] + sum(model.s0_nb[flow_mapping[(51, i)]] for i in out_flow[51])  == ini_self[51])
   
    model.stock_nb = ConstraintList()
    for j in model.state_set:
        for dum, i in enumerate(out_flow[j]):
            f = flow_mapping[(i, j)] 
            model.stock_nb.add(model.s0_nb[f] - model.x[f] == ini_nb[j][dum])
            
    model.delta_bound = ConstraintList()
    for j in model.state_set:
        for tt in model.time_set:
            total_stock = model.s0[j] + sum(model.s0_nb[flow_mapping[(i, j)]] for i in out_flow[j])
            model.delta_bound.add(model.delta[j, tt] + total_stock >= model.demand[j, tt])
            

    model.s0_bound = Constraint(model.state_set, \
                                    rule=lambda model, j: model.s0[j] >= model.G[j])
    model.x_bound = Constraint(model.flow_no_sns_set, \
                                    rule=lambda model, j: model.x[j] <= model.U[j])
    model.s0_nb_bound = Constraint(list(range(F - S, F)), \
                                    rule=lambda model, j: model.s0_nb[j] <= model.U[j])

    #Objective
    model.unmet = Expression(initialize=(sum(model.delta[s, tt] for s in model.state_set for tt in model.time_set)))
    model.penalty = Expression(initialize=(sum(lbd*model.s0_nb[i] for i in model.flow_set)))
    model.obj = Objective(expr = model.unmet + model.penalty, sense=minimize)

    start_time = time.time()
    SolverFactory('cplex_direct').solve(model)
    runing_time = time.time() - start_time
    with open('{}/time_{}.csv'.format(now, date.strftime('%m%d')), 'w') as f:
        f.write('Running time for {} is {}'.format(date.strftime('%m%d'), runing_time))


    #write result
    for i in model.state_p_set:
        data.loc[i, 'stock_self'] = round(model.s0[i].value)
        
    for i in model.state_set:
        l = eval(data.loc[i, 'nb_idx'])
        new_nb = []
        for n in l:
            j = flow_mapping[(n, i)]
            new_nb.append(round(model.s0_nb[j].value))
        data.loc[i, 'stock_nb'] = str(new_nb)
        

    for i in model.state_set:
        j = flow_mapping[(51, i)]
        data.loc[i, 'stock_sns'] += round(model.s0_nb[j].value)

    data.to_csv(now + '/' + date.strftime('%m%d') + '.csv')
    


#solve the first period
##first iteration values
prev_date = pd.to_datetime(first_date) - timedelta(t)
data = raw.rename(columns={'Available capacity': "stock_self"})
data['stock_self'] = round(init_ratio * data['stock_self'])
data['stock_nb'] = None
for i in range(S):
    data.loc[i, 'stock_nb'] = str([0 for _ in eval(data.loc[i, 'nb'])])
data['stock_sns'] = [0 for _ in range(S+1)]
data.to_csv(now + '/' + prev_date.strftime('%m%d') + '.csv')

for i in range(P):
    date = pd.to_datetime(first_date) + timedelta(i * t)
    solve_model(date)


new_first_date = '2020/11/19'
new_P = 4
prev_date = pd.to_datetime(new_first_date) - timedelta(t)

old_date = pd.to_datetime(first_date) + timedelta((P-1) * t)
df = pd.read_csv('{}/{}.csv'.format(now, old_date.strftime('%m%d')), header=0, index_col=0)
df.to_csv('{}/{}.csv'.format(now, prev_date.strftime('%m%d')))

for i in range(new_P):
    date = pd.to_datetime(new_first_date) + timedelta(i * t)
    solve_model(date)


param = 'flow{}_stock{}_ini{}_lbd{}'.format(flow_bound_ratio, stock_bound_ratio, init_ratio, lbd)
if not os.path.exists('../res_all/' + param):
        os.makedirs('../res_all/' + param)
source = "{}".format(now)
destination = '../res_all/' + param 
shutil.move(source, destination)
os.rename('../res_all/{}/{}'.format(param, now), '../res_all/{}/{}'.format(param, 'det_no_SNS'))