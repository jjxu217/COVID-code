import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from math import ceil
from datetime import timedelta
from datetime import datetime

# read parameters
with open('temp.txt', 'r') as f:
    exec(f.read())
with open(now + '/config.txt', 'r') as f:
    exec(f.read())

# read data
t = 7
date = pd.to_datetime(date)
prev_date = date - timedelta(t)
data = pd.read_csv('{}/{}.csv'.format(now, prev_date.strftime("%m%d")), header=0, index_col=0)



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

in_flow[51] = []
out_flow[51] = list(range(51))
for n in range(51):
    flow_mapping[(51, n)] = idx
    flow_mapping_rev[idx] = (51, n)
    idx += 1


# Farmer: Annotated with location of stochastic matrix entries
#         for use with pysp2smps conversion tool.
#
# Imports
#
from pyomo.core import *
from pyomo.pysp.annotations import StochasticConstraintBoundsAnnotation
from pyomo.environ import RangeSet
#
# Model
#

model = ConcreteModel()

#
# Sets
#
S = 51
F = idx
model.time_set = list(range(1, t + 1))
model.state_set = list(range(S))
model.state_p_set = list(range(S + 1))
model.flow_set = list(range(F))
model.flow_no_sns_set = list(range(F - S))
#
# Parameters
#
ini_self = [round(i) for i in data['stock_self']]
ini_nb = defaultdict(list)
for i in range(S):
    ini_nb[i] = eval(data.loc[i, 'stock_nb'])

# Parameters
model.demand = Param(model.state_set, model.time_set, within=NonNegativeReals, mutable=True)

model.U = [round(flow_bound_ratio * ini_self[flow_mapping_rev[j][0]]) for j in model.flow_no_sns_set] + [round(flow_bound_ratio * init_ratio * SNS_stock) for _ in range(S)]
model.G = [round(stock_bound_ratio * i) for i in ini_self]

# first-stage variables
model.s0 = Var(model.state_p_set, within=NonNegativeReals)
model.s0_nb = Var(model.flow_set, within=NonNegativeReals)
model.x = Var(model.flow_no_sns_set, within=Reals)

# second-stage variables
model.delta = Var(model.state_set, model.time_set, within=NonNegativeReals)
model.dummy = Var(model.state_set, model.time_set, within=NonNegativeReals)
# model.theta = Var(model.state_set, within=NonNegativeReals)
# model.mu = Var(model.state_set, within=NonNegativeReals)

# always define the objective as the sum of the stage costs
model.FirstStageCost = Expression(initialize=(sum(lbd*model.s0_nb[i] for i in model.flow_no_sns_set)))
model.SecondStageCost = \
    Expression(initialize=(sum(model.delta[s, t] for s in model.state_set for t in model.time_set)))

# always define the objective as the sum of the stage costs
model.obj = Objective(expr=model.FirstStageCost + model.SecondStageCost)


# first stage constraints
model.stock_self = Constraint(model.state_set, rule=lambda model, j: model.s0[j] + sum(model.x[flow_mapping[(j, i)]] for i in out_flow[j]) == ini_self[j])
model.stock_SNS_self = Constraint(rule=lambda model: model.s0[51] + sum(model.s0_nb[flow_mapping[(51, i)]] for i in out_flow[51]) == init_ratio * SNS_stock)

model.stock_nb = ConstraintList()
for j in model.state_set:
    for dum, i in enumerate(out_flow[j]):
        f = flow_mapping[(i, j)] 
        model.stock_nb.add(model.s0_nb[f] - model.x[f] == ini_nb[j][dum])
    # f = flow_mapping[(51, j)] 
    # model.stock_nb.add(model.s0_nb[f] - model.x[f] == ini_SNS[j])
        

        

model.s0_bound = Constraint(model.state_set, \
                                rule=lambda model, j: model.s0[j] >= model.G[j])
model.x_bound = Constraint(model.flow_no_sns_set, \
                                rule=lambda model, j: model.x[j] <= model.U[j])
model.s0_nb_bound = Constraint(list(range(F - S, F)), \
                                rule=lambda model, j: model.s0_nb[j] <= model.U[j])

# second stage constraints
#model.delta_s_d = Constraint(model.state_set, model.time_set, rule=lambda model, j, t: model.delta[j, t] + model.s0[j] >= model.dummy[j, t])
model.delta_bound = ConstraintList()
for j in model.state_set:
    for tt in model.time_set:
        total_stock = model.s0[j] + model.s0_nb[flow_mapping[(51, j)] ] + sum(model.s0_nb[flow_mapping[(i, j)]] for i in out_flow[j])
        model.delta_bound.add(model.delta[j, tt] + total_stock - model.dummy[j, tt] >= 0)

# 
# these two constraints have stochastic right-hand-sides
#

model.dummy_d = Constraint(model.state_set, model.time_set, rule=lambda model, j, t: model.dummy[j, t] == model.demand[j, t])



model.demand_table = {}
for j in model.state_set:
    for t in model.time_set:
        model.demand_table[j, t] = [0]