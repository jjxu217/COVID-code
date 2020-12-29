from pyomo.pysp.annotations import StochasticConstraintBoundsAnnotation
from GRAbasemodel import model
model.stoch_rhs = StochasticConstraintBoundsAnnotation()

model.stoch_rhs.declare(model.dummy_d)
model.stoch_rhs.declare(model.dummy_d_k)

num_scenarios = 1
one_secnario_data = ([0 for j in model.state_set for t in model.time0_set],\
    [0 for j in model.state_set for t in model.time1_set for k in model.K_set])
scenario_data = {'Scenario1' : one_secnario_data}

def pysp_scenario_tree_model_callback():
    from pyomo.pysp.scenariotree.tree_structure_model import \
        CreateConcreteTwoStageScenarioTreeModel

    st_model = CreateConcreteTwoStageScenarioTreeModel(num_scenarios)

    first_stage = st_model.Stages.first()
    second_stage = st_model.Stages.last()

    # First Stage
    st_model.StageCost[first_stage] = 'FirstStageCost'
    st_model.StageVariables[first_stage].add('s0')
    st_model.StageVariables[first_stage].add('x')
    st_model.StageVariables[first_stage].add('s0_nb')

    # Second Stage
    st_model.StageCost[second_stage] = 'SecondStageCost'
    st_model.StageVariables[second_stage].add('delta')
    st_model.StageVariables[second_stage].add('dummy')

    st_model.StageVariables[second_stage].add('s1')
    st_model.StageVariables[second_stage].add('y')
    st_model.StageVariables[second_stage].add('s1_nb')


    return st_model

def pysp_instance_creation_callback(scenario_name, node_names):

    #
    # Clone a new instance and update the stochastic
    # parameters from the sampled scenario
    #

    instance = model.clone()

    demands, demands_k = scenario_data[scenario_name]
    i = 0
    for j in instance.state_set:
        for t in instance.time0_set:
            instance.demand[j, t].value = demands[i]
            i += 1
    
    i = 0
    for j in instance.state_set:
        for t in instance.time1_set:
            for k in instance.K_set:
                instance.demand_k[j, t, k].value = demands_k[i]
                i += 1

    return instance
