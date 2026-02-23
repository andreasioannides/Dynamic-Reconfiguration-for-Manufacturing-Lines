import pandas as pd
import simpy
from typing import List, Literal, Union
import random
import numpy as np
import copy
import pyomo.environ as pyo


class Operation:
    def __init__(self, name: str, duration: float, nextOperation: Union[object, None] = None, partOfStation: Union[object, None] = None, partOfLine: Union[object, None] = None, need_capability: str = None):
        self.name = name
        self.duration = duration
        self.nextOperation = nextOperation
        self.partOfStation = partOfStation
        self.partOfLine = partOfLine
        self.need_capability = need_capability

class Agent:
    def __init__(self, name: str, env: Union[simpy.Environment, None], capacity: int, baseSpeed_mean: float, baseSpeed_std: float, has_capability: List[str] = None):
        self.name = name

        if env:
            self.resource = simpy.Resource(env, capacity)
        else:
            self.resource = None

        self.capacity = capacity
        self.baseSpeed_mean = baseSpeed_mean
        self.baseSpeed_std = baseSpeed_std
        self.base_speed = random.gauss(baseSpeed_mean, baseSpeed_std)
        self.used = False  # whether is used or not in the plant
        if has_capability is not None and len(has_capability) > 0:
            self.has_capability = has_capability
        else:
            self.has_capability = []

    def operation_time(self, operation: Operation, t: float):
        '''Real operation time of an Agent based on the operation's expected time, it's speed, and it's fatigue if the agent is a Human.'''
        time = operation.duration

        if isinstance(self, Human):
            time *= self.human_time_factor(t)
        else:
            time *= self.base_speed

        return time

class Human(Agent):
    def __init__(self, name: str, env: Union[simpy.Environment, None], capacity: int, baseSpeed_mean: float, baseSpeed_std: float, fatigue_rate_a: float, fatigue_rate_b: float, partial_recovery_a: float, partial_recovery_b: float, lunch_time: float, has_capability: List[str] = None):
        super().__init__(name, env, capacity, baseSpeed_mean, baseSpeed_std, has_capability)

        self.fatigue_rate_a = fatigue_rate_a
        self.fatigue_rate_b = fatigue_rate_b
        self.fatigue_rate = random.uniform(fatigue_rate_a, fatigue_rate_b)
        self.partial_recovery_a = partial_recovery_a
        self.partial_recovery_b = partial_recovery_b
        self.partial_recovery = random.uniform(partial_recovery_a, partial_recovery_b)
        self.lunch_time = lunch_time

    def human_time_factor(self, t: float):
        if t < self.lunch_time:
            fatigue = 1 + self.fatigue_rate * (t / self.lunch_time)
            time_factor = self.base_speed * fatigue
        else:
            fatigue = 1 + self.fatigue_rate * ((t - self.lunch_time) / self.lunch_time)
            time_factor = (self.base_speed + self.partial_recovery) * fatigue

        return time_factor

class Robot(Agent):
    def __init__(self, name: str, env: Union[simpy.Environment, None], capacity: int, baseSpeed_mean: float, baseSpeed_std: float, has_capability: List[str] = None):
        super().__init__(name, env, capacity, baseSpeed_mean, baseSpeed_std, has_capability)

class Machine(Agent):
    def __init__(self, name: str, env: Union[simpy.Environment, None], capacity: int, baseSpeed_mean: float, baseSpeed_std: float, has_capability: List[str] = None):
        super().__init__(name, env, capacity, baseSpeed_mean, baseSpeed_std)

class Station:
    def __init__(self, name: str, agent: Agent, operations: List[Operation]):
        self.name = name
        self.agent = agent
        self.operations = operations
        self.duration = self.get_duration()

    def get_duration(self):
        duration = 0
        for op in self.operations:
            duration += op.duration

        return duration

class Line:
    def __init__(self, name: str):
        self.name = name
        self.processing_stats = [0, 0, 0]  # [total_processed_products, total_precessing_time, average_processing_time]

    def update_processing_stats(self, new_station_time: float):
        '''For each new product, update the Average Processing Time of the line.'''
        self.processing_stats[1] += new_station_time
        self.processing_stats[0] += 1
        self.processing_stats[2] = self.processing_stats[1] / self.processing_stats[0]

class Plant:
    def __init__(self, name: str, starting_operations: list[Operation], agents: list[Agent], stations: list[Station], small_delay_bounds: tuple = (0.05, 0.1)):
        self.name = name
        self.startingOperations = starting_operations
        self.agents = agents
        self.stations = stations
        # self.bottleneckTime = self.get_bottleneck_time()
        # self.triggerTime = (
        #     self.bottleneckTime + self.bottleneckTime * small_delay_bounds[0],
        #     self.bottleneckTime + self.bottleneckTime * small_delay_bounds[1]
        # )
        self.triggerTime = self.create_trigger_time(small_delay_bounds)

    # def get_bottleneck_time(self):
    #     return max(st.duration for st in self.stations)

    def create_trigger_time(self, small_delay_bounds: tuple = (0.05, 0.1)):
        bottleneckTime = max(st.duration for st in self.stations)
        return (bottleneckTime + bottleneckTime * small_delay_bounds[0],
                bottleneckTime + bottleneckTime * small_delay_bounds[1])

class Optimizer:
    def __init__(self, initial_plant: Plant):
        self.initial_plant = initial_plant
        self.disturbance_info = {
            "type" : None,   
            "agent" : None,
            "station" : None,
            "operation" : None
            }
        self.triggered = False

    def get_agents(self) -> List[Agent]:
        agents = []
        for ag in self.initial_plant.agents:
            if not (self.disturbance_info['type'] == 'small' and not ag.used):
                agents.append(ag)

        return agents
    
    def get_operations(self, plant: Plant) -> List[Operation]:
        operations = []
        for st in plant.stations:
            for op in st.operations:
                operations.append(op)

        return operations

    def new_plant(self, solution: np.ndarray) -> Plant:
        saved_data = {}
        for ag in self.initial_plant.agents:
            saved_data[ag.name] = {
                'resource': ag.resource,
                'env': getattr(ag, 'env', None)
            }
            ag.resource = None
            if hasattr(ag, 'env'): ag.env = None

        new_plant = copy.deepcopy(self.initial_plant)

        for ag in self.initial_plant.agents:
            ag.resource = saved_data[ag.name]['resource']
            if hasattr(ag, 'env'): ag.env = saved_data[ag.name]['env']

        agents = self.get_agents()
        operations = self.get_operations(new_plant)

        for ag in agents:
            ag.used = False

        new_stations = []
        for k, ag in enumerate(agents):
            k_ops = []
            for j, op in enumerate(operations):
                if solution[k, j] == 1:
                    k_ops.append(op)

            if k_ops:
                ag.used = True
                new_st = Station(f'Station_{ag.name}', ag, k_ops)
                new_stations.append(new_st)
                for op in k_ops:
                    op.partOfStation = new_st

        new_plant.agents = agents
        new_plant.stations = new_stations
        new_plant.triggerTime = new_plant.create_trigger_time()

        return new_plant

    def optimize(self, current_time: float, plant_store):
        print(f"OPTIMIZATION STARTED t={current_time:.2f}")

        agents = self.get_agents()
        operations = self.get_operations(self.initial_plant)
        num_agents = len(agents)
        num_ops = len(operations)

        Ct = 1.0
        Cz = 0.3

        model = pyo.ConcreteModel()
        model.K = pyo.RangeSet(0, num_agents - 1)
        model.J = pyo.RangeSet(0, num_ops - 1)

        model.x = pyo.Var(model.K, model.J, domain=pyo.Binary)
        model.L = pyo.Var(domain=pyo.NonNegativeReals)
        model.z = pyo.Var(model.K, domain=pyo.NonNegativeReals)
        model.S = pyo.Var(model.K, model.J, domain=pyo.NonNegativeReals)
        model.y = pyo.Var(model.K, model.J, domain=pyo.NonNegativeReals)
        M = sum(op.duration for op in operations) * 10

        # Constraints

        # 1. Assignment
        def assignment_rule(m, j):
            return sum(m.x[k, j] for k in m.K) == 1
        model.assignment = pyo.Constraint(model.J, rule=assignment_rule)

        # 2. Capabilities
        def capability_rule(m, k, j):
            required = operations[j].need_capability
            if required and (required not in agents[k].has_capability):
                return m.x[k, j] == 0
            return pyo.Constraint.Skip
        model.capability_check = pyo.Constraint(model.K, model.J, rule=capability_rule)

        # 3. Station Accumulation (Real-Time + Station Logic)
        def station_accumulation_rule(m, k, j):
            agent = agents[k]
            # Real-Time Speed/Fatigue
            if isinstance(agent, Human):
                factor = agent.human_time_factor(current_time)
            else:
                factor = agent.base_speed

            real_duration = operations[j].duration * factor

            if j == 0:
                return m.S[k, j] >= real_duration * m.x[k, j]
            else:
                return m.S[k, j] >= real_duration * m.x[k, j] + m.S[k, j - 1] - M * (1 - m.x[k, j])
        model.station_accum_ctr = pyo.Constraint(model.K, model.J, rule=station_accumulation_rule)

        # 4. Station Reset
        def station_reset_rule(m, k, j):
            return m.S[k, j] <= M * m.x[k, j]
        model.station_reset_ctr = pyo.Constraint(model.K, model.J, rule=station_reset_rule)

        # 5. Bottleneck
        def bottleneck_rule(m, k, j):
            return m.L >= m.S[k, j]
        model.bottleneck_ctr = pyo.Constraint(model.K, model.J, rule=bottleneck_rule)

        # 6. Continuity Rule
        def startup_detector_rule(m, k, j):
            if j == 0:
                return pyo.Constraint.Skip
            return m.y[k, j] >= m.x[k, j] - m.x[k, j - 1]
        model.startup_detector = pyo.Constraint(model.K, model.J, rule=startup_detector_rule)

        def continuity_sum_rule(m, k):
            return m.z[k] == m.x[k, 0] + sum(m.y[k, j] for j in range(1, num_ops))
        model.continuity = pyo.Constraint(model.K, rule=continuity_sum_rule)

        # Objective
        def obj_rule(m):
            return Ct * m.L + Cz * sum(m.z[k] for k in m.K)
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        solver = pyo.SolverFactory('glpk')
        results = solver.solve(model, tee=True)

        if (results.solver.status != pyo.SolverStatus.ok) or (results.solver.termination_condition != pyo.TerminationCondition.optimal):
            print("Warning: Solver could not find an optimal solution.")
            return

        solution_matrix = np.zeros((num_agents, num_ops))
        for k in model.K:
            for j in model.J:
                if pyo.value(model.x[k, j]) > 0.5:
                    solution_matrix[k, j] = 1

        new_plant = self.new_plant(solution_matrix)
        yield plant_store.put({'new_plant': new_plant})

        print("\n OPTIMIZATION SUCCESFULLY COMPLETED")

def create_ontology_excel(ontology_model_path: str) -> Plant:
    '''Read the excel and construct the ontology model.'''

    df_instances = pd.read_excel(ontology_model_path, sheet_name="Instances")
    df_relationships = pd.read_excel(ontology_model_path, sheet_name="Relationships")
    df_data_properties_values = pd.read_excel(ontology_model_path, sheet_name="DataProperties Values")

    global_agents = {}
    global_operations = {}
    global_stations = {}
    global_lines = {}

    data_properties_values_property = df_data_properties_values['Property']
    data_properties_values_instance = df_data_properties_values['Instance']
    relationships_property = df_relationships['Property']
    relationships_subject = df_relationships['Subject']

    # Create Operations
    data_ops = df_instances[df_instances['Class'] == 'Operation']
    for _, row in data_ops.iterrows():
        inst = row['Instance']
        duration = df_data_properties_values.loc[(data_properties_values_instance == inst) & (data_properties_values_property == 'operationDuration'), 'Value'].values[0]
        need_cap = df_data_properties_values.loc[(data_properties_values_instance == inst) & (data_properties_values_property == 'needCapability'), 'Value'].values[0]
        global_operations[inst] = Operation(name=row['Label'], duration=duration, nextOperation=None, partOfStation=None, partOfLine=None, need_capability=need_cap)

    # Assign 'nextOperation' attribute for Operations
    for _, row in data_ops.iterrows():
        inst = row['Instance']
        nxts = df_relationships.loc[(relationships_subject == inst) & (relationships_property == 'nextOperation'), 'Object']
        if not nxts.empty and pd.notna(nxts.iloc[0]):
            nextOperation = nxts.values[0]
        else:
            nextOperation = None

        global_operations[inst].nextOperation = global_operations.get(nextOperation)

    # Create Agents
    data_agents = df_instances[df_instances['Class'].isin(['Human', 'Robot', 'Machine'])]
    for _, row in data_agents.iterrows():
        inst = row['Instance']
        mask = (data_properties_values_instance == inst)
        has_cap = df_data_properties_values.loc[(mask) & (data_properties_values_property == 'hasCapability'), 'Value'].values
        baseSpeed_mean = df_data_properties_values.loc[(mask) & (data_properties_values_property == 'baseSpeed_mean'), 'Value'].values[0]
        baseSpeed_std = df_data_properties_values.loc[(mask) & (data_properties_values_property == 'baseSpeed_std'), 'Value'].values[0]
        
        if row['Class'] == 'Human':
            fatigue_rate_a = df_data_properties_values.loc[(mask) & (data_properties_values_property == 'fatigue_rate_a'), 'Value'].values[0]
            fatigue_rate_b = df_data_properties_values.loc[(mask) & (data_properties_values_property == 'fatigue_rate_b'), 'Value'].values[0]
            lunch_time = df_data_properties_values.loc[(mask) & (data_properties_values_property == 'lunch_time'), 'Value'].values[0]
            partial_recovery_a = df_data_properties_values.loc[(mask) & (data_properties_values_property == 'partial_recovery_a'), 'Value'].values[0]
            partial_recovery_b = df_data_properties_values.loc[(mask) & (data_properties_values_property == 'partial_recovery_b'), 'Value'].values[0]
            global_agents[inst] = Human(row['Label'], None, 1, baseSpeed_mean, baseSpeed_std, fatigue_rate_a, fatigue_rate_b, partial_recovery_a, partial_recovery_b, lunch_time, has_cap)
        elif row['Class'] == 'Robot':
            global_agents[inst] = Robot(row['Label'], None, 1, baseSpeed_mean, baseSpeed_std, has_cap)
        elif row['Class'] == 'Machine':
            global_agents[inst] = Machine(row['Label'], None, 1, baseSpeed_mean, baseSpeed_std, has_cap)

    # Create Stations
    data_stations = df_instances[df_instances['Class'] == 'Station']
    for _, row in data_stations.iterrows():
        inst = row['Instance']
        agent = df_relationships.loc[(relationships_subject == inst) & (relationships_property == 'hasAgent'), 'Object'].values[0]
        ops = df_relationships.loc[(relationships_subject == inst) & (relationships_property == 'hasOperation'), 'Object'].values
        ops_objects = [global_operations[op] for op in ops]
        global_stations[inst] = Station(name=row['Label'], agent=global_agents[agent], operations=ops_objects)
        global_agents[agent].used = True
    
    # Assign 'partOfStation' attribute for Operations
    for st in global_stations.values():
        for op in st.operations:
            op.partOfStation = st

    # Create Lines & Assign 'partOfLine' attribute for Operations
    data_lines = df_instances[df_instances['Class'] == 'Line']
    for _, row in data_lines.iterrows():
        inst = row['Instance']
        global_lines[inst] = Line(row['Label'])

        line_ops = df_relationships.loc[(relationships_subject == inst) & (relationships_property == 'hasOperation'), 'Object'].values
        for op in line_ops:
            global_operations[op].partOfLine = global_lines[inst]

    # Create Plants
    data_plants = df_instances[df_instances['Class'] == 'Plant']
    for _, row in data_plants.iterrows():
        inst = row['Instance']
        # ls = df_relationships.loc[(relationships_subject == inst) & (relationships_property == 'hasLine'), 'Object'].values
        # ls_objects = [global_lines[l] for l in ls]
        starting_ops = df_relationships.loc[(relationships_subject == inst) & (relationships_property == 'startingOperation'), 'Object'].values
        starting_ops_objects = [global_operations[op] for op in starting_ops] 
        agents = [ag for ag in global_agents.values()]
        stations = [st for st in global_stations.values()]
        global_plant = Plant(row['Label'], starting_ops_objects, agents, stations) 

    return global_plant

def choose_first_operation(plant: Plant, criterion: Literal['shortest_queue', 'shortest_queue_processing_time']) -> Station:
    '''Choose in which of the parallel lines a new product will enter.
    
    criterion: a) shortest_queue: choose the station/line with the shortest queue, b) shortest_queue_processing_time: choose the station/line with the shortest queue and the smallest average product processing time of the line
    '''

    if criterion == 'shortest_queue':
        return min(plant.startingOperations, key=lambda op: len(op.partOfStation.agent.resource.queue) + op.partOfStation.agent.resource.count)  # queue + occupied slots in the resource
    elif criterion == 'shortest_queue_processing_time':
        return min(plant.startingOperations, key=lambda op: len(op.partOfStation.agent.resource.queue) + op.partOfStation.agent.resource.count + op.partOfLine.processing_stats[2])  # queue + occupied slots in the resource + avg processing time

def process_product(env: simpy.Environment, product_id: str, starting_operation: Operation, product_tracker, optimizer: Union[Optimizer, None], pause_event: Union[simpy.Event, None]):
    '''Process a product through the manufacturing line.'''

    current_op = starting_operation
    current_st = current_op.partOfStation
    current_line = current_op.partOfLine

    while current_st:
        with current_st.agent.resource.request() as req:
            yield req

            # print(f"{env.now}: {product_id} starts {current_st.name}")

            station_starttime = env.now
            while current_op:
                print(f"{env.now}: {product_id} starts {current_op.name}")
                yield env.timeout(current_st.agent.operation_time(current_op, env.now)) 

                if optimizer:
                    current_station_time = env.now - station_starttime

                    if (current_station_time >= plant.triggerTime[0]) and (current_station_time < plant.triggerTime[1]):  # small disturbance
                        print(f'\nSmall delay for agent {current_st.agent.name} at operations {current_op.name}\n')
                        optimizer.triggered = True
                        optimizer.disturbance_info['type'] = 'small'
                        optimizer.disturbance_info['agent'] = current_st.agent
                        optimizer.disturbance_info['station'] = current_st
                        optimizer.disturbance_info['operation'] = current_op
                        yield pause_event
                    elif (current_station_time >= plant.triggerTime[1]):  # large disturbance
                        print(f'\nLarge delay for agent {current_st.agent.name} at operations {current_op.name}\n')
                        optimizer.triggered = True
                        optimizer.disturbance_info['type'] = 'large'
                        optimizer.disturbance_info['agent'] = current_st.agent
                        optimizer.disturbance_info['station'] = current_st
                        optimizer.disturbance_info['operation'] = current_op
                        yield pause_event

                current_op = current_op.nextOperation
                if current_op is None:
                    current_st = None
                    yield product_tracker.put({
                        "product_id": product_id,
                        "completion_time": env.now
                        })
                elif current_op.partOfStation.name != current_st.name:
                    station_total_time = env.now - station_starttime
                    current_st = current_op.partOfStation
                    break
                 
        if current_st and current_op.partOfLine.name != current_line.name:
            current_line.update_processing_stats(station_total_time)
            current_line = current_op.partOfLine
    
def product_generator(env: simpy.Environment, plant: Plant, mean_inflow_rate: float, std_inflow_rate: float, product_tracker: simpy.Store, optimizer: Union[Optimizer, None], pause_event: Union[simpy.Event, None], plant_store):
    '''Creates products over time.'''

    i = 0
    while True:
        product_id = f"Product_{i+1}"

        if len(plant_store.items) > 0:
            # print(plant_store.items[-1]['new_plant'].name)
            starting_operation = choose_first_operation(plant_store.items[-1]['new_plant'], 'shortest_queue_processing_time')
        else:
            starting_operation = choose_first_operation(plant, 'shortest_queue_processing_time')

        env.process(process_product(env, product_id, starting_operation, product_tracker, optimizer, pause_event))

        if optimizer and optimizer.triggered:
            yield pause_event

        arrival_time = max(1e-3, random.gauss(mean_inflow_rate, std_inflow_rate))
        yield env.timeout(delay=arrival_time)
        i += 1

def digital_twin_controller(env, optimizer: Optimizer, pause_event, plant_store):
    while True:
        if optimizer.triggered:
            print('OPTIMIZATION TRIGGERED')
            env.process(optimizer.optimize(env.now, plant_store))
            optimizer.triggered = False
            pause_event.succeed()  # resume paused products
            pause_event = env.event()  # reset the pause_event for the next disturbance
        yield env.timeout(0.5)

def simulation(plant: Plant, simulation_time: int, mean_inflow_rate: float, std_inflow_rate: float, digital_twin: bool):
    '''Generate and process products in the Production Lines for a specific time.'''

    env = simpy.Environment()
    product_tracker = simpy.Store(env)
    for ag in plant.agents:
        ag.resource = simpy.Resource(env, ag.capacity)

    if digital_twin:
        pause_event = env.event()
        optimizer = Optimizer(plant)
        plant_store = simpy.Store(env)
        env.process(digital_twin_controller(env, optimizer, pause_event, plant_store))
    else:
        pause_event = None
        optimizer = None
        plant_store = None

    env.process(product_generator(env, plant, mean_inflow_rate, std_inflow_rate, product_tracker, optimizer, pause_event, plant_store))
    env.run(until=simulation_time)

    completed_products = len(product_tracker.items)
    throughput = round(completed_products/simulation_time, 3)
    print(f'\nTotal completed products: {completed_products}')
    print(f'Throughput: {throughput}')

    return throughput

def multiple_simulations(plant: Plant, simulation_time: int, mean_inflow_rate: float, std_inflow_rate: float, digital_twin: bool) -> float:
    '''Digital twin means trigger optimization when a disturbance event occurs.'''

    # seeds = [42, 102, 202, 302, 402]
    seeds = [202]

    throughputs = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        # create a different sequence of random numbers for each seed 
        for ag in plant.agents:
            if isinstance(ag, Human):
                ag.fatigue_rate = random.uniform(ag.fatigue_rate_a, ag.fatigue_rate_b)
                ag.partial_recovery = random.uniform(ag.partial_recovery_a, ag.partial_recovery_b) 
            if isinstance(ag, Agent):
                ag.base_speed = random.gauss(ag.baseSpeed_mean, ag.baseSpeed_std)

        throughputs.append(simulation(plant, simulation_time, mean_inflow_rate, std_inflow_rate, digital_twin))

    mean_throughput = np.mean(throughputs)
    print(f'\nThroughputs for {len(seeds)} seeds: {throughputs}')

    return mean_throughput


ONTOLOGY_MODEL_PATH = 'manufacturing_ontology.xlsx'
plant = create_ontology_excel(ONTOLOGY_MODEL_PATH)
simulation_time = 8 * 3600  # shift of 8 hours
mean_inflow_rate = 3  # 60
std_inflow_rate = 0.5  # 10

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# throughput = simulation(plant, simulation_time, mean_inflow_rate, std_inflow_rate, digital_twin=True)

mean_throughput = multiple_simulations(plant, simulation_time, mean_inflow_rate, std_inflow_rate, digital_twin=True)
print(f'Mean throughput: {mean_throughput}')