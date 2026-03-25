import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

class BikeRebalanceModel:

    def __init__(self, station_file_path, dist_matrix_file_path, loss_demand_cost, over_capacity_cost, movement_cost):
        
        #Initialize Inputs
        stations = pd.read_parquet(station_file_path, engine='pyarrow')
        distance = pd.read_parquet(dist_matrix_file_path, engine='pyarrow')
        #Sort by id 
        stations.sort_values(by='Id', inplace=True)
        distance.sort_index(inplace=True)
        self.capacity = stations['Total Docks'].values
        self.num_stations = len(self.capacity)
        self.dist_matrix = distance.values
        self.loss_demand_cost = loss_demand_cost
        self.over_capacity_cost = over_capacity_cost
        self.movement_cost = movement_cost

        #Create Model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        self.model = gp.Model("Bike_Rebalancing", env=env)

        #Create Decision Variables
        #Flow Between stations
        self.x = self.model.addVars(
            [(i, j) for i in range(self.num_stations) for j in range(self.num_stations) if i != j], 
            vtype=GRB.INTEGER, 
            lb=0, 
            name="flow"
        )

        #Lost Demand at each station
        self.d = self.model.addVars(
            self.num_stations,  
            vtype=GRB.INTEGER, 
            lb=0, 
            name="lost_demand"
        )

        #Over Capacity at each station
        self.c = self.model.addVars(
            self.num_stations,  
            vtype=GRB.INTEGER, 
            lb=0, 
            name="over_cap"
        )

        #Set the objective
        penalty_costs = gp.quicksum(self.d[i] * self.loss_demand_cost + self.c[i] * self.over_capacity_cost for i in range(self.num_stations))
        routing_costs = gp.quicksum(self.x[i, j] * self.dist_matrix[i,j] * self.movement_cost for i in range(self.num_stations) for j in range(self.num_stations) if i != j)
        self.model.setObjective(penalty_costs + routing_costs, GRB.MINIMIZE)

        #Initialize Constraint
        self.shortage_constrs = []
        self.capacity_constrs = []
        self.inv_out_constrs = []
        self.inv_in_constrs = []

        for i in range(self.num_stations):
            bikes_arriving = gp.quicksum(self.x[j, i] for j in range(self.num_stations) if i != j)
            bikes_leaving = gp.quicksum(self.x[i, j] for j in range(self.num_stations) if i != j)
        
            #Lower Bound Constraint (shortage)
            self.shortage_constrs.append(self.model.addConstr(
                bikes_arriving - bikes_leaving + self.d[i] >= 0,
                name=f"shortage_constraint_station_{i}"
            ))

            #Upper Bound Constraint (over capacity)
            self.capacity_constrs.append(self.model.addConstr(
                bikes_arriving - bikes_leaving - self.c[i] <= 0,
                name=f"capacity_constraint_station_{i}"
            ))

            #Cannot ship more than you have out
            self.inv_out_constrs.append(self.model.addConstr(
                bikes_leaving <= 0,
                name=f"inventory_out_station_{i}"
                ))
            
            #Cannot ship more in than you have capacity 
            self.inv_in_constrs.append(self.model.addConstr(
                bikes_arriving - bikes_leaving  <= 0,
                name=f"inventory_in_station_{i}"
                ))

    def update_constraints(self, current_inventory, demand):
        
        for i in range(self.num_stations):
            self.shortage_constrs[i].RHS = -demand[i] - current_inventory[i]
            self.capacity_constrs[i].RHS = self.capacity[i] - demand[i] - current_inventory[i]
            self.inv_out_constrs[i].RHS = current_inventory[i]
            self.inv_in_constrs[i].RHS = self.capacity[i] -current_inventory[i]

        self.model.update()
    
    def solve(self, current_inventory, demand):
        
        self.update_constraints(current_inventory, demand)
        
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            
            #Extract the flow matrix
            flow = np.zeros((self.num_stations, self.num_stations))
            for i, j in self.x.keys():
                            flow[i, j] = self.x[i, j].X
                    
            #Extract the shortage
            shortage = np.zeros(self.num_stations)
            for i in range(self.num_stations):
                shortage[i] = self.d[i].X
            
            #Extract the over capacity
            capacity = np.zeros(self.num_stations)
            for i in range(self.num_stations):
                capacity[i] = self.c[i].X

            return self.model.objVal, flow, shortage, capacity
        elif self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"Conflicting Constraint: {c.ConstrName}")
        
        else:
            print(f"Solver failed! Gurobi status code: {self.model.status}")
            return None, None, None