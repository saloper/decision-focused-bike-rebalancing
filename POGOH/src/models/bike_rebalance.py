import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

class BikeRebalanceModel:

    def __init__(self, dist_matrix_file_path, distance_cost, loss_demand_cost):
        
        #Initialize Inputs
        self.dist_matrix = pd.read_csv(dist_matrix_file_path, index_col=0).values
        self.num_stations = self.dist_matrix.shape[0]
        self.distance_cost = distance_cost
        self.loss_demand_cost = loss_demand_cost

        #Create Model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        self.model = gp.Model("Bike_Rebalancing", env=env)

        #Create Decision Variables
        #Flow Between stations
        self.x = self.model.addVars(
            self.num_stations, 
            self.num_stations, 
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

        #Set the objective
        self.obj = gp.quicksum(
            self.distance_cost * self.dist_matrix[i, j] * self.x[i, j] 
            for i in range(self.num_stations) 
            for j in range(self.num_stations)
            if i !=j
        ) + gp.quicksum(
            self.d[i] * self.loss_demand_cost 
            for i in range(self.num_stations)
        )
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def set_constraints(self, current_inventory, demand):
        
        for i in range(self.num_stations):
            bikes_arriving = gp.quicksum(self.x[j, i] for j in range(self.num_stations))
            bikes_leaving = gp.quicksum(self.x[i, j] for j in range(self.num_stations))
            
            #Flow Constraint
            self.model.addConstr(
                current_inventory[i] + bikes_arriving - bikes_leaving + self.d[i] >= demand[i],
                name=f"flow_constraint_station_{i}"
            )
            
            #Cannot ship more than you have out
            self.model.addConstr(
                bikes_leaving <= current_inventory[i],
                name=f"inventory_limit_station_{i}"
            )
    
    def solve(self):
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            return self.model.objVal
        else:
            print(f"Solver failed! Gurobi status code: {self.model.status}")
            return None