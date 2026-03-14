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

        #Set the objective
        self.obj = gp.quicksum(
            self.distance_cost * self.dist_matrix[i, j] * self.x[i, j] 
            for i, j in self.x.keys()
        ) + gp.quicksum(
            self.d[i] * self.loss_demand_cost 
            for i in range(self.num_stations)
        )
        self.model.setObjective(self.obj, GRB.MINIMIZE)

        #Initialize Constraint
        self.flow_constrs = []
        self.inv_constrs = []
        for i in range(self.num_stations):
            bikes_arriving = gp.quicksum(self.x[j, i] for j in range(self.num_stations) if i != j)
            bikes_leaving = gp.quicksum(self.x[i, j] for j in range(self.num_stations) if i != j)
        
            #Flow Constraint
            self.flow_constrs.append(self.model.addConstr(
                bikes_arriving - bikes_leaving + self.d[i] >= 0,
                name=f"flow_constraint_station_{i}"
            ))

            #Cannot ship more than you have out
            self.inv_constrs.append(self.model.addConstr(
                bikes_leaving <= 0,
                name=f"inventory_limit_station_{i}"
                ))

    def update_constraints(self, current_inventory, demand):
        
        for i in range(self.num_stations):
            self.flow_constrs[i].RHS = demand[i] - current_inventory[i]
            self.inv_constrs[i].RHS = current_inventory[i]

        self.model.update()
    
    def solve(self):
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            
            #Extract the flow matrix
            flow = np.zeros((self.num_stations, self.num_stations))
            for i, j in self.x.keys():
                            flow[i, j] = self.x[i, j].X
                    
            #Extract the lost demand
            loss_demand = np.zeros(self.num_stations)
            for i in range(self.num_stations):
                loss_demand[i] = self.d[i].X

            return self.model.objVal, flow, loss_demand
        else:
            print(f"Solver failed! Gurobi status code: {self.model.status}")
            return None, None, None