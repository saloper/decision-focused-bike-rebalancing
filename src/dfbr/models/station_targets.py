import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel

class BikeStationTargets(optGrbModel):
    def __init__(self, num_stations, max_cap, total_inventory):
        self.num_stations = num_stations
        self.max_cap = max_cap
        self.total_inventory = total_inventory
        super().__init__()
    
    def _getModel(self):
        #Create optmization model
        m = gp.Model()
        #Define decision variables
        x = m.addVars(self.num_stations * (self.max_cap + 1), name='x', vtype=GRB.BINARY) #Flatten vector of decisions for entire station, cost matrix
        #Set model objective
        m.ModelSense = GRB.MINIMIZE
        #Constraints
        #Choose one target per station
        for i in range(self.num_stations):
            m.addConstr(gp.quicksum([x[(i * (self.max_cap + 1)) + j] for j in range(self.max_cap + 1)]) == 1)
        #Total targets equal total inventory
        m.addConstr(gp.quicksum([x[(i * (self.max_cap + 1)) + j] * j for i in range(self.num_stations) for j in range(self.max_cap + 1)]) == self.total_inventory)
        #Return model
        return m, x