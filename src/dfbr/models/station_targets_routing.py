import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel
import numpy as np
import pandas as pd
from dfbr.utils.files import get_path
import matplotlib.pyplot as plt
import networkx as nx

class BikeStationTargetsRouting(optGrbModel):
    def __init__(self, num_stations, max_cap, total_inventory, distance_matrix, current_inventory, truck_cap, max_travel_time):
        self.num_stations = num_stations
        self.max_cap = max_cap
        self.total_inventory = total_inventory
        
        # Physical constraints for the shift
        self.current_inventory = current_inventory 
        self.truck_cap = truck_cap                 
        self.max_travel_time = max_travel_time     
        
        # --- DEPOT SETUP ---
        # The depot is represented as the last index (num_stations)
        self.routing_nodes = num_stations + 1 
        self.depot_idx = num_stations 
        
        # Pad the distance matrix with 0s for the depot (Row N and Col N)
        # This makes travel from/to the depot cost 0 time.
        df = pd.read_parquet(get_path(distance_matrix), engine='pyarrow')
        dist_array = df.to_numpy()
        self.padded_distance = np.zeros((self.routing_nodes, self.routing_nodes))
        self.padded_distance[:self.num_stations, :self.num_stations] = dist_array
        
        super().__init__()
    
    def _getModel(self):
        m = gp.Model()
        m.setParam('OutputFlag', 0) 
        m.Params.Threads = 1
        
        # -----------------------------------------------------------------------
        # 1. Decision Variables
        # -----------------------------------------------------------------------
        # x: The ML targets (PyEPO learns the costs for these) - Only for actual stations
        x = m.addVars(self.num_stations * (self.max_cap + 1), name='x', vtype=GRB.BINARY)
        
        # y: Routing edges for ALL nodes including depot
        y = m.addVars(self.routing_nodes, self.routing_nodes, name='y', vtype=GRB.BINARY)
        
        # v: Visitation flag for ALL nodes
        v = m.addVars(self.routing_nodes, name='v', vtype=GRB.BINARY)
        
        # L: Truck load tracking for ALL nodes
        L = m.addVars(self.routing_nodes, name='L', vtype=GRB.CONTINUOUS, lb=0, ub=self.truck_cap)
        
        # u: Sequence variable for subtour elimination (Applies to stations ONLY, not depot)
        u = m.addVars(self.num_stations, name='u', vtype=GRB.CONTINUOUS, lb=1, ub=self.num_stations)
        
        m.ModelSense = GRB.MINIMIZE
        
        # -----------------------------------------------------------------------
        # 2. Constraints
        # -----------------------------------------------------------------------
        
        # --- A. Target Feasibility (Stations Only) ---
        for i in range(self.num_stations):
            m.addConstr(gp.quicksum([x[(i * (self.max_cap + 1)) + j] for j in range(self.max_cap + 1)]) == 1)
            
        m.addConstr(gp.quicksum([x[(i * (self.max_cap + 1)) + j] * j 
                                 for i in range(self.num_stations) 
                                 for j in range(self.max_cap + 1)]) == self.total_inventory)
        # --- Strict No-Tourist Rule ---
        for i in range(self.num_stations):
            # Get the starting inventory for this specific station
            # Use min() just as a safety net in case a station starts over capacity
            start_inv = min(int(self.current_inventory[i]), self.max_cap)
            
            # Find the binary 'x' variable that corresponds to choosing no change
            target_is_current = x[(i * (self.max_cap + 1)) + start_inv]
            
            # If the solver chooses the target that equals the starting inventory, 
            # this forces v[i] <= 0, mathematically banning the truck from visiting.
            m.addConstr(v[i] <= 1 - target_is_current, name=f"no_tourist_{i}")

                           
        # --- B. Target vs. Visitation Linkage ---
        # Calculate Delta (Net Change) for all nodes
        # Delta > 0 means dropoff. Delta < 0 means pickup.
        Delta = {}
        for i in range(self.num_stations):
            T_i = gp.quicksum([x[(i * (self.max_cap + 1)) + j] * j for j in range(self.max_cap + 1)])
            Delta[i] = T_i - self.current_inventory[i]
            
            # Big-M to force v_i = 1 if inventory changes
            m.addConstr(Delta[i] <= self.max_cap * v[i])
            m.addConstr(-Delta[i] <= self.max_cap * v[i])

        # Depot inherently has 0 change, but MUST be visited
        Delta[self.depot_idx] = 0
        m.addConstr(v[self.depot_idx] == 1, name="force_depot_visit")
        #-- Force the truck to start empty ---
        m.addConstr(L[self.depot_idx] == 0, name="empty_start")

        # --- C. Routing Flow (All Nodes) ---
        for i in range(self.routing_nodes):
            # Total flow out of node i equals v[i] (including potential self-loops)
            m.addConstr(gp.quicksum(y[i, j] for j in range(self.routing_nodes)) == v[i])
            
            # Total flow into node i equals v[i] (including potential self-loops)
            m.addConstr(gp.quicksum(y[j, i] for j in range(self.routing_nodes)) == v[i])
            
            # Ban self-loops ONLY for actual stations. The depot is allowed to stay parked!
            if i < self.num_stations:
                m.addConstr(y[i, i] == 0)

        # --- D. Maximum Shift Time ---
        m.addConstr(gp.quicksum(self.padded_distance[i][j] * y[i, j] 
                                for i in range(self.routing_nodes) 
                                for j in range(self.routing_nodes)) <= self.max_travel_time)

        # --- E. Cargo Load Tracking (All Nodes) ---
        M_truck = self.truck_cap + self.max_cap 
        for i in range(self.routing_nodes):
            for j in range(self.routing_nodes):
                if i != j:
                    m.addConstr(L[j] >= L[i] - Delta[j] - M_truck * (1 - y[i, j]))
                    m.addConstr(L[j] <= L[i] - Delta[j] + M_truck * (1 - y[i, j]))

        # --- F. Sequence Subtour Elimination (Stations Only) ---
        # This forces a single continuous loop anchored to the depot
        for i in range(self.num_stations):
            for j in range(self.num_stations):
                if i != j:
                    m.addConstr(u[j] >= u[i] + 1 - self.num_stations * (1 - y[i, j]))

        # Save references
        self._x_vars = x
        self._y_vars = y
        self._v_vars = v
        self._L_vars = L
        self._gurobi_model = m

        return m, x

    def extract_solution(self):
        """Extracts the routing and inventory plan after optimization."""
        if self._gurobi_model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            print("No valid solution to extract.")
            return None

        route_edges = []
        node_stats = {}
        
        # Calculate derived targets for actual stations
        T = {i: sum(k for k in range(self.max_cap + 1) if self._x_vars[(i * (self.max_cap + 1)) + k].X > 0.5) 
             for i in range(self.num_stations)}

        # Add station data
        for i in range(self.num_stations):
            start_inv = self.current_inventory[i]
            target_inv = T[i]
            net_change = target_inv - start_inv
            visited = self._v_vars[i].X > 0.5
            
            node_stats[i] = {
                'start': start_inv,
                'target': target_inv,
                'change': net_change,
                'visited': visited,
                'is_depot': False
            }
            
        # Add depot data
        node_stats[self.depot_idx] = {
            'start': 0, 'target': 0, 'change': 0, 
            'visited': True, 'is_depot': True
        }

        # Extract edges
        for i in range(self.routing_nodes):
            for j in range(self.routing_nodes):
                if i != j and self._y_vars[i, j].X > 0.5:
                    # FIX: Use L_vars[i] to show the load while driving from i to j
                    route_edges.append((i, j, {'load_on_edge': self._L_vars[i].X}))

        return {'nodes': node_stats, 'edges': route_edges}


def plot_rebalancing_route(solution, coords=None):
    """Visualizes the 1-PDP truck route and station inventory changes."""
    if not solution:
        return

    nodes = solution['nodes']
    edges = solution['edges']
    
    G = nx.DiGraph()
    for i, data in nodes.items():
        G.add_node(i, **data)
    G.add_edges_from(edges)

    plt.figure(figsize=(12, 8))
    pos = coords if coords else nx.spring_layout(G, seed=42)

    dropoff_nodes = [n for n, d in G.nodes(data=True) if d['change'] > 0 and d['visited']]
    pickup_nodes = [n for n, d in G.nodes(data=True) if d['change'] < 0 and d['visited']]
    skipped_nodes = [n for n, d in G.nodes(data=True) if not d['visited']]

    nx.draw_networkx_nodes(G, pos, nodelist=skipped_nodes, node_color='lightgray', node_size=500, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=dropoff_nodes, node_color='lightgreen', node_size=800, edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=pickup_nodes, node_color='lightcoral', node_size=800, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=20, edge_color='black', width=2)

    labels = {n: f"S{n}\n{'+' if d['change'] > 0 else ''}{d['change']}" for n, d in G.nodes(data=True) if d['visited']}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")

    edge_labels = {(u, v): f"Load: {d['load_on_edge']:.0f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='blue')

    plt.title("Optimized Rebalancing Route\nGreen = Drop-off, Red = Pick-up, Blue = Truck Cargo Load", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def print_route_sequence(sol_data):
    """
    Traces the edges of a valid 1-PDP solution and prints the sequence.
    Places the load directly on the arrow between stops.
    """
    if not sol_data or not sol_data['edges']:
        print("The route is empty (the truck stayed parked).")
        return

    # 1. Find the depot ID
    depot_id = None
    for node_id, data in sol_data['nodes'].items():
        if data.get('is_depot', False):
            depot_id = node_id
            break

    if depot_id is None:
        depot_id = sol_data['edges'][0][0]

    # 2. Build a map of "Where do I go next?" and "What is my load?"
    next_stop = {}
    for u, v, data in sol_data['edges']:
        next_stop[u] = (v, data.get('load_on_edge', 0))

    # 3. Traverse the route starting from the depot
    current_node = depot_id
    route_sequence = [f"Depot (S{current_node})"]
    
    max_steps = len(sol_data['nodes']) + 1 
    steps = 0

    while steps < max_steps:
        if current_node not in next_stop:
            route_sequence.append(" --[DEAD END]")
            break 

        next_node, load = next_stop[current_node]
        
        # Format the arrow to hold the load data
        edge_string = f" --[Load: {load:.0f}]--> "
        
        if next_node == depot_id:
            route_sequence.append(f"{edge_string}Depot (S{next_node})")
        else:
            route_sequence.append(f"{edge_string}S{next_node}")

        current_node = next_node
        steps += 1

        if current_node == depot_id:
            break 

    # 4. Print the final result (using "".join instead of "->" since the arrow is built in)
    print("\n🚚 Shift Route Sequence:")
    print("".join(route_sequence))
    print(f"Total Stops: {steps}\n")