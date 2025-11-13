import pandas as pd 
from math import *
import numpy as np 

vehicles_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/vehicles.csv'
instance1_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_01.csv'

vehicles = pd.read_csv(vehicles_path)
instance1 = pd.read_csv(instance1_path)

############################################################

def gamma(f,t):
    res = 0
    w = (2*pi)/86400
    for n in range (0,4): 
        alpha_f_n = vehicles.iloc[f]['fourier_cos_'+ str(n)]
        beta_f_n = vehicles.iloc[f]['fourier_sin_'+ str(n)]
        res += alpha_f_n*cos(n*w*t) + beta_f_n*sin(n*w*t)
    return res


def convert_x(phi_i, phi_j): 
    ro = 6.371e6
    return ro*((2*pi)/360)*(phi_j-phi_i)


def convert_y(lambda_i, lambda_j): 
    ro = 6.371e6
    phi_0 = 48.764246
    return ro*(cos(((2*pi)/360)*phi_0))*((2*pi/360)*(lambda_j-lambda_i))


def travel_time(f,i,j,t): 
    vehicle_idx = f - 1  
    phi_i, lambda_i = instance1.iloc[i]['latitude'], instance1.iloc[i]['longitude']
    phi_j, lambda_j = instance1.iloc[j]['latitude'], instance1.iloc[j]['longitude']
    speed_factor = gamma(vehicle_idx, t) 
    base_speed = vehicles.iloc[vehicle_idx]['speed']
    actual_speed = base_speed * speed_factor 
    manhattan_dist = abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j)) 
    p_f = vehicles.iloc[vehicle_idx]['parking_time']
    return manhattan_dist/actual_speed + p_f

def delta_m(i,j): 
    phi_i, lambda_i = instance1.iloc[i]['latitude'], instance1.iloc[i]['longitude']
    phi_j, lambda_j = instance1.iloc[j]['latitude'], instance1.iloc[j]['longitude']
    return abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j)) 

def delta_e(i,j): 
    phi_i, lambda_i = instance1.iloc[i]['latitude'], instance1.iloc[i]['longitude']
    phi_j, lambda_j = instance1.iloc[j]['latitude'], instance1.iloc[j]['longitude']
    return sqrt(abs(convert_x(phi_i, phi_j))**2 + abs(convert_y(lambda_i, lambda_j))**2)

def delta_M(instance): 
    """
    outputs matrice of manhattan distances M[i][j]
    """
    n_locations = len(instance)
    M = np.zeros((n_locations, n_locations))
    for i in range(n_locations): 
        for j in range(n_locations): 
            M[i,j] = delta_m(i,j)
    return M


def delta_E(instance): 
    """
    outputs matrice of euclidian distances M[i][j]
    """
    n_locations = len(instance)
    E = np.zeros((n_locations, n_locations))
    for i in range(n_locations): 
        for j in range(n_locations): 
            E[i,j] = delta_e(i,j)
    return E

M1 = delta_M(instance1)
E1 = delta_E(instance1)

############################################################
### Individual routes functions ###
############################################################

def is_feasible(route, f, instance):
    """
    Check if a route is feasible for a given vehicle family f 
    """
    vehicle_idx = f-1
    n = len(route)

    # Start and end at depot
    if route[0] != 0 or route[-1] != 0 : 
        return False 
    
    # Total order weight constraint 
    total_weight = 0
    for i in range(n): 
        total_weight += instance.iloc[i]['order_weight']
    if total_weight > vehicles.iloc[vehicle_idx]['max_capacity']:
        return False 
    
    #### Time constraints ####

    d = 0 # initally t = 0 at depot 

    for k in range(n-1): 
        current_order = route[k]
        next_order = route[k+1]

        arrival_next = d + travel_time(f, current_order, next_order, d)

        if next_order == 0 : 
            pass # if we are at depot : route finished, nothing to check 
        else : 

            # get the delivery times window and duration of delivery 
            start = instance.iloc[next_order]['window_start']
            end = instance.iloc[next_order]['window_end']
            delivery_duration = instance.iloc[next_order]['delivery_duration']

            # Arriving too early -> arrival time becomes start time (wait until start time)
            if arrival_next < start :
                arrival_next = start 
            # Arriving too late 
            if arrival_next > end : 
                return False 
            
            # update duration d by adding delivery duration 
            d = arrival_next + delivery_duration
    
    return True 


def route_cost(route, f, instance): 
    """
    computes objective function for a given route and car family 
    """
    vehicle_idx = f-1
    distancesM = delta_M(instance)
    distancesE = delta_E(instance)

    # rental cost 
    c_rental = vehicles.iloc[vehicle_idx]['rental_cost']

    # Fuel cost 
    fuel_cost_per_meter = vehicles.iloc[vehicle_idx]['fuel_cost']
    c_fuel = 0 
    for k in range (len(route)-1): 
        c_fuel += fuel_cost_per_meter*distancesM[route[k], route[k+1]]
    
    # eucledian radius penalty 
    radius_cost = vehicles.iloc[vehicle_idx]['radius_cost']
    max_euclidian_distance = 0
    delivery_points = [i for i in route if i != 0]
    for i in range(len(delivery_points)):
        for j in range(i + 1, len(delivery_points)):
            a = delivery_points[i]
            b = delivery_points[j]
            max_euclidian_distance = max(max_euclidian_distance, distancesE[a, b])
    c_radius = radius_cost*(0.5*max_euclidian_distance)**2

    return c_rental + c_fuel + c_radius 

def get_deliveries(instance):
    return list(range(1, len(instance)))

############################################################
### Global set of routes functions (solution) ###
############################################################

def solution_cost(R, instance):  
    """
    Computes total cost of a set of routes R
    Typical format for R : 

    R = {
        0: {"family": 1, "route": [0, 12, 5, 19, 0]},
        1: {"family": 2, "route": [0, 8, 21, 7, 0]},
    }
    """
    tot_cost = 0
    for r in R: 
        f = R[r]['family']
        route = R[r]['route']
        tot_cost += route_cost(route, f, instance)
    return tot_cost 

def is_solution_feasible(R, instance): 
    visited = set() # set of all delivery points visited 
    # check each individual route 
    for r in R : 
        # route feasible  
        f = R[r]['family']
        route = R[r]['route']
        if is_feasible(route, f, instance) == False : 
            return False, f"route {r} is infeasible"
    
        # unique service for each drop point 
        for delivery_point in route[1:-1]
            if delivery_point in visited:
                return False, f"delivery {delivery_point} visited more than once"
            visited.add(delivery_point)

    # All delivery points must be visited
    all_deliveries = set(instance[instance['id'] != 0].index)
    missing = all_deliveries - visited
    if len(missing) > 0:
        return False, f"{missing} Missing orders"
    return True 

############################################################
### First simple Heuristic ###
############################################################

def next_feasible_node(previous_node, unvisited, f, current_route, instance):
    """
    Looks for nearest delivery point (feasible) to add to current route 
    """
    M = delta_M(instance)

    distances = []
    for node in unvisited:
        dist = M[previous_node][node] 
        distances.append((dist, node))
    distances.sort()
    
    # try each node from closest to farest 
    for dist, next_node in distances: 
        new_route = current_route[:-1] + [next_node, 0]
        if is_feasible(new_route, f, instance): 
            return next_node 
        
    return None 


def build_solution(instance):
    """
    builds first simple solution 
    """
    R = {}
    r = 0
    f = 1 
    unvisited = set(get_deliveries(instance))

    while unvisited : 
        R[r] = {'family': 1, 'route': [0,0]} # initialize empty route
        current_route = R[r]['route']

        while True : # while its possible to add new nodes to the route : 
            prev_delivery = current_route[-2]
            next_delivery = next_feasible_node(prev_delivery, unvisited, f, current_route, instance)
            
            if next_delivery is None : 
                break 

            current_route.insert(-1, next_delivery) # add just before depot the delivery node 
            unvisited.remove(next_delivery)
        r += 1
        
    return R