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
    if total_weight > vehicles.iloc[vehicle_idx]['max_capacity']
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

    
    