import pandas as pd 
from math import *

vehicles_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/vehicles.csv'
instance1_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_01.csv'

vehicles = pd.read_csv(vehicles_path)
instance1 = pd.read_csv(instance1_path)

def gamma(f,t):
    res = 0
    f = f-1
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
    f -= 1 
    phi_i, lambda_i = instance1.iloc[i]['latitude'], instance1.iloc[i]['longitude']
    phi_j, lambda_j = instance1.iloc[j]['latitude'], instance1.iloc[j]['longitude']
    y = gamma(f,t)
    speed = vehicles.iloc[f]['speed']
    manhattan_dist = abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j)) 
    p_f = vehicles.iloc[f]['parking_time']
    return manhattan_dist/speed + p_f

def delta_m(i,j): 
    phi_i, lambda_i = instance1.iloc[i]['latitude'], instance1.iloc[i]['longitude']
    phi_j, lambda_j = instance1.iloc[j]['latitude'], instance1.iloc[j]['longitude']
    return abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j)) 

def delta_e(i,j): 
    phi_i, lambda_i = instance1.iloc[i]['latitude'], instance1.iloc[i]['longitude']
    phi_j, lambda_j = instance1.iloc[j]['latitude'], instance1.iloc[j]['longitude']
    return sqrt(abs(convert_x(phi_i, phi_j))**2 + abs(convert_y(lambda_i, lambda_j))**2)
