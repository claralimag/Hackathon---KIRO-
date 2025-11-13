import numpy as np
from numpy import pi, cos, sin
import pandas as pd
from dowload_data import dataset_1, vehicles

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
    phi_i, lambda_i = dataset_1.iloc[i]['latitude'], dataset_1.iloc[i]['longitude']
    phi_j, lambda_j = dataset_1.iloc[j]['latitude'], dataset_1.iloc[j]['longitude']
    speed_factor = gamma(vehicle_idx, t) 
    base_speed = vehicles.iloc[vehicle_idx]['speed']
    actual_speed = base_speed * speed_factor 
    manhattan_dist = abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j)) 
    p_f = vehicles.iloc[vehicle_idx]['parking_time']
    return manhattan_dist/actual_speed + p_f

def delta_m(i,j): 
    phi_i, lambda_i = dataset_1.iloc[i]['latitude'], dataset_1.iloc[i]['longitude']
    phi_j, lambda_j = dataset_1.iloc[j]['latitude'], dataset_1.iloc[j]['longitude']
    return abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j)) 

def delta_e(i,j): 
    phi_i, lambda_i = dataset_1.iloc[i]['latitude'], dataset_1.iloc[i]['longitude']
    phi_j, lambda_j = dataset_1.iloc[j]['latitude'], dataset_1.iloc[j]['longitude']
    return np.sqrt(abs(convert_x(phi_i, phi_j))**2 + abs(convert_y(lambda_i, lambda_j))**2)

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

#defining route class
dataset = dataset_1

class Route:
    def __init__(self,family: int, n_orders:int, visited : list[int], arrival_times : list[int], departure_times: list[int]):
        self.family = family
        self.n_orders = n_orders
        self.visited = visited
        self.arrival_times = arrival_times
        self.departure_times = departure_times

    def c_rental(self) -> int:
        return vehicles.loc[vehicles['family'] == self.family, 'rental_cost'].iloc[0]
    
    def c_fuel(self) -> float:
        c_f = vehicles.loc[vehicles['family'] == self.family, 'fuel_cost'].iloc[0]
        total = 0.0
        for i in range(len(self.visited) - 1):
            total += delta_m(self.visited[i], self.visited[i+1])
        return c_f * total
    
    def c_radius(self) -> float:
        c_r = vehicles.loc[vehicles['family'] == self.family, 'radius_cost'].iloc[0]
        # On prend tous les points visités
        if len(self.visited) <= 1:
            return 0.0
        max_val = 0.0
        for i in range(len(self.visited)):
            for j in range(i + 1, len(self.visited)):
                d = delta_e(self.visited[i], self.visited[j])
                if d > max_val:
                    max_val = d
        return (c_r / 4.0) * (max_val ** 2)
    
    def total_cost(self) -> int:
        return self.c_rental() + self.c_fuel() + self.c_radius()
    
    def transported_weight(self) -> float:
        weight = 0.0
        for order in self.visited:
            weight += dataset.loc[dataset['id'] == order, 'order_weight'].iloc[0]
        return weight


def heuristique(dataset, vehicles):
    # Set of unvisited customers
    unvisited = set(dataset['id'].tolist())

    #On commence toujours par le dépôt
    unvisited.remove(0)

    #Liste des routes
    routes = []

    #On parcourt les vehicules disponibles : on cherche à remplir chaque véhicule avant de passer au suivant 
    
    #On prend le vehicule i s'il minimise le cout à l'instant t

    cost_route = np.inf
    best_route = None
    best_vehicle = None

    while unvisited:
        cost_route = np.inf
        best_route = None
        best_vehicle = None
        unvisited_local = unvisited.copy()

        for index, vehicle in vehicles.iterrows():

            visites_routes = [0]  # On commence toujours par le dépôt
            arrival_time = [] 
            departure_time = [0]

            route = Route(vehicle["family"], 0, visites_routes, arrival_time, departure_time)

            capacity_remaining = vehicle["max_capacity"]
            current_time = 0

            cout = np.inf
            best_customer = None

            #Choisir la meilleure route pour le véhicule actuel
            while route.transported_weight() < vehicle["max_capacity"] and unvisited_local:
                for customer_id in unvisited_local:
                    demand = dataset.loc[dataset['id'] == customer_id, 'order_weight'].iloc[0]

                    if route.transported_weight() + demand > vehicle["max_capacity"]:
                        continue     

                    f = route.family
                    i = route.visited[-1]
                    j = customer_id

                    row_i = dataset.loc[dataset['id'] == i]
                    row_j = dataset.loc[dataset['id'] == j]

                    t = current_time

                    travel_cost = travel_time(f, row_i, row_j, t)

                    if travel_cost < cout:
                        cout = travel_cost
                        best_customer = customer_id
                    
                if best_customer is not None:
                    # Mettre à jour la route avec le meilleur client trouvé
                    route.visited.append(best_customer)
                    route.n_orders += 1
                    arrival = current_time + travel_cost
                    route.arrival_times.append(arrival)
                    departure = arrival + dataset.loc[dataset['id'] == best_customer, 'service_time'].iloc[0]
                    route.departure_times.append(departure)

                    current_time = departure
                    capacity_remaining -= dataset.loc[dataset['id'] == best_customer, 'order_weight'].iloc[0]
                    unvisited_local.remove(best_customer)

                    # Réinitialiser pour la prochaine itération
                    cout = np.inf
                    best_customer = None
                else:
                    break  # Aucun client n'a été trouvé, sortir de la boucle
            
            # choose best route found for this vehicle
            route_cost = route.total_cost()
            if route_cost < cost_route:
                cost_route = route_cost
                best_route = route
            best_vehicle = vehicle["id"]
            
        if best_route is not None:
            routes.append(best_route)
            # Remove visited customers from unvisited set
            for customer in best_route.visited[1:]:              
                if customer in unvisited:   
                    unvisited.remove(customer)
        else:
            break  # No feasible route found, exit the loop

    #return to the depot for each route
    for route in routes:
        last_customer = route.visited[-1]
        f = route.family
        i = last_customer
        j = 0  # depot
        t = route.departure_times[-1]
        travel_cost = travel_time(f, i, j, t)
        arrival = t + travel_cost
        route.visited.append(0)
        route.arrival_times.append(arrival)
        route.departure_times.append(arrival)  # assuming no service time at depot
        
    return routes

routes = heuristique(dataset, vehicles)


#Solution file The solution you provide must consist of a single CSV file: routes.csv contains one row per route (|R| in total) with the following columns: family fr order_1 i 1r, order_2 i

#To build this file, first find the maximum number of orders served by a route in your solution N = maxr∈R nr. # type: ignore
#The resulting table should have 2+N columns, and the routes shorter than N should end with empty columns
#(not filled with a space or a null placeholder of any kind).

def save_routes(routes, file_path):
    max_orders = max(len(route.visited) for route in routes)
    columns = ['family'] + [f'order_{i+1}' for i in range(max_orders)]
    
    data = []
    for route in routes:
        row = [route.family] + route.visited + [None] * (max_orders - len(route.visited))
        data.append(row)
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)

# Example usage:
save_routes(routes, '/workspaces/Hackathon---KIRO-/routes.csv')


    

