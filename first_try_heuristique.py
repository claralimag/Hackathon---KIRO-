import numpy as np
from dowload_data import dataset_1, vehicles
from class_route import Route

def travel_time(f,i,j,t):
    pass 

def gamma(f,t):
    pass 

def heuristique(dataset, vehicles):
    # Set of unvisited customers
    unvisited = set(dataset['id'].tolist())

    #On commence toujours par le dépôt
    unvisited.remove(0)

    #Liste des routes
    routes = []

    #On parcourt les vehicules disponibles : on cherche à remplir chaque véhicule avant de passer au suivant 
    
    #On prend le vehicule i s'il minimise le cout à l'instant t

    cout = np.inf
    best_route = None
    best_vehicle = None

    while unvisited:
        for index, vehicle in vehicles.iterrows():

            visites_routes = [0]  # On commence toujours par le dépôt
            arrival_time = [] 
            departure_time = [0]

            route = Route(vehicle["id"], 0, visites_routes, arrival_time, departure_time)

            capacity_remaining = vehicle["capacity"]
            current_time = 0

            #Choisir la meilleure route pour le véhicule actuel
            while transported_weight.route < vehicle["capacity"] and unvisited:
                for customer_id in unvisited:
                    f = route.vehicle_id
                    i = route.visites_routes[-1]
                    j = customer_id
                    t = current_time

                    travel_cost = travel_time(f, i, j, t)

                    if travel_cost < cout:
                        cout = travel_cost
                        best_route = route
                        best_vehicle = vehicle
                        best_customer = customer_id




    

