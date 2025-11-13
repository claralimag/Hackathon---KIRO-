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

    cost_route = np.inf
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

            cout = np.inf
            best_customer = None

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
                        best_customer = customer_id
                    
                if best_customer is not None:
                    # Mettre à jour la route avec le meilleur client trouvé
                    route.visites_routes.append(best_customer)
                    route.n_orders += 1
                    arrival = current_time + travel_time(route.vehicle_id, route.visites_routes[-2], best_customer, current_time)
                    route.arrival_times.append(arrival)
                    departure = arrival + dataset.loc[dataset['id'] == best_customer, 'service_time'].iloc[0]
                    route.departure_times.append(departure)

                    current_time = departure
                    capacity_remaining -= dataset.loc[dataset['id'] == best_customer, 'demand'].iloc[0]
                    unvisited.remove(best_customer)

                    # Réinitialiser pour la prochaine itération
                    cout = np.inf
                    best_customer = None
                else:
                    break  # Aucun client n'a été trouvé, sortir de la boucle
            
            # choose best route found for this vehicle
            if route.total_cost < cost_route:
                cost_route = route.total_cost
                best_route = route
                best_vehicle = vehicle["id"]
            
        if best_route is not None:
            routes.append(best_route)
            # Remove visited customers from unvisited set
            for customer in best_route.visites_routes[1:]:  # Exclude depot
                if customer in unvisited:
                    unvisited.remove(customer)
        else:
            break  # No feasible route found, exit the loop

    return routes




    

