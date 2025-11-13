#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <iomanip>
#include <limits>
#include <thread>
#include <future>
#include <random>
#include <functional>

using namespace std;
using namespace std::chrono;

// Constants
const double PI = 3.14159265358979323846;
const double EARTH_RADIUS = 6.371e6;
const double PHI_0 = 48.764246;

// Data structures
struct Location {
    double latitude;
    double longitude;
    double order_weight;
    double window_start;
    double window_end;
    double delivery_duration;
};

struct Vehicle {
    int family;
    double max_capacity;
    double speed;
    double parking_time;
    double rental_cost;
    double fuel_cost;
    double radius_cost;
    vector<double> fourier_cos;
    vector<double> fourier_sin;
};

struct Route {
    int family;
    vector<int> route;
    vector<double> arrival_times;
};

// Global variables
vector<Vehicle> vehicles;
vector<Location> instance_data;
vector<vector<double>> distance_matrix_M;
vector<vector<double>> distance_matrix_E;
unordered_map<string, double> travel_cache;
vector<vector<int>> nearest_neighbors;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// Memory pool for efficient route allocation
class RoutePool {
private:
    vector<Route> pool;
    size_t next_free = 0;
public:
    RoutePool(size_t size) : pool(size), next_free(0) {}
    
    Route* get_route() {
        if (next_free < pool.size()) {
            return &pool[next_free++];
        }
        return nullptr;
    }
    
    void reset() { next_free = 0; }
};

// Saving structure for savings algorithm
struct Saving {
    int i, j;
    double value;
    bool operator<(const Saving& other) const {
        return value > other.value; // Sort descending
    }
};

// Hash function for travel cache
string make_cache_key(int f, int i, int j, double t) {
    return to_string(f) + "_" + to_string(i) + "_" + to_string(j) + "_" + 
           to_string(round(t * 100) / 100);
}

// Utility functions for safe string conversion
int safe_stoi(const string& str) {
    try {
        // Remove whitespace
        string trimmed = str;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
        trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
        
        if (trimmed.empty()) {
            return 0;
        }
        return stoi(trimmed);
    } catch (...) {
        cerr << "Error parsing int: '" << str << "'" << endl;
        return 0;
    }
}

double safe_stod(const string& str) {
    try {
        // Remove whitespace
        string trimmed = str;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
        trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
        
        if (trimmed.empty()) {
            return 0.0;
        }
        return stod(trimmed);
    } catch (...) {
        cerr << "Error parsing: '" << str << "'" << endl;
        return 0.0;
    }
}

// Random number utilities
double random_double() {
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

int random_int(int min, int max) {
    uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

// CSV reading functions
vector<string> split(const string& line, char delimiter) {
    vector<string> tokens;
    stringstream ss(line);
    string token;
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void load_vehicles(const string& filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: Cannot open vehicles file: " << filepath << endl;
        return;
    }
    
    string line;
    bool first_line = true;
    
    cout << "Loading vehicles from: " << filepath << endl;
    
    while (getline(file, line)) {
        if (first_line) {
            first_line = false;
            cout << "Header: " << line << endl;
            continue;
        }
        
        if (line.empty()) continue;
        
        vector<string> tokens = split(line, ',');
        
        if (tokens.size() < 14) {
            cerr << "Warning: Invalid vehicle line (expected 14+ columns, got " 
                 << tokens.size() << "): " << line << endl;
            continue;
        }
        
        Vehicle v;
        v.family = safe_stoi(tokens[0]);
        v.max_capacity = safe_stod(tokens[1]);
        v.rental_cost = safe_stod(tokens[2]);
        v.fuel_cost = safe_stod(tokens[3]);
        v.radius_cost = safe_stod(tokens[4]);
        v.speed = safe_stod(tokens[5]);
        v.parking_time = safe_stod(tokens[6]);
        
        v.fourier_cos.resize(4);
        v.fourier_sin.resize(4);
        for (int i = 0; i < 4; i++) {
            v.fourier_cos[i] = safe_stod(tokens[7 + i]);
            v.fourier_sin[i] = safe_stod(tokens[11 + i]);
        }
        
        vehicles.push_back(v);
        cout << "Loaded vehicle: capacity=" << v.max_capacity 
             << ", speed=" << v.speed << endl;
    }
    
    cout << "Total vehicles loaded: " << vehicles.size() << endl;
}

void load_instance(const string& filepath) {
    instance_data.clear();
    ifstream file(filepath);
    
    if (!file.is_open()) {
        cerr << "Error: Cannot open instance file: " << filepath << endl;
        return;
    }
    
    string line;
    bool first_line = true;
    int line_num = 0;
    
    cout << "Loading instance from: " << filepath << endl;
    
    while (getline(file, line)) {
        line_num++;
        
        if (first_line) {
            first_line = false;
            cout << "Header: " << line << endl;
            continue;
        }
        
        if (line.empty()) continue;
        
        vector<string> tokens = split(line, ',');
        
        if (tokens.size() < 7) {
            cerr << "Warning: Invalid instance line " << line_num 
                 << " (expected 7 columns, got " << tokens.size() << "): " << line << endl;
            continue;
        }
        
        Location loc;
        // Skip tokens[0] which is the id
        loc.latitude = safe_stod(tokens[1]);
        loc.longitude = safe_stod(tokens[2]);
        loc.order_weight = safe_stod(tokens[3]);
        loc.window_start = safe_stod(tokens[4]);
        loc.window_end = safe_stod(tokens[5]);
        loc.delivery_duration = safe_stod(tokens[6]);
        
        instance_data.push_back(loc);
    }
    
    cout << "Total locations loaded: " << instance_data.size() << endl;
    
    if (!instance_data.empty()) {
        cout << "First location: lat=" << instance_data[0].latitude 
             << ", lon=" << instance_data[0].longitude 
             << ", weight=" << instance_data[0].order_weight << endl;
    }
}

// Core mathematical functions
double gamma_function(int f, double t) {
    if (f < 0 || f >= vehicles.size()) {
        return 1.0; // Default multiplier
    }
    
    double res = 0;
    double w = (2 * PI) / 86400;
    
    for (int n = 0; n < 4; n++) {
        double alpha_f_n = vehicles[f].fourier_cos[n];
        double beta_f_n = vehicles[f].fourier_sin[n];
        res += alpha_f_n * cos(n * w * t) + beta_f_n * sin(n * w * t);
    }
    return res;
}

double convert_x(double phi_i, double phi_j) {
    return EARTH_RADIUS * ((2 * PI) / 360) * (phi_j - phi_i);
}

double convert_y(double lambda_i, double lambda_j) {
    return EARTH_RADIUS * cos(((2 * PI) / 360) * PHI_0) * ((2 * PI / 360) * (lambda_j - lambda_i));
}

double travel_time(int f, int i, int j, double t) {
    if (f <= 0 || f > vehicles.size() || i < 0 || i >= instance_data.size() || 
        j < 0 || j >= instance_data.size()) {
        return 0.0;
    }
    
    int vehicle_idx = f - 1;
    double phi_i = instance_data[i].latitude;
    double lambda_i = instance_data[i].longitude;
    double phi_j = instance_data[j].latitude;
    double lambda_j = instance_data[j].longitude;
    
    double speed_factor = gamma_function(vehicle_idx, t);
    double base_speed = vehicles[vehicle_idx].speed;
    
    if (base_speed <= 0) base_speed = 1.0; // Avoid division by zero
    
    double actual_speed = base_speed * max(speed_factor, 0.1); // Minimum speed
    double manhattan_dist = abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j));
    double p_f = vehicles[vehicle_idx].parking_time;
    
    return manhattan_dist / actual_speed + p_f;
}

double travel_time_fast(int f, int i, int j, double t) {
    string key = make_cache_key(f, i, j, t);
    
    auto it = travel_cache.find(key);
    if (it != travel_cache.end()) {
        return it->second;
    }
    
    double val = travel_time(f, i, j, t);
    travel_cache[key] = val;
    return val;
}

// Distance matrix computation
void compute_distance_matrices() {
    int n = instance_data.size();
    if (n == 0) {
        cerr << "Error: No instance data loaded!" << endl;
        return;
    }
    
    cout << "Computing distance matrices for " << n << " locations..." << endl;
    
    distance_matrix_M.assign(n, vector<double>(n));
    distance_matrix_E.assign(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double phi_i = instance_data[i].latitude;
            double lambda_i = instance_data[i].longitude;
            double phi_j = instance_data[j].latitude;
            double lambda_j = instance_data[j].longitude;
            
            double x_dist = convert_x(phi_i, phi_j);
            double y_dist = convert_y(lambda_i, lambda_j);
            
            distance_matrix_M[i][j] = abs(x_dist) + abs(y_dist);
            distance_matrix_E[i][j] = sqrt(x_dist * x_dist + y_dist * y_dist);
        }
    }
    
    cout << "Distance matrices computed successfully." << endl;
}

// Pre-compute nearest neighbors for faster search
void precompute_neighborhoods() {
    int n = instance_data.size();
    nearest_neighbors.assign(n, vector<int>());
    
    cout << "Pre-computing nearest neighbors..." << endl;
    
    for (int i = 0; i < n; i++) {
        vector<pair<double, int>> distances;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                distances.push_back({distance_matrix_M[i][j], j});
            }
        }
        sort(distances.begin(), distances.end());
        
        // Store top 15 nearest neighbors
        int max_neighbors = min(15, (int)distances.size());
        for (int k = 0; k < max_neighbors; k++) {
            nearest_neighbors[i].push_back(distances[k].second);
        }
    }
    
    cout << "Nearest neighbors computed." << endl;
}

// Feasibility checking
vector<double> compute_arrival_times(const vector<int>& route, int f) {
    vector<double> arrival = {0};
    double t = 0;
    
    for (int k = 0; k < route.size() - 1; k++) {
        int i = route[k];
        int j = route[k + 1];
        
        if (i < 0 || i >= instance_data.size() || j < 0 || j >= instance_data.size()) {
            return {}; // Invalid route
        }
        
        double arr = t + travel_time_fast(f, i, j, t);
        
        if (j != 0) {
            double start = instance_data[j].window_start;
            double end = instance_data[j].window_end;
            double serv = instance_data[j].delivery_duration;
            
            if (arr < start) {
                arr = start;
            }
            if (arr > end) {
                return {}; // Return empty vector for infeasible
            }
            
            t = arr + serv;
        } else {
            t = arr;
        }
        
        arrival.push_back(t);
    }
    
    return arrival;
}

bool is_feasible(const vector<int>& route, int f) {
    if (f <= 0 || f > vehicles.size()) {
        return false;
    }
    
    int vehicle_idx = f - 1;
    
    // Check depot constraints
    if (route.empty() || route[0] != 0 || route.back() != 0) {
        return false;
    }
    
    // Check capacity constraint
    double total_weight = 0;
    for (int i = 1; i < route.size() - 1; i++) {
        if (route[i] < 0 || route[i] >= instance_data.size()) {
            return false;
        }
        total_weight += instance_data[route[i]].order_weight;
    }
    if (total_weight > vehicles[vehicle_idx].max_capacity) {
        return false;
    }
    
    // Check time constraints
    vector<double> arrival_times = compute_arrival_times(route, f);
    return !arrival_times.empty();
}

// Cost calculation
double route_cost(const vector<int>& route, int f) {
    if (f <= 0 || f > vehicles.size() || route.size() < 2) {
        return numeric_limits<double>::infinity();
    }
    
    int vehicle_idx = f - 1;
    
    // Rental cost
    double c_rental = vehicles[vehicle_idx].rental_cost;
    
    // Fuel cost
    double fuel_cost_per_meter = vehicles[vehicle_idx].fuel_cost;
    double c_fuel = 0;
    for (int k = 0; k < route.size() - 1; k++) {
        int i = route[k];
        int j = route[k + 1];
        if (i >= 0 && i < distance_matrix_M.size() && 
            j >= 0 && j < distance_matrix_M.size()) {
            c_fuel += fuel_cost_per_meter * distance_matrix_M[i][j];
        }
    }
    
    // Radius penalty
    double radius_cost = vehicles[vehicle_idx].radius_cost;
    double max_euclidean_distance = 0;
    vector<int> delivery_points;
    for (int i = 1; i < route.size() - 1; i++) {
        delivery_points.push_back(route[i]);
    }
    
    for (int i = 0; i < delivery_points.size(); i++) {
        for (int j = i + 1; j < delivery_points.size(); j++) {
            int a = delivery_points[i];
            int b = delivery_points[j];
            if (a >= 0 && a < distance_matrix_E.size() && 
                b >= 0 && b < distance_matrix_E.size()) {
                max_euclidean_distance = max(max_euclidean_distance, distance_matrix_E[a][b]);
            }
        }
    }
    
    double c_radius = radius_cost * (0.5 * max_euclidean_distance);
    
    return c_rental + c_fuel + c_radius;
}

double solution_cost(const vector<Route>& solution) {
    double total_cost = 0;
    for (const auto& route : solution) {
        total_cost += route_cost(route.route, route.family);
    }
    return total_cost;
}

// Export function
void export_routes_csv(const vector<Route>& solution, const string& filename) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Cannot create output file: " << filename << endl;
        return;
    }
    
    // Find max route length
    int max_len = 0;
    for (const auto& route : solution) {
        int delivery_count = 0;
        for (int i = 1; i < route.route.size() - 1; i++) {
            delivery_count++;
        }
        max_len = max(max_len, delivery_count);
    }
    
    // Write header
    file << "family";
    for (int i = 1; i <= max_len; i++) {
        file << ",order_" << i;
    }
    file << "\n";
    
    // Write routes
    for (const auto& route : solution) {
        file << route.family;
        
        vector<int> deliveries;
        for (int i = 1; i < route.route.size() - 1; i++) {
            deliveries.push_back(route.route[i]);
        }
        
        for (int i = 0; i < max_len; i++) {
            file << ",";
            if (i < deliveries.size()) {
                file << deliveries[i];
            }
        }
        file << "\n";
    }
    
    file.close();
    cout << "Routes exported to: " << filename << endl;
}

// Solution building
vector<int> get_deliveries() {
    vector<int> deliveries;
    for (int i = 1; i < instance_data.size(); i++) {
        deliveries.push_back(i);
    }
    return deliveries;
}

int next_feasible_node(int previous_node, const unordered_set<int>& unvisited, 
                       int f, const vector<int>& current_route) {
    if (previous_node < 0 || previous_node >= distance_matrix_M.size()) {
        return -1;
    }
    
    vector<pair<double, int>> distances;
    for (int node : unvisited) {
        if (node >= 0 && node < distance_matrix_M.size()) {
            double dist = distance_matrix_M[previous_node][node];
            distances.push_back({dist, node});
        }
    }
    sort(distances.begin(), distances.end());
    
    // Try each candidate in order of distance
    for (const auto& [dist, next_node] : distances) {
        vector<int> new_route = current_route;
        new_route.insert(new_route.end() - 1, next_node);
        
        if (is_feasible(new_route, f)) {
            return next_node;
        }
    }
    
    return -1; // Not found
}

vector<Route> build_solution_with_family(int f) {
    vector<Route> solution;
    unordered_set<int> unvisited;
    vector<int> deliveries = get_deliveries();
    for (int d : deliveries) {
        unvisited.insert(d);
    }
    
    cout << "    Building with family " << f << ", " << unvisited.size() << " deliveries to assign" << endl;
    
    int route_count = 0;
    while (!unvisited.empty() && route_count < 1000) {
        Route route;
        route.family = f;
        route.route = {0, 0};
        
        int deliveries_added = 0;
        int initial_unvisited = unvisited.size();
        
        while (true) {
            int prev_delivery = route.route[route.route.size() - 2];
            int next_delivery = next_feasible_node(prev_delivery, unvisited, f, route.route);
            
            if (next_delivery == -1) {
                break;
            }
            
            route.route.insert(route.route.end() - 1, next_delivery);
            unvisited.erase(next_delivery);
            deliveries_added++;
        }
        
        // Only add route if it has deliveries
        if (deliveries_added > 0) {
            route.arrival_times = compute_arrival_times(route.route, f);
            solution.push_back(route);
        }
        
        route_count++;
        
        // If we can't make progress (no deliveries added), try force-adding one
        if (deliveries_added == 0 && !unvisited.empty()) {
            cout << "      No feasible deliveries found, force-adding one..." << endl;
            
            // Take the first unvisited delivery and create a single-delivery route
            int forced_delivery = *unvisited.begin();
            Route forced_route;
            forced_route.family = f;
            forced_route.route = {0, forced_delivery, 0};
            
            // Check if this forced route is feasible
            if (is_feasible(forced_route.route, f)) {
                forced_route.arrival_times = compute_arrival_times(forced_route.route, f);
                solution.push_back(forced_route);
                unvisited.erase(forced_delivery);
                cout << "      Force-added delivery " << forced_delivery << endl;
            } else {
                cout << "      Cannot force-add delivery " << forced_delivery << ", skipping remaining" << endl;
                break;
            }
        }
        
        // Safety check - if we're not making progress, break
        if (unvisited.size() == initial_unvisited && deliveries_added == 0) {
            cout << "      No progress made, stopping" << endl;
            break;
        }
    }
    
    cout << "    Created " << solution.size() << " routes, " << unvisited.size() << " unassigned deliveries" << endl;
    
    return solution;
}

vector<Route> build_solution() {
    if (vehicles.empty()) {
        cerr << "Error: No vehicles loaded!" << endl;
        return {};
    }
    
    cout << "Building solution with " << vehicles.size() << " vehicle families..." << endl;
    
    vector<Route> best_solution;
    double best_cost = numeric_limits<double>::infinity();
    
    for (int f = 1; f <= vehicles.size(); f++) {
        try {
            cout << "  Trying family " << f << "..." << endl;
            vector<Route> solution = build_solution_with_family(f);
            if (!solution.empty()) {
                double cost = solution_cost(solution);
                cout << "    Family " << f << " cost: " << cost << endl;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_solution = solution;
                }
            }
        } catch (const exception& e) {
            cout << "    Error with family " << f << ": " << e.what() << endl;
            continue;
        }
    }
    
    cout << "Best solution has " << best_solution.size() << " routes" << endl;
    return best_solution;
}

// ========== ADVANCED CONSTRUCTION ALGORITHMS ==========

// Savings algorithm construction
vector<Route> build_solution_savings(int f) {
    vector<Route> solution;
    vector<Saving> savings;
    unordered_set<int> unvisited;
    vector<int> deliveries = get_deliveries();
    
    for (int d : deliveries) {
        unvisited.insert(d);
    }
    
    // Calculate all pairwise savings
    for (int i = 1; i < instance_data.size(); i++) {
        for (int j = i + 1; j < instance_data.size(); j++) {
            if (unvisited.count(i) && unvisited.count(j)) {
                double saving = distance_matrix_M[0][i] + distance_matrix_M[0][j] 
                               - distance_matrix_M[i][j];
                savings.push_back({i, j, saving});
            }
        }
    }
    
    sort(savings.begin(), savings.end());
    
    // Create individual routes first
    unordered_map<int, int> customer_route;
    for (int customer : deliveries) {
        Route route;
        route.family = f;
        route.route = {0, customer, 0};
        if (is_feasible(route.route, f)) {
            route.arrival_times = compute_arrival_times(route.route, f);
            customer_route[customer] = solution.size();
            solution.push_back(route);
            unvisited.erase(customer);
        }
    }
    
    // Merge routes based on savings
    for (const auto& saving : savings) {
        int i = saving.i, j = saving.j;
        
        if (customer_route.count(i) && customer_route.count(j)) {
            int route_i = customer_route[i];
            int route_j = customer_route[j];
            
            if (route_i != route_j && route_i < solution.size() && route_j < solution.size()) {
                // Try to merge routes
                vector<int> merged = solution[route_i].route;
                merged.pop_back(); // Remove ending depot
                
                // Add route j (without starting depot)
                for (int k = 1; k < solution[route_j].route.size(); k++) {
                    merged.push_back(solution[route_j].route[k]);
                }
                
                if (is_feasible(merged, f)) {
                    solution[route_i].route = merged;
                    solution[route_i].arrival_times = compute_arrival_times(merged, f);
                    
                    // Update customer mapping
                    for (int k = 1; k < solution[route_j].route.size() - 1; k++) {
                        customer_route[solution[route_j].route[k]] = route_i;
                    }
                    
                    // Remove route j by marking it empty
                    solution[route_j].route.clear();
                }
            }
        }
    }
    
    // Remove empty routes
    solution.erase(remove_if(solution.begin(), solution.end(), 
                            [](const Route& r) { return r.route.empty(); }), 
                   solution.end());
    
    return solution;
}

// Farthest insertion construction
vector<Route> build_solution_farthest_first(int f) {
    vector<Route> solution;
    unordered_set<int> unvisited;
    vector<int> deliveries = get_deliveries();
    
    for (int d : deliveries) {
        unvisited.insert(d);
    }
    
    while (!unvisited.empty()) {
        Route route;
        route.family = f;
        route.route = {0, 0};
        
        // Start with farthest customer from depot
        int farthest = -1;
        double max_dist = -1;
        for (int customer : unvisited) {
            double dist = distance_matrix_M[0][customer];
            if (dist > max_dist) {
                max_dist = dist;
                farthest = customer;
            }
        }
        
        if (farthest != -1) {
            route.route = {0, farthest, 0};
            if (is_feasible(route.route, f)) {
                unvisited.erase(farthest);
                
                // Add more customers using farthest insertion
                while (true) {
                    int best_customer = -1;
                    int best_position = -1;
                    double best_increase = -1;
                    
                    for (int customer : unvisited) {
                        for (int pos = 1; pos < route.route.size(); pos++) {
                            vector<int> test_route = route.route;
                            test_route.insert(test_route.begin() + pos, customer);
                            
                            if (is_feasible(test_route, f)) {
                                // Calculate minimum distance increase
                                double increase = distance_matrix_M[route.route[pos-1]][customer] +
                                                distance_matrix_M[customer][route.route[pos]] -
                                                distance_matrix_M[route.route[pos-1]][route.route[pos]];
                                
                                if (best_customer == -1 || increase < best_increase) {
                                    best_customer = customer;
                                    best_position = pos;
                                    best_increase = increase;
                                }
                            }
                        }
                    }
                    
                    if (best_customer == -1) break;
                    
                    route.route.insert(route.route.begin() + best_position, best_customer);
                    unvisited.erase(best_customer);
                }
                
                route.arrival_times = compute_arrival_times(route.route, f);
                solution.push_back(route);
            } else {
                unvisited.erase(farthest); // Skip infeasible customer
            }
        }
    }
    
    return solution;
}

// Time-oriented construction
vector<Route> build_solution_time_oriented(int f) {
    vector<Route> solution;
    vector<int> deliveries = get_deliveries();
    
    // Sort by time windows
    sort(deliveries.begin(), deliveries.end(), [](int a, int b) {
        return instance_data[a].window_start < instance_data[b].window_start;
    });
    
    unordered_set<int> unvisited(deliveries.begin(), deliveries.end());
    
    while (!unvisited.empty()) {
        Route route;
        route.family = f;
        route.route = {0, 0};
        
        for (int customer : deliveries) {
            if (unvisited.count(customer)) {
                vector<int> test_route = route.route;
                test_route.insert(test_route.end() - 1, customer);
                
                if (is_feasible(test_route, f)) {
                    route.route = test_route;
                    unvisited.erase(customer);
                }
            }
        }
        
        if (route.route.size() > 2) {
            route.arrival_times = compute_arrival_times(route.route, f);
            solution.push_back(route);
        } else {
            break; // No more feasible insertions
        }
    }
    
    return solution;
}

// Capacity-oriented construction
vector<Route> build_solution_capacity_oriented(int f) {
    vector<Route> solution;
    vector<int> deliveries = get_deliveries();
    
    // Sort by order weight (heaviest first)
    sort(deliveries.begin(), deliveries.end(), [](int a, int b) {
        return instance_data[a].order_weight > instance_data[b].order_weight;
    });
    
    unordered_set<int> unvisited(deliveries.begin(), deliveries.end());
    
    while (!unvisited.empty()) {
        Route route;
        route.family = f;
        route.route = {0, 0};
        
        for (int customer : deliveries) {
            if (unvisited.count(customer)) {
                vector<int> test_route = route.route;
                test_route.insert(test_route.end() - 1, customer);
                
                if (is_feasible(test_route, f)) {
                    route.route = test_route;
                    unvisited.erase(customer);
                }
            }
        }
        
        if (route.route.size() > 2) {
            route.arrival_times = compute_arrival_times(route.route, f);
            solution.push_back(route);
        } else {
            break;
        }
    }
    
    return solution;
}

// ========== LOCAL SEARCH OPERATIONS ==========

// 2-opt improvement for a single route
bool two_opt_improve_route(vector<int>& route, int family) {
    for (int i = 1; i < route.size() - 2; i++) {
        for (int j = i + 1; j < route.size() - 1; j++) {
            // Calculate improvement
            double old_cost = distance_matrix_M[route[i-1]][route[i]] + 
                            distance_matrix_M[route[j]][route[j+1]];
            double new_cost = distance_matrix_M[route[i-1]][route[j]] + 
                            distance_matrix_M[route[i]][route[j+1]];
            
            if (new_cost < old_cost) {
                // Create new route and check feasibility
                vector<int> new_route = route;
                reverse(new_route.begin() + i, new_route.begin() + j + 1);
                
                // Only apply if feasible
                if (is_feasible(new_route, family)) {
                    route = new_route;
                    return true;
                }
            }
        }
    }
    return false;
}

// 2-opt improvement for all routes
bool two_opt_all_routes(vector<Route>& solution) {
    bool improved = false;
    for (auto& route : solution) {
        if (route.route.size() > 3) { // Need at least 4 nodes for 2-opt
            if (two_opt_improve_route(route.route, route.family)) {
                route.arrival_times = compute_arrival_times(route.route, route.family);
                improved = true;
            }
        }
    }
    return improved;
}

// Cross exchange between two routes
bool cross_exchange(vector<Route>& solution, int r1, int r2) {
    if (r1 >= solution.size() || r2 >= solution.size() || r1 == r2) return false;
    
    auto& route1 = solution[r1].route;
    auto& route2 = solution[r2].route;
    
    // Try swapping segments between routes
    for (int i1 = 1; i1 < route1.size() - 1; i1++) {
        for (int j1 = i1; j1 < route1.size() - 1; j1++) {
            for (int i2 = 1; i2 < route2.size() - 1; i2++) {
                for (int j2 = i2; j2 < route2.size() - 1; j2++) {
                    // Create new routes by swapping segments
                    vector<int> new_route1 = route1;
                    vector<int> new_route2 = route2;
                    
                    // Remove segments
                    vector<int> segment1(route1.begin() + i1, route1.begin() + j1 + 1);
                    vector<int> segment2(route2.begin() + i2, route2.begin() + j2 + 1);
                    
                    new_route1.erase(new_route1.begin() + i1, new_route1.begin() + j1 + 1);
                    new_route2.erase(new_route2.begin() + i2, new_route2.begin() + j2 + 1);
                    
                    // Insert segments
                    new_route1.insert(new_route1.begin() + i1, segment2.begin(), segment2.end());
                    new_route2.insert(new_route2.begin() + i2, segment1.begin(), segment1.end());
                    
                    // Check feasibility and improvement
                    if (is_feasible(new_route1, solution[r1].family) && 
                        is_feasible(new_route2, solution[r2].family)) {
                        
                        double old_cost = route_cost(route1, solution[r1].family) + 
                                        route_cost(route2, solution[r2].family);
                        double new_cost = route_cost(new_route1, solution[r1].family) + 
                                        route_cost(new_route2, solution[r2].family);
                        
                        if (new_cost < old_cost) {
                            solution[r1].route = new_route1;
                            solution[r2].route = new_route2;
                            solution[r1].arrival_times = compute_arrival_times(new_route1, solution[r1].family);
                            solution[r2].arrival_times = compute_arrival_times(new_route2, solution[r2].family);
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

// Cross exchange for all route pairs
bool cross_exchange_all(vector<Route>& solution) {
    for (int i = 0; i < solution.size(); i++) {
        for (int j = i + 1; j < solution.size(); j++) {
            if (cross_exchange(solution, i, j)) {
                return true;
            }
        }
    }
    return false;
}

// Route merge and split operations
bool route_merge_split(vector<Route>& solution) {
    // Try merging small routes
    for (int i = 0; i < solution.size(); i++) {
        if (solution[i].route.size() <= 4) { // Small route
            for (int j = i + 1; j < solution.size(); j++) {
                if (solution[i].family == solution[j].family) {
                    // Try to merge
                    vector<int> merged = solution[i].route;
                    merged.pop_back(); // Remove ending depot
                    
                    for (int k = 1; k < solution[j].route.size(); k++) {
                        merged.push_back(solution[j].route[k]);
                    }
                    
                    if (is_feasible(merged, solution[i].family)) {
                        double old_cost = route_cost(solution[i].route, solution[i].family) + 
                                        route_cost(solution[j].route, solution[j].family);
                        double new_cost = route_cost(merged, solution[i].family);
                        
                        if (new_cost < old_cost) {
                            solution[i].route = merged;
                            solution[i].arrival_times = compute_arrival_times(merged, solution[i].family);
                            solution.erase(solution.begin() + j);
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

// Variable Neighborhood Search
void variable_neighborhood_search(vector<Route>& solution) {
    vector<function<bool(vector<Route>&)>> neighborhoods = {
        [](vector<Route>& s) { 
            for (auto& route : s) {
                if (route.route.size() > 3 && two_opt_improve_route(route.route, route.family)) {
                    route.arrival_times = compute_arrival_times(route.route, route.family);
                    return true;
                }
            }
            return false;
        },
        [](vector<Route>& s) { return cross_exchange_all(s); },
        [](vector<Route>& s) { return route_merge_split(s); }
    };
    
    bool improvement = true;
    int iterations = 0;
    while (improvement && iterations < 100) {
        improvement = false;
        for (auto& neighborhood : neighborhoods) {
            if (neighborhood(solution)) {
                improvement = true;
                break; // Restart from first neighborhood
            }
        }
        iterations++;
    }
}

// Simulated Annealing
void make_random_move(vector<Route>& solution) {
    if (solution.empty()) return;
    
    int move_type = random_int(0, 2);
    
    switch (move_type) {
        case 0: { // Random 2-opt in a random route
            int route_idx = random_int(0, solution.size() - 1);
            if (solution[route_idx].route.size() > 3) {
                two_opt_improve_route(solution[route_idx].route, solution[route_idx].family);
                solution[route_idx].arrival_times = compute_arrival_times(
                    solution[route_idx].route, solution[route_idx].family);
            }
            break;
        }
        case 1: { // Random relocation
            if (solution.size() > 1) {
                int r1 = random_int(0, solution.size() - 1);
                int r2 = random_int(0, solution.size() - 1);
                if (r1 != r2 && solution[r1].route.size() > 3) {
                    int customer_idx = random_int(1, solution[r1].route.size() - 2);
                    int customer = solution[r1].route[customer_idx];
                    
                    // Remove from r1
                    solution[r1].route.erase(solution[r1].route.begin() + customer_idx);
                    
                    // Add to r2
                    int pos = random_int(1, solution[r2].route.size() - 1);
                    solution[r2].route.insert(solution[r2].route.begin() + pos, customer);
                    
                    // Update arrival times
                    solution[r1].arrival_times = compute_arrival_times(
                        solution[r1].route, solution[r1].family);
                    solution[r2].arrival_times = compute_arrival_times(
                        solution[r2].route, solution[r2].family);
                }
            }
            break;
        }
        case 2: { // Random cross exchange
            if (solution.size() > 1) {
                int r1 = random_int(0, solution.size() - 1);
                int r2 = random_int(0, solution.size() - 1);
                cross_exchange(solution, r1, r2);
            }
            break;
        }
    }
}

bool simulated_annealing_move(vector<Route>& solution, double temperature) {
    vector<Route> new_solution = solution;
    make_random_move(new_solution);
    
    // Check feasibility
    bool feasible = true;
    for (const auto& route : new_solution) {
        if (!is_feasible(route.route, route.family)) {
            feasible = false;
            break;
        }
    }
    
    if (!feasible) return false;
    
    double old_cost = solution_cost(solution);
    double new_cost = solution_cost(new_solution);
    double delta = new_cost - old_cost;
    
    if (delta < 0 || random_double() < exp(-delta / temperature)) {
        solution = new_solution;
        return true;
    }
    return false;
}

void simulated_annealing(vector<Route>& solution) {
    double initial_temp = 1000.0;
    double final_temp = 1.0;
    double cooling_rate = 0.95;
    int iterations_per_temp = 50;
    
    double temperature = initial_temp;
    
    while (temperature > final_temp) {
        for (int i = 0; i < iterations_per_temp; i++) {
            simulated_annealing_move(solution, temperature);
        }
        temperature *= cooling_rate;
    }
}

// Multi-start construction with different strategies
vector<Route> build_solution_multistart() {
    vector<Route> best_solution;
    double best_cost = numeric_limits<double>::infinity();
    
    // Try different construction strategies
    vector<function<vector<Route>(int)>> strategies = {
        build_solution_with_family,
        build_solution_savings,
        build_solution_farthest_first,
        build_solution_time_oriented,
        build_solution_capacity_oriented
    };
    
    vector<string> strategy_names = {
        "Greedy", "Savings", "Farthest", "Time-oriented", "Capacity-oriented"
    };
    
    for (int f = 1; f <= vehicles.size(); f++) {
        for (int s = 0; s < strategies.size(); s++) {
            try {
                cout << "    Trying " << strategy_names[s] << " with family " << f << "..." << endl;
                vector<Route> solution = strategies[s](f);
                
                if (!solution.empty()) {
                    // Apply improvement
                    variable_neighborhood_search(solution);
                    simulated_annealing(solution);
                    
                    double cost = solution_cost(solution);
                    cout << "      Cost: " << fixed << setprecision(2) << cost << endl;
                    
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_solution = solution;
                        cout << "      New best solution!" << endl;
                    }
                }
            } catch (const exception& e) {
                cout << "      Error: " << e.what() << endl;
                continue;
            }
        }
    }
    
    cout << "  Best multistart cost: " << fixed << setprecision(2) << best_cost << endl;
    return best_solution;
}

// Process a single instance
void process_single_instance(const string& instance_file, int instance_num) {
    cout << "\n=== Processing " << instance_file << " ===" << endl;
    
    // Clear cache for each instance
    travel_cache.clear();
    
    // Load instance
    string filepath = "/Users/antoinechosson/Desktop/KIRO2025/instances/" + instance_file;
    load_instance(filepath);
    
    if (instance_data.empty()) {
        cerr << "Error: No instance data loaded for " << instance_file << endl;
        return;
    }
    
    auto start_time = high_resolution_clock::now();
    
    // Compute distance matrices
    auto matrix_start = high_resolution_clock::now();
    compute_distance_matrices();
    precompute_neighborhoods();
    auto matrix_end = high_resolution_clock::now();
    auto matrix_duration = duration_cast<milliseconds>(matrix_end - matrix_start);
    cout << "Distance matrices and neighborhoods computed in " << matrix_duration.count() << "ms" << endl;
    
    // Build solution using multi-start approach
    auto build_start = high_resolution_clock::now();
    vector<Route> solution = build_solution_multistart();
    auto build_end = high_resolution_clock::now();
    auto build_duration = duration_cast<milliseconds>(build_end - build_start);
    
    if (solution.empty()) {
        cout << "No solution found!" << endl;
        return;
    }
    
    double cost = solution_cost(solution);
    cout << "Advanced solution built in " << build_duration.count() << "ms" << endl;
    cout << "Solution cost: " << fixed << setprecision(2) << cost << endl;
    cout << "Number of routes: " << solution.size() << endl;
    cout << "Cache size: " << travel_cache.size() << " entries" << endl;
    
    // Export CSV
    string output_file = "routes" + to_string(instance_num) + ".csv";
    export_routes_csv(solution, output_file);
    
    auto end_time = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Instance " << instance_num << " completed in " << total_duration.count() << "ms" << endl;
}

// Optimized main function
int main() {
    try {
        cout << "=== ADVANCED KIRO VRP SOLVER ===" << endl;
        cout << "Features: Multi-start, VNS, Simulated Annealing, 2-opt, Cross-exchange" << endl;
        
        // Load vehicles
        load_vehicles("/Users/antoinechosson/Desktop/KIRO2025/instances/vehicles.csv");
        
        if (vehicles.empty()) {
            cerr << "Error: No vehicles loaded. Exiting." << endl;
            return 1;
        }
        
        auto total_start = high_resolution_clock::now();
        
        // Process all instances
        vector<string> instance_files = {
            "instance_01.csv", "instance_02.csv", "instance_03.csv", "instance_04.csv",
            "instance_05.csv", "instance_06.csv", "instance_07.csv", "instance_08.csv",
            "instance_09.csv", "instance_10.csv"
        };
        
        // Option 1: Sequential processing (easier debugging)
        cout << "\nProcessing instances sequentially with advanced algorithms..." << endl;
        for (int i = 0; i < instance_files.size(); i++) {
            process_single_instance(instance_files[i], i + 1);
        }
        
        /* Option 2: Parallel processing (uncomment for maximum speed)
        cout << "\nProcessing instances in parallel..." << endl;
        vector<future<void>> futures;
        
        for (int i = 0; i < instance_files.size(); i++) {
            futures.push_back(async(launch::async, [&instance_files, i]() {
                process_single_instance(instance_files[i], i + 1);
            }));
        }
        
        // Wait for all instances to complete
        for (auto& future : futures) {
            future.get();
        }
        */
        
        auto total_end = high_resolution_clock::now();
        auto total_duration = duration_cast<milliseconds>(total_end - total_start);
        cout << "\n=== All instances processed in " << total_duration.count() << "ms! ===" << endl;
        cout << "Advanced algorithms applied: Multi-start + VNS + Simulated Annealing" << endl;
        
    } catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error occurred" << endl;
        return 1;
    }
    
    return 0;
}