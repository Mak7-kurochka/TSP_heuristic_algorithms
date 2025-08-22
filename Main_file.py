import random
import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations
import time
import math
from scipy import stats

class Node:
    def __init__(self,value):
        self.value = value
        self.edges = []
        self.parents = []

class Edge:
    def __init__(self,direction=None,parent=None,weight=1):
        self.dir = direction
        self.parent = parent
        self.weight = weight
        self.visibility = 1/weight
        self.pheromone = 1
        self.ants = []

class PathNode:
    def __init__(self,node,parent=None):
        self.node = node
        self.parent = parent

class Ant:
    def __init__(self):
        self.passed = []
        self.dist = 0
        self.routes = []

class Graph:
    def __init__(self,matrix=None):
        self.nodes = {}
        self.edges = {}
        self.matrix = matrix

    def add_or_get(self,name):
        if name in self.nodes.keys():
            return self.nodes[name]
        else:
            self.nodes[name] = Node(name)
            return self.nodes[name]

    def add_edge(self,matrix,names):
        if not self.matrix:
            self.matrix = matrix
        names_vertical = names.reshape(-1,1)
        r=0
        for row in matrix:
            for i in range(len((row))):
                if row[i] != 0:
                    start_node = self.add_or_get(names[r])
                    end_node = self.add_or_get(names_vertical[i][0])
                    try:
                        _ = self.edges[frozenset([start_node.value,end_node.value])]
                    except:
                        edge = Edge(end_node,start_node,row[i])
                        self.edges[frozenset([start_node.value,end_node.value])] = edge
                        start_node.edges.append(edge)
                        end_node.edges.append(edge)
                        end_node.parents.append(start_node)  
                        start_node.parents.append(end_node)                  
            r += 1

    def depth_traversal(self,passed=[]):
        values = list(self.nodes.values())
        stack = []
        for value in values:
            if value not in passed:
                stack.append(value)
                self.depth_traversal_wrap(passed,stack)

    def depth_traversal_wrap(self,passed,stack):
        node = stack[-1]
        passed.append(node)
        print(node.value)
        for edge in node.edges:
            if edge.dir not in passed and edge.dir not in stack:
                stack.append(edge.dir)
                self.depth_traversal_wrap(passed,stack)
        stack.pop()
    
    def breadth_traversal(self,passed=[]):
        values = list(self.nodes.values())
        queue = []
        for value in values:
            if value not in passed:
                queue.append(value)
                self.breadth_traversal_wrap(passed,queue)

    def breadth_traversal_wrap(self,passed,queue):
        if len(queue) == 0:
            return
        node = queue[0]
        queue.remove(node)
        passed.append(node)
        print(node.value)
        for edge in node.edges:
            if edge.dir not in passed and edge.dir not in queue:
                queue.append(edge.dir)
        self.breadth_traversal_wrap(passed,queue)

    def find_path(self, start, end):
        start_node = self.nodes[start]
        end_node = self.nodes[end]
        path = []
        return self.get_path(start_node,end_node,path,[])

    def get_path(self,start,end,path,passed):
        passed.append(start)
        if start == end:
            path.append(start.value)
            return path
        for edge in start.edges:
            if edge.dir not in passed:
                if self.get_path(edge.dir,end,path,passed):
                    path.insert(0,start.value)
                    return path
        passed.pop()

    def find_path_all(self,start,end):
        start_node = self.nodes[start]
        end_node = self.nodes[end]
        paths = []
        self.get_path_all(start_node,end_node,paths,[])
        
        return [[node.value for node in path] for path in paths]

    def get_path_all(self,start,end,paths,passed):
        passed.append(start)
        if start == end:
            paths.append(passed.copy())
        for edge in start.edges:
            if edge.dir not in passed:
                self.get_path_all(edge.dir,end,paths,passed)
        passed.pop()

    def delete_edge(self,start,end):
        start_node = self.nodes[start]
        end_node = self.nodes[end]
        end_node.parents.remove(start_node)
        start_node.parents.remove(end_node)
        for edge in start_node.edges:
            if edge.dir == end_node:
                start_node.edges.remove(edge)
                break
            elif edge.parent == end_node:
                start_node.edges.remove(edge)
                break
        for edge in end_node.edges:
            if edge.dir == start_node:
                end_node.edges.remove(edge)
                break
            elif edge.parent == start_node:
                end_node.edges.remove(edge)
                break  
        del self.edges[frozenset([start_node.value,end_node.value])]
           
    def delete_node(self,value):
        node = self.nodes[value]
        for parent in node.parents.copy():
            self.delete_edge(parent.value,value)
        del(self.nodes[value])

    def find_path_shortest(self,start,end):
        start_node = PathNode(self.nodes[start])
        end_node = self.nodes[end]
        queue = [start_node]
        path = self.get_path_shortest(end_node,queue,[])
        path_value = []
        
        while path:
            path_value.insert(0,path.node.value)
            path = path.parent

        return path_value               

    def get_path_shortest(self,end,queue,passed):
        path_node = queue[0]
        queue.remove(path_node)
        passed.append(path_node.node)
        if path_node.node == end:
            return path_node
        for edge in path_node.node.edges:
            if edge.dir not in passed and edge.dir not in queue:
                queue.append(PathNode(edge.dir,path_node))
                passed.append(edge.dir)
        return self.get_path_shortest(end,queue,passed)
    
    def calculate_distance(self,route):
        dist = 0
        for i in range(len(route)-1):
            dist += self.matrix[route[i]][route[i+1]]
        return dist

###Algorithms start

##NN start
    def find_minimum_edge(self,path):
        edges = []
        current_town = path[-1]
        towns = list(set(self.nodes.keys())-set(path))
        if not towns:
            return None
        else:
            for town in towns:
                edges.append(self.edges[frozenset([town,current_town])].weight)
        min_edge = min(edges)
        return towns[edges.index(min_edge)]

    def nearest_neighbour(self):
        start_node = random.choice(list(self.nodes.values()))
        path = [start_node.value]
        current_town = start_node.value
        
        while len(path) < len(self.nodes):
            current_town = self.find_minimum_edge(path)
            path.append(current_town)
        
        path.append(start_node.value)

        return self.calculate_distance(path),path
###NN end
    
###brute force start
    def brute_force(self):
        names = list(self.nodes.keys())
        start_city = self.nodes[random.choice(names)].value
        del(names[names.index(start_city)])
        routes = []
        for route in permutations(names):
            routes.append([start_city] + list(route) + [start_city])
        routes_weights = []
        for route in routes:
            routes_weights.append(self.calculate_distance(route))
        return min(routes_weights)
###brute force end

###random swapping start
    def swap_nodes(self,route):
        local_route = route.copy()
        del(local_route[0])
        del(local_route[-1])
        first_city, second_city = 0,0
        while first_city == second_city:
            first_city = random.choice(local_route)
            second_city = random.choice(local_route)

        first_city_index = route.index(first_city)
        second_city_index = route.index(second_city)
        route[first_city_index] = second_city
        route[second_city_index] = first_city

        return route

    def random_swaping(self,route_llst=None,batch=1000,s=0.05):
        if not route_llst:
            route = (list(range(0,len(self.nodes.keys()))))
            random.shuffle(route)
            route.append(route[0])
            route_dist = self.calculate_distance(route)
        else:
            route_dist, route = route_llst
            
        i = 0
        Y = 0
        while i < 10**6:
            
            if Y >= batch:
                break
            
            new_route = self.swap_nodes(route.copy())
            new_route_dist = self.calculate_distance(new_route)

            
            if new_route_dist < route_dist:
                route = new_route
                route_dist = new_route_dist
                Y = 0
            elif (new_route_dist - route_dist)/route_dist < s:
                Y += 1

            i += 1

        return route_dist,route
###random swapping end

###2-opt start
    def two_opt(self,path=None):
        if not path:
            route = (list(range(0,len(self.nodes.keys()))))
            random.shuffle(route)
            route.append(route[0])
        else:
            route = path[1]
        iteration = 0
        i = 0
        improved = True
        while i<len(route)-1:

            if iteration >= 10**6:
                
                break

            if improved:
                i=0
                improved=False
                
            A = route[i]
            B = route[i+1]
            j=i+2
            while j<len(route)-1:
                C = route[j]
                D = route[j+1]
                org_dist = self.calculate_distance([A,B]) + self.calculate_distance([C,D])
                new_dist = self.calculate_distance([A,C]) + self.calculate_distance([B,D])
                if new_dist < org_dist:
                    route = route[:i]+[A,C]+route[i+2:j][::-1]+[B,D]+route[j+2:]
                    improved = True
                    break
                j +=1
            i += 1
            iteration += 1
        return self.calculate_distance(route),route
###2-opt end

###Simulated annealing starts
    def simulated_annealing(self,path=None,t_max=10000,t_min=0.1,a=0.995,i_max=500):##t_max was set to have acceptance rate > 0.9
        if not path:
            current_route = (list(range(0,len(self.nodes.keys()))))
            random.shuffle(current_route)
            current_route.append(current_route[0])
        else:
            current_route = path[1]

        t = t_max
        best_distance = self.calculate_distance(current_route)
        best_route = current_route.copy()
        while t>t_min:
            
            for _ in range(i_max):
                
                try:
                    i = random.randint(0,len(current_route)-5)
                    j = random.randint(i+2,len(current_route)-3)
                except:
                    return "Graph is to small"
                
                org_dist = self.calculate_distance([current_route[i], current_route[i+1]]) + self.calculate_distance([current_route[j], current_route[j+1]])
                new_dist = self.calculate_distance([current_route[i], current_route[j]]) + self.calculate_distance([current_route[i+1], current_route[j+1]])
                
                delta = new_dist - org_dist

                if delta < 0 or random.random() <= math.exp(-delta/t):
                    current_route = current_route[:i+1] + current_route[i+1:j+1][::-1] + current_route[j+1:]
                    current_distance = self.calculate_distance(current_route)

                    if current_distance < best_distance:
                        best_route = current_route.copy()
                        best_distance = current_distance

            t *= a

        return best_distance, best_route
    
    def choose_the_town(self,ant,a=2,b=5):
        current_node = ant.passed[-1]
        towns = list(set(self.nodes.keys())-set(ant.passed))
        t_n = []
        for town in towns:
            edge = self.edges[frozenset([town,current_node])]
            t_n.append((edge.pheromone**a) * (edge.visibility**b))
        total = sum(t_n)
        probabilities = [t_n[i]/total for i in range(len(t_n))]   
        town = np.random.choice(towns, p=probabilities)
        return town


    def update_pheromones(self,p=0.3,Q=1000):
        for edge in self.edges.values():
            delta_trail = 0
            if edge.ants:
                for ant in edge.ants:
                    delta_trail += Q/ant.dist
            edge.pheromone = (1-p)*edge.pheromone + delta_trail
            edge.ants = []

    def ant_system(self,m=100,s=0.02,batch=10):
        ants = [Ant() for _ in range(m)]
        keys = list(self.nodes.keys())
        best_route = []
        best_dist = np.inf
        Y = 0
        i = 0
        while i < 2500:
            
            if Y >= batch:
                break
                
            for ant in ants:
                
                ant.routes.append(ant.passed)
                ant.passed = []
                ant.dist = 0
                ant.passed.append(random.choice(keys))

                while len(ant.passed) < len(keys):
                    town = self.choose_the_town(ant)
                    edge = self.edges[frozenset([town,ant.passed[-1]])]
                    ant.passed.append(town)
                    ant.dist += edge.weight
                    edge.ants.append(ant)
            for ant in ants:
                ant.dist += self.edges[frozenset([ant.passed[-1],ant.passed[0]])].weight
                ant.passed.append(ant.passed[0])
                if ant.dist < best_dist:
                    best_route = ant.passed
                    best_dist = ant.dist
                    Y = 0
                elif ((ant.dist - best_dist)/best_dist) < s:
                    Y += 1
            self.update_pheromones()
            i += 1

        return (best_dist,best_route)




def create_matrix(n):
    matrix = np.random.randint(400, 5000, size=(n, n))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return matrix

def sample_size(sample,alpha=0.05):
    u_alpha = stats.norm.ppf(1 - alpha)
    mean = np.average(sample)
    std = np.std(sample)
    allow_eror = alpha * mean

    h = ((u_alpha ** 2) * (std ** 2))/allow_eror ** 2
    return h

def confirm_normal_distribution(sample,N,alpha=0.05):
    obs_frequency, ints, _ = plt.hist(sample)
    intervals = list(zip(ints, ints[1:]))

    mean = np.average(sample)
    std = np.std(sample)

    if std == 0:
        return True

    cdf = [stats.norm.cdf(intervals[i][1],loc=mean,scale=std)-stats.norm.cdf(intervals[i][0],loc=mean,scale=std) for i in range(len(intervals))]

    exp_frequency = [N*cdf[i] for i in range(len(obs_frequency))]


    chi2 = sum([((obs_frequency[i]-exp_frequency[i])**2)/exp_frequency[i] for i in range(len(obs_frequency))])

    dof = len(intervals) - 2

    critical_value = stats.chi2.ppf(1 - alpha, dof)

    print(chi2,critical_value)

    return chi2<critical_value

def pilot_experiment(lof,distance_matrix,minimum,step):
    file_name = input('Please enter the name of .txt file for distances: ')
    file_name_time = input('Please enter the name of .txt file for times: ')
    results_file = open(file_name,'a',buffering=1)
    results_time_file = open(file_name_time,'a',buffering=1)
    results = {}
    results_time = {}
    for func in lof:
        results_file.write(f'{func}\n')
        results_time_file.write(f'{func}\n')
        print(f'{func}')
        results[func] = []
        results_time[func] = []
        for n in range(minimum,len(distance_matrix)+1,step):
            submatrix = distance_matrix[0:n,0:n]
            names = np.arange(0,n)
            sample = []
            sample_times = []
            N = 30
            for _ in range(N):
                graph_temp = Graph(submatrix)
                graph_temp.add_edge(submatrix,names)
                time_before = time.time()
                res, _ = getattr(graph_temp,func)()
                sample_times.append(time.time() - time_before)
                sample.append(res)
            print(confirm_normal_distribution(sample,N))
            if sample_size(sample) <= N:
                results[func].append(np.mean(sample))
                results_time[func].append(np.mean(sample_times))
                results_file.write(f'{np.mean(sample)}\n')
                results_time_file.write(f'{np.mean(sample_times)}\n')
            else:
                while sample_size(sample) > N:
                    N = sample_size(sample)
                    sample_ns = []
                    sample_times_ns = []
                    print(f"For {func} sample is too small, it has to be {sample_size(sample)}")
                    for _ in range(N):
                        graph_temp = Graph(submatrix)
                        graph_temp.add_edge(submatrix,names)
                        time_before_ns = time.time()
                        res, _ = getattr(graph_temp,func)()
                        sample_times_ns.append(time.time() - time_before_ns)
                        sample_ns.append(res)
                results[func].append(np.mean(sample))
                results_time[func].append(np.mean(sample))
                results_file.write(f'{np.mean(sample)}\n')
                results_time_file.write(f'{np.mean(sample_times)}\n')

    results_file.close()
    results_time_file.close()
    
    for func in lof:
        plt.plot(list(range(minimum,len(distance_matrix)+1,step)),results[func],label=func)

    plt.xlim(minimum,len(distance_matrix))
    plt.legend()
    plt.show()
    
    for func in lof:
        plt.plot(list(range(minimum,len(distance_matrix)+1,step)),results_time[func],label=func)
        
    plt.xlim(minimum,len(distance_matrix))
    plt.legend()

    plt.show()
