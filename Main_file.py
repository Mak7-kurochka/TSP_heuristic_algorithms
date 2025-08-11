import random
import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations
import math

class Node:
    def __init__(self,value):
        self.value = value
        self.edges = []
        self.parents = []

class Edge:
    def __init__(self,direction=None,weight=1):
        self.dir = direction
        self.weight = weight
        self.pheromone = 1
        self.ants = []

class PathNode:
    def __init__(self,node,parent=None):
        self.node = node
        self.parent = parent

class Ant:
    def __init__(self,start_city):
        self.passed = [start_city]
        self.dist = 0

class Graph:
    def __init__(self,matrix):
        self.nodes = {}
        self.matrix = matrix

    def add_or_get(self,name):
        if name in self.nodes.keys():
            return self.nodes[name]
        else:
            self.nodes[name] = Node(name)
            return self.nodes[name]

    def add_edge(self,matrix,names):
        names_vertical = names.reshape(-1,1)
        r=0
        for row in matrix:
            for i in range(len((row))):
                if row[i] != 0:
                    if names[r] in self.nodes.keys() and names_vertical[i][0] in self.nodes.keys():
                        start_node = self.nodes[names[r]]
                        end_node = self.nodes[names_vertical[i][0]]
                        exist = False
                        for edge in start_node.edges:
                            if edge.dir == end_node:
                                exist = True
                                edge.weight = row[i]
                                break
                        if not exist:
                            start_node.edges.append(Edge(end_node,row[i]))
                            end_node.parents.append(start_node)
                    else:
                        start_node = self.add_or_get(names[r])
                        end_node = self.add_or_get(names_vertical[i][0])
                        start_node.edges.append(Edge(end_node,row[i]))
                        end_node.parents.append(start_node)
            r += 1

    def depth_traversal(self,passed=[]):
        values = list(self.nodes.values())
        stack = [values[0]]
        self.depth_traversal_wrap(passed,stack)
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
    
    def breath_traversal(self,passed=[]):
        values = list(self.nodes.values())
        queue = [values[0]]
        self.breath_traversal_wrap(passed,queue)
        for value in values:
            if value not in passed:
                queue.append(value)
                self.breath_traversal_wrap(passed,queue)

    def breath_traversal_wrap(self,passed,queue):
        if len(queue) == 0:
            return
        node = queue[0]
        queue.remove(node)
        passed.append(node)
        print(node.value)
        for edge in node.edges:
            if edge.dir not in passed and edge.dir not in queue:
                queue.append(edge.dir)
        self.breath_traversal_wrap(passed,queue)

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
        for edge in end_node.edges:
            if edge.dir == start_node:
                end_node.edges.remove(edge)
                break
           
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

    def edge_contrection(self,start,end):
        self.delete_edge(start,end)
        parent_to_add = list(set(self.nodes[end].parents)-set(self.nodes[start].parents))
        node_parent_weights = [0]
        parent_node_weights = [0]
        names = np.array([self.nodes[start].value])

        for parent in parent_to_add:
            for edge in parent.edges:
                if edge.dir == end:
                    parent_node_weights.append(edge.weight)
            for edge in end.edges:
                if edge.dir == parent:
                    node_parent_weights.append(edge.weight)
            np.append(names,parent.value)
        
        self.delete_node(end)
        
        matrix = np.zeros((len(names), len(names)),dtype=int)
        matrix[0] = node_parent_weights

        for i in range(len(parent_node_weights)):
            matrix[i][0] = parent_node_weights[i]

        self.add_edge(matrix,names)

    def edge_breaking(self,start,end):
        self.delete_edge(start,end)
        names = np.array([max(self.nodes.keys())+1,start,end])
        matrix = np.zeros((len(names),len(names)),dtype=int)
        _ = [1 for i in range(len(names))]
        _[0] = 0
        matrix[0] = _
        for i in range(len(_)):
            matrix[i][0] = _[i]
        self.add_edge(matrix,names)

    def calculate_distance(self,route):
        dist = 0
        for i in range(len(route)-1):
            dist += self.matrix[route[i]][route[i+1]]
        return dist

###

###Algorithms start

##NN start
    def find_minimum_edge(self,node,passed,path):
        edges = []
        for index, edge in enumerate(node.edges):
            if edge.dir not in passed:
                edges.append([index,edge.weight])
        if len(edges)==0:
            return None
        min_edge = min(edges, key=lambda e: e[1])
        index = min_edge[0]
        path.append(node.edges[index].dir.value)
        return node.edges[index].dir

    def nearest_neighbour(self):
        start_node = random.choice(list(self.nodes.values()))
        passed = []
        path = [start_node.value]
        current_node = start_node
        
        while len(passed) < len(self.nodes):
            passed.append(current_node)
            current_node = self.find_minimum_edge(current_node,passed,path)
        for edge in self.nodes[path[-1]].edges:
            if edge.dir == start_node:
                path.append(start_node.value)
                break
        return self.calculate_distance(path),path
###NN end
    
###ACO start a - 20, b=100
    def choose_the_edge_ACO(self,node,ant,a=2,b=10):
        edges = []
        for edge in node.edges:
            if edge.dir not in ant.passed:
                edges.append(edge)
        t_n = [(edge.pheromone**a) *((1/(edge.weight))**b) for edge in edges]
        total = sum(t_n)
        probabilites = np.array([t_n[i]/total for i in range(len(t_n))])
        #print(edges,t_n,probabilites)
        return np.random.choice(edges,p=probabilites)
    
    def pheromone_update(self,visited_edges, p=0.4,Q=1000):
        for edge in visited_edges:
            delta_tau = sum(Q/ant.dist for ant in edge.ants)
            edge.pheromone = (1-p)*edge.pheromone + delta_tau
            edge.ants.clear()

    def ant_colony(self,k=200,s=10,batch=10):
        city = random.choice(list(self.nodes.values()))
        ants = [Ant(city) for _ in range(k)]
        best_solution = [float('inf'),[]]
        visited_edges = []
        Y = 0
        i = 0
        while i < 10**6:
            if Y > batch:
                break
            else:
                while len(ants[0].passed) < len(self.nodes):
                    for ant in ants:
                        edge = self.choose_the_edge_ACO(ant.passed[-1],ant)
                        ant.passed.append(edge.dir)
                        ant.dist += edge.weight
                        edge.ants.append(ant)
                        if edge not in visited_edges:
                            visited_edges.append(edge)
                for ant in ants:
                    end_city = ant.passed[-1]
                    for edge in end_city.edges:
                        if edge.dir == ant.passed[0]:
                            ant.passed.append(ant.passed[0])
                            ant.dist += edge.weight
                            edge.ants.append(ant)
                            visited_edges.append(edge)
                            break
                    if ant.dist < best_solution[0]:
                        Y = 0
                        best_solution = ant.dist,[city.value for city in ant.passed.copy()]
                    elif (best_solution[0] - ant.dist) < s:
                        Y += 1
                self.pheromone_update(visited_edges)
                for ant in ants:
                    ant.passed=[random.choice(list(self.nodes.values()))]
                    ant.dist = 0
                i += 1
        return best_solution
###ACO end
    
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
        first_city = random.choice(local_route)
        del(local_route[local_route.index(first_city)])
        second_city = random.choice(local_route)

        first_city_index = route.index(first_city)
        second_city_index = route.index(second_city)
        route[first_city_index] = second_city
        route[second_city_index] = first_city

        return route

    def random_swaping(self,route_llst=None,batch=1000,k=0.1):
        if not route_llst:
            route_dist, route = self.nearest_neighbour()
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
            elif (new_route_dist - route_dist) < k:
                Y += 1

            i += 1

        return route_dist,route
###random swapping end

###2-opt start
    def two_opt(self,path=None):
        distt = []
        if not path:
            res = self.nearest_neighbour()
            route = res[1]
        else:
            route = path[1]
        iteration = 0
        i = 1
        improved = True
        while i<len(route)-2:

            if iteration >= 10**6:
                break

            if improved:
                i=1
                improved=False
                
            A = route[i]
            B = route[i+1]
            j=i+2
            while j<len(route)-2:
                C = route[j]
                D = route[j+1]
                org_dist = self.calculate_distance([A,B]) + self.calculate_distance([C,D])
                new_dist = self.calculate_distance([A,C]) + self.calculate_distance([B,D])
                if new_dist < org_dist:
                    route = route[:i]+[A,C]+route[i+2:j][::-1]+[B,D]+route[j+2:]
                    improved = True
                    distt.append(self.calculate_distance(route))
                    break
                j +=1

            i += 1
            iteration += 1
        return self.calculate_distance(route),route
###2-opt end

###3-opt start
    def three_opt_swaps(self,route,i,j,k):
        segm_a = route[1:i+1]
        segm_b = route[i+1:j+1]
        segm_c = route[j+1:k+1]
        segm_d = route[k+1:]

        segm_a_rev = segm_a[::-1]
        segm_b_rev = segm_b[::-1]
        segm_c_rev = segm_c[::-1]
        options = [
            [route[0]]+segm_a_rev+segm_b+segm_c+segm_d, #first case(reverse segment 1)
            [route[0]]+segm_a+segm_b_rev+segm_c+segm_d, #second case(reserve segment 2)
            [route[0]]+segm_a+segm_b+segm_c_rev+segm_d, #third case(reverse segment 3)
            [route[0]]+segm_b+segm_c+segm_a+segm_d, #fourth case B-C-A
            [route[0]]+segm_c+segm_b+segm_a+segm_d, #C-B-A
            [route[0]]+segm_a+segm_c_rev+segm_b_rev+segm_d, #sixth case A-C'-B'
            [route[0]]+segm_a+segm_b_rev+segm_c_rev+segm_d, #seventh case A-B'-C'
        ]
        
        return options
    
    def three_opt(self,path=None,max_iter=10**6):
        if not path:
            res = self.nearest_neighbour()
            route = res[1]
            dist = res[0]
        else:
            dist = path[0]
            route = path[1]
        i=0
        _=0
        j=i+2
        improved = True
        while i<j-1:
            if _>=max_iter:
                break
            if improved:
                i = 0
                j = 2
                k = 4
                improved = False
            else:
                if k == len(route)-2:
                    if j == len(route) - 4:
                        i +=1
                    else:
                        j += 1
                else:
                    k += 1
            options = self.three_opt_swaps(route,i,j,k)
            for option in options:
                new_distance = self.calculate_distance(option)
                if new_distance<dist:
                    dist = new_distance
                    route = option
                    improved = True
            _+=1
        return dist, route
###3-opt end

###Simulated annealing starts
    def simulated_annealing(self,path=None,t_max=10000,t_min=1,a=0.995,i_max=500):##t_max was set to have acceptance rate > 0.9
        if not path:
            _, current_route = self.nearest_neighbour()
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
                    return "Graph is too small"
                
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
        

def create_matrix(n):
    matrix = np.random.randint(400, 1000, size=(n, n))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return matrix
