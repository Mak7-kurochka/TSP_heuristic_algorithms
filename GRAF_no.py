import random
import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations
import time
import pandas as pd

class Node:
    def __init__(self,value):
        self.value = value
        self.edges = []
        self.parents = []

class Edge:
    def __init__(self,diraction=None,weight=1):
        self.dir = diraction
        self.weight = weight
        self.pheromone = 0.1
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
    def __init__(self):
        self.nodes = {}

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
        keys = list(self.nodes.keys())
        stack = [self.nodes[keys[0]]]
        self.depth_traversal_wrap(passed,stack)
        for key in keys:
            if self.nodes[key] not in passed:
                stack.append(self.nodes[key])
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
        keys = list(self.nodes.keys())
        queue = [self.nodes[keys[0]]]
        self.breath_traversal_wrap(passed,queue)
        for key in keys:
            if self.nodes[key] not in passed:
                queue.append(self.nodes[key])
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
        #start_node = self.nodes[random.choice(list(self.nodes.keys()))]
        start_node = self.nodes[1]
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
        return [self.calculate_distance(path),path]
    
    def ant_colony(self,k=100,s=10,batch=10):
        #city = self.nodes[random.choice(list(self.nodes.keys()))]
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
                        best_solution[0] = ant.dist
                        best_solution[1] = [city.value for city in ant.passed.copy()]
                        continue
                    if (best_solution[0] - ant.dist) < s:
                        Y += 1
                self.pheromone_update(visited_edges)
                for ant in ants:
                    ant.passed=[random.choice(list(self.nodes.values()))]
                    ant.dist = 0
                i += 1
        return best_solution

    def choose_the_edge_ACO(self,node,ant,a=5,b=0.8):
        edges = []
        for edge in node.edges:
            if edge.dir not in ant.passed:
                edges.append(edge)
        t_n = [(edge.pheromone**a) *((1/(edge.weight+1))**b) for edge in edges]
        total = sum(t_n)
        probabilites = np.array([t_n[i]/total for i in range(len(t_n))])
        return np.random.choice(edges,p=probabilites)
    
    def pheromone_update(self,visited_edges,p=0.2,Q=1000):
        #for node in self.nodes.values():
        #    for edge in node.edges:
        #        edge.pheromone *= (1 - p)
        #for edge in visited_edges:
        #    delta_tau = 0
        #    for ant in edge.ants:
        #        delta_tau += Q/ant.dist
        #    edge.pheromone = (1-p)*edge.pheromone + delta_tau
        #    edge.ants.clear()
        for node in self.nodes.values():
            for edge in node.edges:
                delta_tau = sum(Q/ant.dist for ant in edge.ants if ant.dist>0)
                edge.pheromone = (1-p)*edge.pheromone + delta_tau
                edge.ants.clear()

    def calculate_distance(self,route):
        dist = 0
        for i in range(len(route)-1):
            for edge in self.nodes[route[i]].edges:
                if edge.dir.value == route[(i+1)]:
                    dist += edge.weight
                    break
        return dist
    
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

    def random_swaping(self,route_llst,batch=1000):
        print([route_llst])
        route_dist = route_llst[0]
        route = route_llst[1]
        i = 0
        Y=0
        while i < 10**6:
            
            if Y < batch:
                break

            new_route = self.swap_nodes(route.copy())
            new_route_dist = self.calculate_distance(new_route)

            if new_route_dist < route_dist:
                route = new_route
                route_dist = new_route_dist
                Y = 0
            Y += 1
            i += 1

        return [route_dist,route]

    def k_opt(self,n,route):
        pass
        for i in range(len(route)):
            city = route[i]
            city_2 = 0
        #check is route better?
        #if yes, T = T'

    def clear_pheromones(self):
        for node in self.nodes.values():
            for edge in node.edges:
                edge.pheromone = 0.1

def create_matrix(n):
    matrix = np.random.randint(400, 1000, size=(n, n))
    np.fill_diagonal(matrix, 0)
    return matrix

graph = Graph()
n = 5
matrix = create_matrix(n)
names = np.arange(0,n)
graph.add_edge(matrix,names)
#graph.depth_traversal()
#graph.breath_traversal()
#print(graph.find_path(1,5))
#print(graph.find_path_all(1,5))
#print(graph.find_path_shortest(1,5))
#graph.edge_contrection(3,4)
#graph.edge_breaking(2,3)
#graph.breath_traversal()
#print(graph.random_swaping(graph.nearest_neighbour()))
#print(graph.ant_colony())
#print(graph.brute_force())
print(graph.k_opt(2,graph.nearest_neighbour))
#time_llst = []
#res = []
#for i in range(30):
#    start_time = time.time()
#    res.append(graph.ant_colony()[0])
#    end_time = time.time()
#    time_llst.append(end_time-start_time)
#    graph.clear_pheromones()
#
#print((sum(time_llst)/30))
#
#df = pd.DataFrame(res,columns=["Distance a = 0.2, b=1"])
#df.to_excel('first.xlsx',index = False)