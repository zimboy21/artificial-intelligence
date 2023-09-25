#Ziman David zdim1981 524/2

import numpy as np
import sys
import heapq
from matplotlib import cm, pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Node: 
    def __init__(self, x, y, z, b, index):
        self.x = x
        self.y = y
        self.z = z
        self.b = b
        self.index = index

def row_count() :
    rc = 0
    y = nodes[rc].y
    rc += 1
    while y == nodes[rc].y :
        rc += 1
    return rc

def column_count() :
    cc = 0
    x = nodes[cc].x
    cc += 1
    while x != nodes[cc].x :
        cc += 1
    return cc

def euclid_dist(act, next) :
    return np.sqrt((act.x - next.x) ** 2 + (act.y - next.y) ** 2 + (act.z - next.z) ** 2)

def step(act, next) :
    return 1

def getPoint(index) :
    return nodes[index].x, nodes[index].y, nodes[index].z

def getPath(prev_nodes, start, end) :
    act = end.index
    path = []
    while act != start.index :
        path.append(act)
        act = prev_nodes[act]
    path.append(start.index)
    path.reverse()
    return list(map(getPoint, path))

def getNeighbours(act) :
    neighbours = []
    if act == 0 :                                   #bal felso sarok
        neighbours.append(nodes[act + 1])
        neighbours.append(nodes[act + r_count])
        neighbours.append(nodes[act + r_count + 1])
    elif act == maxPos :                            #jobb also sarok
        neighbours.append(nodes[act - 1])
        neighbours.append(nodes[act - r_count])
        neighbours.append(nodes[act - r_count - 1])
    elif act == r_count - 1:                        #bal also sarok
        neighbours.append(nodes[act - 1])
        neighbours.append(nodes[act + r_count])
        neighbours.append(nodes[act + r_count - 1])
    elif act == (maxPos - r_count + 1) :            #jobb felso sarok
        neighbours.append(nodes[act + 1])
        neighbours.append(nodes[act - r_count])
        neighbours.append(nodes[act - r_count + 1])
    elif 1 <= act < (r_count - 1) :                 #bal oszlop
        neighbours.append(nodes[act - 1])
        neighbours.append(nodes[act + 1])
        neighbours.append(nodes[act + r_count - 1])
        neighbours.append(nodes[act + r_count])
        neighbours.append(nodes[act + r_count + 1])
    elif (maxPos - r_count + 1) < act < maxPos :    #jobb oszlop
        neighbours.append(nodes[act - 1])
        neighbours.append(nodes[act + 1])
        neighbours.append(nodes[act - r_count - 1])
        neighbours.append(nodes[act - r_count])
        neighbours.append(nodes[act - r_count + 1])
    elif (act + 1) % r_count == 0 :                 #also sor
        neighbours.append(nodes[act - 1])
        neighbours.append(nodes[act - r_count])
        neighbours.append(nodes[act - r_count - 1])
        neighbours.append(nodes[act + r_count])
        neighbours.append(nodes[act + r_count - 1])
    elif act % r_count == 0 :                       #felso sor
        neighbours.append(nodes[act + 1])
        neighbours.append(nodes[act - r_count])
        neighbours.append(nodes[act - r_count + 1])
        neighbours.append(nodes[act + r_count])
        neighbours.append(nodes[act + r_count + 1])
    else :
        neighbours.append(nodes[act - r_count])
        neighbours.append(nodes[act + r_count])
        neighbours.append(nodes[act - r_count - 1])
        neighbours.append(nodes[act - r_count + 1])
        neighbours.append(nodes[act + r_count + 1])
        neighbours.append(nodes[act + r_count - 1])
        neighbours.append(nodes[act + 1])
        neighbours.append(nodes[act - 1])

    return neighbours

def A(start, end, dist_heuristic) :
    prev_nodes = {}
    act_cost = {}
    prev_nodes[start.index] = None
    act_cost[start.index] = 0
    open_heap = []
    heapq.heappush(open_heap, (0, start.index))

    while len(open_heap) != 0 :
        act = heapq.heappop(open_heap)[1]
        if act == end.index :
            return getPath(prev_nodes, start, end), act_cost[end.index]
        for i in getNeighbours(act) :
            if i.b == 0 :
                new_cost = act_cost[act] + dist_heuristic(nodes[act], i)
                if i.index not in act_cost or new_cost < act_cost[i.index] :
                    priority = new_cost + dist_heuristic(i, end)
                    heapq.heappush(open_heap, (priority, i.index))
                    act_cost[i.index] = new_cost
                    prev_nodes[i.index] = act
    return False

file = open("points_100x100.txt", "r")
coords = file.readline().split()
start = Node(float(coords[0]), float(coords[1]), 0, 0, -1)

coords = file.readline().split()
end = Node(float(coords[0]), float(coords[1]), 0, 0, -1)

file = open("surface_100x100.txt", "r")
nodes = []
maxPos = 0
z = []
for i in file :
    line = i.split()
    z.append(float(line[2]))
    nodes.append(Node(float(line[0]), float(line[1]), float(line[2]), int(line[3]), maxPos))
    if nodes[maxPos].x == start.x and nodes[maxPos].y == start.y :
        start.z = nodes[maxPos].z
        start.index = maxPos
    if nodes[maxPos].x == end.x and nodes[maxPos].y == end.y :
        end.z = nodes[maxPos].z
        end.index = maxPos
    maxPos += 1

c_count = column_count()
r_count = row_count()


fig = pyplot.figure()
ax = Axes3D(fig)

orig_stdout = sys.stdout

file = open("steps.txt", "w")
path, cost = A(start, end, step)
sys.stdout = file
print("Path = ", path, "\nSteps = ", cost)

for point in path:
    ax.plot3D([point[0]], [point[1]], [point[2]], marker='.', color='blue')

file = open("euclid.txt", "w")
path, cost = A(start, end, euclid_dist)
sys.stdout = file
print("Path = ", path, "\nDistance = ", cost)

sys.stdout = orig_stdout

for point in path:
    ax.plot3D([point[0]], [point[1]], [point[2]], marker='.', color='red')

for i in range(len(nodes)):
    if nodes[i].b :
        ax.plot3D([nodes[i].x], [nodes[i].y], [nodes[i].z], color='black', marker='.')

x = list(range(int(nodes[0].x), int(c_count + nodes[0].x)))
y = list(range(int(nodes[0].y), int(r_count + nodes[0].y)))
z = np.array(z).reshape(r_count, c_count)
x, y = np.meshgrid(x, y)
surf = ax.plot_surface(x, y, z, cmap=cm.terrain, edgecolors='black', antialiased=False, linewidth=0.03, alpha=0.5)
fig.colorbar(surf, shrink=0.5, aspect=5)

pyplot.show()
