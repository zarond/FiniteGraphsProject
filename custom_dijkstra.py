import osmnx as ox
import networkx as nx
import numpy as np
import time as tm
import heapq as hp
import pandas as pd
import matplotlib.pyplot as plt

def load_graph():
    global G
    try:
        G = ox.load_graphml('network.graphml') #load
        print("loaded from graphml file")   
    except: pass
    if (G is None):
        try:
            G = ox.graph_from_place('Ижевск, Россия', network_type='drive',simplify=True)
            print("downloaded from internet")
        except: pass
    if (G is None):
        print('failed to load map. To download network from internet check internet connection. To load from file it\'s name should be \'network.graphml\' or \'network.xml\'')
        raise 0
    return

def custom_dijkstra(G, origin, weight='length'):
    G_succ = G._succ if G.is_directed() else G._adj
    dist = {}
    pred = {}
    front = []
    
    push = hp.heappush
    pop = hp.heappop
    
    push(front, (0, origin))
    while front:
        (l, v) = pop(front)
        if v in dist:
            continue
        dist[v] = l
        for u, e in G_succ[v].items():
            vu_dist = dist[v] + e[weight]
            if u in dist:
                continue
            tmp = (True,0.0)
            for e in front:
                if e[1]==u:
                    tmp = (False,e[0])
                    break
            if (tmp[0] or vu_dist < tmp[1]):
                push(front, (vu_dist, u))
                pred[u] = [v]
  
    return pred, dist

def perform_tests(files):
    Time_nx = []
    Time_custom = []
    Num = []
    for file in files:
        input_data = pd.read_csv(file, index_col=0)
        G = nx.DiGraph(input_data.values)
        N = len(G.nodes())
        Num.append(N)
        #D_truth = np.empty([N,N])
        #D = np.empty([N,N])
        D_truth = []
        D = []
        t0 = tm.time()
        for i in range(N):
            D_truth.append(nx.dijkstra_predecessor_and_distance(G, i, weight='weight')[1])
        t1 = tm.time()
        Time_nx.append(t1-t0)

        t0 = tm.time()
        for i in range(N):
            D.append(custom_dijkstra(G, i, weight='weight')[1])
        t1 = tm.time()
        Time_custom.append(t1-t0)
        Dist = np.zeros([N,N])
        Dist_truth = np.zeros([N,N])
        for i in range(N):
            for e in D[i]:
                Dist[i,e]=D[i][e]
        for i in range(N):
            for e in D_truth[i]:
                Dist_truth[i,e]=D_truth[i][e]
                
        print(np.max(Dist-Dist_truth)==0.0)
        Dist = np.append([np.arange(N)],Dist,axis=0)
        Dist = np.append(np.transpose([np.concatenate(([0],np.arange(N)))]),Dist,axis=1)
        
        np.savetxt(file[0:-4]+'_result.csv',Dist.astype(int),delimiter=',',fmt='%5.0f',)

    return Time_nx,Time_custom,Num
try:
    #load_graph()
    G=G.to_directed()
    G = nx.DiGraph(G)

    largest = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest)
    N=100
    house_nodes = np.random.choice(list(G.nodes), size = N, replace = False) # get houses as random nodes

    t0 = tm.time()
    A = nx.dijkstra_predecessor_and_distance(G, house_nodes[0], weight='length')
    t1 = tm.time()
    print(t1-t0)

    t0 = tm.time()
    B = custom_dijkstra(G, house_nodes[0])
    t1 = tm.time()
    print(t1-t0)

    C = []
    for nod in G.nodes():
        C.append(A[1][nod]-B[1][nod])
    print(np.max(C)==0.0)
    #print("if 0.0 0.0 that means outputs of nx.dijkstra_predecessor_and_distance() and custom_dijkstra() are equal")
except: print("skip")

files = ['test/export_test2.csv','test/export_test3.csv','test/export_test1.csv','test/export_test.csv','test/export_test4.csv']
#files = files[0:-1]
T_nx,T_custom, Num = perform_tests(files)

plt.plot(Num,T_nx)
plt.plot(Num,T_custom)
plt.legend(['dijkstra from networkx','custom dijkstra'])
plt.show()
