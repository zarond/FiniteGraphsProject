import networkx as nx
import heapq as hp

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


if __name__ == "__main__":
    import numpy as np
    import time as tm
    import pandas as pd
    import matplotlib.pyplot as plt
    files = ['test/export_test2.csv','test/export_test3.csv','test/export_test1.csv','test/export_test.csv','test/export_test4.csv']
    #files = files[0:-1]
    T_nx,T_custom, Num = perform_tests(files)

    plt.plot(Num,T_nx)
    plt.plot(Num,T_custom)
    plt.legend(['dijkstra from networkx','custom dijkstra'])
    plt.show()
