import osmnx as ox
import networkx as nx
import numpy as np
import time as tm
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
import copy
import scipy as sp
import scipy.cluster.hierarchy as sch
import lxml.etree as ET
import warnings
warnings.filterwarnings("ignore")

def adjust_weights(house_nd,object_nd): # make weights going to objects a little bigger
    for nd in object_nd:
        for e in G.in_edges(nd):
            rnd = np.random.random() + 1.0
            #G.edges[(e[0],e[1],0)]['length']*=rnd
            G.edges[e]['length']*=rnd
    return

def Part1_1(house_nd,object_nd,X=np.inf):
    N = len(house_nd)
    M = len(object_nd)
    dist2objects = np.empty([N,M])
    dist2houses = np.empty([M,N])
    Grev = G.reverse()
    for j in range(M):
        length_house2object = nx.single_source_dijkstra_path_length(Grev, object_nd[j], cutoff=None, weight='length')
        length_object2house = nx.single_source_dijkstra_path_length(G, object_nd[j], cutoff=None, weight='length')
        for i in range(N):
            dist2objects[i,j] = length_house2object[house_nd[i]]
            dist2houses[j,i] = length_object2house[house_nd[i]]

    Min_indicesObTo = np.argmin(dist2objects, axis=1) # indices   
    NearestObjectsTo = object_nd[Min_indicesObTo] # nearest objects to houses (per house)
    Min_distObTo = dist2objects[np.arange(N),Min_indicesObTo] # minimum distances to objects from houses

    Min_indicesObFrom = np.argmin(dist2houses, axis=0) # indices   
    NearestObjectsFrom = object_nd[Min_indicesObFrom] # nearest objects to houses (per house) going from object to house
    Min_distObFrom = dist2objects[np.arange(N),Min_indicesObFrom] # minimum distances to houses from objects

    ThereAndBack = np.empty(N) # nearest objects to houses there-and-back
    ThereAndBackDist = np.empty([N,M]) # distance to nearest objects to houses there-and-back
    ThereAndBackDist = dist2objects + dist2houses.transpose()
    
    ThereAndBack = np.argmin(ThereAndBackDist, axis=1) #indices
    NearestThereAndBack = object_nd[ThereAndBack]
    MinDistThereAndBack = ThereAndBackDist[np.arange(N),ThereAndBack]

    #ReachableNearestObjects = NearestObjects[Min_distOb<X] # there
    IsReachableObjects = (dist2objects<X) # there
    #ReachableNearestHouses = NearestHouses[Min_distH<X] # back
    IsReachableHouses = (dist2houses<X) # back
    #ReachableThereAndBack = ThereAndBack[ThereAndBackDist<X] # there and back
    IsReachableThereAndBack = (ThereAndBackDist<X) # there and back
    
    return NearestObjectsTo, NearestObjectsFrom, ThereAndBack, IsReachableObjects, IsReachableHouses, IsReachableThereAndBack, dist2objects, dist2houses 

def Part1_2(house_nd,object_nd, dist2objects=None, dist2houses=None):
    if (dist2objects is None or dist2houses is None):
        N = len(house_nd)
        M = len(object_nd)
        dist2objects = np.empty([N,M])
        dist2houses = np.empty([M,N])
        Grev = G.reverse()
        for j in range(M):
            length_house2object = nx.single_source_dijkstra_path_length(Grev, object_nd[j], cutoff=None, weight='length')
            length_object2house = nx.single_source_dijkstra_path_length(G, object_nd[j], cutoff=None, weight='length')
            for i in range(N):
                dist2objects[i,j] = length_house2object[house_nd[i]]
                dist2houses[j,i] = length_object2house[house_nd[i]]

    #------------ можно обойтись только тем, что ниже, если использовать инфу из первой функции            
    #ThereAndBack = np.empty(N) # nearest objects to houses there-and-back
    #ThereAndBackDist = np.empty([N,M]) # distance to nearest objects to houses there-and-back
    ThereAndBackDist = dist2objects + dist2houses.transpose()
   
    A = np.argmin(np.max(dist2objects, axis=0)) # максимальное расстояние до объектов ???
    B = np.argmin(np.max(dist2houses.transpose(), axis=0)) # максимальное расстояние до домов ???
    C = np.argmin(np.max(ThereAndBackDist, axis=0)) # максимальное расстояние туда и обратно ???
    a = np.argmax(dist2objects, axis=0)[A]
    b = np.argmax(dist2houses.transpose(), axis=0)[B]
    c = np.argmax(ThereAndBackDist, axis=0)[C]
    
    return A, B, C, a, b, c
    
def Part1_3(house_nd,object_nd, dist2houses=None):
    if (dist2houses is None):
        N = len(house_nd)
        M = len(object_nd)
        dist2houses = np.empty([M,N])
        #Grev = G.reverse()
        for j in range(M):
            length_object2house = nx.single_source_dijkstra_path_length(G, object_nd[j], cutoff=None, weight='length')
            for i in range(N):
                dist2houses[j,i] = length_object2house[house_nd[i]]

    #------------ можно обойтись только тем, что ниже, если использовать инфу из первой функции
    sums = np.sum(dist2houses,axis=1)
    arg = np.argmin(sums)
    return arg

def build_paths_tree(pred,dest_nds,origin):
    N = len(dest_nds)
    sub = []
    for i in range(N):
        nd = dest_nds[i]
        edge = ()
        while True:
            p = pred.get(nd)
            if len(p)==0:
                break
            #edge = (p[0],nd,0)
            edge = (p[0],nd)
            if (edge not in sub):
                sub.append(edge)
            else:
                break
            nd = p[0]
    Tree = G.edge_subgraph(sub)   
    return Tree

def make_paths_tree(nodes,origin):
    pred, length = nx.dijkstra_predecessor_and_distance(G, origin, cutoff=None, weight='length')
    tree = build_paths_tree(pred,nodes,origin)
    N = len(nodes)
    dist = np.empty(N)
    for i in range(N):
        dist[i] = length[nodes[i]]
    return tree, dist
    
def Part1_4(house_nd,object_nd):
    N = len(house_nd)
    M = len(object_nd)
    tree_weights = np.empty(M)
    for j in range(M):
        pred, length_object2house = nx.dijkstra_predecessor_and_distance(G, object_nd[j], cutoff=None, weight='length')
        tree = build_paths_tree(pred,house_nd,object_nd[j])
        tree_weights[j] = tree.size(weight='length')

    arg = np.argmin(tree_weights)
    return arg 


def Part2_1(house_nd,object_nd):
    tree, dist2houses = make_paths_tree(house_nd,object_nd)
    tree_weight = tree.size(weight='length')
    sum_dist = np.sum(dist2houses)
    return tree, tree_weight, sum_dist

def getDistHouse2House(house_nd):
    N = len(house_nd)
    dist_house2house = np.empty([N,N])
    for j in range(N):
        length = nx.single_source_dijkstra_path_length(G, house_nd[j], cutoff=None, weight='length')
        for i in range(N):
            dist_house2house[j,i] = length[house_nd[i]]
    return dist_house2house

def Part2_2(house_nd=None,dist_house2house=None):
    if (dist_house2house is None and house_nd is not None):
        dist_house2house =  getDistHouse2House(house_nd)

    N = len(house_nd) 
    # all distances between houses computed
    ClusterDistances = (dist_house2house + dist_house2house.transpose())/2 #make simetrical
    Z = np.empty([N-1,4])
    indices = np.arange(N)
    number = np.ones(N)
    
    for i in range(N):
        ClusterDistances[i,i]=np.inf
    
    for c in range(N-1):
        a = np.argmin(ClusterDistances)
        i = a//ClusterDistances.shape[1]
        j = a%ClusterDistances.shape[1]

        Z[c,0]=indices[i]
        Z[c,1]=indices[j]
        Z[c,2]=ClusterDistances[i,j]
        Z[c,3]=number[i]+number[j]
        
        for k in range(ClusterDistances.shape[1]):
            ClusterDistances[i,k] = ClusterDistances[k,i] = max(ClusterDistances[i,k],ClusterDistances[j,k])

        ClusterDistances = np.delete(ClusterDistances,j,0)
        ClusterDistances = np.delete(ClusterDistances,j,1)
        indices[i] = c + N
        number[i] = number[i]+number[j]
        indices = np.delete(indices,j)
        number = np.delete(number,j)
        
    return Z

def Part2_2_scipy(house_nd = None,dist_house2house=None):
    if (dist_house2house is None and house_nd is not None):
        dist_house2house =  getDistHouse2House(house_nd)

    ClusterDistances = (dist_house2house + dist_house2house.transpose())/2 #make simetrical        
    Z = sch.linkage(ClusterDistances, method='complete', metric='chebyshev')
    return Z

def getClusters(k, house_nodes, Cl=None, Z=None):
    if (Cl is not None):
        Clusters = [[house_nodes[j] for j in i] for i in Cl[-k+1]]
        return Clusters
    if (Z is not None):
        div = sch.fcluster(Z,k,criterion='maxclust')
        Clusters = [[] for i in range(max(div))]
        for i in range(len(div)):
            Clusters[div[i]-1].append(house_nodes[i])
        return Clusters
    return None

def Part2_3(clusters,house_nd,object_nd):
    k = len(clusters)
    center_nd = np.empty(k, dtype=int)
    tree_weight = np.empty(k)
    sum_dist = np.empty(k)
    Trees = []
    for i in range(k):
        #nds_in_cluster = (house_nd[clusters[i]])
        nds_in_cluster = clusters[i]
        S = G.subgraph(nds_in_cluster)
        X = list(nx.get_node_attributes(S,'x').values()) # lon
        Y = list(nx.get_node_attributes(S,'y').values()) # lat
        X = np.average(X)
        Y = np.average(Y)
        center_nd[i] = ox.get_nearest_node(G, (Y, X))
        tree, dist = make_paths_tree(nds_in_cluster,center_nd[i])
        tree_weight[i] = tree.size(weight='length')
        sum_dist[i] = np.sum(dist)
        Trees.append(tree)

    TreeFromObject, dist = make_paths_tree(center_nd,object_nd)
    return TreeFromObject, Trees, center_nd, tree_weight, sum_dist, dist

def plot_stuff(a,Clusters=None,house_nodes=None,objects_nodes=None, Cl_centers=[],tree=None,origin=None,tree_targets=None,src_nds=None,dst_nds=None,Trees=None,Z=None, twoWay = False):
    if (0 in a): # plot just map
        #fig, ax = ox.plot_graph(G)
        fig, ax = ox.plot_graph(G_o, node_size=0,show=False, close=False)
    if (1 in a): # plot houses and objects
        color = [[0.0,0.0,0.0,0.0] for node in G.nodes()]
        for el in house_nodes:
             color[list(G.nodes()).index(el)] = [1.0,0.0,0.0,1.0]
        for el in objects_nodes:
             color[list(G.nodes()).index(el)] = [0.0,0.0,1.0,1.0]
        for el in Cl_centers:
             color[list(G.nodes()).index(el)] = [0.0,1.0,0.0,1.0] 
        fig, ax = ox.plot_graph(G_o, node_color = color, node_zorder = 3, node_alpha=None, show=False, close=False)
    if (2 in a): # plot tree
        color = [[1.0,0.0,0.0,1.0] if (node in tree_targets) else [0.0,0.0,0.0,0.0] for node in G.nodes()]
        color[list(G.nodes()).index(origin)] = [0.0,0.0,1.0,1.0]
        edge_color = [[1.0,0.0,0.1,0.8] if (edge in tree.edges()) else [0.5,0.5,0.5,0.5] for edge in G.edges()]
        edge_linewidth = [1 if (edge in tree.edges()) else 0.5 for edge in G.edges()]
        fig, ax = ox.plot_graph(G_o, node_color = color, node_zorder = 3, edge_color = edge_color,edge_linewidth = edge_linewidth,node_alpha=None, edge_alpha=None, show=False, close=False)
    if (3 in a): # plot different clusters
        color = [[0.0,0.0,0.0,0.0] for node in G.nodes()]
        cltones = []
        for k in range(len(Clusters)):
            #col = np.random.random(size=3)
            col = colorsys.hsv_to_rgb(k/len(Clusters), 1.0, 0.7)
            for nd in Clusters[k]:
                color[list(G.nodes()).index(nd)] = [col[0],col[1],col[2],1.0]
            cltones.append(col)
        for el in Cl_centers:
             color[list(G.nodes()).index(el)] = [0.5,0.5,0.5,1.0]
        #-----------------------------------------------------------------
        edge_color='#999999'
        edge_linewidth = 1
        if (origin is not None):
            color[list(G.nodes()).index(origin)] = [0.0,0.0,0.0,1.0]
        if (tree is not None):
            #edge_color = [[1.0,0.0,0.1,0.8] if (edge in tree.edges()) else [0.5,0.5,0.5,0.5] for edge in G.edges()]
            edge_color = [[0.0,0.0,0.0,1.0] if (edge in tree.edges()) else [0.5,0.5,0.5,0.5] for edge in G.edges()]
            edge_linewidth = [1 if (edge in tree.edges()) else 0.5 for edge in G.edges()]
        if (Trees is not None):
            listG = list(G.edges())
            for j in range(len(Trees)):
                tr = Trees[j]
                edge_color = [cltones[j] if (listG[i] in tr.edges()) else edge_color[i] for i in range(len(listG))]
                edge_linewidth = [1 if (listG[i] in tr.edges()) else edge_linewidth[i] for i in range(len(listG))]
        #fig, ax = ox.plot_graph(G, node_color = color, node_zorder = 3, edge_color = edge_color,edge_linewidth = edge_linewidth,node_alpha=None, edge_alpha=None, show=False, close=False)
        #-----------------------------------------------------------------    
        fig, ax = ox.plot_graph(G_o, node_color = color, edge_color = edge_color,edge_linewidth = edge_linewidth, node_zorder = 3, node_alpha=None,node_size=100,show=False, close=False)
    if (4 in a): # plot routes
        color = [[0.0,0.0,1.0,1.0] if (node in src_nds) else [0.0,0.0,0.0,0.0] for node in G.nodes()]
        for el in dst_nds:
             color[list(G.nodes()).index(el)] = [0.0,1.0,0.0,1.0]
        Routes = [nx.shortest_path(G, src_nds[i], dst_nds[i], weight='length') for i in range(len(dst_nds))]
        #route_color = ['r' for i in dst_nds]
        route_color = ['r']*sum([len(i) for i in Routes])
        if twoWay:
            Routes2 = [nx.shortest_path(G, dst_nds[i], src_nds[i], weight='length') for i in range(len(dst_nds))]
            Routes.extend(Routes2)
            #route_color.extend(['b' for i in dst_nds])
            route_color = route_color +  ['b']*sum([len(i) for i in Routes2])
        fig, ax = ox.plot_graph_routes(G_o, Routes, node_color=color,route_alpha=0.7,orig_dest_node_alpha=1.0,orig_dest_node_size=20,route_linewidth=3, node_zorder=5, node_alpha=None, route_color=route_color, show=False, close=False)
    if (5 in a): # plot dendrogram
        dn = sch.dendrogram(Z)

    plt.show()
    return

def save_stuff(Graph,a):
    pass
##    #ox.save_graph_shapefile(Graph, filename='network-shape')
##    ox.save_graphml(Graph, filename='network.graphml')
##    fig, ax = ox.plot_graph(Graph, show=False, save=True, filename='network', file_format='svg')
##
##    #G2 = ox.load_graphml('network.graphml') #load
##    fig, ax = ox.plot_graph(G2)
##
##    gdf = ox.footprints_from_place(place='Piedmont, California, USA') #save building footprints
##    gdf.drop(labels='nodes', axis=1).to_file('data/piedmont_bldgs')
    
    if (0 in a):
        ox.save_graphml(Graph,'network.graphml')
    if (1 in a):
        fh=open("test.adjlist",'wb')
        nx.write_adjlist(Graph, fh, delimiter=',')
        fh.close()
    if (2 in a):
        A = nx.adjacency_matrix(Graph)
        #A=A.todense()
        #np.savetxt('file1',A,delimiter=',')
        sp.sparse.save_npz('file1', A, compressed=True)
    if (3 in a):
        fire.to_pickle('fire.pkl')
        healthcare.to_pickle('healthcare.pkl')
        shops.to_pickle('shops.pkl')
    return

def boolMat2Pairs(boolMat,house_nd,object_nd):
    srcs = []
    dstns = []
    for i in range(len(boolMat)):
        tmp = [i for i, x in enumerate(boolMat[i]) if x]
        srcs.extend([house_nd[i]]*len(tmp))
        dstns.extend(object_nd[tmp])
    return srcs, dstns

##############################################################################################################
##############################################################################################################
# define what to search
healthcare_list = ['hospital','dentist','baby_hatch','clinic','doctors']
fire_list = ['fire_station']
shop_list = ['convenience','department_store','general','mall','supermarket']
##############################################################################################################
##############################################################################################################
def load_graph(a):
    global G
    if (a==0):
        try:
            G = ox.load_graphml('network.graphml') #load
            print("loaded from graphml file")   
        except: pass
    elif (a==1):
        try:
            G = ox.graph_from_xml('network.osm', bidirectional=False, simplify=True, retain_all=False)
            e = list(G.edges(data=True))
            sub_ed = []
            for el in e:
                if 'highway' in el[2]:
                    if (el[2]['highway'] in ['primary','motorway','trunk','secondary','tertiary','residential','road','unclassified']):
                        sub_ed.append(tuple([el[0],el[1],0]))

            G = nx.edge_subgraph(G,sub_ed)
            print("loaded from osm xml")
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

def load_objects(a):
    global healthcare, fire, shops
    if (a==0):
        try:
            healthcare = pd.read_pickle('healthcare.pkl')
            print('loaded saved healthcare objects from file')
        except: healthcare = None
        try:
            fire = pd.read_pickle('fire.pkl')
            print('loaded saved fire station objects from file')
        except: fire = None
        try:
            shops = pd.read_pickle('shops.pkl')
            print('loaded saved shops objects from file')
        except: shops = None
    elif (a==1):
        try:
            #load from xml
            tree = ET.parse('network.osm')
            root = tree.getroot()
            healthcare_set = []
            for i in healthcare_list:
                healthcare_set.extend(root.findall(".//way/tag/[@k='amenity'][@v='"+i+"']/.."))
                healthcare_set.extend(root.findall(".//node/tag/[@k='amenity'][@v='"+i+"']/.."))
            healthcare_n = []
            for i in healthcare_set:
                try:    healthcare_n.append(i.find("./nd").get('ref'))
                except: pass
                try:    healthcare_n.append(i.get('id'))
                except: pass

            fire_set = []
            for i in fire_list:
                fire_set.extend(root.findall(".//way/tag/[@k='amenity'][@v='"+i+"']/.."))
                fire_set.extend(root.findall(".//node/tag/[@k='amenity'][@v='"+i+"']/.."))
            fire_n = []
            for i in fire_set:
                try:    fire_n.append(i.find("./nd").get('ref'))
                except: pass
                try:    fire_n.append(i.get('id'))
                except: pass

            shops_set = []
            for i in shop_list:
                shops_set.extend(root.findall(".//way/tag/[@k='shop'][@v='"+i+"']/.."))
                shops_set.extend(root.findall(".//node/tag/[@k='shop'][@v='"+i+"']/.."))
            shops_n = []
            for i in fire_set:
                try:    shops_n.append(i.find("./nd").get('ref'))
                except: pass
                try:    shops_n.append(i.get('id'))
                except: pass

            X = []
            Y = []
            for i in healthcare_n:
                ans = root.find(".//node[@id='"+i+"']")
                try:
                    Y.append(ans.get('lat'))
                    X.append(ans.get('lon'))
                except: pass
            healthcare = pd.DataFrame({'osmid':healthcare_n,'Latitude':Y,'Longitude':X})

            X = []
            Y = []
            for i in fire_n:
                ans = root.find(".//node[@id='"+i+"']")
                try:
                    Y.append(ans.get('lat'))
                    X.append(ans.get('lon'))
                except: pass
            fire = pd.DataFrame({'osmid':fire_n,'Latitude':Y,'Longitude':X})

            X = []
            Y = []
            for i in shops_n:
                ans = root.find(".//node[@id='"+i+"']")
                try:
                    Y.append(ans.get('lat'))
                    X.append(ans.get('lon'))
                except: pass
            shops = pd.DataFrame({'osmid':shops_n,'Latitude':Y,'Longitude':X})   
##            a = root.findall(".//way/tag/[@k='amenity']/..")
##            b = root.findall(".//way/tag/[@k='amenity']/..")
##            c = root.findall(".//way/tag/[@k='shop']/..")
##            a_tree = ET.ElementTree(a)
##            for ae in a:
##                ae.find("./tag/[@v='hospital']/..")

            
            print('loaded objects from osm file')
        except: healthcare = fire = shops = None
    if (healthcare is None):
        healthcare = ox.pois_from_place('Ижевск, Россия', {'amenity':healthcare_list}, which_result=1)
        print('downloaded healthcare objects from internet')
    if (fire is None):
        fire = ox.pois_from_place('Ижевск, Россия', {'amenity':fire_list}, which_result=1)
        print('downloaded fire station objects from internet')
    if (shops is None):
        shops = ox.pois_from_place('Ижевск, Россия', {'shop':shop_list}, which_result=1)
        print('downloaded shops objects from internet')
    return
#-----------------------
G = None
G_o = None
healthcare = fire = shops = None

  
def main(N=100,M=10,k=6,X=5000,choice=0):
    global G, G_o
    global healthcare, fire, shops
    global objects
    global objects_nodes
    global house_nodes
    global NearestObjectsTo, NearestObjectsFrom, ThereAndBack, IsReachableObjects, IsReachableHouses, IsReachableThereAndBack, dist2objects, dist2houses
    global A, B, C, a, b, c, km
    global D
    global E
    global Xm
    global srcs, dstns, srcs1, dstns1, srcs2, dstns2
    global new_object, tree, tree_weight, sum_dist
    global tree, tree_weight, sum_dis, Clustering, Cl, Z, Z1
    global  Clusters, Clusters1, TreeFromObject, Trees, center_nd, tree_weight1, sum_dist1, treedist

    if (G is None):
        load_graph(0)

    #fig, ax = ox.plot_graph(G_simple)

    t0 = tm.time()

    largest = max(nx.strongly_connected_components(G), key=len)

    G = G.subgraph(largest) # to work with only strongly connected graph
    if G_o is None:
        G_o = G
    G = G.to_directed()
    G = nx.DiGraph(G) # to work with DiGraphs only

    # search
    #healthcare = ox.pois_from_place('Ижевск, Россия', amenities=healthcare_list, which_result=1)
    
##    healthcare = ox.pois_from_place('Ижевск, Россия', {'amenity':healthcare_list}, which_result=1)
##    fire = ox.pois_from_place('Ижевск, Россия', {'amenity':fire_list}, which_result=1)
##    shops = ox.pois_from_place('Ижевск, Россия', {'shop':shop_list}, which_result=1)

    if (healthcare is None or fire is None or shops is None):
        load_objects(0)

    if (choice == 0):
        objects = healthcare
    elif (choice == 1):
        objects = fire
    else:
        objects = shops
    
    # get objects coordinates
    objects = objects.centroid
    objects_coord = (np.array([objects.x,objects.y]))
    # get subset of objects
    objects_coord = objects_coord[:,np.random.choice(len(objects_coord[0]),size = M, replace = False)]

    #find nodes near coordinates
   
    objects_nodes = ox.get_nearest_nodes(G, objects_coord[0],objects_coord[1])
    # ? maybe merge close nodes ?

    # get houses
    ##houses = ox.footprints.footprints_from_place(place='Ижевск, Россия')
    ###gdf_proj = ox.project_gdf(houses)
    ####fig, ax = ox.footprints.plot_footprints(gdf_proj, bgcolor='#333333', color='w', 
    ###                            save=True, show=False, close=True,  filename='piedmont_bldgs', dpi=40)
    ##houses = houses.centroid
    ##houses_coord = (np.array([houses.x,houses.y]))
    ### get subset of houses
    ##houses_coord = houses_coord[:,np.random.choice(len(houses_coord[0]),size = 100, replace = False)]
    ##
    ##house_nodes = ox.get_nearest_nodes(G, houses_coord[0],houses_coord[1])
    
    house_nodes = np.random.choice(list(G.nodes), size = N, replace = False) # get houses as random nodes

    adjust_weights(house_nodes,objects_nodes) # adjust weights

    t1 = tm.time()
    Xm = X
    km=k
    NearestObjectsTo, NearestObjectsFrom, ThereAndBack, IsReachableObjects, IsReachableHouses, IsReachableThereAndBack, dist2objects, dist2houses = Part1_1(house_nodes,objects_nodes,X=Xm)
    t2 = tm.time()
    print(t2-t0,' ',t2-t1)

    t1 = tm.time()
    
    A, B, C, a, b, c = Part1_2(house_nodes,objects_nodes, dist2objects, dist2houses)
    t2 = tm.time()
    print(t2-t0,' ',t2-t1)

    t1 = tm.time()
   
    D = Part1_3(house_nodes,objects_nodes, dist2houses)
    t2 = tm.time()
    print(t2-t0,' ',t2-t1)

    t1 = tm.time()
    E = Part1_4(house_nodes,objects_nodes)
    t2 = tm.time()
    print(t2-t0,' ',t2-t1)

    
    srcs, dstns = boolMat2Pairs(IsReachableObjects,house_nodes,objects_nodes)
    srcs1, dstns1 = boolMat2Pairs(IsReachableHouses,objects_nodes,house_nodes,)
    srcs2, dstns2 = boolMat2Pairs(IsReachableThereAndBack,house_nodes,objects_nodes)

    #------------------------------------------------------------part 2
    
    new_object = np.random.choice(list(G.nodes), size = 1, replace = False)[0]
    t1 = tm.time()
    tree, tree_weight, sum_dist = Part2_1(house_nodes,new_object)
    t2 = tm.time()
    print(t2-t0,' ',t2-t1)

    t1 = tm.time()
    
    dist_hous2house = getDistHouse2House(house_nodes)
    Z1 = Part2_2(house_nodes,dist_hous2house) # this one is used
    Z = Part2_2_scipy(house_nodes,dist_hous2house) #is not used
    t2 = tm.time()

    print(t2-t0,' ',t2-t1)
    t1 = tm.time()
    
    Clusters = getClusters(k, house_nodes, Z = Z1)
    Clusters1 = getClusters(k, house_nodes, Z = Z)
    TreeFromObject, Trees, center_nd, tree_weight1, sum_dist1, treedist = Part2_3(Clusters,house_nodes,new_object)
    t2 = tm.time()
    print(t2-t0,' ',t2-t1)

    print ('calculations done')
    #G = G_o
    return

#------------------------------------------------------------------------------------------------------
def Plot_results_Part0():
    print('plotting city road network. Red nodes - houses, blue nodes - objects')
    plot_stuff([1], house_nodes = house_nodes, objects_nodes = objects_nodes) #plot map,houses and objects
    return

def Plot_results_Part1_1a():
    print('plotting shortest paths from every house to a nearest object. Blue nodes - houses, green nodes - objects')
    plot_stuff([4], src_nds=house_nodes,dst_nds=NearestObjectsTo) #there
    print('plotting shortest paths to every house from a nearest object. Green nodes - houses, blue nodes - objects')
    plot_stuff([4], src_nds=NearestObjectsFrom,dst_nds=house_nodes) #back
    print('plotting shortest paths from every house to a nearest object and back. blue nodes - houses, green nodes - objects')
    plot_stuff([4], src_nds=house_nodes,dst_nds=objects_nodes[ThereAndBack],twoWay = True) #there and back
    return

def Plot_results_Part1_1b():
    print('plotting paths from every house to objects where path length is less than ',Xm,'m. Blue nodes - houses, green nodes - objects')
    plot_stuff([4], src_nds=srcs,dst_nds=dstns) #there reachable
    print('plotting paths to every house from objects where path length is less than ',Xm,'m. Green nodes - houses, blue nodes - objects')
    plot_stuff([4], src_nds=srcs1,dst_nds=dstns1) #back reachable
    print('plotting paths from every house to objects and back where path length is less than ',Xm,'m. Blue nodes - houses, green nodes - objects')
    plot_stuff([4], src_nds=srcs2,dst_nds=dstns2,twoWay = True) #there and back reachable
    return

def Plot_results_Part1_2():
    #print(A,B,C,a,b,c)
    print('plotting route from house id=',house_nodes[a],' to object id=',objects_nodes[A],' that the longest path from any house is shorter than for any other object. Blue node - house, green node - object')
    plot_stuff([4], src_nds=[house_nodes[a]],dst_nds=[objects_nodes[A]]) #there
    print('plotting route to house id=',house_nodes[b],' from object id=',objects_nodes[B],' that the longest path to any house is shorter than for any other object. Green node - house, blue node - object')
    plot_stuff([4], src_nds=[objects_nodes[B]],dst_nds=[house_nodes[b]]) #back
    print('plotting route from house id=',house_nodes[a],' to object id=',objects_nodes[A],' and back that the longest path from any house is shorter than for any other object. Blue node - house, green node - object')
    plot_stuff([4], src_nds=[house_nodes[c]],dst_nds=[objects_nodes[C]],twoWay = True) #there and back
    return

def Plot_results_Part1_3():
    print('plotting shortest paths tree from object id=', objects_nodes[D],' to all houses. The sum of distances to houses for this object is less than for any other object')
    tree1, dist1 = make_paths_tree(house_nodes,objects_nodes[D])
    plot_stuff([2],tree=tree1,origin=objects_nodes[D],tree_targets=house_nodes)
    return

def Plot_results_Part1_4():
    print('plotting shortest paths tree from object id=', objects_nodes[E],' to all houses. Tree\'s weight for this object is less than for any other object')
    tree2, dist2 = make_paths_tree(house_nodes,objects_nodes[E])
    plot_stuff([2],tree=tree2,origin=objects_nodes[E],tree_targets=house_nodes)
    return

def Plot_results_Part2_1():
    print('plotting shortest paths tree from new object id=', new_object,' to all houses. The sum of distances to houses is ',sum_dist,'m. Tree\'s weight is ',tree_weight)
    #print(tree_weight, sum_dist)
    plot_stuff([2],tree=tree,origin=new_object,tree_targets=house_nodes)
    return

def Plot_results_Part2_2():
    print('plotting dendrogram for house clustering. All houses are presented by their indices in house_nodes array.')
    plot_stuff([5],Z=Z1)
    return

def Plot_results_Part2_3():
    print('plotting',km,' clusters, shortest paths from new object to centers of clusters and trees from centers to nodes in cluster. New object - black node, centers - gray nodes')
    print('clusters tree\'s weights: ',tree_weight1)
    print('clusters tree\'s sum of distances: ',sum_dist1)
    print('tree from new object distances: ',treedist,'\ntree from new object weight: ',TreeFromObject.size(weight='length'))
    #plot_stuff([3],Clusters=Clusters1,house_nodes=house_nodes,Cl_centers=center_nd)
    plot_stuff([3],Clusters=Clusters,house_nodes=house_nodes,origin=new_object,tree=TreeFromObject,Cl_centers=center_nd,Trees=Trees)
    #plot_stuff([1], house_nodes = house_nodes, objects_nodes = [], Cl_centers=center_nd)
