# FiniteGraphsProject# City Network Analysis Tool

CityNAT is a python library designed to analyse city's network using Finite Graph Theory. 

## Requirements
**Environment:**\
Python 3.6, pip

**Dependencies:**
* osmnx 0.13.0
* networkx 2.4
* numpy 1.18.4
* matplotlib 3.2.1
* lxml 4.5.1
* scipy 1.4.1

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install CityNAT.

```bash
pip install citynat
```
Known bug during installation: module [OSMnx](https://github.com/gboeing/osmnx/blob/master/README.md) may require you to manually install its dependent library [Rtree](https://pypi.org/project/Rtree/) first. To install it simply use:
```bash
pip install Rtree
```

## Usage
This example loads road network from file, gets objects with given tags from file and plots shortest paths to objects from a random node.

```python
import citynalib as cn
import numpy as np

file = open('network.osm')
G = cn.load_graph(file)   # ejects road network and returns networkx's MultiDiGraph object 
                          # from OpenStreetMap .osm format file
tags = {'shop':['department_store','general','mall','supermarket']}
objects = cn.load_objects(G,tags,file) # get osm nodes in graph G which correspond to 
                                       # objects from file with given tags
origin = np.random.choice(list(G.nodes), size = 1)[0] # get random node in G
tree, dist = cn.make_paths_tree(G,objects,origin) # get shortest paths tree from origin to all objects
cn.plot_stuff(G,[2],tree=tree,origin=origin) # plot this tree on a map using osmnx and matplotlib
```

![image](https://github.com/zarond/FiniteGraphsProject/blob/master/Figure_14.png)\
**_Program result_**

## Documentation
Main functions:
#### 1. Load network from file
```python
load_graph(file = None, place = None)
```
>Parameters:
>1. **file** (file (optional)) – if file is given, then extract road network from it, otherwise if place is not None, download network from internet using OSMnx functionality.
>2. **place** (string (optional)) – name of place to download data from, if extracting from file failed. Example: place='Moskow, Russia'.

>Returns: **G** (networkx.MultiDiGraph) – Graph created from given file or place.

#### 2. Load objects from file
```python
load_objects(graph, tags, file = None, place = None)
```
>Parameters:
>1. **graph** (networkx.MultiDiGraph) – loaded Graph.
>2. **tags** (dictionary) – dictionary of OSM tags – key and value pairs used to search for objects. Multiple values for one key are allowed.
>3. **file** (file (optional)) – if file is given, then search within the file, otherwise if place is not None, download objects using OSMnx functionality.
>4. **place** (string (optional)) – name of place to download objects from, if extracting from file failed.

>Returns: **objects** (list[integer]) – list of nodes ids which correspond with found objects.


#### 3. Make shortest paths tree
```python
make_paths_tree(graph,destinations,origin)
```
>Parameters:
>1. **graph** (networkx.MultiDiGraph) – Graph.
>2. **destinations** (list[integer]) – list of nodes ids from graph.
>3. **origin** (integer) – origin node id.

>Returns: 
>1. **tree** (networkx.DiGraph) – Graph's subtree with shortest paths.
>2. **dist** (list[double]) – distances from origin to objects.

#### 4. Compute Hierarchical Clustering
```python
clustering(graph,objects,dist = None)
```
>Parameters:
>1. **graph** (networkx.MultiDiGraph) – Graph.
>2. **objects** (list of integers) – list of nodes ids from graph.
>3. **dist** (np.array(optional)) – precomputed distances between all pairs of objects. If dist is None, then compute dist within the function itself.

>Returns: **linkage** (Zndarray) – The hierarchical clustering encoded as a linkage matrix computed with scipy.cluster.hierarchy.linkage using nodes from graph.

#### 5. Plot trees, objects, clusters on map
```python
plot_stuff(graph,modes,objects=None,tree=None,origin=None,Z=None,K=None)
```
>Parameters:
>1. **graph** (networkx.MultiDiGraph) – Graph.
>2. **modes** (list[integer]) – list of modes, in which the function must work. Depending on mode, the function will plot different things and accept different arguments. It is possible to use multiple modes to plot multiple things on one figure. List of modes:
>    - **modes = [0]** Plot graph and objects (if not None) only.
>    - **modes = [1]** Plot tree from origin node.
>    - **modes = [2]** Plot clusters of objects with different colors. Number of clusters is defined by variable **K**.
>3. **objects** (list[integer]\(optional)) – list of nodes ids from graph.
>4. **tree** (networkx.DiGraph(optional)) – Graph's subtree with shortest paths.
>5. **origin** (integer(optional)) – origin node id.
>6. **Z** (Zndarray(optional)) – The hierarchical clustering encoded as a linkage matrix.
>7. **K** (integer(optional)) – number of clusters to divide into.

## Plot examples:
![image](https://github.com/zarond/FiniteGraphsProject/blob/master/Figure_11.png)\
**_100 random objects in Ижевск, Россия._**

![image](https://github.com/zarond/FiniteGraphsProject/blob/master/Figure_14.png)\
**_Shortest paths tree from one object to all others_**

![image](https://github.com/zarond/FiniteGraphsProject/blob/master/Figure_13.png)\
**_Divide object into 6 clusters_**

![image](https://github.com/zarond/FiniteGraphsProject/blob/master/Figure_12.png)\
**_For each cluster find median node and find shortest paths from origin node (black node) to all median nodes (gray nodes) and from median nodes to all nodes in cluster._**



## License
[MIT](https://choosealicense.com/licenses/mit/)
