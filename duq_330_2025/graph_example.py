import matplotlib.pyplot as plt
import networkx as nx

nodes = list(range(9))
edges = [(1, 0), (2, 1), (3, 2), (4, 1), (5, 0),
         (0, 5), (6, 3), (7, 3), (8, 0)] # a connects to b

# if you want an undirected graph, use nx.Graph()
gr = nx.DiGraph()
gr.add_nodes_from(nodes)
gr.add_edges_from(edges)

pr = nx.pagerank(gr, max_iter=1000)
print(pr)

pos = nx.spring_layout(gr)
nx.draw_networkx_nodes(gr,pos,node_size=[pr[node]*300 for node in nodes])
nx.draw_networkx_edges(gr,pos)
plt.show()