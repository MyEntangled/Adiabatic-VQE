import networkx as nx
from Generating_Problems import MIS
from Calculating_Expectation_Values import ExpectationValues
from QIRO_MIS import QIRO

import pennylane as qml
import matplotlib.pyplot as plt

import random
import numpy as np

seed = 123
random.seed(seed)
np.random.seed(seed)

graph = nx.random_geometric_graph(n=10, radius=0.5)
print(graph.edges)
nx.draw(graph, with_labels = True)
plt.show()
alpha = 1.1
mis_problem = MIS(graph, alpha)

qiro_obj = QIRO(mis_problem, nc=1, strategy="Max", no_correlation=1, temperature=10)
qiro_obj.execute()
print(qiro_obj.solution)


