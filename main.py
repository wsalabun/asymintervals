from asymintervals import AIN, GraphAIN
import matplotlib.pyplot as plt
import pandas as pd


# # Tworzenie instancji AIN
# x = AIN(0, 10, 4)
# y = AIN(5, 15, 12)
# z = AIN(3, 13, 8)
#
# # Tworzenie grafu
# graph = GraphAIN(directed=False, edge_threshold=0.5)
# graph.add_node("X", x)
# graph.add_node("Y", y)
# graph.add_node("Z", z)
#
# # Wizualizacja
# graph.plot()
# plt.show()
#
# # Statystyki
# print(graph.summary())

df = pd.read_csv("epi.csv", delimiter=";")
lista =[]
for i, row in df.iterrows():
    lista.append((row["Name"], AIN(row["low"], row["upper"], row["expected"])))

g = GraphAIN(directed=False)
s = lista
# s = sorted(lista, key=lambda x: x[1].expected)[-100:]
i = 0
for name, ain in s:
    i=i+1
    g.add_node(f"{i}", ain)
# g.plot(layout="circular")
# plt.show()
# M = g.get_adjacency_matrix()
# print(M)
print(g.average_uncertainty())
print(g.graph_entropy())