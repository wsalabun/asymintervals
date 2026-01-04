from asymintervals import AIN, GraphAIN
import matplotlib.pyplot as plt

# Tworzenie instancji AIN
x = AIN(0, 10, 4)
y = AIN(5, 15, 12)
z = AIN(3, 13, 8)

# Tworzenie grafu
graph = GraphAIN(directed=False, edge_threshold=0.5)
graph.add_node("X", x)
graph.add_node("Y", y)
graph.add_node("Z", z)

# Wizualizacja
graph.plot()
plt.show()

# Statystyki
print(graph.summary())
