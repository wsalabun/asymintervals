import matplotlib.pyplot as plt
import networkx as nx
try:
    from .ain import AIN
except ImportError:
    from asymintervals import AIN

class GraphAIN:
    """
    A class for creating and visualizing graphs with AIN (Asymmetric Interval Number) nodes.

    Supports both directed and undirected graphs where edges are weighted based on
    the probability relationships between AIN instances.

    Parameters
    ----------
    directed : bool, optional
        If True, creates a directed graph. If False, creates an undirected graph.
        Default is False.

    Attributes
    ----------
    graph : networkx.Graph or networkx.DiGraph
        The underlying NetworkX graph object.
    nodes_data : dict
        Dictionary mapping node names to their AIN instances.
    directed : bool
        Whether the graph is directed or undirected.

    Examples
    --------
    Creating an undirected graph:
    >>> A = AIN(0, 10, 2)
    >>> B = AIN(2, 8, 3)
    >>> C = AIN(4, 12, 5)
    >>> D = AIN(6, 14, 11)
    >>>
    >>> g = GraphAIN(directed=False)
    >>> g.add_node("A", A)
    >>> g.add_node("B", B)
    >>> g.add_node("C", C)
    >>> g.add_node("D", D)
    >>>
    >>> g.plot() # doctest: +SKIP

    Creating a directed graph:
    >>> g_directed = GraphAIN(directed=True)
    >>> g_directed.add_node("A", A)
    >>> g_directed.add_node("B", B)
    >>> _ = g_directed.plot() # doctest: +SKIP
    """

    def __init__(self, directed=False, edge_threshold=0.0, dominance_only=False):
        """
        Initialize a GraphAIN instance.

        Parameters
        ----------
        directed : bool, optional
            If True, creates a directed graph. Default is False (undirected).
        edge_threshold : float, optional
            Minimum edge weight required to add an edge (epsilon).
            Edges with weight <= edge_threshold are ignored.
            Default is 0.0.
        dominance_only: bool, optional
            If True (directed graphs only), for each pair of nodes (A, B)
            only the direction with the larger weight is added

        Examples
        --------
        >>> g = GraphAIN(directed=True)
        >>> print(g.directed)
        True
        >>> g2 = GraphAIN()
        >>> print(g2.directed)
        False
        """

        if not isinstance(directed, bool):
            raise TypeError("directed must be a boolean")
        if not isinstance(dominance_only, bool):
            raise TypeError("dominance_only must be a boolean")
        if not isinstance(edge_threshold, float):
            raise TypeError("edge_threshold must be a float")
        if edge_threshold < 0.0 or edge_threshold > 1.0:
            raise ValueError("edge_threshold must be between 0.0 and 1.0")
        if dominance_only and not directed:
            raise ValueError("dominance_only can only be True for directed graphs")

        self.directed = directed
        self.edge_threshold = edge_threshold
        self.dominance_only = dominance_only
        if directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

        self.nodes_data = {}

    def add_node(self, name, ain_instance):
        """
        Add a node to the graph with an associated AIN instance.

        Parameters
        ----------
        name : str
            The name/label of the node.
        ain_instance : AIN
            The AIN instance associated with this node.

        Raises
        ------
        TypeError
            If name is not a string or ain_instance is not an AIN instance.
        ValueError
            If a node with this name already exists.

        Examples
        --------
        >>> g = GraphAIN()
        >>> a = AIN(0, 10, 5)
        >>> g.add_node("A", a)
        >>> "A" in g.nodes_data
        True
        """
        if not isinstance(name, str):
            raise TypeError("Node name must be a string")
        if not isinstance(ain_instance, AIN):
            raise TypeError("ain_instance must be an AIN instance")
        if name in self.nodes_data:
            raise ValueError(f"Node '{name}' already exists in the graph")

        self.graph.add_node(name)
        self.nodes_data[name] = ain_instance

        for other, other_ain in self.nodes_data.items():
            if other == name:
                continue

            if self.directed:
                if self.dominance_only:
                    self._add_directed_edge_max(name, other)

                else:
                    self._add_directed_edge(name, other)
                    self._add_directed_edge(other, name)
            else:
                self._add_undirected_edge(name, other)

    def _add_directed_edge_max(self, u, v):
        p_uv = self.nodes_data[u] > self.nodes_data[v]
        p_vu = self.nodes_data[v] > self.nodes_data[u]

        if p_uv > p_vu:
            weight = p_uv
            src, dst = u, v
        elif p_vu > p_uv:
            weight = p_vu
            src, dst = v, u
        else:
            src, dst = (u, v) if u < v else (v, u)
            weight = p_uv  # == p_vu

        if weight > self.edge_threshold:
            self.graph.add_edge(src, dst, weight=weight)

    def _add_directed_edge(self, u, v):
        p = self.nodes_data[u] > self.nodes_data[v]
        w = float(f"{p:.4f}")
        if w > self.edge_threshold:
            self.graph.add_edge(u, v, weight=w)

    def _add_undirected_edge(self, u, v):
        p = self.nodes_data[v] > self.nodes_data[u]
        w = float(f"{4 * p * (1 - p):.4f}")
        if w > self.edge_threshold:
            self.graph.add_edge(u, v, weight=w)

    def plot(self, figsize=(5, 4), node_size=1000, font_size=12,
             layout='spring', seed=42, save_path=None, dpi=300, edge_decimals = 2):
        """
        Visualize the graph using matplotlib and networkx.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Default is (5, 4).
        node_size : int, optional
            Size of the nodes. Default is 1000.
        font_size : int, optional
            Font size for node labels. Default is 12.
        layout : str, optional
            Layout algorithm: 'spring', 'circular', 'kamada_kawai', 'random'.
            Default is 'spring'.
        seed : int, optional
            Random seed for layout algorithms. Default is 42.
        save_path : str, optional
            If provided, saves the figure to this path. Default is None.
        dpi : int, optional
            Resolution for saved figure. Default is 300.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.

        Examples
        --------
        >>> g = GraphAIN()
        >>> # ... add nodes ...
        >>> _ = g.plot(layout='circular', save_path='my_graph.pdf')  # doctest: +SKIP
        """

        if len(self.graph.nodes()) == 0:
            raise ValueError("Graph has no nodes to plot")

        fig, ax = plt.subplots(figsize=figsize)

        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, seed=seed)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == 'random':
            pos = nx.random_layout(self.graph, seed=seed)
        else:
            raise ValueError(f"Unknown layout: '{layout}'")

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_size,
            node_color='lightblue'
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=font_size
        )

        if self.directed:
            # Draw directed edges with curvature for bidirectional pairs
            for (u, v) in self.graph.edges():
                rad = 0.12 if self.graph.has_edge(v, u) else 0.0
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=[(u, v)],
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=20,
                    edge_color='gray',
                    connectionstyle=f"arc3,rad={rad}",
                    min_source_margin=15,
                    min_target_margin=15
                )
        else:
            # Draw undirected edges
            nx.draw_networkx_edges(
                self.graph, pos,
                edge_color='gray'
            )

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        if edge_labels:
            if self.directed:
                # Place labels slightly off the edge to avoid overlap
                for (u, v), w in edge_labels.items():
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]

                    xm = 0.5 * (x1 + x2)
                    ym = 0.5 * (y1 + y2)

                    rad = 0.12 if self.graph.has_edge(v, u) else 0.0
                    dx = (y2 - y1) * rad * 0.7
                    dy = -(x2 - x1) * rad * 0.7

                    plt.text(
                        xm + dx, ym + dy,
                        f"{w:.{edge_decimals}f}",
                        fontsize=font_size - 2,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9)
                    )
            else:
                nx.draw_networkx_edge_labels(
                    self.graph, pos,
                    edge_labels=edge_labels,
                    font_size=font_size - 2
                )

        plt.gca().set_axis_off()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        plt.show()
        plt.close(fig)
        return fig

    def get_adjacency_matrix(self):
        """
        Get the adjacency matrix of the graph.

        Returns
        -------
        numpy.ndarray
            The adjacency matrix with edge weights.

        Examples
        --------
        >>> g = GraphAIN(directed=True)
        >>> a = AIN(0, 10, 2)
        >>> b = AIN(2, 8, 3)
        >>> g.add_node("A", a)
        >>> g.add_node("B", b)
        >>> M = g.get_adjacency_matrix()
        >>> M.shape
        (2, 2)
        >>> float(M[0, 1]) == (g.get_edge_weight("A", "B") or 0.0)
        True
        """
        return nx.adjacency_matrix(self.graph).todense()

    def get_edge_weight(self, node1, node2):
        """
        Get the weight of an edge between two nodes.

        Parameters
        ----------
        node1 : str
            Name of the first node.
        node2 : str
            Name of the second node.

        Returns
        -------
        float or None
            The edge weight, or None if no edge exists.

        Examples
        --------
        >>> g = GraphAIN()
        >>> # ... add nodes and edges ...
        >>> weight = g.get_edge_weight("A", "B")
        """
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]['weight']
        return None

    def get_node_degree(self, node):
        """
        Get the degree of a node.

        Parameters
        ----------
        node : str
            Name of the node.

        Returns
        -------
        int
            The degree of the node.

        Examples
        --------
        >>> g = GraphAIN()
        >>> # ... add nodes and edges ...
        >>> degree = g.get_node_degree("A")
        """
        return self.graph.degree(node)

    def summary(self):
        """
        Print a summary of the graph.

        Examples
        --------
        >>> g = GraphAIN(directed=True)
        >>> a = AIN(0, 10, 2)
        >>> b = AIN(2, 8, 3)
        >>> g.add_node("A", a)
        >>> g.add_node("B", b)
        >>> g.summary()
        ==================================================
        Graph Type: Directed
        Number of Nodes: 2
        Number of Edges: 2
        ==================================================
        Nodes:
          A: [0.0000, 10.0000]_{2.0000}
          B: [2.0000, 8.0000]_{3.0000}
        ==================================================
        Edges (with weights):
          A -> B: 0.1750
          B -> A: 0.8250
        ==================================================
        """
        print("=" * 50)
        print(f"Graph Type: {'Directed' if self.directed else 'Undirected'}")
        print(f"Number of Nodes: {self.graph.number_of_nodes()}")
        print(f"Number of Edges: {self.graph.number_of_edges()}")
        print("=" * 50)
        print("Nodes:")
        for node, ain in self.nodes_data.items():
            print(f"  {node}: {ain}")
        print("=" * 50)
        print("Edges (with weights):")
        for u, v, data in self.graph.edges(data=True):
            print(f"  {u} -> {v}: {data['weight']:.4f}")
        print("=" * 50)

    def __repr__(self):
        """String representation of the GraphAIN instance."""
        graph_type = "Directed" if self.directed else "Undirected"
        return (f"GraphAIN({graph_type}, "
                f"nodes={self.graph.number_of_nodes()}, "
                f"edges={self.graph.number_of_edges()})")

    def average_uncertainty(self):
        """
        Calculate the average graph uncertainty for an undirected graph.

        For an undirected UAIG (Uncertain Asymmetric Interval Graph) with n vertices,
        the average uncertainty measures the overall uncertainty across all node pairs.

        Returns
        -------
        float
            The average graph uncertainty value in range [0, 1].
            Returns 0.0 for graphs with fewer than 2 nodes.

        Raises
        ------
        ValueError
            If called on a directed graph (this metric is only defined for undirected graphs).

        Examples
        --------
        >>> A = AIN(0, 10, 2)
        >>> B = AIN(2, 8, 3)
        >>> C = AIN(4, 12, 5)
        >>> g = GraphAIN(directed=False)
        >>> g.add_node("A", A)
        >>> g.add_node("B", B)
        >>> g.add_node("C", C)
        >>> uncertainty = g.average_uncertainty()
        >>> print(f"Average uncertainty: {uncertainty:.4f}")
        Average uncertainty: 0.4643
        """
        if self.directed:
            raise ValueError("Average uncertainty is only defined for undirected graphs")

        n = self.graph.number_of_nodes()

        # Edge case: graphs with 0 or 1 node have no uncertainty
        if n < 2:
            return 0.0

        # Sum all edge weights
        total_weight = 0.0
        node_names = list(self.nodes_data.keys())

        for i in range(n):
            for j in range(i + 1, n):
                node_i = node_names[i]
                node_j = node_names[j]

                # Calculate weight even if edge doesn't exist in graph
                # (it might be filtered by edge_threshold)
                ain_i = self.nodes_data[node_i]
                ain_j = self.nodes_data[node_j]
                p = ain_j > ain_i
                weight = 4 * p * (1 - p)

                total_weight += weight

        # Apply the normalization factor
        avg_uncertainty = (2 * total_weight) / (n * (n - 1))

        return avg_uncertainty


    def graph_entropy(self):
        """
        Calculate the graph entropy for an undirected graph.

        For an undirected UAIG (Uncertain Asymmetric Interval Graph) with n vertices,
        the graph entropy measures the overall information-theoretic uncertainty
        across all node pairs using the binary entropy function.

        Returns
        -------
        float
            The graph entropy value in range [0, 1].
            Returns 0.0 for graphs with fewer than 2 nodes.

        Raises
        ------
        ValueError
            If called on a directed graph (this metric is only defined for undirected graphs).

        Examples
        --------
        >>> A = AIN(0, 10, 2)
        >>> B = AIN(2, 8, 3)
        >>> C = AIN(4, 12, 5)
        >>> g = GraphAIN(directed=False)
        >>> g.add_node("A", A)
        >>> g.add_node("B", B)
        >>> g.add_node("C", C)
        >>> entropy = g.graph_entropy()
        >>> print(f"Graph entropy: {entropy:.4f}")
        Graph entropy: 0.5663
        """
        import numpy as np

        if self.directed:
            raise ValueError("Graph entropy is only defined for undirected graphs")

        n = self.graph.number_of_nodes()

        # Edge case: graphs with 0 or 1 node have no entropy
        if n < 2:
            return 0.0

        def binary_entropy(p):
            """
            Binary entropy function h(p) = -(p*log2(p) + (1-p)*log2(1-p))
            with the convention that 0*log2(0) = 0.
            """
            # Handle edge cases where p is 0 or 1
            if p <= 0.0 or p >= 1.0:
                return 0.0

            # Calculate binary entropy
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        # Sum binary entropy for all pairs
        total_entropy = 0.0
        node_names = list(self.nodes_data.keys())

        for i in range(n):
            for j in range(i + 1, n):
                node_i = node_names[i]
                node_j = node_names[j]

                # Calculate p_ij = P(X_j > X_i)
                ain_i = self.nodes_data[node_i]
                ain_j = self.nodes_data[node_j]
                p_ij = ain_j > ain_i

                # Add binary entropy of this probability
                total_entropy += binary_entropy(p_ij)

        # Apply the normalization factor
        graph_ent = (2 * total_entropy) / (n * (n - 1))

        return graph_ent


if __name__=='__main__':
    A = AIN(0, 10, 2)
    B = AIN(2, 8, 3)
    C = AIN(4, 12, 5)
    # D = AIN(6, 14, 11)
    g = GraphAIN(directed=False, edge_threshold=0.0)
    g.add_node("A", A)
    g.add_node("B", B)
    g.add_node("C", C)
    # g.add_node("D", D)
    _ = g.plot(layout='circular')
    print(f"Average uncertainty: {g.graph_entropy():.4f}")



    # # A = AIN(0, 10, 2)
    # B = AIN(2, 8, 3)
    # C = AIN(4, 12, 5)
    # D = AIN(6, 14, 11)
    # g = GraphAIN(directed=True, edge_threshold=0.0, dominance_only=True)
    # g.add_node("A", A)
    # g.add_node("B", B)
    # g.add_node("C", C)
    # g.add_node("D", D)
    # g.summary()
    # _ = g.plot(layout='spring', edge_decimals=3)