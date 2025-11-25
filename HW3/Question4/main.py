import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_graph(
    G, title, path_length=None, diameter=None, clustering=None, filename=None
):
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray")
    plt.title(title)
    # Add statistics as text box in the figure
    stats = (
        f"Path Length: {path_length:.3f}\n"
        f"Diameter: {diameter}\n"
        f"Clustering: {clustering:.3f}"
    )
    plt.gcf().text(
        0.02, 0.80, stats, fontsize=10, bbox=dict(facecolor="white", alpha=0.7)
    )
    if filename:
        plt.savefig(filename)
    plt.show()
    plt.clf()


def generate_ws_graphs(n, c, p_values, num_graphs=2):
    graphs = {}
    for p in p_values:
        graphs[p] = []
        for i in range(num_graphs):
            G = nx.watts_strogatz_graph(n, c, p)
            graphs[p].append(G)
    return graphs


def analyze_graphs(graphs):
    results = {}
    for p, gs in graphs.items():
        results[p] = []
        for G in gs:
            path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
            clustering = nx.average_clustering(G)
            results[p].append(
                {
                    "path_length": path_length,
                    "diameter": diameter,
                    "clustering": clustering,
                }
            )
    return results


def part_d_ws_path_length_vs_p():
    n = 100
    c = 6
    p_values = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    num_graphs = 3

    avg_path_lengths = []
    std_path_lengths = []

    for p in p_values:
        path_lengths = []
        for i in range(num_graphs):
            G = nx.watts_strogatz_graph(n, c, p)
            # Ensure the graph is connected for path length calculation
            if not nx.is_connected(G):
                # Take the largest connected component
                Gc = max(nx.connected_components(G), key=len)
                G = G.subgraph(Gc)
            pl = nx.average_shortest_path_length(G)
            path_lengths.append(pl)
        avg = np.mean(path_lengths)
        std = np.std(path_lengths)
        avg_path_lengths.append(avg)
        std_path_lengths.append(std)

    # Plot l(p) with error bars
    plt.errorbar(p_values, avg_path_lengths, yerr=std_path_lengths, fmt="o-", capsize=5)
    plt.xscale("log")
    plt.xlabel("Rewiring probability p (log scale)")
    plt.ylabel("Average path length l(p)")
    plt.title("Average Path Length vs Rewiring Probability (n=100, c=6)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "ws_path_length_vs_p.png"))
    plt.show()
    plt.clf()


if __name__ == "__main__":
    n = 20
    c = 4
    p_values = [0, 0.2, 0.4]
    graphs = generate_ws_graphs(n, c, p_values, num_graphs=2)

    # (B) Calculate path length, diameter, and clustering for all graphs
    results = analyze_graphs(graphs)

    # Create a directory for figures
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)

    # (A) Plot graphs and save figures, including statistics in the figure
    for p, gs in graphs.items():
        for idx, G in enumerate(gs):
            title = f"Watts-Strogatz n={n}, c={c}, p={p} (Graph {idx+1})"
            filename = os.path.join(fig_dir, f"ws_n{n}_c{c}_p{p}_g{idx+1}.png")
            stats = results[p][idx]
            plot_graph(
                G,
                title,
                path_length=stats["path_length"],
                diameter=stats["diameter"],
                clustering=stats["clustering"],
                filename=filename,
            )

    # Print results to console as before
    for p in p_values:
        print(f"\nResults for p={p}:")
        for idx, res in enumerate(results[p]):
            print(
                f"  Graph {idx+1}: Path Length = {res['path_length']:.3f}, Diameter = {res['diameter']}"
            )

    print("\nClustering coefficients:")
    for p in p_values:
        for idx, res in enumerate(results[p]):
            print(
                f"  p={p}, Graph {idx+1}: Clustering Coefficient = {res['clustering']:.3f}"
            )

    # (D) Plot average path length vs p for n=100, c=6
    part_d_ws_path_length_vs_p()
