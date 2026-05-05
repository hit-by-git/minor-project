from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D


INPUT_CSV = Path("gpt-120-enriched.csv")
OUTPUT_DIR = Path("results")
OUTPUT_PNG = OUTPUT_DIR / "top_50_coauthorship_network.png"
OUTPUT_METRICS = OUTPUT_DIR / "top_50_author_metrics.csv"


def split_authors(raw_authors: object) -> list[str]:
    """Parse the mixed author formats present in the enriched CSV."""
    if pd.isna(raw_authors):
        return []

    text = str(raw_authors).strip()
    if not text:
        return []

    if ":::" in text:
        parts = text.split(":::")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = re.split(r"\s+\band\b\s+", text)

    authors = []
    for part in parts:
        author = re.sub(r"\s+", " ", part).strip(" .,\t\n\r")
        if author:
            authors.append(author)
    return authors


def build_coauthorship_graph(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    publication_counts: Counter[str] = Counter()

    for _, row in df.iterrows():
        authors = list(dict.fromkeys(split_authors(row.get("author"))))
        if not authors:
            continue

        title = str(row.get("title", "")).strip()
        publication_id = row.get("doi") or row.get("id") or title

        for author in authors:
            publication_counts[author] += 1
            graph.add_node(author)

        for source, target in itertools.combinations(authors, 2):
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1
                graph[source][target]["papers"].append(publication_id)
            else:
                graph.add_edge(source, target, weight=1, papers=[publication_id])

    nx.set_node_attributes(graph, dict(publication_counts), "publications")
    return graph


def top_author_subgraph(graph: nx.Graph, n: int = 50) -> nx.Graph:
    ranking = sorted(
        graph.nodes,
        key=lambda node: (
            graph.nodes[node].get("publications", 0),
            graph.degree(node, weight="weight"),
            graph.degree(node),
            node,
        ),
        reverse=True,
    )
    return graph.subgraph(ranking[:n]).copy()


def assign_louvain_communities(graph: nx.Graph) -> dict[str, int]:
    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes]
    else:
        communities = nx.community.louvain_communities(
            graph, weight="weight", seed=42, resolution=1.0
        )

    community_by_author = {}
    for community_id, members in enumerate(communities, start=1):
        for author in members:
            community_by_author[author] = community_id
    nx.set_node_attributes(graph, community_by_author, "community_id")
    return community_by_author


def spread_dense_communities(
    graph: nx.Graph, pos: dict[str, tuple[float, float]], community_by_author: dict[str, int]
) -> dict[str, tuple[float, float]]:
    grouped_nodes: dict[int, list[str]] = defaultdict(list)
    for node, community_id in community_by_author.items():
        grouped_nodes[community_id].append(node)

    x_values = [coords[0] for coords in pos.values()]
    y_values = [coords[1] for coords in pos.values()]
    span = max(max(x_values) - min(x_values), max(y_values) - min(y_values), 1.0)
    adjusted = dict(pos)

    for community_id, nodes in grouped_nodes.items():
        if len(nodes) < 4:
            continue

        density = nx.density(graph.subgraph(nodes))
        if density < 0.35:
            continue

        center_x = sum(pos[node][0] for node in nodes) / len(nodes)
        center_y = sum(pos[node][1] for node in nodes) / len(nodes)
        radius = span * (0.065 + 0.018 * math.sqrt(len(nodes)))

        nodes_by_angle = sorted(
            nodes,
            key=lambda node: (
                math.atan2(pos[node][1] - center_y, pos[node][0] - center_x),
                node,
            ),
        )
        for index, node in enumerate(nodes_by_angle):
            angle = 2 * math.pi * index / len(nodes_by_angle)
            adjusted[node] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )

    return adjusted


def draw_network(graph: nx.Graph, community_by_author: dict[str, int]) -> None:
    degree_centrality = nx.degree_centrality(graph)
    weighted_degree = dict(graph.degree(weight="weight"))
    nx.set_node_attributes(graph, degree_centrality, "degree_centrality")
    nx.set_node_attributes(graph, weighted_degree, "degree")

    communities = sorted(set(community_by_author.values()))
    cmap = plt.get_cmap("tab20", max(len(communities), 1))
    color_for_community = {
        community_id: cmap(index % cmap.N)
        for index, community_id in enumerate(communities)
    }

    node_colors = [
        color_for_community[community_by_author[node]] for node in graph.nodes
    ]
    centrality_values = [degree_centrality[node] for node in graph.nodes]
    min_cent = min(centrality_values) if centrality_values else 0
    max_cent = max(centrality_values) if centrality_values else 1
    node_sizes = [
        220 + 1150 * ((degree_centrality[node] - min_cent) / (max_cent - min_cent or 1))
        for node in graph.nodes
    ]
    edge_widths = [
        0.45 + 1.1 * math.log1p(data.get("weight", 1))
        for _, _, data in graph.edges(data=True)
    ]

    pos = nx.spring_layout(graph, k=0.5, iterations=250, seed=42, weight="weight")
    pos = spread_dense_communities(graph, pos, community_by_author)

    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color="#9aa0a6",
        alpha=0.34,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=1.1,
        edgecolors="white",
        alpha=0.95,
    )

    labels = {}
    for author in graph.nodes:
        if len(author) > 28:
            labels[author] = author[:25] + "..."
        else:
            labels[author] = author

    for author, (x_pos, y_pos) in pos.items():
        ax.annotate(
            labels[author],
            xy=(x_pos, y_pos),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.3,
            color="#202124",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_for_community[community_id],
            markeredgecolor="white",
            markersize=9,
            label=f"Community {community_id}",
        )
        for community_id in communities
    ]
    ax.legend(
        handles=legend_handles,
        title="Louvain Communities",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    ax.set_title(
        "Top 50 Most Prolific Authors Co-authorship Network",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metrics(graph: nx.Graph) -> None:
    rows = []
    for author, attrs in graph.nodes(data=True):
        rows.append(
            {
                "author": author,
                "community_id": attrs.get("community_id"),
                "degree": attrs.get("degree"),
                "degree_centrality": attrs.get("degree_centrality"),
                "publications": attrs.get("publications"),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["degree", "degree_centrality", "publications", "author"],
        ascending=[False, False, False, True],
    )
    metrics.to_csv(OUTPUT_METRICS, index=False)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    graph = build_coauthorship_graph(df)
    top_graph = top_author_subgraph(graph, n=50)
    community_by_author = assign_louvain_communities(top_graph)
    draw_network(top_graph, community_by_author)
    write_metrics(top_graph)

    print(f"Loaded {len(df):,} records from {INPUT_CSV}")
    print(
        f"Full graph: {graph.number_of_nodes():,} authors, "
        f"{graph.number_of_edges():,} co-authorship edges"
    )
    print(
        f"Top-50 graph: {top_graph.number_of_nodes():,} authors, "
        f"{top_graph.number_of_edges():,} edges, "
        f"{len(set(community_by_author.values())):,} Louvain communities"
    )
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()
