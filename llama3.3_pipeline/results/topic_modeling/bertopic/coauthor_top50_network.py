#!/usr/bin/env python3
"""Create a top-50 coauthorship network visualization from a CSV file.

The script supports two common inputs:
1. A node/author metrics CSV with columns such as author, community_id, degree.
2. The project paper CSV, where each row is a paper and column 3 contains authors.

For the project CSV, coauthorship edges are derived by pairing authors who appear on
the same paper. Communities are then detected from the resulting graph.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("result/coauthorship/.mplconfig")))

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


INPUT_CSV = Path("llama-prescreening-enriched.csv")
OUTPUT_DIR = Path("result/coauthorship")
OUTPUT_PNG = OUTPUT_DIR / "top_50_authors_coauthorship_network.png"
OUTPUT_NODES = OUTPUT_DIR / "top_50_authors_nodes.csv"
OUTPUT_EDGES = OUTPUT_DIR / "top_50_authors_edges.csv"

AUTHOR_COLUMNS = {"author", "authors", "name", "node", "source"}
METRIC_COLUMNS = {"degree", "degree_centrality", "betweenness_centrality"}

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def clean_author_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip().strip("\"'")).strip("[] ")
    if "," in name:
        parts = [part.strip() for part in name.split(",")]
        if len(parts) == 2 and all(parts):
            name = f"{parts[1]} {parts[0]}"
    return name


def parse_authors(author_field: str) -> list[str]:
    value = author_field.strip()
    if not value:
        return []

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        raw_authors = [str(item) for item in parsed] if isinstance(parsed, list) else [value]
    elif ":::" in value:
        raw_authors = value.split(":::")
    elif ";" in value:
        raw_authors = value.split(";")
    elif " and " in value:
        raw_authors = value.split(" and ")
    else:
        raw_authors = [value]

    authors: list[str] = []
    seen: set[str] = set()
    for raw_author in raw_authors:
        author = clean_author_name(raw_author)
        key = author.casefold()
        if author and key not in seen:
            authors.append(author)
            seen.add(key)
    return authors


def read_csv_rows(path: Path) -> tuple[list[str] | None, list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row]

    first = [cell.strip().casefold() for cell in rows[0]]
    looks_headered = bool(set(first) & (AUTHOR_COLUMNS | METRIC_COLUMNS | {"community_id"}))
    return (rows[0] if looks_headered else None), (rows[1:] if looks_headered else rows)


def build_graph_from_papers(rows: list[list[str]]) -> nx.Graph:
    graph = nx.Graph()
    for row in rows:
        if len(row) < 4:
            continue
        authors = parse_authors(row[3])
        for author in authors:
            graph.add_node(author)
        for author_1, author_2 in itertools.combinations(sorted(authors), 2):
            if graph.has_edge(author_1, author_2):
                graph[author_1][author_2]["weight"] += 1
                graph[author_1][author_2]["papers"] += 1
            else:
                graph.add_edge(author_1, author_2, weight=1, papers=1)
    return graph


def build_graph_from_author_metrics(header: list[str], rows: list[list[str]]) -> nx.Graph:
    columns = {name.strip().casefold(): index for index, name in enumerate(header)}
    author_column = next((columns[name] for name in AUTHOR_COLUMNS if name in columns), None)
    if author_column is None:
        raise ValueError("Could not find an author column in the headered CSV.")

    graph = nx.Graph()
    for row in rows:
        if author_column >= len(row):
            continue
        author = clean_author_name(row[author_column])
        if not author:
            continue
        graph.add_node(author)
        for metric in METRIC_COLUMNS | {"community_id"}:
            if metric in columns and columns[metric] < len(row):
                graph.nodes[author][metric] = row[columns[metric]]
    return graph


def keep_top_authors(graph: nx.Graph, limit: int) -> nx.Graph:
    weighted_degree = dict(graph.degree(weight="weight"))
    top_authors = sorted(
        graph.nodes,
        key=lambda author: (weighted_degree.get(author, 0), graph.degree(author), author.casefold()),
        reverse=True,
    )[:limit]
    return graph.subgraph(top_authors).copy()


def annotate_graph(graph: nx.Graph) -> None:
    weighted_degree = dict(graph.degree(weight="weight"))
    degree = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)

    nx.set_node_attributes(graph, weighted_degree, "weighted_degree")
    nx.set_node_attributes(graph, degree, "degree")
    nx.set_node_attributes(graph, degree_centrality, "degree_centrality")
    nx.set_node_attributes(graph, betweenness, "betweenness_centrality")

    if graph.number_of_edges() > 0:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    else:
        communities = [{node} for node in graph.nodes]

    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            graph.nodes[node]["community_id"] = community_id


def write_graph_tables(graph: nx.Graph) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_NODES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "author",
                "community_id",
                "degree",
                "weighted_degree",
                "degree_centrality",
                "betweenness_centrality",
            ]
        )
        for author, data in sorted(
            graph.nodes(data=True),
            key=lambda item: (-item[1]["weighted_degree"], item[0].casefold()),
        ):
            writer.writerow(
                [
                    author,
                    data["community_id"],
                    data["degree"],
                    data["weighted_degree"],
                    f"{data['degree_centrality']:.6f}",
                    f"{data['betweenness_centrality']:.6f}",
                ]
            )

    with OUTPUT_EDGES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["author_1", "author_2", "shared_paper_count"])
        for author_1, author_2, data in sorted(
            graph.edges(data=True),
            key=lambda item: (-item[2].get("weight", 1), item[0].casefold(), item[1].casefold()),
        ):
            writer.writerow([author_1, author_2, data.get("weight", 1)])


def draw_graph(graph: nx.Graph, output_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(graph, k=0.5, seed=42, weight="weight", iterations=300)
    communities = sorted({data["community_id"] for _, data in graph.nodes(data=True)})
    node_colors = [
        COLORS[graph.nodes[node]["community_id"] % len(COLORS)]
        for node in graph.nodes
    ]
    node_sizes = [
        180 + 55 * graph.nodes[node]["weighted_degree"]
        for node in graph.nodes
    ]
    edge_widths = [
        0.8 + 0.55 * graph[author_1][author_2].get("weight", 1)
        for author_1, author_2 in graph.edges
    ]

    fig, ax = plt.subplots(figsize=(18, 13), facecolor="white")
    ax.set_facecolor("white")

    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color="#9aa6b2",
        alpha=0.42,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=1.6,
        alpha=0.96,
        ax=ax,
    )

    label_pos = {node: (x, y + 0.025) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        graph,
        label_pos,
        labels={node: node for node in graph.nodes},
        font_size=7,
        font_family="DejaVu Sans",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.72},
        ax=ax,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Community {community_id}",
            markerfacecolor=COLORS[community_id % len(COLORS)],
            markeredgecolor="white",
            markersize=10,
        )
        for community_id in communities
    ]
    ax.legend(
        handles=legend_handles,
        title="Louvain Communities",
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d8dee6",
    )

    ax.set_title(
        "Top 50 Authors Co-authorship Network",
        fontsize=22,
        color="#20272e",
        pad=24,
    )
    ax.text(
        0.01,
        0.01,
        "Node size = weighted degree; edge width = shared paper count; layout = spring_layout(k=0.5)",
        transform=ax.transAxes,
        fontsize=10,
        color="#5f6b73",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    header, rows = read_csv_rows(args.input)
    if header:
        graph = build_graph_from_author_metrics(header, rows)
        if graph.number_of_edges() == 0:
            raise ValueError(
                "The CSV appears to contain author metrics but no edges. "
                "Provide a paper-level CSV or add an edge list to build coauthorship links."
            )
    else:
        graph = build_graph_from_papers(rows)

    top_graph = keep_top_authors(graph, args.top)
    annotate_graph(top_graph)
    write_graph_tables(top_graph)
    draw_graph(top_graph, OUTPUT_PNG)

    print(f"Input graph authors: {graph.number_of_nodes()}")
    print(f"Input graph coauthorship edges: {graph.number_of_edges()}")
    print(f"Visualized top authors: {top_graph.number_of_nodes()}")
    print(f"Visualized edges among top authors: {top_graph.number_of_edges()}")
    print(f"Wrote: {OUTPUT_PNG}")
    print(f"Wrote: {OUTPUT_NODES}")
    print(f"Wrote: {OUTPUT_EDGES}")


if __name__ == "__main__":
    main()
