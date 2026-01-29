#!/usr/bin/env python3
"""Export LadybugDB graph to interactive HTML visualization."""

from pathlib import Path
import real_ladybug as lb

# Option 1: NetworkX + PyVis (interactive HTML)
try:
    import networkx as nx
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("Install pyvis for interactive visualization: pip install pyvis networkx")

# Option 2: Matplotlib (static)
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("Install matplotlib for static visualization: pip install matplotlib")


def export_to_pyvis(db_path: Path, output_file: str = "graph.html"):
    """Export graph to interactive HTML using PyVis."""
    if not PYVIS_AVAILABLE:
        print("PyVis not available. Install with: pip install pyvis networkx")
        return

    conn = lb.Database(str(db_path))

    # Create network graph
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # Query all nodes
    for label in ("File", "Section", "Entity", "Tag"):
        result = conn.execute(f"MATCH (n:{label}) RETURN n.id, n")
        df = result.get_as_df()

        color_map = {
            "File": "#ff6b6b",
            "Section": "#4ecdc4",
            "Entity": "#45b7d1",
            "Tag": "#96ceb4"
        }

        for _, row in df.iterrows():
            node_id = str(row.iloc[0])
            net.add_node(
                node_id,
                label=node_id[:30],  # Truncate long IDs
                title=f"{label}: {node_id}",  # Tooltip
                color=color_map.get(label, "#gray"),
                size=15 if label == "File" else 10
            )

    # Query all relationships
    rel_tables = [
        "FILE_SECTION", "SECTION_SECTION", "SECTION_ENTITY",
        "SECTION_TAG", "ENTITY_ENTITY", "FILE_TAG", "FILE_ENTITY"
    ]

    for rel_table in rel_tables:
        try:
            result = conn.execute(
                f"MATCH (a)-[r:{rel_table}]->(b) RETURN a.id, b.id, r.rel_type, r.weight"
            )
            df = result.get_as_df()

            for _, row in df.iterrows():
                source = str(row.iloc[0])
                target = str(row.iloc[1])
                rel_type = str(row.iloc[2]) if len(row) > 2 else rel_table
                weight = float(row.iloc[3]) if len(row) > 3 else 1.0

                net.add_edge(
                    source,
                    target,
                    title=rel_type,
                    label=rel_type[:15],
                    width=max(1, weight * 2)
                )
        except Exception as e:
            print(f"Error querying {rel_table}: {e}")

    # Save interactive HTML
    net.show(output_file)
    print(f"Interactive graph saved to: {output_file}")
    print(f"Open in browser: file://{Path(output_file).absolute()}")


def export_stats(db_path: Path):
    """Print graph statistics."""
    conn = lb.Database(str(db_path))

    print("\nGraph Statistics\n" + "="*50)

    # Node counts
    for label in ("File", "Section", "Entity", "Tag"):
        try:
            result = conn.execute(f"MATCH (n:{label}) RETURN count(n)")
            df = result.get_as_df()
            count = int(df.iloc[0, 0])
            print(f"  {label:12} nodes: {count:5}")
        except Exception:
            print(f"  {label:12} nodes: 0")

    print()

    # Edge counts
    rel_tables = [
        "FILE_SECTION", "SECTION_SECTION", "SECTION_ENTITY",
        "SECTION_TAG", "ENTITY_ENTITY", "FILE_TAG", "FILE_ENTITY"
    ]

    total_edges = 0
    for rel_table in rel_tables:
        try:
            result = conn.execute(f"MATCH ()-[r:{rel_table}]->() RETURN count(r)")
            df = result.get_as_df()
            count = int(df.iloc[0, 0])
            if count > 0:
                print(f"  {rel_table:20} edges: {count:5}")
                total_edges += count
        except Exception:
            pass

    print(f"\n  {'Total edges':20}: {total_edges:5}")
    print("="*50)


if __name__ == "__main__":
    db_path = Path.home() / ".dotmd" / "graphdb"

    if not db_path.exists():
        print(f"Database not found at: {db_path}")
        print("Run 'dotmd index ../data' first")
        exit(1)

    # Show stats
    export_stats(db_path)

    # Export to interactive visualization
    print("\nGenerating visualization...")
    export_to_pyvis(db_path, "dotmd_graph.html")
