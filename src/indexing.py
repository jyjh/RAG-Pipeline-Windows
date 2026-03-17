import os
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import networkx as nx
import lancedb

from lightrag.operate import chunking_by_token_size
from lightrag.utils import compute_mdhash_id

from src.embeddings import EmbeddingEngine, embedding_mode
from src.utils import create_lightrag_instance, get_deepseek_tokenizer

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, working_dir="./db", model="deepseek-r1:32b"):
        self.working_dir = working_dir
        self.engine = EmbeddingEngine()
        self.rag = create_lightrag_instance(
            working_dir=working_dir, model=model, engine=self.engine
        )
        self._tokenizer = get_deepseek_tokenizer()

        # LanceDB connection for bridge edge ANN (separate from LightRAG's tables)
        lance_dir = os.path.join(working_dir, "lancedb")
        self._lance_db = lancedb.connect(lance_dir)

    # ------------------------------------------------------------------
    #  Custom chunking_func with LanceDB ANN deduplication
    # ------------------------------------------------------------------
    def _dedup_chunking_func(
        self,
        tokenizer,
        content: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        chunk_overlap_token_size: int = 200,
        chunk_token_size: int = 1200,
    ) -> list[dict[str, Any]]:
        """
        Custom chunking function injected into LightRAG via the chunking_func
        constructor parameter.

        1. Delegates to LightRAG's default chunking_by_token_size (using the
           DeepSeek Qwen2 tokenizer for accurate token counts).
        2. For each chunk, queries the existing LanceDB chunks_vdb table for
           near-duplicates (cosine > 0.90).
        3. Filters out near-duplicate chunks before NER dispatch.
        4. Embeds kept chunks into the EmbeddingEngine cache so LightRAG's
           subsequent embedding call is a cache hit (zero redundant compute).

        LightRAG's llm_response_cache handles exact-text duplicates automatically.
        This dedup targets *near-duplicate* chunks — e.g., two textbooks explaining
        the same theorem with different notation.
        """
        chunks = chunking_by_token_size(
            tokenizer,
            content,
            split_by_character,
            split_by_character_only,
            chunk_overlap_token_size,
            chunk_token_size,
        )

        if not chunks:
            return chunks

        # Embed chunks at 768d for dedup comparison (populates _cache)
        texts = [c["content"] for c in chunks]
        vecs = self.engine.get_mrl_embeddings(texts, truncate_dim=768)

        # Check for near-duplicates against existing chunks in LanceDB
        chunks_table_name = self._find_chunks_table()
        if chunks_table_name is not None:
            try:
                table = self._lance_db.open_table(chunks_table_name)
                keep = []
                for i, chunk in enumerate(chunks):
                    results = (
                        table.search(vecs[i].tolist())
                        .metric("cosine")
                        .limit(1)
                        .to_list()
                    )
                    if results:
                        # cosine distance: 0 = identical, 2 = opposite
                        similarity = 1.0 - results[0].get("_distance", 1.0)
                        if similarity > 0.90:
                            logger.debug(
                                f"Dedup: skipping near-duplicate chunk "
                                f"(sim={similarity:.3f})"
                            )
                            continue
                    keep.append(i)

                if len(keep) < len(chunks):
                    removed = len(chunks) - len(keep)
                    logger.info(
                        f"Dedup: removed {removed} near-duplicate chunks "
                        f"from document ({len(keep)} kept)"
                    )
                    chunks = [chunks[i] for i in keep]
                    # Re-index chunk_order_index on survivors
                    for idx, c in enumerate(chunks):
                        c["chunk_order_index"] = idx
            except Exception as e:
                logger.warning(f"Dedup check failed (non-fatal): {e}")

        return chunks

    def _find_chunks_table(self) -> str | None:
        """Find the LanceDB table name used by LightRAG for chunk vectors."""
        for name in self._lance_db.table_names():
            if "chunk" in name.lower():
                return name
        return None

    # ------------------------------------------------------------------
    #  Full-document insertion
    # ------------------------------------------------------------------
    async def _async_index_all(self, md_dir: str):
        """
        Inserts full markdown documents into LightRAG.

        Full-document insertion (not pre-chunked) preserves document-level
        hierarchy and long-range entity co-occurrence. LightRAG's internal
        chunking (via our custom chunking_func) handles splitting and dedup.
        """
        files = sorted(
            f for f in os.listdir(md_dir) if f.endswith(".md")
        )
        if not files:
            logger.warning(f"No markdown files found in {md_dir}")
            return

        embedding_mode.set("document")

        for filename in files:
            file_path = os.path.join(md_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                continue

            logger.info(f"Inserting full document: {filename}")
            if asyncio.iscoroutinefunction(self.rag.ainsert):
                await self.rag.ainsert(content, file_paths=[file_path])
            else:
                self.rag.insert(content, file_paths=[file_path])

    # ------------------------------------------------------------------
    #  Bridge edges — dual injection (.graphml + relationships_vdb)
    # ------------------------------------------------------------------
    async def _async_build_bridge_edges(self, threshold: float = 0.82):
        """
        Adds cross-domain edges between entity nodes that never co-occurred
        in a chunk. Uses LanceDB ANN (O(N log N)) instead of O(N²) pairwise.

        Bridge edges are injected into BOTH locations for LightRAG hybrid query:
        1. NetworkX .graphml — reachable via local mode graph traversal
        2. relationships_vdb — reachable via global/hybrid ANN search
        """
        logger.info("Building MRL bridge edges at 256d...")

        graphml_files = list(Path(self.working_dir).glob("*.graphml"))
        if not graphml_files:
            logger.warning("No graphml files found — skipping bridge edges")
            return

        G = nx.read_graphml(str(graphml_files[0]))
        entity_names = list(G.nodes())
        if len(entity_names) < 2:
            logger.warning("Fewer than 2 entities — skipping bridge edges")
            return

        # Embed entity names at 256d for bridge computation
        logger.info(f"Embedding {len(entity_names)} entity names at 256d")
        vecs_256 = self.engine.get_mrl_embeddings(
            entity_names, truncate_dim=256
        )

        # Store in a dedicated LanceDB table for ANN search
        bridge_table_name = "mrl_bridge_entities"
        bridge_data = [
            {"id": name, "vector": vecs_256[i].tolist()}
            for i, name in enumerate(entity_names)
        ]
        bridge_table = self._lance_db.create_table(
            bridge_table_name, bridge_data, mode="overwrite"
        )

        existing_edges = set(G.edges())
        bridge_edges = []

        # ANN radius search: O(N log N) instead of O(N²) pairwise
        for i, name in enumerate(entity_names):
            results = (
                bridge_table.search(vecs_256[i].tolist())
                .metric("cosine")
                .limit(20)
                .to_list()
            )
            for row in results:
                target = row["id"]
                if target == name:
                    continue
                similarity = 1.0 - row.get("_distance", 1.0)
                if similarity <= threshold:
                    continue
                if (name, target) in existing_edges or (target, name) in existing_edges:
                    continue
                # Avoid duplicate bridge edges (keep canonical order)
                edge_key = tuple(sorted([name, target]))
                if edge_key not in {tuple(sorted([e["source"], e["target"]])) for e in bridge_edges}:
                    bridge_edges.append({
                        "source": edge_key[0],
                        "target": edge_key[1],
                        "similarity_256d": float(similarity),
                    })

        if not bridge_edges:
            logger.info("No bridge edges found above threshold")
            return

        # --- Injection 1: .graphml (for local mode graph traversal) ---
        for edge in bridge_edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["similarity_256d"],
                description=(
                    f"Cross-domain semantic bridge: {edge['source']} ↔ "
                    f"{edge['target']} (MRL 256d similarity: "
                    f"{edge['similarity_256d']:.3f})"
                ),
                source_id="mrl_bridge",
                keywords=f"{edge['source']}, {edge['target']}",
            )
        nx.write_graphml(G, str(graphml_files[0]))
        logger.info(
            f"Injected {len(bridge_edges)} bridge edges into .graphml"
        )

        # --- Injection 2: relationships_vdb (for global/hybrid ANN search) ---
        vdb_data = {}
        for edge in bridge_edges:
            src, tgt = edge["source"], edge["target"]
            description = (
                f"Cross-domain semantic bridge: {src} ↔ {tgt} "
                f"(MRL 256d similarity: {edge['similarity_256d']:.3f})"
            )
            keywords = f"{src}, {tgt}"
            rel_id = compute_mdhash_id(src + tgt, prefix="rel-")
            vdb_data[rel_id] = {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": "mrl_bridge",
                "content": f"{keywords}\t{src}\n{tgt}\n{description}",
                "file_path": "",
            }

        await self.rag.relationships_vdb.upsert(vdb_data)
        await self.rag.relationships_vdb.index_done_callback()
        logger.info(
            f"Injected {len(bridge_edges)} bridge edges into relationships_vdb"
        )

        # Write JSON for inspection/debugging
        out_path = os.path.join(self.working_dir, "mrl_bridge_edges.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bridge_edges, f, indent=2)
        logger.info(f"Bridge edge manifest: {out_path}")

    # ------------------------------------------------------------------
    #  Quotient graph
    # ------------------------------------------------------------------
    def build_quotient_graph(self, n_clusters: int = None):
        """
        Constructs a coarse-grained quotient graph over entity clusters.

        Written to db/quotient_graph.json — intentionally kept separate from
        the primary .graphml to avoid namespace pollution. Requires a custom
        two-stage query pre-processor for retrieval (future work).
        """
        from sklearn.cluster import KMeans

        logger.info("Building quotient graph at 256d...")

        graphml_files = list(Path(self.working_dir).glob("*.graphml"))
        if not graphml_files:
            logger.warning("No graphml files found — skipping quotient graph")
            return

        G = nx.read_graphml(str(graphml_files[0]))
        entity_names = list(G.nodes())
        if len(entity_names) < 2:
            logger.warning("Fewer than 2 entities — skipping quotient graph")
            return

        # Load bridge edges into working graph copy
        G_enriched = G.copy()
        bridge_path = os.path.join(self.working_dir, "mrl_bridge_edges.json")
        if os.path.exists(bridge_path):
            with open(bridge_path, "r", encoding="utf-8") as f:
                bridge_edges = json.load(f)
            for edge in bridge_edges:
                G_enriched.add_edge(
                    edge["source"], edge["target"],
                    weight=edge["similarity_256d"],
                    edge_type="mrl_bridge",
                )
            logger.info(f"Loaded {len(bridge_edges)} bridge edges into enriched graph")

        logger.info(f"Embedding {len(entity_names)} entity names at 256d")
        vecs_256 = self.engine.get_mrl_embeddings(entity_names, truncate_dim=256)

        if n_clusters is None:
            n_clusters = max(4, min(int(np.sqrt(len(entity_names))), 100))

        labels = KMeans(
            n_clusters=n_clusters, random_state=42, n_init="auto"
        ).fit_predict(vecs_256)

        entity_to_cluster = {
            name: int(labels[i]) for i, name in enumerate(entity_names)
        }

        quotient_edges: dict[tuple[int, int], int] = {}
        for u, v in G_enriched.edges():
            cu, cv = entity_to_cluster.get(u), entity_to_cluster.get(v)
            if cu is not None and cv is not None and cu != cv:
                key = (min(cu, cv), max(cu, cv))
                quotient_edges[key] = quotient_edges.get(key, 0) + 1

        cluster_members: dict[int, list[str]] = {}
        for name, cid in entity_to_cluster.items():
            cluster_members.setdefault(cid, []).append(name)

        output = {
            "n_clusters": n_clusters,
            "entity_to_cluster": entity_to_cluster,
            "clusters": {
                str(cid): {
                    "representative": members[0],
                    "size": len(members),
                    "members": members,
                }
                for cid, members in cluster_members.items()
            },
            "quotient_edges": [
                {"cluster_a": k[0], "cluster_b": k[1], "edge_count": v}
                for k, v in quotient_edges.items()
            ],
        }

        out_path = os.path.join(self.working_dir, "quotient_graph.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        logger.info(
            f"Quotient graph: {n_clusters} clusters, "
            f"{len(quotient_edges)} inter-cluster edges → {out_path}"
        )

    # ------------------------------------------------------------------
    #  Top-level pipeline
    # ------------------------------------------------------------------
    def index_markdown(self, markdown_dir: str):
        """
        Full pipeline: insert documents → bridge edges → quotient graph.

        Documents are inserted as full text (not pre-chunked). LightRAG's
        chunking_func handles splitting using the DeepSeek tokenizer, with
        LanceDB ANN dedup filtering near-duplicate chunks before NER.
        """
        logger.info(f"Indexing markdown files from {markdown_dir}")

        # Inject our custom chunking_func with dedup
        self.rag.chunking_func = self._dedup_chunking_func

        # Step 1–3: Insert full documents (chunking + NER + embedding)
        asyncio.run(self._async_index_all(markdown_dir))
        logger.info("LightRAG indexing complete")

        # Step 4: Bridge edges (dual injection)
        asyncio.run(self._async_build_bridge_edges(threshold=0.82))

        # Step 5: Quotient graph
        self.build_quotient_graph()


def run_indexing(md_dir: str, db_dir: str):
    indexer = Indexer(working_dir=db_dir)
    indexer.index_markdown(md_dir)


if __name__ == "__main__":
    run_indexing("processed_docs", "db")
