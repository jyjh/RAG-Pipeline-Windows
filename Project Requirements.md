# Project Requirements

### Purpose of this Documentation

This document defines the project goal and requirements for a local RAG (Retrieval-Augmented Generation) pipeline. It serves as a planning document to ensure the final implementation is clear and replicable.

## Goal

Create a local RAG pipeline to consolidate similar and related information and theory from multiple STEM sources (mathematics, physics, engineering, statistics, computer science). The tool acts as a learning aid to reduce search and retrieval time, allowing users to focus mental capacity on analysis while maintaining high information quality.

### Requirements

- **Query Capability**: Able to query accurate information from textbooks, PDF documents, research papers, and potentially other document formats.
- **Documentation**: Clear, detailed, and comprehensive developer documentation on how to set up and utilize the RAG pipeline, including technical explanations for underlying mechanisms at each step.
- **Hardware Compatibility**: Optimized for Ryzen 9 9950X3D, RTX 4000 ada 20GB VRAM, 128GB RAM, ROG Strix X870E-E motherboard.
- **Locality Constraint**: Do not use cloud models or APIs; keep the entire RAG pipeline local.
- **Domain Focus**: Primarily STEM fields (mathematics, physics, engineering, statistics, computer science).
- **Input Formats**: Primarily PDFs, with support for other document formats as needed.
- **Output Formats**: Structured knowledge base components (e.g., knowledge graphs, consolidated summaries, query responses with citations) to support learning and analysis.

### Technical Stack (Local Models)

- **Document Ingestion**: IBM Docling for layout analysis, table reconstruction, and LaTeX extraction; Qwen2.5-VL-7B (adapted for Windows) as fallback for image/chart processing.
- **Knowledge Graph**: LightRAG for incremental indexing and dual-level retrieval.
- **Inference Models**: DeepSeek-R1-Distill-Qwen-32B for indexing, reasoning, and generation.
- **Vector Database**: LanceDB for local storage.
- **Orchestration**: Python-based orchestrator for model loading/unloading to manage RAM.

### Remarks

Local models must be selected to run efficiently on the specified hardware. If knowledge base quality is insufficient, hardware upgrades may be considered, but cloud alternatives are not permitted.