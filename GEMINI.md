# GEMINI.md

## Project Overview

This project aims to build a **Local RAG (Retrieval-Augmented Generation) Pipeline** designed to consolidate information from textbooks, PDF documents, and research papers. The primary goal is to create a high-quality learning aid that reduces search and retrieval time, allowing for more efficient analysis of complex information.

### Key Technologies & Constraints
- **Local-Only**: Must not use cloud-based models or external APIs. All processing (embeddings, inference) must be done locally.
- **Target Hardware**: Optimized for Ryzen 9 9950X3D, RTX 4000 ada 20GB VRAM, 128GB RAM, ROG Strix X870E-E motherboard.
- **Format**: Supports querying textbooks, PDFs, and research papers, primarily in STEM fields (mathematics, physics, engineering, statistics, computer science).

## Directory Overview

This directory currently holds the foundational documentation for the project. It is in the late planning phase. Next steps is to iterate and optimise the pipeline.

### Key Files
- `Project Requirements.md`: Defines the goals, hardware specifications, and core constraints (local-first).
- `Plan Actionables.md`: Outlines the project overall plan, roadmap, technical workflow, and milestones (currently in progress).

## Development Guidelines

- **Privacy & Locality**: Always prioritize local models (e.g., Ollama, llama.cpp) over cloud services.
- **Hardware Optimization**: Leverage Ryzen 9 CPU and RTX 4000 ada GPU capabilities.
- **Documentation**: All usage and replication steps must be clearly documented in `.md` files as part of the project requirements.

## Usage

As the project is in the planning stage, refer to `Project Requirements.md` for the core vision and `Plan Actionables.md` for current tasks. Future development will involve setting up a local vector database and an LLM orchestration layer.

## Future Works

Consider GUNDAM (Graph Understanding for Natural Language Driven Analytical Model) to interpret knowledge graphs for LLMs (this does not have to be considered currently, ignore this for any current works)
