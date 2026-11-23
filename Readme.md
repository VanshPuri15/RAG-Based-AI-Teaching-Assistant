# RAG-Based AI Teaching Assistant

An AI-powered teaching assistant built using Retrieval-Augmented Generation (RAG) to help students interact with educational content and retrieve relevant information.

## Project Overview

This project aims to build an intelligent teaching assistant that processes educational material, converts it into searchable chunks, and generates embeddings to support semantic retrieval.

## Current Progress

- Implemented text processing and chunk handling.
- Prepared structured JSON files containing text chunks.
- Integrated Ollama for local embedding generation.
- Using the `bge-m3` embedding model.
- Implemented batch-wise embedding generation through the Ollama API.
- Storing generated embeddings alongside their corresponding chunks using Pandas.

## Technology Stack

- **Language:** Python
- **Embedding Model:** BGE-M3
- **Local Model Runtime:** Ollama
- **Libraries:** Requests, Pandas, JSON, OS
- **Approach:** Retrieval-Augmented Generation (RAG)

## Project Status

Currently under development. Further work includes building the vector retrieval pipeline, integrating an LLM for response generation, and developing the user interface.

## Future Improvements

- Implement vector similarity search.
- Integrate LLM-based answer generation.
- Add source-aware responses.
- Develop an interactive interface.
