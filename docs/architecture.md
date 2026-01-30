# dotMD Architecture

## Pipeline Flowchart

```mermaid
flowchart TD
    subgraph Ingestion
        A[Markdown Files] --> B[Reader]
        B -->|FileInfo: path, title,\nchecksum, size| C[Chunker]
        C -->|Split by headings,\nmax 512 tokens,\n50 token overlap| D[Chunks]
    end

    subgraph Storage["Storage & Indexing"]
        D --> E[SQLite Metadata Store]
        D --> F[Sentence-Transformer\nEncoder]
        F -->|Dense vectors| G[LanceDB Vector Store]
        D --> H[BM25 Tokenizer]
        H -->|Sparse index| I[BM25 Pickle Index]
    end

    subgraph Extraction
        D --> J[Structural Extractor]
        J -->|Wikilinks, tags,\nfrontmatter, heading hierarchy| L[Entities + Relations]
        D --> K[GLiNER NER Extractor]
        K -->|Person, org, tech,\nconcept, location +\nCO_OCCURS / MENTIONS| L
    end

    subgraph GraphStorage["Graph Indexing"]
        L --> M[LadybugDB Graph Store]
        M -->|Nodes: File, Section,\nEntity, Tag| M
        M -->|Edges: LINKS_TO, HAS_TAG,\nPARENT_OF, CO_OCCURS,\nMENTIONS, HAS_FRONTMATTER| M
    end

    subgraph Query["Query Pipeline"]
        Q[User Query] --> QE[Query Expansion]
        QE -->|Acronym lookup +\nheading structure expansion| QX[Expanded Query]

        QX --> S1[Semantic Search\nLanceDB]
        QX --> S2[BM25 Search\nrank_bm25]

        S1 -->|seed chunk IDs| S3[Graph Search\nLadybugDB]
        S2 -->|seed chunk IDs| S3

        S1 -->|ranked list| RRF[Reciprocal Rank Fusion\nscore = Σ 1/k + rank]
        S2 -->|ranked list| RRF
        S3 -->|ranked list\nhop-1: weight/1²\nhop-2: weight/2²| RRF
    end

    subgraph Rerank["Reranking & Results"]
        RRF -->|top 100 candidates| RR[Cross-Encoder Reranker\nms-marco-MiniLM-L-6-v2]
        RR -->|score threshold > -8.0\n+ length penalty for\nshort chunks| SR[Search Results]
        SR -->|chunk_id, snippet,\nfused_score, heading_path,\nper-engine scores| OUT[Top-K Results]
    end

    G -.->|vector similarity| S1
    I -.->|term scores| S2
    M -.->|neighbor traversal\nmax 2 hops| S3
    E -.->|chunk text for\nreranking + snippets| RR
```

## Pipeline Stages

### 1. Ingestion

- **Reader** discovers `.md` files recursively, extracts title from first H1 or filename
- **Chunker** splits by ATX headings, tracks heading hierarchy, applies sentence-level sliding window for sections exceeding 512 tokens (50 token overlap)
- Chunk IDs are deterministic: `MD5(file_path:chunk_index)`

### 2. Storage & Indexing

| Store | Technology | Contents |
|-------|-----------|----------|
| SQLite | `~/.dotmd/metadata.db` | Chunk text, heading hierarchy, file path, offsets |
| LanceDB | `~/.dotmd/lancedb/` | Dense vectors from `all-MiniLM-L6-v2` |
| BM25 | `~/.dotmd/bm25_index.pkl` | Tokenized corpus for keyword scoring |

### 3. Extraction

- **Structural** (always on): wikilinks, tags, YAML frontmatter, markdown links, heading parent-child relations
- **NER** (optional, `--extract-depth ner`): GLiNER zero-shot extraction for person, organization, technology, concept, location entities plus co-occurrence and mention relations

### 4. Graph Indexing

LadybugDB stores a property graph with four node types (File, Section, Entity, Tag) and seven edge types (FILE_SECTION, SECTION_SECTION, SECTION_ENTITY, SECTION_TAG, ENTITY_ENTITY, FILE_TAG, FILE_ENTITY).

### 5. Query Pipeline

1. **Query Expansion** — acronym dictionary lookup (exact + fuzzy) and heading-structure expansion for domain terms
2. **Three parallel engines:**
   - Semantic search (dense vector similarity via LanceDB)
   - BM25 keyword search (sparse term matching)
   - Graph search (1-2 hop neighbor traversal seeded by semantic + BM25 hits)
3. **Reciprocal Rank Fusion** — `score = Σ 1/(k + rank)` with `k=60`, no learned weights needed

### 6. Reranking

- Cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores top 100 candidates
- Length penalty applied to chunks under 100 characters: `factor = 0.5 + 0.5 × (len/100)`
- Score threshold filter at `-8.0`
- Final top-K returned with per-engine scores, snippets, and heading paths