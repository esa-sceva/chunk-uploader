# Metadata Fields Added to Qdrant

When chunks are uploaded to Qdrant, the following metadata fields are automatically added:

## Core Fields

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `id` | int | Unique incremental ID | Auto-generated counter |
| `content` | str | Full text content of the chunk | From chunk file |
| `chunk_name` | str | Name of the chunk file | From metadata |
| `original_file_name` | str | Original document filename | From metadata |
| `sub_folder` | str | Subfolder/category | From metadata |

## URL and Source

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `url` | str | Source URL (renamed from source_url) | From metadata |
| `publisher` | str | Publisher name | From source_json_file |
| `source_json_file` | str | Original JSON metadata file | From metadata |

## Scholarly Metadata (Placeholders)

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `doi` | None | Digital Object Identifier | null |
| `title` | None | Document title | null |
| `journal` | None | Journal name | null |
| `reference_count` | int | Number of references | 0 |
| `n_citations` | int | Citation count | 0 |
| `influential_citation_count` | int | Influential citations | 0 |
| `header` | list | Document headers | [] |

## Quality Metrics

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `score` | float | Quality/relevance score | From metadata |

## Example Payload

```json
{
  "id": 12345,
  "content": "Full text content of the chunk...",
  "chunk_name": "chunk_001.md",
  "original_file_name": "document.pdf",
  "sub_folder": "arxiv",
  "url": "https://arxiv.org/abs/1234.5678",
  "publisher": "arxiv",
  "source_json_file": "arxiv.json",
  "score": 0.95,
  "doi": null,
  "title": null,
  "journal": null,
  "reference_count": 0,
  "n_citations": 0,
  "influential_citation_count": 0,
  "header": []
}
```

## Usage

These fields are automatically added by the uploader. You don't need to do anything special - just run:

```bash
export QDRANT_URL="..."
export QDRANT_API_KEY="..."
export S3_CHUNKS_PATH="s3://bucket/chunks/"
python main.py
```

The metadata enrichment happens automatically before uploading to Qdrant.

## Updating Scholarly Metadata

If you want to populate the scholarly fields (doi, title, journal, etc.) later, you can update them using a separate script that:

1. Queries external APIs (e.g., Crossref, Semantic Scholar)
2. Updates the Qdrant points with the retrieved metadata

This allows you to upload chunks quickly first, then enrich with external data later.

