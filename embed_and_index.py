"""
Standalone script for embedding and indexing TED talks to Pinecone.
Builds chunks using specified schemes and uploads to vector database.
"""

import os
import re
import unicodedata
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Iterable

import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# =============================
# Configuration
# =============================

EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

MAX_CHUNK_SIZE = 2048
MAX_OVERLAP_RATIO = 0.3

REQUIRED_COLS = [
    "talk_id", "title", "transcript", "speaker_1", "all_speakers", "topics", "description"
]


# =============================
# Data Models
# =============================

@dataclass(frozen=True)
class ChunkScheme:
    """Configuration for text chunking strategy."""
    scheme_id: str
    chunk_tokens: int
    overlap_ratio: float
    
    def __post_init__(self):
        """Validate hyperparameters against assignment constraints."""
        if not self.scheme_id:
            raise ValueError("scheme_id cannot be empty")
        if self.chunk_tokens <= 0:
            raise ValueError(f"chunk_tokens must be positive, got {self.chunk_tokens}")
        if self.chunk_tokens > MAX_CHUNK_SIZE:
            raise ValueError(f"chunk_tokens {self.chunk_tokens} exceeds max {MAX_CHUNK_SIZE}")
        if not 0 <= self.overlap_ratio <= MAX_OVERLAP_RATIO:
            raise ValueError(f"overlap_ratio must be between 0 and {MAX_OVERLAP_RATIO}, got {self.overlap_ratio}")


@dataclass(frozen=True)
class ChunkRecord:
    """A single text chunk with associated metadata for vector storage."""
    chunk_id: str
    text: str
    metadata: Dict


# =============================
# Text Processing
# =============================

_WS = re.compile(r"\s+")

def strip_weird_unicode(text: str) -> str:
    """Remove control, format, and non-printable Unicode characters."""
    if not text:
        return ""
    
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in {"Cc", "Cf", "Cs", "Co"}:
            continue
        if ch == "\uFFFD":
            continue
        cleaned.append(ch)
    
    return "".join(cleaned)


def preprocess_text(text: str) -> str:
    """Clean text before chunking - removes TED artifacts and normalizes."""
    if not text:
        return ""
    
    text = unicodedata.normalize("NFC", text)
    text = strip_weird_unicode(text)
    
    # Remove TED-specific artifacts
    text = re.sub(r'\([A-Z][a-z]+\)', '', text)
    text = re.sub(r'\([A-Z][a-z]+ [a-z]+\)', '', text)
    text = re.sub(r'\[\d{1,2}:\d{2}(:\d{2})?\]', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'([.!?]){3,}', r'\1\1', text)
    text = re.sub(r'\b\d+\b(?!\s*[a-zA-Z])', '', text)
    
    text = _WS.sub(" ", text).strip()
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    
    return text


def chunk_text(text: str, max_tokens: int, overlap_ratio: float) -> List[str]:
    """Split text into overlapping chunks respecting sentence boundaries."""
    text = preprocess_text(text)
    
    if not text:
        return []
    
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    
    sentence_tokens = []
    for sent in sentences:
        words = sent.split(" ")
        tokens = sum(max(1, len(w) // 4) for w in words)
        sentence_tokens.append(tokens)
    
    chunks: List[str] = []
    overlap_tokens = int(max_tokens * overlap_ratio)
    
    n = len(sentences)
    start_idx = 0
    
    while start_idx < n:
        current_tokens = 0
        end_idx = start_idx
        chunk_sentences = []
        
        while end_idx < n and current_tokens + sentence_tokens[end_idx] <= max_tokens:
            chunk_sentences.append(sentences[end_idx])
            current_tokens += sentence_tokens[end_idx]
            end_idx += 1
        
        if end_idx == start_idx:
            # Fallback to word-level chunking for oversized sentences
            words = sentences[start_idx].split(" ")
            word_tokens = [max(1, len(w) // 4) for w in words]
            
            word_end = 0
            word_tokens_sum = 0
            while word_end < len(words) and word_tokens_sum + word_tokens[word_end] <= max_tokens:
                word_tokens_sum += word_tokens[word_end]
                word_end += 1
            
            if word_end == 0:
                word_end = 1
            
            chunk = " ".join(words[:word_end]).strip()
            if chunk:
                chunks.append(chunk)
            
            back_words = 0
            back_tokens = 0
            i = word_end - 1
            while i >= 0 and back_tokens < overlap_tokens:
                back_tokens += word_tokens[i]
                back_words += 1
                i -= 1
            
            sentences[start_idx] = " ".join(words[max(1, word_end - back_words):])
            sentence_tokens[start_idx] = sum(word_tokens[max(1, word_end - back_words):])
            
            if not sentences[start_idx].strip():
                start_idx += 1
            continue
        
        chunk = " ".join(chunk_sentences).strip()
        if chunk:
            chunks.append(chunk)
        
        if end_idx >= n:
            break
        
        back_idx = end_idx - 1
        back_tokens = 0
        while back_idx >= start_idx and back_tokens < overlap_tokens:
            back_tokens += sentence_tokens[back_idx]
            back_idx -= 1
        
        start_idx = max(start_idx + 1, back_idx + 1)
    
    return chunks


# =============================
# Configuration Loading
# =============================

def load_best_config() -> Dict:
    """Load best configuration from JSON file."""
    config_path = "best_config.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found. "
            f"Please run the hyperparameter tuning experiment first to generate it."
        )
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    required_keys = ["scheme_id", "chunk_size", "overlap_ratio"]
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required keys in best_config.json: {missing_keys}")
    
    return config


# =============================
# Data Loading
# =============================

def load_talks(csv_path: str, sample_n: int = None) -> pd.DataFrame:
    """Load TED talks from CSV, optionally sampling."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"TED talks CSV not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.dropna(subset=REQUIRED_COLS)
    df = df.sort_values('talk_id').reset_index(drop=True)
    
    if sample_n and sample_n < len(df):
        df = df.head(sample_n)
    
    print(f"Loaded {len(df)} talks")
    return df


# =============================
# Chunk Building
# =============================

def build_chunks_for_scheme(df: pd.DataFrame, scheme: ChunkScheme, 
                           store_chunk_text_in_metadata: bool = True) -> List[ChunkRecord]:
    """Build content chunks for all talks using a specific chunking scheme."""
    out: List[ChunkRecord] = []
    
    for _, row in df.iterrows():
        talk_id = str(row.get("talk_id", "")).strip()
        title = str(row.get("title", "")).strip()
        speaker = str(row.get("speaker_1", "")).strip()
        topics = str(row.get("topics", "")).strip()
        description = str(row.get("description", "")).strip()
        transcript = str(row.get("transcript", "")).strip()
        
        # Additional metadata field
        all_speakers = str(row.get("all_speakers", "")).strip()
        
        chunks = chunk_text(transcript, max_tokens=scheme.chunk_tokens, overlap_ratio=scheme.overlap_ratio)
        
        for i, ch in enumerate(chunks):
            chunk_id = f"{talk_id}:{scheme.scheme_id}:{i}"
            md = {
                "talk_id": talk_id,
                "title": title,
                "speaker_1": speaker,
                "topics": topics,
                "description": description,
                "chunk_index": i,
                "scheme_id": scheme.scheme_id,
                "all_speakers": all_speakers,
            }
            if store_chunk_text_in_metadata:
                md["chunk"] = ch
            out.append(ChunkRecord(chunk_id=chunk_id, text=ch, metadata=md))
    
    return out


# =============================
# Embedding & Pinecone Upload
# =============================

def embed_texts(embedding_client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    resp = embedding_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def batched(xs: List, batch_size: int) -> Iterable[List]:
    """Yield successive batches from a list."""
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]


def upsert_chunks(index, namespace: str, embedding_client: OpenAI, 
                 chunks: List[ChunkRecord], batch_size: int = 96) -> None:
    """Embed and upload chunks to Pinecone."""
    for batch in batched(chunks, batch_size):
        vecs = embed_texts(embedding_client, [c.text for c in batch])
        payload = [(c.chunk_id, v, c.metadata) for c, v in zip(batch, vecs)]
        index.upsert(vectors=payload, namespace=namespace)


def ensure_index_exists(pc: Pinecone, index_name: str, dimension: int = 1536) -> None:
    """Create Pinecone index if it doesn't exist."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists")
        return
    
    print(f"Creating Pinecone index '{index_name}' with dimension={dimension}...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    print("Waiting for index to be ready...")
    while not pc.describe_index(index_name).status.ready:
        time.sleep(1)
    
    print(f"✓ Index '{index_name}' created and ready!")


def build_and_upsert_schemes(index, embedding_client: OpenAI, df: pd.DataFrame, 
                             schemes: List[ChunkScheme], force_reembed: bool = False) -> None:
    """Build chunks for all schemes and upload to Pinecone."""
    for i, scheme in enumerate(schemes, 1):
        print(f"\n[{i}/{len(schemes)}] Processing scheme: {scheme.scheme_id}")
        
        skip_content = False
        
        if not force_reembed:
            try:
                stats = index.describe_index_stats()
                namespace_stats = stats.get('namespaces', {})
                
                if scheme.scheme_id in namespace_stats and namespace_stats[scheme.scheme_id].get('vector_count', 0) > 0:
                    print(f"  ⏭️  Namespace '{scheme.scheme_id}' already has {namespace_stats[scheme.scheme_id]['vector_count']} vectors")
                    print(f"     (Set FORCE_REEMBED=true to override)")
                    skip_content = True
                    continue
            except Exception as e:
                print(f"  Warning: Could not check namespace stats: {e}")
        else:
            print(f"  🔄 Force re-embedding enabled, will overwrite existing data")
            try:
                stats = index.describe_index_stats()
                namespace_stats = stats.get('namespaces', {})
                
                if scheme.scheme_id in namespace_stats and namespace_stats[scheme.scheme_id].get('vector_count', 0) > 0:
                    index.delete(delete_all=True, namespace=scheme.scheme_id)
                    print(f"  🗑️  Deleted {namespace_stats[scheme.scheme_id]['vector_count']} existing vectors from '{scheme.scheme_id}'")
            except Exception as e:
                print(f"  Warning: Could not delete from namespace: {e}")
        
        # Build and upload content chunks (transcript with metadata)
        if not skip_content:
            chunks = build_chunks_for_scheme(df, scheme, store_chunk_text_in_metadata=True)
            print(f"  Created {len(chunks)} chunks, uploading to Pinecone...")
            upsert_chunks(index, scheme.scheme_id, embedding_client, chunks)
            print(f"  ✓ Uploaded to namespace: {scheme.scheme_id}")


# =============================
# Main
# =============================

def main():
    """Main embedding and indexing pipeline."""
    print("=" * 60)
    print("TED Talk RAG - Embedding & Indexing")
    print("=" * 60)
    
    # Validate environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please set them before running this script."
        )
    
    # Load best configuration
    print("\nLoading best configuration from best_config.json...")
    try:
        best_config = load_best_config()
        print(f"✓ Loaded configuration: {best_config['scheme_id']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    # Configuration
    csv_path = os.getenv("TED_CSV_PATH", "ted_talks_en.csv")
    index_name = os.getenv("PINECONE_INDEX")
    sample_n = int(os.getenv("SAMPLE_N", "0")) or None  # 0 means use all data
    force_reembed = os.getenv("FORCE_REEMBED", "false").lower() in ("true", "1", "yes")
    
    # Create chunking scheme from best config
    scheme = ChunkScheme(
        scheme_id=best_config["scheme_id"],
        chunk_tokens=best_config["chunk_size"],
        overlap_ratio=best_config["overlap_ratio"]
    )
    schemes = [scheme]  # Single scheme based on best config
    
    print(f"\nConfiguration:")
    print(f"  CSV Path: {csv_path}")
    print(f"  Pinecone Index: {index_name}")
    print(f"  Sample Size: {'All data' if not sample_n else sample_n}")
    print(f"  Force Re-embed: {force_reembed}")
    print(f"\nChunking Scheme (from best_config.json):")
    print(f"  Scheme ID: {scheme.scheme_id}")
    print(f"  Chunk Size: {scheme.chunk_tokens} tokens")
    print(f"  Overlap Ratio: {scheme.overlap_ratio}")
    
    # Load data
    print(f"\n{'=' * 60}")
    print("Loading TED Talks Data")
    print(f"{'=' * 60}")
    df = load_talks(csv_path, sample_n)
    
    # Initialize clients
    print(f"\n{'=' * 60}")
    print("Initializing API Clients")
    print(f"{'=' * 60}")
    
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", os.environ["OPENAI_API_KEY"])
    embedding_client = OpenAI(api_key=embedding_api_key, base_url=LLMOD_BASE_URL)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    print("✓ Clients initialized")
    
    # Ensure Pinecone index exists
    ensure_index_exists(pc, index_name, dimension=1536)
    index = pc.Index(index_name)
    
    # Embed and upload
    print(f"\n{'=' * 60}")
    print("Embedding & Uploading Chunks")
    print(f"{'=' * 60}")
    
    if force_reembed:
        print("⚠️  FORCE_REEMBED enabled - will re-embed all data")
        time.sleep(3)
    
    build_and_upsert_schemes(index, embedding_client, df, schemes, force_reembed=force_reembed)
    
    print(f"\n{'=' * 60}")
    print("✓ Embedding and Indexing Complete!")
    print(f"{'=' * 60}")
    print(f"\nProcessed scheme: {scheme.scheme_id}")
    print(f"Total talks embedded: {len(df)}")
    print(f"Namespace: {scheme.scheme_id}")


if __name__ == "__main__":
    main()
