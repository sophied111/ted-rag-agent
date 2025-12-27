import os
import re
import unicodedata
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
import time

import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# =============================
# Constants / Defaults
# =============================

EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

# LLMOD AI base URL for custom API endpoint
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

# Assignment constraints
MAX_CHUNK_SIZE = 2048
MAX_OVERLAP_RATIO = 0.3
MAX_TOP_K = 30

SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and "
    "only based on the TED dataset context provided to you (metadata "
    "and transcript passages). You must not use any external "
    "knowledge, the open internet, or information that is not explicitly "
    "contained in the retrieved context. If the answer cannot be "
    "determined from the provided context, respond: “I don’t know "
    "based on the provided TED data.” Always explain your answer "
    "using the given context, quoting or paraphrasing the relevant "
    "transcript or metadata when helpful."
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", OPENAI_API_KEY)

# =============================
# Data Models
# =============================

@dataclass(frozen=True)
class ChunkScheme:
    """
    Configuration for text chunking strategy.
    
    Attributes:
        scheme_id: Unique identifier for this chunking configuration
        chunk_tokens: Maximum tokens per chunk (must be <= MAX_CHUNK_SIZE)
        overlap_ratio: Fraction of overlap between consecutive chunks (0 to MAX_OVERLAP_RATIO)
    """
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
    """
    A single text chunk with associated metadata for vector storage.
    
    Attributes:
        chunk_id: Unique identifier (format: talk_id:scheme_id:chunk_index)
        text: The actual chunk text to be embedded
        metadata: Talk metadata (title, speaker, topics, etc.) and chunk info
    """
    chunk_id: str
    text: str
    metadata: Dict

@dataclass(frozen=True)
class EvalQuery:
    """
    Evaluation query with expected results for measuring retrieval quality.
    
    Attributes:
        question: Natural language query
        expected_talk_id: Expected talk ID for binary evaluation (optional)
        expected_keywords: Keywords that should appear in results for scoring (optional)
    """
    question: str
    expected_talk_id: Optional[str] = None
    expected_keywords: Optional[List[str]] = None


# =============================
# Text utils + Chunking
# =============================

_WS = re.compile(r"\s+")

def normalize_ws(text: str) -> str:
    """Normalize whitespace by collapsing multiple spaces into one."""
    return _WS.sub(" ", (text or "").strip())

def strip_weird_unicode(text: str) -> str:
    """
    Remove control, format, and non-printable Unicode characters
    while keeping normal punctuation and letters.
    
    Args:
        text: Input text that may contain problematic Unicode
        
    Returns:
        Cleaned text with only printable characters
    """
    if not text:
        return ""

    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)

        # Drop control chars, format chars, private use, surrogates
        if cat in {"Cc", "Cf", "Cs", "Co"}:
            continue

        # Drop replacement character �
        if ch == "\uFFFD":
            continue

        cleaned.append(ch)

    return "".join(cleaned)

_WS = re.compile(r"\s+")




def preprocess_text(text: str) -> str:
    """
    Canonical preprocessing before chunking.
    Removes TED-specific artifacts and cleans text for better embedding quality.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned text ready for chunking
    """
    if not text:
        return ""

    # Optional but safe: normalize Unicode
    text = unicodedata.normalize("NFC", text)

    # Remove weird / invisible chars
    text = strip_weird_unicode(text)
    
    # Remove TED-specific artifacts (audience reactions, stage directions)
    # Common patterns: (Laughter), (Applause), (Music), (Video), (Audio), etc.
    text = re.sub(r'\([A-Z][a-z]+\)', '', text)  # Remove (Laughter), (Applause), etc.
    text = re.sub(r'\([A-Z][a-z]+ [a-z]+\)', '', text)  # Remove (Laughter and applause)
    
    # Remove timestamps if present: [00:00] or [0:00:00]
    text = re.sub(r'\[\d{1,2}:\d{2}(:\d{2})?\]', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive punctuation (3+ repeated chars)
    text = re.sub(r'([.!?]){3,}', r'\1\1', text)
    
    # Remove standalone numbers that might be artifacts
    text = re.sub(r'\b\d+\b(?!\s*[a-zA-Z])', '', text)

    # Normalize whitespace (collapse multiple spaces, newlines, tabs)
    text = _WS.sub(" ", text).strip()
    
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)

    return text


def chunk_text(text: str, max_tokens: int, overlap_ratio: float) -> List[str]:
    """
    Split text into overlapping chunks using token approximation.
    
    Uses a simple heuristic: ~4 characters ≈ 1 token. This is deterministic
    and dependency-free (no tiktoken required).
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_ratio: Fraction of overlap between consecutive chunks (0.0 to 0.3)
        
    Returns:
        List of text chunks with specified overlap
    """
    text = preprocess_text(text)

    if not text:
        return []

    words = text.split(" ")
    # crude token estimate per word
    word_tokens = [max(1, len(w) // 4) for w in words]

    chunks: List[str] = []
    n = len(words)
    start = 0
    overlap_tokens = int(max_tokens * overlap_ratio)

    while start < n:
        total = 0
        end = start

        while end < n and total + word_tokens[end] <= max_tokens:
            total += word_tokens[end]
            end += 1

        if end == start:  # pathological case
            end = min(n, start + 1)

        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        # step forward with overlap
        back = 0
        back_tokens = 0
        i = end - 1
        while i >= start and back_tokens < overlap_tokens:
            back_tokens += word_tokens[i]
            back += 1
            i -= 1

        start = max(start + 1, end - back)

    return chunks


# =============================
# Clients
# =============================

def make_clients() -> Tuple[OpenAI, OpenAI, Pinecone]:
    """
    Initialize API clients for embeddings, chat, and vector storage.
    Validates API keys by making test calls before returning clients.
    
    Requires environment variables:
        - OPENAI_API_KEY: For chat completions (required)
        - PINECONE_API_KEY: For vector database (required)
        - EMBEDDING_API_KEY: For embeddings (optional, defaults to OPENAI_API_KEY)
    
    Returns:
        Tuple of (embedding_client, chat_client, pinecone_client)
        
    Raises:
        RuntimeError: If required API keys are missing or invalid
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY env var. Set it before running.")
    if not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("Missing PINECONE_API_KEY env var. Set it before running.")
    
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", os.environ["OPENAI_API_KEY"])
    embedding_client = OpenAI(api_key=embedding_api_key, base_url=LLMOD_BASE_URL)
    chat_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=LLMOD_BASE_URL)
    pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # Validate API keys with test calls
    print("Validating API keys...")
    
    try:
        # Test embedding client
        print("  Testing embedding API...", end=" ")
        test_embed = embedding_client.embeddings.create(
            model=EMBED_MODEL, 
            input=["test"]
        )
        if len(test_embed.data) != 1 or len(test_embed.data[0].embedding) != 1536:
            raise RuntimeError("Embedding API returned unexpected format")
        print("✓")
    except Exception as e:
        raise RuntimeError(f"Embedding API validation failed: {e}")
    
    try:
        # Test chat client
        print("  Testing chat API...", end=" ")
        test_chat = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        # Just verify the call succeeds and returns a response object
        if not test_chat.choices:
            raise RuntimeError("Chat API returned no choices")
        print("✓")
    except Exception as e:
        raise RuntimeError(f"Chat API validation failed: {e}")
    
    try:
        # Test Pinecone client
        print("  Testing Pinecone API...", end=" ")
        indexes = pinecone_client.list_indexes()
        # Just successfully listing is enough validation
        print("✓")
    except Exception as e:
        raise RuntimeError(f"Pinecone API validation failed: {e}")
    
    print("All API keys validated successfully!\n")
    
    return embedding_client, chat_client, pinecone_client


# =============================
# Embedding + Pinecone
# =============================

def embed_texts(embedding_client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts using OpenAI API."""
    resp = embedding_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def batched(xs: List, batch_size: int) -> Iterable[List]:
    """Yield successive batches from a list."""
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]

def upsert_chunks(index, namespace: str, embedding_client: OpenAI, chunks: List[ChunkRecord], batch_size: int = 96) -> None:
    """
    Embed and upload chunks to Pinecone vector database.
    
    Args:
        index: Pinecone index instance
        namespace: Namespace for this chunking scheme
        embedding_client: OpenAI client for embeddings
        chunks: List of chunks to upload
        batch_size: Number of chunks to process per batch (default: 96)
    """
    for batch in batched(chunks, batch_size):
        vecs = embed_texts(embedding_client, [c.text for c in batch])
        payload = [(c.chunk_id, v, c.metadata) for c, v in zip(batch, vecs)]
        index.upsert(vectors=payload, namespace=namespace)

def query_index(index, namespace: str, embedding_client: OpenAI, question: str, top_k: int):
    """
    Search Pinecone index for relevant chunks using semantic similarity.
    
    Args:
        index: Pinecone index instance
        namespace: Namespace to search within
        embedding_client: OpenAI client for query embedding
        question: User's natural language query
        top_k: Number of results to return
        
    Returns:
        List of matching chunks with metadata and similarity scores
    """
    q_vec = embed_texts(embedding_client, [question])[0]
    res = index.query(vector=q_vec, top_k=top_k, include_metadata=True, namespace=namespace)
    return res.matches


# =============================
# Chunk construction
# =============================

REQUIRED_COLS = ["talk_id", "title", "transcript"]

def load_sample_talks(csv_path: str, sample_n: int, seed: int) -> pd.DataFrame:
    """
    Load and sample TED talks from CSV with strategic topic coverage.
    
    Ensures the sample contains diverse topics to support all evaluation query types,
    particularly multi-result queries that need multiple talks per topic.
    
    Args:
        csv_path: Path to ted_talks_en.csv
        sample_n: Number of talks to sample
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled talks
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"TED talks CSV not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.dropna(subset=REQUIRED_COLS)
    
    # CRITICAL: Sort by talk_id to ensure consistent ordering across runs
    # This ensures DataFrame iteration order is deterministic even if CSV or dropna() changes
    df = df.sort_values('talk_id').reset_index(drop=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define key topics to ensure coverage (for Type 2 multi-result queries)
    key_topics = [
        ("brain", ["brain", "neuroscience", "cognitive", "memory", "mind"]),
        ("technology", ["technology", "innovation", "design", "tech"]),
        ("health", ["health", "medicine", "medical", "disease"]),
        ("climate", ["climate", "environment", "sustainability", "carbon"]),
        ("social", ["social", "society", "culture", "humanity"]),
    ]
    
    # Sample with topic diversity
    min_per_topic = 5  # Ensure at least 5 talks per key topic
    sampled_ids = set()
    topic_samples = []
    
    # First, ensure we have diverse coverage
    for topic_name, keywords in key_topics:
        topic_matches = []
        for idx, row in df.iterrows():
            if str(row['talk_id']) in sampled_ids:
                continue
            
            title_lower = str(row.get('title', '')).lower()
            desc_lower = str(row.get('description', '')).lower()
            topics_lower = str(row.get('topics', '')).lower()
            
            if any(kw in title_lower or kw in desc_lower or kw in topics_lower for kw in keywords):
                topic_matches.append(idx)
        
        # Sample min_per_topic from this topic
        # CRITICAL: Sort to ensure deterministic random.sample() behavior
        topic_matches_sorted = sorted(topic_matches)
        if len(topic_matches_sorted) >= min_per_topic:
            selected = random.sample(topic_matches_sorted, min_per_topic)
        else:
            selected = topic_matches_sorted
        
        for idx in selected:
            sampled_ids.add(str(df.loc[idx, 'talk_id']))
            topic_samples.append(idx)
    
    # Fill remaining with random samples
    remaining_needed = sample_n - len(topic_samples)
    if remaining_needed > 0:
        remaining_pool = [idx for idx in df.index if str(df.loc[idx, 'talk_id']) not in sampled_ids]
        # CRITICAL: Sort to ensure deterministic random.sample() behavior
        remaining_pool_sorted = sorted(remaining_pool)
        if len(remaining_pool_sorted) > remaining_needed:
            additional = random.sample(remaining_pool_sorted, remaining_needed)
        else:
            additional = remaining_pool_sorted
        topic_samples.extend(additional)
    
    # Create final sample
    sample_df = df.loc[topic_samples[:sample_n]].reset_index(drop=True)
    
    if len(sample_df) < sample_n:
        print(f"Warning: Requested {sample_n} talks but only {len(sample_df)} available after filtering.")
    
    return sample_df

def build_chunks_for_scheme(df: pd.DataFrame, scheme: ChunkScheme, store_chunk_text_in_metadata: bool = True) -> List[ChunkRecord]:
    """
    Build chunks for all talks using a specific chunking scheme.
    
    Args:
        df: DataFrame with TED talks
        scheme: Chunking configuration
        store_chunk_text_in_metadata: If True, include chunk text in metadata
        
    Returns:
        List of ChunkRecord objects ready for embedding
    """
    out: List[ChunkRecord] = []
    for _, row in df.iterrows():
        talk_id = str(row.get("talk_id", "")).strip()
        title = str(row.get("title", "")).strip()
        speaker = str(row.get("speaker_1", "")).strip()
        topics = str(row.get("topics", "")).strip()
        description = str(row.get("description", "")).strip()
        transcript = str(row.get("transcript", "")).strip()

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
            }
            if store_chunk_text_in_metadata:
                md["chunk"] = ch
            out.append(ChunkRecord(chunk_id=chunk_id, text=ch, metadata=md))
    return out


# =============================
# Evaluation (retrieval-only)
# =============================

def score_matches(matches, eq: EvalQuery) -> float:
    """
    Score retrieval results based on evaluation criteria.
    
    Scoring methods (in priority order):
    1. Binary: 1.0 if expected_talk_id is in results, else 0.0
    2. Keyword coverage: fraction of expected_keywords found in results
    3. Fallback: average similarity score from Pinecone
    
    Args:
        matches: List of Pinecone match objects
        eq: Evaluation query with expected results
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not matches:
        return 0.0

    if eq.expected_talk_id:
        return 1.0 if any(str((m.metadata or {}).get("talk_id")) == str(eq.expected_talk_id) for m in matches) else 0.0

    if eq.expected_keywords:
        blob = " ".join(
            normalize_ws(
                " ".join(
                    [
                        (m.metadata or {}).get("title", ""),
                        (m.metadata or {}).get("description", ""),
                        (m.metadata or {}).get("topics", ""),
                        (m.metadata or {}).get("speaker_1", ""),
                        (m.metadata or {}).get("chunk", ""),
                    ]
                )
            ).lower()
            for m in matches
        )
        kws = [k.lower() for k in eq.expected_keywords]
        hit = sum(1 for k in kws if k in blob)
        return hit / max(1, len(kws))

    # fallback: mean similarity score (relative comparisons only)
    return sum(float(m.score) for m in matches) / len(matches)

def evaluate_grid(index, embedding_client: OpenAI, df: pd.DataFrame, schemes: List[ChunkScheme], topk_grid: List[int], eval_set: List[EvalQuery]) -> List[Dict]:
    """
    Evaluate all configurations and save detailed results to text files.
    
    For each configuration, creates a file like 'eval_cs512_ol20_topk5.txt'
    containing queries, retrieved context, and similarity scores.
    
    Args:
        df: DataFrame with sampled TED talks (to log sample info)
    """
    results: List[Dict] = []
    total_configs = len(schemes) * len(topk_grid)
    config_num = 0
    
    # Create output directory with timestamp subfolder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    for scheme in schemes:
        for top_k in topk_grid:
            config_num += 1
            scores = []
            
            # Prepare detailed output file
            output_file = f"{output_dir}/eval_{scheme.scheme_id}_topk{top_k}.txt"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"EVALUATION RESULTS - Configuration {config_num}/{total_configs}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Scheme ID: {scheme.scheme_id}\n")
                f.write(f"Chunk Size: {scheme.chunk_tokens} tokens\n")
                f.write(f"Overlap Ratio: {scheme.overlap_ratio}\n")
                f.write(f"Top-K: {top_k}\n")
                f.write("=" * 80 + "\n\n")
                
                # Add sample dataset information
                f.write("SAMPLE DATASET INFORMATION:\n")
                f.write(f"Total talks in sample: {len(df)}\n\n")
                f.write("Sample Talk IDs and Titles:\n")
                f.write("-" * 80 + "\n")
                for idx, row in df.iterrows():
                    talk_id = str(row.get("talk_id", "")).strip()
                    title = str(row.get("title", "")).strip()
                    speaker = str(row.get("speaker_1", "")).strip()
                    f.write(f"  [{talk_id}] {title}\n")
                    f.write(f"           Speaker: {speaker}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, eq in enumerate(eval_set, 1):
                    # Retrieval only - no LLM
                    try:
                        matches = query_index(index, scheme.scheme_id, embedding_client, eq.question, top_k)
                        score = score_matches(matches, eq)
                        scores.append(score)
                        
                    except Exception as e:
                        matches = []
                        score = 0.0
                        scores.append(score)
                    
                    # Write detailed results
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"QUERY {i}/{len(eval_set)}\n")
                    f.write(f"{'=' * 80}\n\n")
                    f.write(f"Question: {eq.question}\n\n")
                    
                    if eq.expected_talk_id:
                        f.write(f"Expected Talk ID: {eq.expected_talk_id}\n")
                    if eq.expected_keywords:
                        f.write(f"Expected Keywords: {', '.join(eq.expected_keywords)}\n")
                    f.write(f"Retrieval Score: {score:.4f}\n\n")
                    
                    f.write(f"{'-' * 80}\n")
                    f.write("RETRIEVED CONTEXT:\n")
                    f.write(f"{'-' * 80}\n\n")
                    
                    for j, m in enumerate(matches, 1):
                        md = m.metadata or {}
                        f.write(f"[{j}] Similarity: {float(m.score):.4f}\n")
                        f.write(f"    Talk ID: {md.get('talk_id')}\n")
                        f.write(f"    Title: {md.get('title')}\n")
                        chunk_text = md.get('chunk', '')
                        f.write(f"    Chunk: {chunk_text[:200]}...\n\n")
            
            mean_score = sum(scores) / max(1, len(scores))
            results.append(
                {
                    "scheme_id": scheme.scheme_id,
                    "chunk_size": scheme.chunk_tokens,
                    "overlap_ratio": scheme.overlap_ratio,
                    "top_k": top_k,
                    "mean_eval_score": mean_score,
                    "output_file": output_file,
                }
            )
            print(f"[{config_num}/{total_configs}] {scheme.scheme_id} top_k={top_k}: score={mean_score:.4f} → {output_file}")
    
    results.sort(key=lambda x: x["mean_eval_score"], reverse=True)
    return results


# =============================
# End-to-end RAG answer
# =============================

def build_prompt(question: str, matches) -> str:
    evidence_lines = []
    for i, m in enumerate(matches, start=1):
        md = m.metadata or {}
        evidence_lines.append(
            f"[{i}] talk_id={md.get('talk_id')} | title={md.get('title')} | speaker={md.get('speaker_1')} "
            f"| topics={md.get('topics')} | description={md.get('description')}\n"
            f"CHUNK:\n{md.get('chunk','')}"
        )
    evidence = "\n\n".join(evidence_lines)

    return (
        "Use ONLY the following TED dataset context (metadata + transcript passages). "
        "Answer the question. If the answer cannot be determined from this context, say: "
        "“I don’t know based on the provided TED data.”\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{evidence}\n"
    )

def rag_answer(embedding_client: OpenAI, chat_client: OpenAI, index, scheme_id: str, top_k: int, question: str) -> Dict:
    matches = query_index(index, scheme_id, embedding_client, question, top_k)
    user_prompt = build_prompt(question, matches)

    chat = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    context = []
    for m in matches:
        md = m.metadata or {}
        context.append(
            {
                "talk_id": md.get("talk_id"),
                "title": md.get("title"),
                "chunk": md.get("chunk", ""),
                "score": float(m.score),
            }
        )

    return {
        "response": chat.choices[0].message.content,
        "context": context,
        "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
    }


# =============================
# Orchestration
# =============================

def build_and_upsert_all_schemes(index, embedding_client: OpenAI, df: pd.DataFrame, schemes: List[ChunkScheme], force_reembed: bool = False) -> None:
    """
    Build chunks for all schemes and upload to Pinecone.
    
    Args:
        index: Pinecone index instance
        embedding_client: OpenAI client for embeddings
        df: DataFrame with TED talks
        schemes: List of chunking schemes to process
        force_reembed: If True, re-embed data even if namespace already has vectors.
                      If False, skip namespaces that already have vectors.
    """
    for i, scheme in enumerate(schemes, 1):
        print(f"[{i}/{len(schemes)}] Processing scheme: {scheme.scheme_id}")
        
        # Check if namespace already has vectors (unless force_reembed is True)
        if not force_reembed:
            try:
                stats = index.describe_index_stats()
                namespace_stats = stats.get('namespaces', {})
                if scheme.scheme_id in namespace_stats and namespace_stats[scheme.scheme_id].get('vector_count', 0) > 0:
                    print(f"  ⏭️  Namespace '{scheme.scheme_id}' already has {namespace_stats[scheme.scheme_id]['vector_count']} vectors, skipping")
                    print(f"     (Set FORCE_REEMBED=true to override)")
                    continue
            except Exception as e:
                print(f"  Warning: Could not check namespace stats: {e}")
        else:
            print(f"  🔄 Force re-embedding enabled, will overwrite existing data")
            # Delete all vectors in the namespace if it exists
            try:
                stats = index.describe_index_stats()
                namespace_stats = stats.get('namespaces', {})
                if scheme.scheme_id in namespace_stats and namespace_stats[scheme.scheme_id].get('vector_count', 0) > 0:
                    index.delete(delete_all=True, namespace=scheme.scheme_id)
                    print(f"  🗑️  Deleted {namespace_stats[scheme.scheme_id]['vector_count']} existing vectors from namespace '{scheme.scheme_id}'")
                else:
                    print(f"  ℹ️  Namespace '{scheme.scheme_id}' is empty or doesn't exist yet")
            except Exception as e:
                print(f"  Warning: Could not delete from namespace: {e}")
        
        chunks = build_chunks_for_scheme(df, scheme, store_chunk_text_in_metadata=True)
        print(f"  Created {len(chunks)} chunks, uploading to Pinecone...")
        upsert_chunks(index, scheme.scheme_id, embedding_client, chunks)
        print(f"  ✓ Uploaded to namespace: {scheme.scheme_id}")

def choose_best(results: List[Dict]) -> Dict:
    """
    Select best configuration from evaluation results.
    
    Args:
        results: List of evaluation results (sorted by score, descending)
        
    Returns:
        Best configuration dictionary
        
    Raises:
        RuntimeError: If results list is empty
    """
    if not results:
        raise RuntimeError("No evaluation results. Cannot determine best configuration.")
    return results[0]

def validate_environment() -> None:
    """Validate required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please set them before running this script."
        )

def ensure_index_exists(pc: Pinecone, index_name: str, dimension: int = 1536) -> None:
    """
    Create Pinecone index if it doesn't exist.
    
    Args:
        pc: Pinecone client instance
        index_name: Name of the index to create
        dimension: Vector dimension (1536 for text-embedding-3-small)
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists, skipping creation.")
        return
    
    print(f"Creating Pinecone index '{index_name}' with dimension={dimension}...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    # Wait for index to be ready
    print("Waiting for index to be ready...")
    while not pc.describe_index(index_name).status.ready:
        time.sleep(1)
    
    print(f"✓ Index '{index_name}' created and ready!")


def main():
    """Main hyperparameter tuning pipeline."""
    print("=" * 60)
    print("TED Talk RAG - Hyperparameter Tuning Experiment")
    print("=" * 60)
    
    # Validate environment
    try:
        validate_environment()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return
    
    # Configuration
    csv_path = os.getenv("TED_CSV_PATH", "ted_talks_en.csv")
    index_name = os.getenv("PINECONE_INDEX")

    seed = int(os.getenv("SEED", "7"))
    sample_n = int(os.getenv("SAMPLE_N", "100"))

    # Strategic chunking schemes: 6 combinations balancing cost and coverage
    # Testing chunk sizes: 512, 1024, 1536 with overlaps: 0.1 (low), 0.2 (medium), 0.25 (high)
    schemes = [
        ChunkScheme("cs512_ol10", 512, 0.10),           # Small baseline
        ChunkScheme("cs1024_ol10", 1024, 0.10),         # Medium, low overlap
        ChunkScheme("cs1024_ol20", 1024, 0.20),         # Medium baseline
        ChunkScheme("cs1024_ol25", 1024, 0.25),         # Medium, high overlap
        ChunkScheme("cs1536_ol10", 1536, 0.10),         # Large, low overlap
        ChunkScheme("cs1536_ol20", 1536, 0.20),         # Large baseline
    ]
    
    # Top-k grid: 4 values × 6 schemes = 24 total configurations
    topk_grid = [5, 10, 15, 20]
    
    # Validate hyperparameters
    for k in topk_grid:
        if k > MAX_TOP_K:
            raise ValueError(f"top_k {k} exceeds max {MAX_TOP_K}")

    # Load sample talks
    # Note: load_sample_talks() sets random.seed internally, no need to set it here
    df = load_sample_talks(csv_path, sample_n=sample_n, seed=seed)
    print(f"Loaded {len(df)} talks")
    print(f"Testing {len(schemes)} schemes × {len(topk_grid)} top_k = {len(schemes) * len(topk_grid)} configurations")
    
    # Print sample talk IDs and titles
    print("\n" + "=" * 60)
    print("SAMPLED TALKS")
    print("=" * 60)
    for idx, row in df.iterrows():
        talk_id = str(row.get("talk_id", "")).strip()
        title = str(row.get("title", "")).strip()
        speaker = str(row.get("speaker_1", "")).strip()
        print(f"[{talk_id}] {title}")
        print(f"         Speaker: {speaker}")
    print("=" * 60 + "\n")

    # Clients
    embedding_client, chat_client, pc = make_clients()
    
    # Ensure index exists (create if needed)
    ensure_index_exists(pc, index_name, dimension=1536)
    
    index = pc.Index(index_name)

    # 1) Embed+upsert once per namespace (scheme)
    print("\n=== Embedding & Upserting Chunks ===")
    force_reembed = os.getenv("FORCE_REEMBED", "false").lower() in ("true", "1", "yes")
    if force_reembed:
        print("⚠️  FORCE_REEMBED enabled - will re-embed all data even if it exists")
    time.sleep(5)
    build_and_upsert_all_schemes(index, embedding_client, df, schemes, force_reembed=force_reembed)

    # 2) Expanded evaluation set: 18 queries covering all 4 capability types
    # Queries specifically match talks in the 100-talk sample (seed=7, sample_n=100)
    print("\n=== Building Evaluation Set ===")
    eval_set = [
        # Type 1: Precise Fact Retrieval (5 queries)
        EvalQuery("Find a TED talk about experience versus memory. Provide the title and speaker.", 
                  expected_talk_id="779",
                  expected_keywords=["experience", "memory", "riddle"]),
        EvalQuery("Which TED talk discusses gravitational waves and LIGO?", 
                  expected_talk_id="2886",
                  expected_keywords=["LIGO", "gravitational", "waves"]),
        EvalQuery("Find a TED talk about growing fresh air with plants.", 
                  expected_talk_id="490",
                  expected_keywords=["fresh", "air", "plants"]),
        EvalQuery("Which speaker talks about stress and its effects on the brain?", 
                  expected_talk_id="24275",
                  expected_keywords=["stress", "brain", "cortisol"]),
        EvalQuery("Find a TED talk about mental illness and comedy.", 
                  expected_talk_id="1584",
                  expected_keywords=["mental", "illness", "funny"]),
        
        # Type 2: Multi-Result Topic Listing (5 queries - requesting 3 results)
        EvalQuery("Which TED talks focus on brain science and cognition? Return 3 talk titles.", 
                  expected_keywords=["brain", "cognitive", "mind"]),
        EvalQuery("List 3 TED talks about technology and innovation.", 
                  expected_keywords=["technology", "innovation", "design"]),
        EvalQuery("Find 3 TED talks discussing health and medicine.", 
                  expected_keywords=["health", "medicine", "disease"]),
        EvalQuery("Which 3 TED talks address climate and environment?", 
                  expected_keywords=["climate", "environment", "sustainability"]),
        EvalQuery("Show me 3 TED talks about social issues and humanity.", 
                  expected_keywords=["social", "humanity", "culture"]),
        
        # Type 3: Key Idea Summary Extraction (4 queries)
        EvalQuery("Find a TED talk about asking for help. Provide the title and a short summary of the key idea.", 
                  expected_talk_id="2712",
                  expected_keywords=["help", "strength", "weakness"]),
        EvalQuery("What is the main idea presented in a TED talk about living to be 100?", 
                  expected_talk_id="727",
                  expected_keywords=["longevity", "100", "centenarian"]),
        EvalQuery("Summarize the key message from a TED talk about the art of choosing.", 
                  expected_talk_id="924",
                  expected_keywords=["choice", "choosing", "decisions"]),
        EvalQuery("What's the central theme of a TED talk discussing whether the world is getting better?", 
                  expected_talk_id="15274",
                  expected_keywords=["world", "better", "worse", "numbers"]),
        
        # Type 4: Recommendation with Evidence-Based Justification (4 queries)
        EvalQuery("I'm interested in learning about augmented reality in medicine. Which talk would you recommend?", 
                  expected_talk_id="6477",
                  expected_keywords=["augmented", "reality", "surgery"]),
        EvalQuery("Recommend a TED talk that could help me understand how to work with intelligent machines.", 
                  expected_talk_id="2787",
                  expected_keywords=["intelligent", "machines", "fear"]),
        EvalQuery("I want to learn about solving medical mysteries. Which TED talk should I watch?", 
                  expected_talk_id="445",
                  expected_keywords=["medical", "mysteries", "diagnosis"]),
        EvalQuery("Suggest a TED talk for someone interested in experience versus memory.", 
                  expected_talk_id="779",
                  expected_keywords=["experience", "memory", "happiness"]),
    ]
    print(f"Created {len(eval_set)} evaluation queries")

    # 3) Retrieval-only grid evaluation
    print("\n=== Grid Search Evaluation ===")
    results = evaluate_grid(index, embedding_client, df, schemes, topk_grid, eval_set)
    
    print("\n=== Top 5 Configurations ===")
    for i, config in enumerate(results[:5], 1):
        print(f"{i}. {config['scheme_id']} | top_k={config['top_k']} | score={config['mean_eval_score']:.4f}")
    
    best = choose_best(results)
    print("\n=== BEST CONFIG ===")
    print(json.dumps(best, indent=2))

    # 4) Example end-to-end: Test all 4 query types with best config
    print("\n=== Testing Best Configuration with Example Queries ===")
    
    example_queries = [
        ("Type 1: Fact Retrieval", "Find a TED talk about gravitational waves and LIGO. Provide the title and speaker."),
        ("Type 2: Multi-Result Listing", "List 3 TED talks about technology and innovation."),
        ("Type 3: Summary Extraction", "Find a TED talk about asking for help. Provide the title and a short summary of the key idea."),
        ("Type 4: Recommendation", "Recommend a TED talk about how stress affects the brain."),
    ]
    
    output_file = "best_config_examples.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BEST CONFIGURATION - END-TO-END EXAMPLES\n")
        f.write("=" * 80 + "\n\n")
        f.write("CONFIGURATION DETAILS:\n")
        f.write(f"Scheme ID: {best['scheme_id']}\n")
        f.write(f"Chunk Size: {best['chunk_size']} tokens\n")
        f.write(f"Overlap Ratio: {best['overlap_ratio']}\n")
        f.write(f"Top-K: {best['top_k']}\n")
        f.write(f"Mean Evaluation Score: {best['mean_eval_score']:.4f}\n")
        f.write("=" * 80 + "\n\n")
        
        # Add sample dataset information
        f.write("SAMPLE DATASET INFORMATION:\n")
        f.write(f"Total talks in sample: {len(df)}\n\n")
        f.write("Sample Talk IDs and Titles:\n")
        f.write("-" * 80 + "\n")
        for idx, row in df.iterrows():
            talk_id = str(row.get("talk_id", "")).strip()
            title = str(row.get("title", "")).strip()
            speaker = str(row.get("speaker_1", "")).strip()
            f.write(f"  [{talk_id}] {title}\n")
            f.write(f"           Speaker: {speaker}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (query_type, question) in enumerate(example_queries, 1):
            print(f"Testing {query_type}...")
            
            try:
                out = rag_answer(embedding_client, chat_client, index, best["scheme_id"], best["top_k"], question)
                
                f.write(f"\n{'=' * 80}\n")
                f.write(f"EXAMPLE {i}: {query_type}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(f"Question: {question}\n\n")
                
                f.write(f"{'-' * 80}\n")
                f.write("RETRIEVED CONTEXT:\n")
                f.write(f"{'-' * 80}\n\n")
                
                for j, ctx in enumerate(out["context"], 1):
                    f.write(f"[{j}] Similarity: {ctx.get('score', 0):.4f}\n")
                    f.write(f"    Talk ID: {ctx.get('talk_id')}\n")
                    f.write(f"    Title: {ctx.get('title')}\n")
                    chunk_text = ctx.get('chunk', '')
                    f.write(f"    Chunk: {chunk_text[:200]}...\n\n")
                
                f.write(f"{'-' * 80}\n")
                f.write("LLM RESPONSE:\n")
                f.write(f"{'-' * 80}\n\n")
                f.write(f"{out['response']}\n\n")
                
                # Also print to console
                print(f"✓ {query_type} completed")
                
            except Exception as e:
                f.write(f"\n[ERROR: {e}]\n\n")
                print(f"✗ {query_type} failed: {e}")
    
    print(f"\n✓ Example results saved to: {output_file}")
    print(f"\nSample answer (Type 4):")
    print(out["response"][:1000] + "...")

    # Save best config for later use (e.g., /api/stats)
    with open("best_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scheme_id": best["scheme_id"],
                "chunk_size": best["chunk_size"],
                "overlap_ratio": best["overlap_ratio"],
                "top_k": best["top_k"],
            },
            f,
            indent=2,
        )
    print("\nWrote best_config.json")


if __name__ == "__main__":
    main()
