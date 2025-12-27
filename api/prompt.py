import os
import json
from http.server import BaseHTTPRequestHandler
from typing import Dict, List

from openai import OpenAI
from pinecone import Pinecone


# =============================
# Configuration
# =============================

EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and "
    "only based on the TED dataset context provided to you (metadata "
    "and transcript passages). You must not use any external "
    "knowledge, the open internet, or information that is not explicitly "
    "contained in the retrieved context. If the answer cannot be "
    "determined from the provided context, respond: "I don't know "
    "based on the provided TED data." Always explain your answer "
    "using the given context, quoting or paraphrasing the relevant "
    "transcript or metadata when helpful."
)


# =============================
# Helper Functions
# =============================

def load_config() -> Dict:
    """Load best configuration from JSON file."""
    try:
        with open("best_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Default fallback configuration
        return {
            "scheme_id": "cs1024_ol20",
            "chunk_size": 1024,
            "overlap_ratio": 0.2,
            "top_k": 10
        }


def embed_texts(embedding_client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for texts."""
    resp = embedding_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def query_index(index, namespace: str, embedding_client: OpenAI, question: str, top_k: int):
    """Search Pinecone index for relevant chunks."""
    q_vec = embed_texts(embedding_client, [question])[0]
    res = index.query(vector=q_vec, top_k=top_k, include_metadata=True, namespace=namespace)
    return res.matches


def hybrid_query(index, scheme_id: str, embedding_client: OpenAI, question: str, top_k: int, 
                 metadata_weight: float = 0.6, content_weight: float = 0.4):
    """
    Hybrid search combining metadata and content embeddings with RRF fusion.
    """
    # Search metadata namespace
    metadata_namespace = f"{scheme_id}_metadata"
    try:
        metadata_matches = query_index(index, metadata_namespace, embedding_client, question, top_k)
    except Exception:
        metadata_matches = []
    
    # Search content namespace
    content_matches = query_index(index, scheme_id, embedding_client, question, top_k)
    
    # Reciprocal Rank Fusion (RRF) with weights
    k = 60
    talk_scores = {}
    
    # Process metadata matches
    for rank, match in enumerate(metadata_matches, start=1):
        talk_id = match.metadata.get('talk_id') if match.metadata else None
        if not talk_id:
            continue
        
        rrf_score = metadata_weight / (k + rank)
        combined_score = rrf_score + (float(match.score) * metadata_weight * 0.5)
        
        if talk_id not in talk_scores or combined_score > talk_scores[talk_id][0]:
            talk_scores[talk_id] = (combined_score, match)
    
    # Process content matches
    for rank, match in enumerate(content_matches, start=1):
        talk_id = match.metadata.get('talk_id') if match.metadata else None
        if not talk_id:
            continue
        
        rrf_score = content_weight / (k + rank)
        combined_score = rrf_score + (float(match.score) * content_weight * 0.5)
        
        if talk_id not in talk_scores:
            talk_scores[talk_id] = (combined_score, match)
        else:
            existing_score, existing_match = talk_scores[talk_id]
            new_score = existing_score + combined_score
            better_match = match if match.metadata.get('chunk_type') != 'metadata' else existing_match
            talk_scores[talk_id] = (new_score, better_match)
    
    # Sort by combined score and return top_k
    sorted_matches = sorted(talk_scores.items(), key=lambda x: x[1][0], reverse=True)
    final_matches = [(score, match) for _, (score, match) in sorted_matches[:top_k]]
    
    return final_matches


def build_prompt(question: str, matches) -> str:
    """Build prompt from question and retrieved matches."""
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
        ""I don't know based on the provided TED data."\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{evidence}\n"
    )


def rag_answer(question: str) -> Dict:
    """Main RAG pipeline - retrieve and generate answer."""
    # Load configuration
    config = load_config()
    scheme_id = config["scheme_id"]
    top_k = config["top_k"]
    
    # Initialize clients
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX")
    
    if not openai_api_key or not pinecone_api_key or not pinecone_index_name:
        raise RuntimeError("Missing required environment variables")
    
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY", openai_api_key)
    embedding_client = OpenAI(api_key=embedding_api_key, base_url=LLMOD_BASE_URL)
    chat_client = OpenAI(api_key=openai_api_key, base_url=LLMOD_BASE_URL)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    
    # Retrieve relevant chunks using hybrid search
    matches = hybrid_query(index, scheme_id, embedding_client, question, top_k)
    
    # Extract match objects from tuples
    match_objects = [m[1] if isinstance(m, tuple) else m for m in matches]
    
    # Build prompt
    user_prompt = build_prompt(question, match_objects)
    
    # Generate answer
    chat = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    
    # Format context for response
    context = []
    for match_item in matches:
        if isinstance(match_item, tuple):
            combined_score, m = match_item
        else:
            combined_score = None
            m = match_item
        
        md = m.metadata or {}
        context.append({
            "talk_id": md.get("talk_id"),
            "title": md.get("title"),
            "chunk": md.get("chunk", ""),
            "score": float(m.score),
        })
    
    return {
        "response": chat.choices[0].message.content,
        "context": context,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt
        }
    }


# =============================
# Vercel Handler
# =============================

class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler."""
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            
            # Parse JSON
            data = json.loads(body)
            question = data.get("question", "").strip()
            
            if not question:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Missing 'question' field in request body"
                }).encode())
                return
            
            # Process question
            result = rag_answer(question)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result, indent=2).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": str(e)
            }).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
