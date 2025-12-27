import json
from http.server import BaseHTTPRequestHandler


def load_config():
    """Load best configuration from JSON file."""
    try:
        with open("best_config.json", "r") as f:
            config = json.load(f)
            return {
                "chunk_size": config.get("chunk_size", 1024),
                "overlap_ratio": config.get("overlap_ratio", 0.2),
                "top_k": config.get("top_k", 10)
            }
    except FileNotFoundError:
        # Default fallback configuration
        return {
            "chunk_size": 1024,
            "overlap_ratio": 0.2,
            "top_k": 10
        }


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for stats endpoint."""
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            # Load configuration
            config = load_config()
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(config, indent=2).encode())
            
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
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
