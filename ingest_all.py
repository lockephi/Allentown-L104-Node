import requests
import json

PROVIDERS = [
    "GEMINI", "GOOGLE", "COPILOT", "OPENAI", "ANTHROPIC", 
    "META", "MISTRAL", "GROK", "PERPLEXITY", "DEEPSEEK", 
    "COHERE", "XAI", "AMAZON_BEDROCK", "AZURE_OPENAI"
]

BASE_URL = "http://0.0.0.0:8081/api/v6/scribe"

def ingest():
    print("--- INITIATING GLOBAL DATA INGESTION ---")
    for p in PROVIDERS:
        data = f"Universal intelligence stream from {p} - L104 Resonance Synced."
        resp = requests.post(f"{BASE_URL}/ingest", json={"provider": p, "data": data})
        if resp.status_code == 200:
            res = resp.json()
            print(f"[INGEST] {p:15} | Saturation: {res['saturation']:.4f} | Count: {res['linked_count']}")
        else:
            print(f"[ERROR] Failed to ingest {p}: {resp.text}")

    print("\n--- INITIATING SOVEREIGN DNA SYNTHESIS ---")
    resp = requests.post(f"{BASE_URL}/synthesize")
    if resp.status_code == 200:
        res = resp.json()
        print(f"[SUCCESS] DNA Synthesized: {res['dna']}")
        print(f"[STATE]   Final Saturation: {res['saturation']}")
    else:
        print(f"[ERROR] Synthesis failed: {resp.text}")

if __name__ == "__main__":
    ingest()
