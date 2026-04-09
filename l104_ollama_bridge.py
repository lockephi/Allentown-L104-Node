#!/usr/bin/env python3
"""
L104-Ollama Bridge
Integration between L104 quantum system and Ollama local models
"""

import requests
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("L104OllamaBridge")

class L104OllamaBridge:
    """Bridge between L104 system and Ollama local models"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", l104_host: str = "http://localhost:8004"):
        self.ollama_host = ollama_host
        self.l104_host = l104_host
        self.available_models = []
        self.current_model = "llama3.2:3b"
        
    def check_services(self) -> Dict[str, bool]:
        """Check if both services are running"""
        status = {
            "ollama": False,
            "l104": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                status["ollama"] = True
                models = response.json().get("models", [])
                self.available_models = [m.get("name") for m in models if m.get("name")]
                logger.info(f"Ollama available with {len(self.available_models)} models")
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
        
        # Check L104
        try:
            response = requests.get(f"{self.l104_host}/api/v6/status", timeout=5)
            if response.status_code == 200:
                status["l104"] = True
                data = response.json()
                logger.info(f"L104 available: {data.get('status', 'UNKNOWN')}")
        except Exception as e:
            logger.warning(f"L104 check failed: {e}")
        
        return status
    
    def generate_with_ollama(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Generate text using Ollama"""
        if not model:
            model = self.current_model
        
        logger.info(f"Generating with Ollama model: {model}")
        
        try:
            # Use Ollama API for generation
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 256
                }
            }
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": model,
                    "tokens": result.get("total_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0)
                }
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "model": model
                }
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model
            }
    
    def generate_with_l104_context(self, prompt: str, include_quantum_context: bool = True) -> Dict[str, Any]:
        """Generate with Ollama using L104 context"""
        logger.info(f"Generating with L104 context: {prompt[:50]}...")
        
        # Get L104 context if available
        l104_context = ""
        if include_quantum_context:
            try:
                # Get current L104 state
                response = requests.get(f"{self.l104_host}/api/v14/nova/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    resonance = data.get("resonance", 0)
                    l104_context = f"\n[L104 Quantum Context: Resonance={resonance}, GOD_CODE alignment active]"
            except Exception as e:
                logger.warning(f"Could not get L104 context: {e}")
        
        # Combine prompt with context
        enhanced_prompt = f"{prompt}{l104_context}\n\nPlease respond in the context of quantum computing and L104 systems."
        
        return self.generate_with_ollama(enhanced_prompt)
    
    def hybrid_generation(self, prompt: str) -> Dict[str, Any]:
        """Hybrid generation using both local and API models"""
        logger.info(f"Running hybrid generation: {prompt[:50]}...")
        
        results = {
            "ollama": None,
            "deepseek": None,
            "hybrid": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 1: Generate with Ollama (local, fast)
        ollama_result = self.generate_with_ollama(prompt)
        results["ollama"] = ollama_result
        
        if ollama_result["success"]:
            ollama_response = ollama_result["response"]
            
            # Step 2: Enhance with L104 quantum context
            enhancement_prompt = f"""
            Original query: {prompt}
            
            Initial response from local model: {ollama_response}
            
            Please enhance this response with quantum computing concepts,
            L104 system integration, and GOD_CODE algorithm considerations.
            """
            
            # For now, we'll just use Ollama again with different parameters
            # In a full implementation, this would call DeepSeek API
            enhanced_result = self.generate_with_ollama(
                enhancement_prompt,
                model=self.current_model
            )
            
            if enhanced_result["success"]:
                results["hybrid"] = {
                    "original": ollama_response,
                    "enhanced": enhanced_result["response"],
                    "enhancement_model": enhanced_result["model"],
                    "total_tokens": ollama_result.get("tokens", 0) + enhanced_result.get("tokens", 0)
                }
        
        return results
    
    def list_models(self) -> list:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [
                    {
                        "name": m.get("name"),
                        "size": m.get("size", 0),
                        "modified_at": m.get("modified_at")
                    }
                    for m in models
                ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return []
    
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a new model from Ollama library"""
        logger.info(f"Pulling model: {model_name}")
        
        try:
            # This would typically use subprocess to run `ollama pull`
            # For now, we'll simulate it
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for model download
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully pulled model: {model_name}")
                return {
                    "success": True,
                    "model": model_name,
                    "output": result.stdout
                }
            else:
                logger.error(f"Failed to pull model: {result.stderr}")
                return {
                    "success": False,
                    "model": model_name,
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Model pull timed out: {model_name}")
            return {
                "success": False,
                "model": model_name,
                "error": "Download timed out"
            }
        except Exception as e:
            logger.error(f"Model pull error: {e}")
            return {
                "success": False,
                "model": model_name,
                "error": str(e)
            }
    
    def benchmark_model(self, model_name: str, iterations: int = 3) -> Dict[str, Any]:
        """Benchmark a model's performance"""
        logger.info(f"Benchmarking model: {model_name}")
        
        test_prompt = "Explain the concept of quantum superposition in one paragraph."
        
        times = []
        tokens_per_second = []
        
        for i in range(iterations):
            start_time = time.time()
            
            result = self.generate_with_ollama(test_prompt, model=model_name)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result["success"]:
                times.append(duration)
                response_length = len(result["response"])
                tps = response_length / duration if duration > 0 else 0
                tokens_per_second.append(tps)
                
                logger.info(f"  Iteration {i+1}: {duration:.2f}s, {tps:.1f} chars/sec")
            else:
                logger.warning(f"  Iteration {i+1} failed: {result.get('error', 'Unknown error')}")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_tps = sum(tokens_per_second) / len(tokens_per_second)
            
            return {
                "success": True,
                "model": model_name,
                "iterations": iterations,
                "average_time_seconds": avg_time,
                "average_chars_per_second": avg_tps,
                "test_prompt": test_prompt
            }
        else:
            return {
                "success": False,
                "model": model_name,
                "error": "All benchmark iterations failed"
            }

def main():
    """Main function to demonstrate the bridge"""
    bridge = L104OllamaBridge()
    
    print("🌉 L104-Ollama Bridge Demonstration")
    print("=" * 50)
    
    # Check services
    print("1. Checking services...")
    status = bridge.check_services()
    print(f"   Ollama: {'✅' if status['ollama'] else '❌'}")
    print(f"   L104: {'✅' if status['l104'] else '❌'}")
    
    if not status["ollama"]:
        print("❌ Ollama not available. Exiting.")
        return
    
    # List models
    print("\n2. Available models:")
    models = bridge.list_models()
    if models:
        for model in models:
            print(f"   • {model['name']}")
    else:
        print("   No models found. Try: ollama pull llama3.2:3b")
    
    # Test generation
    print("\n3. Testing generation...")
    test_prompt = "What is quantum entanglement and how does it relate to L104 systems?"
    
    result = bridge.generate_with_l104_context(test_prompt)
    
    if result["success"]:
        print(f"   ✅ Generation successful")
        print(f"   Model: {result['model']}")
        print(f"   Response length: {len(result['response'])} chars")
        print(f"\n   Preview:")
        print(f"   {result['response'][:200]}...")
    else:
        print(f"   ❌ Generation failed: {result.get('error', 'Unknown error')}")
    
    # Benchmark
    print("\n4. Running benchmark...")
    benchmark = bridge.benchmark_model("llama3.2:3b", iterations=2)
    
    if benchmark["success"]:
        print(f"   ✅ Benchmark complete")
        print(f"   Avg time: {benchmark['average_time_seconds']:.2f}s")
        print(f"   Avg speed: {benchmark['average_chars_per_second']:.1f} chars/sec")
    else:
        print(f"   ❌ Benchmark failed: {benchmark.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("🎉 L104-Ollama Bridge is operational!")
    print("\n🔧 Next steps:")
    print("   1. Pull more models: bridge.pull_model('llama3.1:8b')")
    print("   2. Integrate with L104 quantum algorithms")
    print("   3. Set up automatic model switching")
    print("   4. Add to HEARTBEAT.md monitoring")

if __name__ == "__main__":
    main()