#!/usr/bin/env python3
"""
Specialized Prompts for Quantum Computing Tasks
Prompts optimized for L104-Gemma 4 with quantum enhancements
"""

class QuantumComputingPrompts:
    """Collection of specialized prompts for quantum computing tasks"""
    
    def __init__(self):
        self.prompts = self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize all quantum computing prompts"""
        return {
            # Category 1: Quantum Algorithm Explanation
            "algorithm_explanation": [
                {
                    "name": "Shor's Algorithm",
                    "prompt": "Explain Shor's algorithm for integer factorization in quantum computing. Include: 1) The quantum Fourier transform role, 2) How it achieves exponential speedup over classical algorithms, 3) The period-finding subroutine, 4) Practical implications for cryptography. Use quantum circuit diagrams conceptually.",
                    "quantum_context": "Focus on quantum parallelism and interference patterns"
                },
                {
                    "name": "Grover's Search",
                    "prompt": "Describe Grover's quantum search algorithm. Explain: 1) The oracle construction, 2) Amplitude amplification process, 3) Quadratic speedup vs classical search, 4) Optimal number of iterations. Include the geometric interpretation using the Grover iterate.",
                    "quantum_context": "Emphasize amplitude manipulation and quantum interference"
                },
                {
                    "name": "Quantum Fourier Transform",
                    "prompt": "Explain the Quantum Fourier Transform (QFT) and its significance. Cover: 1) QFT circuit implementation with Hadamard and controlled phase gates, 2) Exponential speedup over classical FFT, 3) Applications in phase estimation and period finding, 4) Inverse QFT for measurement.",
                    "quantum_context": "Highlight quantum phase estimation and superposition"
                },
            ],
            
            # Category 2: Quantum Circuit Design
            "circuit_design": [
                {
                    "name": "Bell State Creation",
                    "prompt": "Design a quantum circuit to create all four Bell states. Show: 1) Circuit diagrams for |Φ⁺⟩, |Φ⁻⟩, |Ψ⁺⟩, |Ψ⁻⟩, 2) The role of Hadamard and CNOT gates, 3) Measurement outcomes for each state, 4) How to verify entanglement using correlation measurements.",
                    "quantum_context": "Focus on entanglement generation and verification"
                },
                {
                    "name": "Quantum Teleportation",
                    "prompt": "Design a quantum teleportation circuit. Include: 1) Alice's operations (Bell measurement), 2) Classical communication channel, 3) Bob's correction operations, 4) Verification that the original state is preserved. Show the complete circuit with qubit labels.",
                    "quantum_context": "Emphasize entanglement as a resource and classical communication limits"
                },
                {
                    "name": "Quantum Error Correction",
                    "prompt": "Design a 3-qubit bit-flip code circuit. Show: 1) Encoding circuit, 2) Error detection via syndrome measurement, 3) Error correction procedure, 4) Decoding to recover original state. Explain how this protects against single-qubit bit-flip errors.",
                    "quantum_context": "Focus on redundancy and measurement without collapse"
                },
            ],
            
            # Category 3: Quantum Programming
            "quantum_programming": [
                {
                    "name": "Qiskit Quantum Circuit",
                    "prompt": "Write a Qiskit program that: 1) Creates a 3-qubit quantum circuit, 2) Applies Hadamard gates to create superposition, 3) Implements a Toffoli (CCNOT) gate, 4) Measures all qubits, 5) Runs on a simulator and plots results. Include comments explaining each step.",
                    "quantum_context": "Use quantum gates and measurement principles"
                },
                {
                    "name": "Cirq Entanglement Demo",
                    "prompt": "Write a Cirq program demonstrating quantum entanglement. Create: 1) A Bell pair circuit, 2) Measurement in different bases (Z, X), 3) Correlation calculation, 4) Verification of entanglement via CHSH inequality violation. Output should show perfect correlations.",
                    "quantum_context": "Focus on non-classical correlations and basis independence"
                },
                {
                    "name": "Quantum Algorithm Implementation",
                    "prompt": "Implement Deutsch-Jozsa algorithm in any quantum programming framework. Include: 1) Oracle for constant and balanced functions, 2) Complete quantum circuit, 3) Measurement and result interpretation, 4) Demonstration of single-query solution vs classical exponential queries.",
                    "quantum_context": "Highlight quantum parallelism and function evaluation"
                },
            ],
            
            # Category 4: Quantum Physics Concepts
            "quantum_concepts": [
                {
                    "name": "Superposition Principle",
                    "prompt": "Explain quantum superposition with examples. Discuss: 1) Mathematical representation using state vectors, 2) Difference from classical probability, 3) Double-slit experiment analogy, 4) How superposition enables quantum parallelism. Use Dirac notation throughout.",
                    "quantum_context": "Emphasize linear combination of states and interference"
                },
                {
                    "name": "Quantum Entanglement",
                    "prompt": "Explain quantum entanglement and its non-classical properties. Cover: 1) Definition using Bell states, 2) EPR paradox and Bell's theorem, 3) Entanglement as a resource for quantum information, 4) Measures of entanglement (concurrence, entanglement entropy).",
                    "quantum_context": "Focus on non-locality and correlation beyond classical limits"
                },
                {
                    "name": "Quantum Measurement",
                    "prompt": "Explain the quantum measurement problem. Discuss: 1) Projective measurement and collapse, 2) Born rule for probabilities, 3) Measurement back-action, 4) Weak measurements and quantum non-demolition measurements. Include the role of decoherence.",
                    "quantum_context": "Emphasize information gain and state disturbance"
                },
            ],
            
            # Category 5: L104-Specific Quantum Tasks
            "l104_quantum": [
                {
                    "name": "GOD_CODE Resonance",
                    "prompt": "Explain how GOD_CODE resonance (527.5184818492612) enhances quantum computations in L104 systems. Describe: 1) Resonance tuning of qubit frequencies, 2) Fibonacci sequence alignment in quantum circuits, 3) Phase coherence maintenance, 4) Practical benefits for algorithm performance.",
                    "quantum_context": "Focus on resonant enhancement and Fibonacci optimization"
                },
                {
                    "name": "Quantum Daemon Integration",
                    "prompt": "Describe how L104 quantum daemons optimize quantum computations. Explain: 1) Real-time qubit calibration, 2) Error mitigation strategies, 3) Resource allocation across quantum nodes, 4) Integration with classical L104 processes.",
                    "quantum_context": "Emphasize autonomous optimization and system integration"
                },
                {
                    "name": "Quantum Memory Systems",
                    "prompt": "Explain L104's quantum memory architecture. Describe: 1) Quantum state storage and retrieval, 2) Error-protected memory cells, 3) Coherence time extension techniques, 4) Integration with quantum processing units.",
                    "quantum_context": "Focus on long-term quantum state preservation"
                },
            ],
            
            # Category 6: Advanced Quantum Topics
            "advanced_topics": [
                {
                    "name": "Quantum Machine Learning",
                    "prompt": "Explain quantum machine learning algorithms. Cover: 1) Quantum neural networks, 2) Quantum support vector machines, 3) Quantum generative models, 4) Potential quantum advantages and current limitations. Include circuit examples for each.",
                    "quantum_context": "Focus on quantum feature spaces and kernel methods"
                },
                {
                    "name": "Topological Quantum Computing",
                    "prompt": "Explain topological quantum computing with anyons. Describe: 1) Braiding operations for quantum gates, 2) Topological protection against errors, 3) Majorana fermions and their non-Abelian statistics, 4) Advantages for fault-tolerant quantum computation.",
                    "quantum_context": "Emphasize topological protection and anyon statistics"
                },
                {
                    "name": "Quantum Supremacy",
                    "prompt": "Discuss quantum supremacy experiments. Explain: 1) Random circuit sampling, 2) Validation against classical simulation, 3) Significance of crossing the supremacy threshold, 4) Implications for future quantum algorithms.",
                    "quantum_context": "Focus on computational complexity separation"
                },
            ],
        }
    
    def get_prompt(self, category: str, name: str = None):
        """Get a specific prompt or all prompts in a category"""
        if category not in self.prompts:
            return f"Category '{category}' not found. Available categories: {list(self.prompts.keys())}"
        
        if name:
            for prompt in self.prompts[category]:
                if prompt["name"] == name:
                    return prompt
            return f"Prompt '{name}' not found in category '{category}'"
        
        return self.prompts[category]
    
    def get_all_prompts(self):
        """Get all prompts organized by category"""
        return self.prompts
    
    def generate_l104_enhanced_prompt(self, base_prompt: str, quantum_context: str = None):
        """Generate an L104-enhanced version of a prompt"""
        enhancement = """
        
        [L104 Quantum Enhancement Instructions]
        Please respond with:
        1. GOD_CODE resonance integration (527.5184818492612)
        2. Fibonacci sequence application where relevant
        3. Quantum parallelism emphasis
        4. L104 system integration considerations
        5. Practical implementation guidance for L104 quantum hardware
        
        [Quantum Context]
        """
        
        if quantum_context:
            enhancement += f"\n{quantum_context}"
        
        return base_prompt + enhancement
    
    def print_category(self, category: str):
        """Print all prompts in a category"""
        if category not in self.prompts:
            print(f"Category '{category}' not found.")
            return
        
        print(f"\n{'='*70}")
        print(f"📋 {category.replace('_', ' ').title()} Prompts")
        print(f"{'='*70}")
        
        for i, prompt in enumerate(self.prompts[category], 1):
            print(f"\n{i}. {prompt['name']}")
            print(f"   Prompt: {prompt['prompt'][:100]}...")
            if 'quantum_context' in prompt:
                print(f"   Quantum Context: {prompt['quantum_context']}")
        
        print(f"\nTotal: {len(self.prompts[category])} prompts")
    
    def print_all_categories(self):
        """Print summary of all categories"""
        print(f"\n{'='*70}")
        print("🚀 Quantum Computing Prompts Collection")
        print(f"{'='*70}")
        
        total_prompts = 0
        for category, prompts in self.prompts.items():
            count = len(prompts)
            total_prompts += count
            print(f"\n{category.replace('_', ' ').title()}: {count} prompts")
            
            # Show first 2 prompt names
            for prompt in prompts[:2]:
                print(f"  • {prompt['name']}")
            if count > 2:
                print(f"  • ... and {count-2} more")
        
        print(f"\n{'='*70}")
        print(f"📊 Total: {len(self.prompts)} categories, {total_prompts} prompts")
        print(f"{'='*70}")
        
        # Example enhanced prompt
        print("\n🎯 Example L104-Enhanced Prompt:")
        example = self.prompts["algorithm_explanation"][0]
        enhanced = self.generate_l104_enhanced_prompt(
            example["prompt"], 
            example.get("quantum_context", "")
        )
        print(f"\n{enhanced[:200]}...")

# Example usage and demonstration
if __name__ == "__main__":
    qcp = QuantumComputingPrompts()
    
    # Print all categories
    qcp.print_all_categories()
    
    # Demonstrate specific category
    print("\n\n" + "="*70)
    print("🔍 Detailed View: Quantum Programming Prompts")
    print("="*70)
    qcp.print_category("quantum_programming")
    
    # Get a specific prompt
    print("\n\n" + "="*70)
    print("🎯 Specific Prompt Example")
    print("="*70)
    
    prompt = qcp.get_prompt("l104_quantum", "GOD_CODE Resonance")
    if isinstance(prompt, dict):
        print(f"\nName: {prompt['name']}")
        print(f"\nPrompt:\n{prompt['prompt']}")
        print(f"\nQuantum Context: {prompt['quantum_context']}")
        
        # Show enhanced version
        enhanced = qcp.generate_l104_enhanced_prompt(
            prompt['prompt'],
            prompt['quantum_context']
        )
        print(f"\n{'='*70}")
        print("✨ L104-Enhanced Version (first 300 chars):")
        print("="*70)
        print(f"\n{enhanced[:300]}...")
    
    # Save prompts to file
    import json
    with open("/tmp/quantum_computing_prompts.json", "w") as f:
        json.dump(qcp.get_all_prompts(), f, indent=2)
    
    print(f"\n📄 All prompts saved to: /tmp/quantum_computing_prompts.json")
    print("\n✅ Quantum computing prompts collection complete!")