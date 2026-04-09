"""
Quantum Oracle Implementation for Deutsch-Jozsa Algorithm
L104 Quantum Processing Unit (QPU) Interface
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

class DeutschJozsaOracle:
    """Configurable quantum oracle for n-qubit D-J problem"""
    
    def __init__(self, n_qubits: int = 3, oracle_type: str = 'balanced'):
        """
        Initialize oracle with specified behavior
        
        Args:
            n_qubits: Number of input qubits (n)
            oracle_type: 'constant' (all 0 or all 1) or 'balanced'
        """
        self.n = n_qubits
        self.type = oracle_type
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Ensure parameters are within L104 QPU specifications"""
        if self.n < 1 or self.n > 10:
            raise ValueError(f"n_qubits {self.n} outside L104 operational range (1-10)")
        if self.type not in ['constant', 'balanced']:
            raise ValueError("oracle_type must be 'constant' or 'balanced'")
    
    def build_constant_oracle(self) -> QuantumCircuit:
        """Constant oracle: f(x) = 0 or f(x) = 1"""
        qr = QuantumRegister(self.n + 1)  # n input + 1 ancilla
        circuit = QuantumCircuit(qr, name='constant_oracle')
        
        # Constant-0: do nothing (identity)
        # Constant-1: flip ancilla
        import random
        if random.random() > 0.5:
            circuit.x(self.n)  # f(x) = 1
            
        return circuit
    
    def build_balanced_oracle(self) -> QuantumCircuit:
        """Balanced oracle: f(x) = 1 for exactly half of inputs"""
        qr = QuantumRegister(self.n + 1)
        circuit = QuantumCircuit(qr, name='balanced_oracle')
        
        # Create balanced function by applying CNOTs to random half of states
        # For demonstration: XOR of first qubit onto ancilla
        circuit.cx(0, self.n)
        
        # Add random phase flips to ensure true balance
        for i in range(1, self.n):
            if np.random.random() > 0.5:
                circuit.cx(i, self.n)
                
        return circuit
    
    def build_circuit(self) -> QuantumCircuit:
        """Construct complete Deutsch-Jozsa circuit"""
        if self.type == 'constant':
            oracle = self.build_constant_oracle()
        else:
            oracle = self.build_balanced_oracle()
        
        # Main circuit
        qr = QuantumRegister(self.n + 1)
        cr = ClassicalRegister(self.n)
        circuit = QuantumCircuit(qr, cr)
        
        # Initialize ancilla to |1⟩ for phase kickback
        circuit.x(self.n)
        
        # Apply Hadamard to all qubits
        for i in range(self.n + 1):
            circuit.h(i)
        
        # Append oracle
        circuit.append(oracle, range(self.n + 1))
        
        # Apply Hadamard to input qubits
        for i in range(self.n):
            circuit.h(i)
        
        # Measure input qubits
        for i in range(self.n):
            circuit.measure(i, i)
            
        return circuit
    
    def execute(self, shots: int = 1024) -> dict:
        """Execute on L104 QPU simulator"""
        circuit = self.build_circuit()
        
        # Use L104's optimized simulator
        backend = AerSimulator(method='statevector')
        result = backend.run(circuit, shots=shots).result()
        counts = result.get_counts()
        
        return {
            'counts': counts,
            'circuit_depth': circuit.depth(),
            'gate_count': circuit.count_ops(),
            'oracle_type': self.type,
            'determination': self.analyze_results(counts)
        }
    
    def analyze_results(self, counts: dict) -> str:
        """Analyze measurement results to determine function type"""
        # All zeros measurement indicates constant function
        zero_state = '0' * self.n
        if zero_state in counts and counts[zero_state] == sum(counts.values()):
            return 'constant'
        else:
            return 'balanced'

# Example usage
if __name__ == '__main__':
    print("Testing Deutsch-Jozsa Oracle on L104 QPU")
    
    # Test constant oracle
    oracle_const = DeutschJozsaOracle(n_qubits=3, oracle_type='constant')
    result_const = oracle_const.execute(shots=1024)
    print(f"\nConstant Oracle Result: {result_const['determination']}")
    
    # Test balanced oracle  
    oracle_bal = DeutschJozsaOracle(n_qubits=3, oracle_type='balanced')
    result_bal = oracle_bal.execute(shots=1024)
    print(f"Balanced Oracle Result: {result_bal['determination']}")