import qiskit_ibm_runtime
print("runtime:", qiskit_ibm_runtime.__version__)
import qiskit
print("qiskit:", qiskit.__version__)

import inspect
from qiskit_ibm_runtime import SamplerV2
sig = inspect.signature(SamplerV2.__init__)
print("SamplerV2.__init__ params:", list(sig.parameters.keys()))
sig2 = inspect.signature(SamplerV2.run)
print("SamplerV2.run params:", list(sig2.parameters.keys()))

# Check what 'mode' expects
print("\nSamplerV2.__init__ full sig:", sig)
print("SamplerV2.run full sig:", sig2)

# Check result structure
print("\nChecking data attribute names...")
from qiskit_ibm_runtime import QiskitRuntimeService
print("QiskitRuntimeService channels available - checking help...")
sig3 = inspect.signature(QiskitRuntimeService.__init__)
print("QiskitRuntimeService.__init__:", sig3)
