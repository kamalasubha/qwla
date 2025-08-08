from state import State
from math import acos, sqrt

def quantum_random_bits(n_bits: int) -> str:
    # Initialize quantum state with n_bits qubits and classical bits
    state = State(n_bits, n_bits)
    
    # Apply Hadamard gates to create superposition
    for i in range(n_bits):
        state.h(i)
    
    # Measure all qubits
    for i in range(n_bits):
        state.measure(i, i)
    
    # Return the classical bits as a string
    return ''.join(str(bit) for bit in state.cbits)

def bell_state(n_bits: int):
    # Quantum entanglement (Bellstate Generation)
    state = State(n_bits, n_bits)
    for i in range(0, n_bits, 2):
        state.h(i).cx(i, i + 1)
    # Measure all qubits
    for i in range(n_bits):
        state.measure(i, i)
    #print(state)
    return ''.join(str(bit) for bit in state.cbits)

def quantum_sqrt(a: float, t: float = 0.5, C: float = 1.0) -> tuple:
    if a <= 0:
        raise ValueError("Input a must be positive")
    
    # Initialize state: 1 state qubit, 2 phase qubits, 1 ancilla
    state = State(4,4)
    
    # Apply Hadamard to phase estimation qubits
    state.h(1).h(2)
    
    # Unitary: phase gate with φ = 2 * arccos(1/sqrt(a))
    phi = 2 * acos(1 / sqrt(a))
    # Controlled-U: apply phase φ to state qubit
    state.cx(1, 0).cp(0, 0, phi * t).cx(1, 0)  # First controlled phase
    state.cx(2, 0).cp(0, 0, phi * t * 2).cx(2, 0)  # Second controlled phase (2φ)
    
    # Inverse QFT (2 qubits)
    state.h(2).cp(2, 1, -pi/2).h(1)
    
    # Step 3: Controlled rotation to encode sqrt(a)
    # Approximate sqrt(a) via phase: φ = 2 * arccos(1/sqrt(a)) => sqrt(a) = 1/cos(φ/2)
    theta = 2 * acos(C / sqrt(a))
    state.cx(1, 3).ry(3, theta).cx(1, 3)
    state.cx(2, 3).ry(3, theta).cx(2, 3)
    
    # Step 4: Inverse QPE
    state.h(1).cp(2, 1, pi/2).h(2)
    
    # Step 5: Measure qubits
    state.measure(3, 3)  # Ancilla (post-select on 1)
    state.measure(1, 1).measure(2, 2)  # Phase qubits
    state.measure(0, 0)  # State qubit for verification
    
    # Estimate sqrt(a) from phase register
    phase_bits = state.cbits[1:3]
    phase = (phase_bits[0] + 2 * phase_bits[1]) / 4.0  # 2-bit phase estimate
    estimated_sqrt = 1 / cos(pi * phase)
    
    print("\nFinal state:")
    print(state)
    print(f"Phase register measurement: {phase_bits}")
    print(f"Estimated phase: {phase:.3f}")
    print(f"Estimated square root of {a}: {estimated_sqrt:.3f}")
    
    return state, estimated_sqrt

if __name__ == "__main__":
    # Generate 8 random bits
    random_bits = quantum_random_bits(8)
    print(f"\nGenerated random bits: {random_bits}")
    # Convert to integer for demonstration
    random_number = int(random_bits, 2)
    print(f"Random number (decimal): {random_number}")
    print(f"\n")
    bell_number = int(bell_state(8), 2)
    print(f"Bell bits: {bell_number}")
    a = 9.0
    state, sqrt_estimate = quantum_sqrt(a, t=0.5, C=1.0)
    print(f"Actual square root of {a}: {sqrt(a):.3f}")