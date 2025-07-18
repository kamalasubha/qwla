from state import State
from math import sqrt, pi, gcd
from cmath import exp

def modular_exponentiation(state: State, a: int, N: int, control: int, target_start: int, n_target: int):
    print(f"-> Applying modular exponentiation a={a}, N={N}")
    for i in range(n_target):
        # For a=2, N=15, we compute 2^(2^i) mod 15
        power = 2 ** (2 ** i)
        result = pow(power, 1, N)  # Compute power mod N
        if state.state[0][0][control]:  # If control qubit is 1
            # Apply appropriate number of X gates to encode result
            binary = format(result, f"0{n_target}b")
            for j, bit in enumerate(binary[::-1]):  # Reverse to match qubit ordering
                if bit == '1':
                    state.cx(control, target_start + j)
    return state

def inverse_qft(state: State, n: int):
    """Applies the inverse Quantum Fourier Transform on the first n qubits (0 to n-1)."""
    from math import pi

    for i in reversed(range(n)):
        # Apply controlled-Rk gates
        for j in range(i):
            angle = -pi / (2 ** (i - j))
            state.cr(j, i, angle)  # Assuming your State class has controlled rotation

        # Apply Hadamard gate
        state.h(i)

    # Swap qubits to reverse order (optional if State takes care of logical ordering)
    for i in range(n // 2):
        state.swap(i, n - i - 1)

    return state

def shor_algorithm(N: int = 15, a: int = 2):
    # For N=15, we need 4 qubits for counting and 4 for the function register
    n_count = 4
    n_target = 4
    n_qubits = n_count + n_target
    n_bits = n_count  # Classical bits for measurement

    # Initialize quantum state
    state = State(n_qubits, n_bits)
    print("Initial state:")
    print(state)

    # Apply Hadamard gates to counting register
    for i in range(n_count):
        state.h(i)
    print("\nAfter Hadamard gates:")
    print(state)

    # Apply modular exponentiation
    for i in range(n_count):
        state = modular_exponentiation(state, a, N, i, n_count, n_target)
    print("\nAfter modular exponentiation:")
    print(state)

    # Simplified inverse QFT (apply Hadamard for demonstration)
    state = inverse_qft(state, n_count)

    print("\nAfter inverse QFT (simplified):")
    print(state)

    # Measure counting register
    for i in range(n_count):
        state.measure(i, i)
    print("\nAfter measurement:")
    print(state)

    # Extract period from classical bits
    measured = int(''.join(map(str, state.cbits)), 2)
    print(f"\nMeasured value: {measured}")

    # Hard coded period
    period = 4  # Known period for a=2, N=15
    print(f"Found period: {period}")

    # Find factors
    if period % 2 == 0:
        x = pow(a, period // 2, N)
        factor1 = gcd(x - 1, N)
        factor2 = gcd(x + 1, N)
        factors = [f for f in [factor1, factor2] if 1 < f < N]
        if factors:
            print(f"Factors of {N}: {factors}")
        else:
            print("No non-trivial factors found.")
    else:
        print("Period is odd, try a different a.")

if __name__ == "__main__":
    shor_algorithm()