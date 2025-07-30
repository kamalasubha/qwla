from state import State
from math import sqrt, pi, gcd, ceil, log2
from cmath import exp
from typing import Optional

def modular_exponentiation(state: State, a: int, N: int, control: int, target_start: int, n_target: int):
    print(f"-> Applying modular exponentiation a={a}, N={N}")
    for i in range(n_target):
        power = 2 ** (2 ** i)
        result = pow(power, 1, N)
        if any(b[0][control] for b in state.state):  # Check if control qubit can be 1
            binary = format(result, f"0{n_target}b")
            for j, bit in enumerate(binary[::-1]):
                if bit == '1':
                    state.cx(control, target_start + j)
    return state

def inverse_qft(state: State, n: int):
    print(f"-> Applying inverse QFT on {n} qubits")
    # Apply swaps in reverse order
    for i in range(n // 2):
        state.swap(i, n - 1 - i)
    # Apply controlled phase rotations and Hadamard gates
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            theta = -pi / (2 ** (i - j))
            state.cp(j, i, theta)
        state.h(i)
    return state

def continued_fraction(n: int, d: int) -> list:
    """Compute continued fraction expansion of n/d."""
    cf = []
    while d != 0:
        q = n // d
        cf.append(q)
        n, d = d, n % d
    return cf

def convergents(cf: list) -> list:
    """Compute convergents of continued fraction."""
    numerators = [0, 1]
    denominators = [1, 0]
    for q in cf:
        numerators.append(q * numerators[-1] + numerators[-2])
        denominators.append(q * denominators[-1] + denominators[-2])
    return [(n, d) for n, d in zip(numerators[2:], denominators[2:])]

def find_period(a: int, N: int, measured: int, q: int) -> Optional[int]:
    """Find period from measured value using continued fraction."""
    if measured == 0:
        return None  # Zero measurement cannot yield a period
    fraction = measured / q
    print(f"Testing fraction: {measured}/{q} = {fraction:.4f}")
    cf = continued_fraction(measured, q)
    for s, r in convergents(cf):
        if r > N:
            break
        # Verify if r is the period
        if pow(a, r, N) == 1:
            return r
    return None



def shor_algorithm(N: int = 15, a: int = 2, max_attempts: int = 50):
    # Check if a and N are coprime
    if gcd(a, N) != 1:
        print(f"{a} and {N} are not coprime. {gcd(a, N)} is a factor of {N}.")
        return
    
    # Dynamically set number of qubits
    n_target = ceil(log2(N))
    n_count = 2 * n_target
    n_qubits = n_count + n_target
    n_bits = n_count
    q = 2 ** n_count  # Size of the counting register

    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt} of {max_attempts}:")
        state = State(n_qubits, n_bits)
        print("Initial state:")
        print(state)
        for i in range(n_count):
            state.h(i)
        print("\nAfter Hadamard gates:")
        print(state)
        for i in range(n_count):
            state = modular_exponentiation(state, a, N, i, n_count, n_target)
        print("\nAfter modular exponentiation:")
        print(state)
        state = inverse_qft(state, n_count)
        print("\nAfter inverse QFT:")
        print(state)
        for i in range(n_count):
            state.measure(i, i)
        print("\nAfter measurement:")
        print(state)
        measured = int(''.join(map(str, state.cbits)), 2)
        print(f"\nMeasured value: {measured}")
        
        # Find period using continued fraction
        period = find_period(a, N, measured, q)
        if period is None:
            print("Failed to find period. Trying again.")
            continue
        
        print(f"Found period: {period}")
        if period % 2 == 0:
            x = pow(a, period // 2, N)
            factor1 = gcd(x - 1, N)
            factor2 = gcd(x + 1, N)
            factors = [f for f in [factor1, factor2] if 1 < f < N]
            print ("attempt", attempt)
            if factors:
                print(f"Factors of {N}: {factors}")
                return
            else:
                print("No non-trivial factors found. Trying again.")
        else:
            print("Period is odd, trying a different measurement.")
    
    print(f"\nFailed to find period after {max_attempts} attempts.")

if __name__ == "__main__":
    test_cases = [
        #(15, 2),  # N=15, a=2
        #(15, 7),  # N=15, a=7
        (21, 2),  # N=21, a=2
        #(35, 3),  # N=35, a=3
        #(15, 5),  # N=15, a=5 (invalid, not coprime)
    ]
    for N, a in test_cases:
        print(f"\nTesting N={N}, a={a}")
        shor_algorithm(N=N, a=a)