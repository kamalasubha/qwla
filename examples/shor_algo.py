from state import State

def shor_algorithm(N: int, a: int, n_qubits: int):
    m_qubits = N.bit_length()  # output register size
    state_1 = state_1(n_qubits + m_qubits)

    # Apply Hadamard to input qubits
    for i in range(n_qubits):
        state_1.h(i)

    # Apply modular exponentiation: |x⟩|0⟩ → |x⟩|a^x mod N⟩
    def modular_exp_mock(x):
        return pow(a, x, N)

    # Replace amplitudes based on mapping x → f(x)
    new_state_1 = []
    for (b, amp) in state_1.state_1:
        x = int(b[:n_qubits].to01(), 2)
        f_x = modular_exp_mock(x)
        f_x_bits = bitarray(format(f_x, f"0{m_qubits}b"))
        full_bits = b[:n_qubits] + f_x_bits
        new_state_1.append((full_bits, amp))
    state_1.state_1 = seq(new_state_1)

    # Apply inverse QFT to input register - Need to fix  gate
    # inverse_qft(state_1, n_qubits)

    # Measure input qubits
    for i in range(n_qubits):
        state_1.measure(i)

    print(state_1)
    return state_1.cbits  # binary measurement result = phase → used to estimate period

