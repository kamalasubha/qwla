import time
import math
import random
from typing import List, Tuple, Optional
from state import State

class GroverSearchComparison:
    def __init__(self):
        self.results = []
    
    # Classical Search Methods
    
    def linear_search(self, database: List[int], target: int) -> Tuple[Optional[int], int]:
        """Classical linear search - O(N) complexity"""
        comparisons = 0
        for i, item in enumerate(database):
            comparisons += 1
            if item == target:
                return i, comparisons
        return None, comparisons
    
    def random_search(self, database: List[int], target: int, max_attempts: int = None) -> Tuple[Optional[int], int]:
        """Random search strategy"""
        if max_attempts is None:
            max_attempts = len(database)
        
        attempts = 0
        checked = set()
        
        while attempts < max_attempts and len(checked) < len(database):
            idx = random.randint(0, len(database) - 1)
            attempts += 1
            
            if idx not in checked:
                checked.add(idx)
                if database[idx] == target:
                    return idx, attempts
        
        return None, attempts
    
    # Quantum Grover's Algorithm
    
    def create_oracle(self, state: State, target_idx: int, n_qubits: int):
        """Create oracle that flips phase of target state"""
        # Convert target index to binary representation
        target_binary = format(target_idx, f'0{n_qubits}b')
        
        # Apply X gates to qubits that should be 0 in target state
        for i, bit in enumerate(target_binary):
            if bit == '0':
                state.x(i)
        
        # Apply controlled-Z gate (using multiple controlled operations)
        if n_qubits == 1:
            state.s(0).s(0)  # Z gate = S^2
        elif n_qubits == 2:
            state.cx(0, 1).s(1).cx(0, 1)
        else:
            # Multi-controlled Z gate approximation
            # For simplicity, using a cascade of controlled operations
            for i in range(n_qubits - 1):
                state.cx(i, n_qubits - 1)
            state.s(n_qubits - 1).s(n_qubits - 1)  # Z gate
            for i in range(n_qubits - 2, -1, -1):
                state.cx(i, n_qubits - 1)
        
        # Undo X gates
        for i, bit in enumerate(target_binary):
            if bit == '0':
                state.x(i)
    
    def diffusion_operator(self, state: State, n_qubits: int):
        """Apply diffusion operator (inversion about average)"""
        # H gates
        for i in range(n_qubits):
            state.h(i)
        
        # Apply oracle for |00...0⟩ state (flip phase of all-zero state)
        for i in range(n_qubits):
            state.x(i)
        
        # Multi-controlled Z gate
        if n_qubits == 1:
            state.s(0).s(0)  # Z gate
        elif n_qubits == 2:
            state.cx(0, 1).s(1).cx(0, 1)
        else:
            for i in range(n_qubits - 1):
                state.cx(i, n_qubits - 1)
            state.s(n_qubits - 1).s(n_qubits - 1)  # Z gate
            for i in range(n_qubits - 2, -1, -1):
                state.cx(i, n_qubits - 1)
        
        # Undo X gates
        for i in range(n_qubits):
            state.x(i)
        
        # H gates
        for i in range(n_qubits):
            state.h(i)
    
    def grovers_algorithm(self, database_size: int, target_idx: int) -> Tuple[Optional[int], int]:
        if database_size <= 0 or target_idx >= database_size:
            return None, 0
        
        # Calculate number of qubits needed
        n_qubits = math.ceil(math.log2(database_size))
        
        # Calculate optimal number of iterations
        optimal_iterations = math.floor(math.pi / 4 * math.sqrt(2 ** n_qubits))
        
        # Initialize quantum state
        state = State(n_qubits, n_qubits)
        
        # Create equal superposition
        for i in range(n_qubits):
            state.h(i)
        
        
        # Apply Grover iterations
        for iteration in range(optimal_iterations):
            
            # Apply oracle
            self.create_oracle(state, target_idx, n_qubits)
            
            # Apply diffusion operator
            self.diffusion_operator(state, n_qubits)
        
        # Measure all qubits
        for i in range(n_qubits):
            state.measure(i, i)
        
        # Convert measurement result to index
        measured_idx = 0
        for i in range(n_qubits):
            measured_idx += state.cbits[i] * (2 ** i)
        
        
        # Check if we found the target (within valid database range)
        if measured_idx < database_size and measured_idx == target_idx:
            return measured_idx, optimal_iterations
        elif measured_idx < database_size:
            # Found a valid index but not the target
            return measured_idx, optimal_iterations
        else:
            # Measured index outside database range
            return None, optimal_iterations
    
    # Benchmarking and Comparison
    
    def benchmark_search_algorithms(self, database_sizes: List[int], num_trials: int = 10):
        """Benchmark classical vs quantum search algorithms"""
        print("=" * 80)
        print("GROVER'S ALGORITHM vs CLASSICAL SEARCH BENCHMARK")
        print("=" * 80)
        
        results = []
        
        for db_size in database_sizes:
            print(f"\nTesting database size: {db_size}")
            print("-" * 50)
            
            # Create test database
            database = list(range(db_size))
            
            # Classical search results
            linear_times = []
            linear_comparisons = []
            random_times = []
            random_comparisons = []
            
            # Quantum search results
            quantum_times = []
            quantum_iterations = []
            quantum_success_rate = 0
            
            for trial in range(num_trials):
                # Choose random target
                target_idx = random.randint(0, db_size - 1)
                target_value = database[target_idx]
                
                # Classical Linear Search
                start_time = time.time()
                found_idx, comparisons = self.linear_search(database, target_value)
                linear_time = time.time() - start_time
                linear_times.append(linear_time)
                linear_comparisons.append(comparisons)
                
                # Classical Random Search
                start_time = time.time()
                found_idx, attempts = self.random_search(database, target_value, max_attempts=db_size)
                random_time = time.time() - start_time
                random_times.append(random_time)
                random_comparisons.append(attempts)
                
                # Quantum Grover's Search
                start_time = time.time()
                found_idx, iterations = self.grovers_algorithm(db_size, target_idx)
                quantum_time = time.time() - start_time
                quantum_times.append(quantum_time)
                quantum_iterations.append(iterations)
                
                if found_idx == target_idx:
                    quantum_success_rate += 1
            
            # Calculate averages
            avg_linear_time = sum(linear_times) / num_trials
            avg_linear_comparisons = sum(linear_comparisons) / num_trials
            avg_random_time = sum(random_times) / num_trials
            avg_random_comparisons = sum(random_comparisons) / num_trials
            avg_quantum_time = sum(quantum_times) / num_trials
            avg_quantum_iterations = sum(quantum_iterations) / num_trials
            quantum_success_rate = quantum_success_rate / num_trials
            
            # Print results
            print(f"Linear Search:")
            print(f"  Average time: {avg_linear_time:.6f}s")
            print(f"  Average comparisons: {avg_linear_comparisons:.1f}")
            print(f"  Complexity: O(N) = O({db_size})")
            
            print(f"Random Search:")
            print(f"  Average time: {avg_random_time:.6f}s")
            print(f"  Average attempts: {avg_random_comparisons:.1f}")
            
            print(f"Grover's Search:")
            print(f"  Average time: {avg_quantum_time:.6f}s")
            print(f"  Average iterations: {avg_quantum_iterations:.1f}")
            print(f"  Expected iterations: {math.floor(math.pi / 4 * math.sqrt(db_size))}")
            print(f"  Success rate: {quantum_success_rate:.1%}")
            print(f"  Complexity: O(√N) = O({math.sqrt(db_size):.1f})")
            
            # Calculate theoretical speedup
            theoretical_speedup = db_size / math.sqrt(db_size)
            actual_speedup = avg_linear_comparisons / avg_quantum_iterations if avg_quantum_iterations > 0 else 0
            
            print(f"Theoretical Speedup: {theoretical_speedup:.2f}x")
            print(f"Actual Speedup: {actual_speedup:.2f}x")
            
            results.append({
                'database_size': db_size,
                'linear_comparisons': avg_linear_comparisons,
                'quantum_iterations': avg_quantum_iterations,
                'theoretical_speedup': theoretical_speedup,
                'actual_speedup': actual_speedup,
                'quantum_success_rate': quantum_success_rate
            })
        
        return results
    
    def demonstrate_grovers_algorithm(self):
        """Demonstrate Grover's algorithm with detailed output"""
        print("=" * 60)
        print("GROVER'S ALGORITHM DEMONSTRATION")
        print("=" * 60)
        
        # Small example for detailed demonstration
        database_size = 8
        target_idx = 5
        
        print(f"Searching for index {target_idx} in database of size {database_size}")
        print(f"Classical linear search would need up to {database_size} comparisons")
        print(f"Grover's algorithm needs ~{math.floor(math.pi / 4 * math.sqrt(database_size))} iterations")
        
        print("\nRunning Grover's algorithm with detailed output:")
        found_idx, iterations = self.grovers_algorithm(database_size, target_idx)
        
        if found_idx == target_idx:
            print(f"\nSUCCESS! Found target at index {found_idx}")
        else:
            print(f"\nTarget not found. Measured index: {found_idx}")
        
        print(f"Iterations used: {iterations}")
        
        # Compare with classical
        database = list(range(database_size))
        target_value = database[target_idx]
        
        found_classical, comparisons = self.linear_search(database, target_value)
        print(f"Classical search comparisons: {comparisons}")
        
        speedup = comparisons / iterations if iterations > 0 else 0
        print(f"Speedup achieved: {speedup:.2f}x")

def main():
    """Main demonstration and benchmarking"""
    random.seed(42)  # For reproducibility
    
    comparison = GroverSearchComparison()
    
    # Detailed demonstration
    comparison.demonstrate_grovers_algorithm()
    
    # Benchmark different database sizes
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BENCHMARK")
    print("="*80)
    
    database_sizes = [4, 8, 16, 32]  # Limited sizes for quantum simulation
    results = comparison.benchmark_search_algorithms(database_sizes, num_trials=5)
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    print(f"{'Size':<8} {'Linear':<8} {'Quantum':<8} {'Theoretical':<12} {'Actual':<8} {'Success':<8}")
    print(f"{'':>8} {'Steps':<8} {'Steps':<8} {'Speedup':<12} {'Speedup':<8} {'Rate':<8}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['database_size']:<8} "
              f"{result['linear_comparisons']:<8.1f} "
              f"{result['quantum_iterations']:<8.1f} "
              f"{result['theoretical_speedup']:<12.2f} "
              f"{result['actual_speedup']:<8.2f} "
              f"{result['quantum_success_rate']:<8.1%}")

if __name__ == "__main__":
    main()