import re
import csv

def parse_shor_results(filename):
    """
    Parse Shor's algorithm benchmark results and extract performance data
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    results = []
    
    # Find all sections that start with "Factoring N ="
    # Use a more comprehensive pattern to capture entire sections
    pattern = r'Factoring N = (\d+) \((\d+) bits\)(.*?)(?=Factoring N =|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        n = int(match[0])
        bits = int(match[1])
        section = match[2]
        
        # Debug: print what we're parsing
        print(f"\n--- Parsing N={n} ---")
        
        # Extract attempt number (if quantum succeeded)
        attempt_match = re.search(r'attempt (\d+)', section)
        quantum_attempts = int(attempt_match.group(1)) + 1 if attempt_match else None
        
        # Check if fallback was used
        fallback_count = section.count('Fallback to pollard_rho')
        used_fallback = fallback_count >= 2
        
        # Extract classical results with more flexible pattern
        classical_match = re.search(r'Classical:\s*\((\d+),\s*(\d+)\)\s*in\s*([\d.]+)s', section)
        if classical_match:
            classical_factor1 = int(classical_match.group(1))
            classical_factor2 = int(classical_match.group(2))
            classical_time = float(classical_match.group(3))
            print(f"Classical: ({classical_factor1}, {classical_factor2}) in {classical_time}s")
        else:
            classical_factor1 = classical_factor2 = None
            classical_time = None
            print("Classical: NOT FOUND")
        
        # Extract quantum results with more flexible pattern
        quantum_match = re.search(r'Quantum:\s*\((\d+),\s*(\d+)\)\s*in\s*([\d.]+)s', section)
        if quantum_match:
            quantum_factor1 = int(quantum_match.group(1))
            quantum_factor2 = int(quantum_match.group(2))
            quantum_time = float(quantum_match.group(3))
            print(f"Quantum: ({quantum_factor1}, {quantum_factor2}) in {quantum_time}s")
        else:
            quantum_factor1 = quantum_factor2 = None
            quantum_time = None
            print("Quantum: NOT FOUND")
        
        # Calculate speedup (if both successful)
        if classical_time and quantum_time and classical_time > 0:
            speedup = classical_time / quantum_time
        else:
            speedup = None
        
        results.append({
            'N': n,
            'Bits': bits,
            'Classical_Factor1': classical_factor1,
            'Classical_Factor2': classical_factor2,
            'Classical_Time_s': classical_time,
            'Quantum_Factor1': quantum_factor1,
            'Quantum_Factor2': quantum_factor2,
            'Quantum_Time_s': quantum_time,
            'Quantum_Attempts': quantum_attempts,
            'Used_Fallback': used_fallback,
            'Speedup': speedup
        })
    
    return results


def save_to_csv(results, output_filename):
    """
    Save results to CSV file
    """
    if not results:
        print("No results to save")
        return
    
    fieldnames = [
        'N', 'Bits', 
        'Classical_Factor1', 'Classical_Factor2', 'Classical_Time_s',
        'Quantum_Factor1', 'Quantum_Factor2', 'Quantum_Time_s', 'Quantum_Attempts',
        'Used_Fallback', 'Speedup'
    ]
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to {output_filename}")


def print_summary_table(results):
    """
    Print a formatted summary table
    """
    print("\n" + "="*120)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*120)
    print(f"{'N':<12} {'Bits':<6} {'Classical':<20} {'Classical_Time':<15} {'Quantum':<20} {'Quantum_Time':<15} {'Attempts':<10}")
    print("-"*120)
    
    for r in results:
        classical_factors = f"({r['Classical_Factor1']}, {r['Classical_Factor2']})" if r['Classical_Factor1'] else "N/A"
        quantum_factors = f"({r['Quantum_Factor1']}, {r['Quantum_Factor2']})" if r['Quantum_Factor1'] else "N/A"
        classical_time = f"{r['Classical_Time_s']:.6f}s" if r['Classical_Time_s'] else "N/A"
        quantum_time = f"{r['Quantum_Time_s']:.6f}s" if r['Quantum_Time_s'] else "N/A"
        attempts = str(r['Quantum_Attempts']) if r['Quantum_Attempts'] else "Fallback"
        
        print(f"{r['N']:<12} {r['Bits']:<6} {classical_factors:<20} {classical_time:<15} {quantum_factors:<20} {quantum_time:<15} {attempts:<10}")
    


def main():
    # Parse the results file
    input_file = 'v1.txt'
    output_file = 'shor_results.csv'
    
    print(f"Parsing {input_file}...")
    results = parse_shor_results(input_file)
    
    # Print summary table
    print_summary_table(results)
    
    # Save to CSV
    save_to_csv(results, output_file)
    
    print(f"\n✓ CSV file created: {output_file}")


if __name__ == "__main__":
    main()