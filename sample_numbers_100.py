import random
import math

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_primes_in_range(start, end):
    """Generate all primes in a given range"""
    primes = []
    for num in range(start, end + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def generate_semiprime(min_bits, max_bits, prefer_distinct_primes=True):
    """Generate a semiprime (product of two primes) within a bit range"""
    # Calculate the range for each prime factor
    min_val = 2 ** (min_bits - 1)
    max_val = 2 ** max_bits - 1
    
    # Get primes in the range
    p1_min = int(math.sqrt(min_val))
    p1_max = int(math.sqrt(max_val))
    
    primes = generate_primes_in_range(p1_min, p1_max)
    
    if len(primes) < 2:
        return None
    
    # Try to find two primes whose product is in range
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts:
        p1 = random.choice(primes)
        # Choose second prime so product is in desired range
        p2_min = max(p1, min_val // p1)
        p2_max = min(max_val // p1, p1_max)
        
        if prefer_distinct_primes:
            # Prefer distinct primes (avoid perfect squares like 9, 25, 49)
            valid_primes = [p for p in primes if p2_min <= p <= p2_max and p > p1]
        else:
            valid_primes = [p for p in primes if p2_min <= p <= p2_max and p >= p1]
        
        if valid_primes:
            p2 = random.choice(valid_primes)
            n = p1 * p2
            if min_val <= n <= max_val:
                return (n, p1, p2)
        
        attempts += 1
    
    # If we couldn't find distinct primes, allow same primes
    if prefer_distinct_primes:
        return generate_semiprime(min_bits, max_bits, prefer_distinct_primes=False)
    
    return None

def generate_test_numbers(count=100):
    """Generate test numbers distributed across different bit sizes"""
    random.seed(42)  # For reproducibility
    
    test_numbers = []
    seen_numbers = set()  # Track generated numbers to avoid duplicates
    
    # Define distribution of bit sizes
    # More numbers in smaller ranges (easier to test)
    # Fewer numbers in larger ranges (harder/slower)
    bit_distribution = [
        (4, 6, 15),    # 4-6 bits: 15 numbers
        (7, 9, 15),    # 7-9 bits: 15 numbers
        (10, 12, 15),  # 10-12 bits: 15 numbers
        (13, 15, 12),  # 13-15 bits: 12 numbers
        (16, 18, 10),  # 16-18 bits: 10 numbers
        (19, 21, 10),  # 19-21 bits: 10 numbers
        (22, 24, 10),  # 22-24 bits: 10 numbers
        (25, 27, 8),   # 25-27 bits: 8 numbers
        (28, 30, 5),   # 28-30 bits: 5 numbers
    ]
    
    for min_bits, max_bits, num_samples in bit_distribution:
        print(f"Generating {num_samples} unique numbers in {min_bits}-{max_bits} bit range...")
        
        attempts = 0
        max_attempts = num_samples * 100  # Allow many attempts to find unique numbers
        generated = 0
        
        while generated < num_samples and attempts < max_attempts:
            result = generate_semiprime(min_bits, max_bits)
            if result:
                n, p1, p2 = result
                
                # Only add if it's unique
                if n not in seen_numbers:
                    seen_numbers.add(n)
                    test_numbers.append({
                        'n': n,
                        'p1': p1,
                        'p2': p2,
                        'bits': n.bit_length()
                    })
                    generated += 1
            
            attempts += 1
        
        if generated < num_samples:
            print(f"  Warning: Only generated {generated}/{num_samples} unique numbers in this range")
    
    # Sort by value
    test_numbers.sort(key=lambda x: x['n'])
    
    return test_numbers

def format_test_numbers(test_numbers):
    """Format test numbers as Python list"""
    print("\ntest_numbers = [")
    
    for item in test_numbers:
        n = item['n']
        p1 = item['p1']
        p2 = item['p2']
        bits = item['bits']
        print(f"    {n:>12},  # {p1} × {p2} ({bits} bits)")
    
    print("]")
    
    # Print statistics
    print(f"\n# Total: {len(test_numbers)} numbers")
    bit_counts = {}
    for item in test_numbers:
        bits = item['bits']
        bit_counts[bits] = bit_counts.get(bits, 0) + 1
    
    print("\n# Distribution by bit size:")
    for bits in sorted(bit_counts.keys()):
        print(f"#   {bits:2d} bits: {bit_counts[bits]:3d} numbers")

def save_to_file(test_numbers, filename='test_numbers_100.py'):
    """Save test numbers to a Python file"""
    with open(filename, 'w') as f:
        f.write("# 100 Semiprime test numbers for Shor's algorithm\n")
        f.write("# Generated with varying bit sizes for comprehensive testing\n\n")
        f.write("test_numbers = [\n")
        
        for item in test_numbers:
            n = item['n']
            p1 = item['p1']
            p2 = item['p2']
            bits = item['bits']
            f.write(f"    {n:>12},  # {p1} × {p2} ({bits} bits)\n")
        
        f.write("]\n")
    
    print(f"\n✓ Test numbers saved to {filename}")

def main():
    print("Generating 100 semiprime test numbers...")
    print("="*60)
    
    test_numbers = generate_test_numbers(100)
    
    print(f"\n✓ Generated {len(test_numbers)} test numbers")
    print("\n" + "="*60)
    
    # Display formatted output
    format_test_numbers(test_numbers)
    
    # Save to file
    save_to_file(test_numbers)
    
    # Show some sample numbers
    print("\n" + "="*60)
    print("Sample numbers (first 10):")
    print("="*60)
    for i, item in enumerate(test_numbers[:10], 1):
        print(f"{i:2d}. {item['n']:>10} = {item['p1']:>5} × {item['p2']:>5} ({item['bits']} bits)")

if __name__ == "__main__":
    main()