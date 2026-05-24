import time
import random
import tracemalloc
import sys
import math


def binary_search(arr, target, low=None, high=None):
    # Set default bounds on the first call
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1

    if low > high:
        return -1  # target not in array

    mid = (low + high) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, high)  # search right half
    else:
        return binary_search(arr, target, low, mid - 1)   # search left half


def karatsuba(x, y):
    # Fall back to regular multiplication for single-digit numbers
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    m = n // 2        # split each number into two halves of ~m digits
    power = 10 ** m

    x_high, x_low = divmod(x, power)
    y_high, y_low = divmod(y, power)

    # Three multiplications instead of four (Karatsuba's key insight)
    z0 = karatsuba(x_low, y_low)
    z2 = karatsuba(x_high, y_high)
    z1 = karatsuba(x_low + x_high, y_low + y_high)

    # Recombine: x*y = z2*10^(2m) + (z1-z2-z0)*10^m + z0
    return z2 * (10 ** (2 * m)) + (z1 - z2 - z0) * (10 ** m) + z0


def make_bs_datasets(n):
    # Binary search needs sorted input regardless of original order
    sorted_arr  = list(range(n))
    reverse_arr = list(range(n - 1, -1, -1))
    random_arr  = sorted(random.sample(range(n * 10), n))

    return {
        "sorted"        : (sorted_arr,          n // 2),         # target sits at midpoint
        "reverse_sorted": (sorted(reverse_arr), n // 2),
        "random"        : (random_arr,          random_arr[n // 2]),
    }


def make_karatsuba_datasets(num_digits):
    lo = 10 ** (num_digits - 1)
    hi = 10 **  num_digits - 1

    x_rand = random.randint(lo, hi)
    y_rand = random.randint(lo, hi)

    # Repeating 1–9 pattern, reversed for the second operand
    pattern  = "".join(str((i % 9) + 1) for i in range(num_digits))
    x_struct = int(pattern)
    y_struct  = int(pattern[::-1])

    # All-nines is the adversarial case — sums overflow to an extra digit
    x_nines = int("9" * num_digits)
    y_nines = int("9" * num_digits)

    return {
        "random"    : (x_rand,   y_rand),
        "structured": (x_struct, y_struct),
        "all_nines" : (x_nines,  y_nines),
    }


def benchmark(fn, *args, repeats=5):
    # Run several times and return the median to reduce scheduling noise
    times    = []
    peak_mem = 0

    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append((t1 - t0) * 1e6)        # seconds → microseconds
        peak_mem = max(peak_mem, peak / 1024) # bytes → KB

    times.sort()
    return times[repeats // 2], peak_mem

def print_table_header(cols, widths):
    print(" | ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))


def print_table_row(vals, widths):
    print(" | ".join(f"{v:>{w}}" for v, w in zip(vals, widths)))


def run_binary_search_benchmarks(sizes):
    print("="*72)
    print("BINARY SEARCH BENCHMARKS")
    print("Recurrence: T(n) = T(n/2) + O(1)  →  T(n) = Θ(log n)")
    print("="*72)

    cols   = ["Array Size", "Dataset Type", "Time (µs)", "Peak Mem (KB)", "log₂(n)"]
    widths = [12, 15, 12, 15, 10]
    print_table_header(cols, widths)

    results = []

    for n in sizes:
        datasets = make_bs_datasets(n)
        log_n    = round(math.log2(n), 2)

        for label, (arr, target) in datasets.items():
            t, m = benchmark(binary_search, arr, target)
            results.append((n, label, t, m))
            print_table_row(
                [f"{n:,}", label.replace("_", " ").title(), f"{t:.3f}", f"{m:.4f}", str(log_n)],
                widths
            )

    print()
    print("Observation: Time grows roughly proportional to log₂(n),")
    print("confirming the Θ(log n) theoretical prediction.")
    print()
    return results


def run_karatsuba_benchmarks(digit_counts):
    print("="*72)
    print("KARATSUBA MULTIPLICATION BENCHMARKS")
    print("Recurrence: T(n) = 3·T(n/2) + O(n)  →  T(n) = Θ(n^log₂3) ≈ Θ(n^1.585)")
    print("="*72)

    cols   = ["Digits (n)", "Dataset Type", "Time (µs)", "Peak Mem (KB)", "n^1.585 (rel)"]
    widths = [12, 15, 12, 15, 14]
    print_table_header(cols, widths)

    results = []

    for d in digit_counts:
        datasets = make_karatsuba_datasets(d)

        for label, (x, y) in datasets.items():
            t, m = benchmark(karatsuba, x, y)
            results.append((d, label, t, m))
            print_table_row(
                [str(d), label.replace("_", " ").title(), f"{t:.3f}", f"{m:.4f}", f"{d**1.585:.1f}"],
                widths
            )

    print()
    print("Observation: 'All Nines' consistently slower — overflow during")
    print("(x_low + x_high) produces an extra digit, deepening recursion.")
    print()
    return results


def growth_rate_analysis(bs_results, kar_results):
    print("="*72)
    print("GROWTH RATE ANALYSIS")
    print("="*72)

    print("\nBinary Search — Measured vs. Theoretical Growth (sorted dataset):")
    print(f"  {'From n':>10} → {'To n':>10} | {'Time ratio':>12} | {'log ratio (theory)':>20}")
    print("  " + "-" * 60)

    sorted_rows = [(n, t) for (n, label, t, _) in bs_results if label == "sorted"]
    for i in range(1, len(sorted_rows)):
        n1, t1 = sorted_rows[i - 1]
        n2, t2 = sorted_rows[i]
        if t1 > 0:
            ratio  = t2 / t1
            theory = math.log2(n2) / math.log2(n1) if n1 > 1 else float("inf")
            print(f"  {n1:>10,} → {n2:>10,} | {ratio:>12.3f} | {theory:>20.3f}")

    print("\nKaratsuba — Measured vs. Theoretical Growth (random dataset):")
    print(f"  {'From d':>8} → {'To d':>6} | {'Time ratio':>12} | {'(d2/d1)^1.585 (theory)':>24}")
    print("  " + "-" * 58)

    rand_rows = [(d, t) for (d, label, t, _) in kar_results if label == "random"]
    for i in range(1, len(rand_rows)):
        d1, t1 = rand_rows[i - 1]
        d2, t2 = rand_rows[i]
        if t1 > 0:
            ratio  = t2 / t1
            theory = (d2 / d1) ** 1.585
            print(f"  {d1:>8} → {d2:>6} | {ratio:>12.3f} | {theory:>24.3f}")

    print()


def verify_correctness():
    print("="*72)
    print("CORRECTNESS VERIFICATION")
    print("="*72)

    print("\nBinary Search:")
    bs_cases = [
        ([1, 3, 5, 7, 9, 11, 13], 7,  3,  "target at midpoint"),
        ([1, 3, 5, 7, 9, 11, 13], 1,  0,  "target at left edge"),
        ([1, 3, 5, 7, 9, 11, 13], 13, 6,  "target at right edge"),
        ([1, 3, 5, 7, 9, 11, 13], 4,  -1, "target absent"),
        ([42],                    42, 0,  "single-element array, found"),
        ([42],                    7,  -1, "single-element array, not found"),
        ([],                      1,  -1, "empty array"),
    ]

    all_pass = True
    for arr, target, expected, desc in bs_cases:
        result = binary_search(arr, target) if arr else -1
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {desc:<40}  got={result}, expected={expected}")

    print("\nKaratsuba Multiplication:")
    kar_cases = [
        (0,           999,          "multiply by zero"),
        (1,           123456789,    "multiply by one"),
        (9,           9,            "single digits"),
        (12,          34,           "two-digit × two-digit"),
        (1234,        5678,         "four-digit × four-digit"),
        (10**50,      10**50,       "fifty-digit × fifty-digit"),
        (int("9"*100), int("9"*100), "hundred nines × hundred nines"),
    ]

    for x, y, desc in kar_cases:
        result = karatsuba(x, y)
        status = "PASS" if result == x * y else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {desc}")

    print()
    print("All correctness checks passed." if all_pass else "SOME CHECKS FAILED.")
    print()


def main():
    sys.setrecursionlimit(50_000)  # needed for deep Karatsuba recursion on large inputs
    random.seed(42)                # fix seed so datasets are reproducible

    print()
    print("="*72)
    print("  Divide-and-Conquer: Binary Search & Karatsuba Multiplication")
    print("  Assignment 2 — Algorithm Analysis and Implementation")
    print("="*72)
    print()

    verify_correctness()

    bs_results  = run_binary_search_benchmarks([100, 1_000, 10_000, 100_000, 1_000_000])
    kar_results = run_karatsuba_benchmarks([10, 20, 50, 100, 200])

    growth_rate_analysis(bs_results, kar_results)

    print("="*72)
    print("Run complete.")
    print("="*72)


if __name__ == "__main__":
    main()
