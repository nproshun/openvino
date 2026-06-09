import sys
import numpy as np
from ml_dtypes import bfloat16

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input1.bin> <input2.bin> <expected.bin>")
        sys.exit(1)

    file_a, file_b, file_expected = sys.argv[1], sys.argv[2], sys.argv[3]

    a = np.fromfile(file_a, dtype=bfloat16)
    b = np.fromfile(file_b, dtype=bfloat16)
    expected = np.fromfile(file_expected, dtype=bfloat16)

    print(a.astype(np.float32))
    print(b.astype(np.float32))

    result = a.astype(np.float32) + b.astype(np.float32)

    expected_f32 = expected.astype(np.float32)

    abs_diff = np.abs(result - expected_f32)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    mismatches = np.count_nonzero(abs_diff > 0)

    print(f"Elements:   {len(a)}")
    print(f"Max diff:   {max_diff}")
    print(f"Mean diff:  {mean_diff}")
    print(f"Mismatches: {mismatches} / {len(a)}")

    if max_diff == 0:
        print("PASS: exact match")
    else:
        print("MISMATCH: results differ")
        sys.exit(1)

if __name__ == "__main__":
    main()
