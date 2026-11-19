#!/usr/bin/env python3
"""Writes the input files of the Rodinia applications that read their input (hotspot, hotspot3D), as Rodinia's own
generators would: one value per line, temperatures around 323-343 K and powers up to 1e-3 W per cell. Deterministic.

With --lud N it also writes two matrices for lud -i (text, NumPy needed, about 2.7 GB each for N = 16384):
- lud_smooth_N: the matrix lud -s N makes itself, 10 exp(-0.001 |i - j|), so its values change only in low bits from
  one element to the next;
- lud_random_N: uniform random values in [-1, 1) with N added on the diagonal (diagonally dominant, so the
  factorisation without pivoting stays well behaved).
Both print every value with 9 significant digits, which gives back the same float. The pair tests whether lud's data
values explain its error (docs/APPLICATIONS.md).

    python3 apps/make_inputs.py OUT_DIR [--lud N]
"""
import os
import random
import sys


def write(path, count, low, high, seed):
    generator = random.Random(seed)
    with open(path, "w") as file:
        file.writelines(f"{generator.uniform(low, high):.6f}\n" for _ in range(count))


def write_matrix(path, rows, size):
    """lud's text format: the size, then the values row by row, separated by spaces."""
    with open(path, "w") as file:
        file.write(f"{size}\n")
        for row in rows:
            row.tofile(file, sep=" ", format="%.9g")
            file.write("\n")


def lud_matrices(out, size):
    import numpy
    # As lud's create_matrix: float lamda * int index, exp in double, stored as float.
    distance = numpy.arange(size, dtype=numpy.float32)
    coefficients = (10.0 * numpy.exp((numpy.float32(-0.001) * distance).astype(numpy.float64))).astype(numpy.float32)
    write_matrix(os.path.join(out, f"lud_smooth_{size}"),
                 (coefficients[numpy.abs(numpy.arange(size) - i)] for i in range(size)), size)
    generator = numpy.random.default_rng(5)

    def random_rows():
        for i in range(size):
            row = generator.uniform(-1.0, 1.0, size).astype(numpy.float32)
            row[i] += numpy.float32(size)
            yield row
    write_matrix(os.path.join(out, f"lud_random_{size}"), random_rows(), size)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    if "--lud" in sys.argv:
        lud_matrices(out, int(sys.argv[sys.argv.index("--lud") + 1]))
        return
    write(os.path.join(out, "temp_2048"), 2048 * 2048, 323.0, 343.0, 1)
    write(os.path.join(out, "power_2048"), 2048 * 2048, 0.0, 1e-3, 2)
    write(os.path.join(out, "temp_512x8"), 512 * 512 * 8, 323.0, 343.0, 3)
    write(os.path.join(out, "power_512x8"), 512 * 512 * 8, 0.0, 1e-3, 4)


if __name__ == "__main__":
    main()
