````markdown id="finalreadme1"
# RV-Sparse Coding Challenge

This repository contains my solution for the RV-Sparse coding challenge.

The implementation performs:

1. Extraction of non-zero elements from a dense row-major matrix into CSR (Compressed Sparse Row) format
2. Sparse matrix-vector multiplication:

\[
y = A \times x
\]

3. Writing results directly into caller-provided output buffers

The solution satisfies the primary challenge constraint:

- **Zero dynamic memory allocation inside `sparse_multiply()`**

---

# Implementation Overview

The `sparse_multiply()` function converts a dense matrix into CSR format while simultaneously computing the matrix-vector product.

Instead of performing separate passes for:
- CSR construction
- Sparse matrix-vector multiplication (SpMV)

both operations are fused into a **single-pass traversal** of the matrix.

This reduces:
- redundant memory traversal
- unnecessary intermediate operations
- extra processing overhead

---

# CSR Representation

The sparse matrix is stored using the CSR (Compressed Sparse Row) format.

## CSR Arrays

### `values[]`
Stores all non-zero matrix values sequentially.

### `col_indices[]`
Stores the column index corresponding to each non-zero value.

### `row_ptrs[]`
Stores offsets indicating where each row begins in the CSR arrays.

---

## Example

Dense Matrix:

```text
[ 0  5  0 ]
[ 1  0  2 ]
[ 0  0  9 ]
```

CSR Representation:

```text
values      = [5, 1, 2, 9]
col_indices = [1, 0, 2, 2]
row_ptrs    = [0, 1, 3, 4]
```

---

# Logic Flow

For each row of matrix `A`:

1. Scan all columns sequentially
2. Identify non-zero values
3. Store non-zero entries into CSR buffers
4. Multiply each non-zero element with the corresponding vector element from `x`
5. Accumulate the result into the output vector `y`

The implementation operates directly on a dense row-major matrix while dynamically constructing sparse CSR output during traversal.

---

# Key Design Decisions

## 1. Zero Dynamic Allocation

The implementation does not use:
- `malloc`
- `calloc`
- `realloc`

inside the `sparse_multiply()` function.

All buffers are pre-allocated by the caller, ensuring predictable memory behavior and reduced runtime overhead.

---

## 2. Single-Pass Processing

CSR extraction and SpMV computation are fused into one traversal.

Benefits:
- lower traversal overhead
- fewer redundant operations
- improved spatial locality during matrix scanning

---

## 3. Sequential Row-Major Access

Rows are accessed sequentially:

```c
const double* row = A + (size_t)i * cols;
```

This improves memory locality and keeps traversal cache-friendly for dense input matrices.

---

# Complexity Analysis

## Time Complexity

Dense matrix traversal:

```text
O(rows × cols)
```

The matrix-vector multiplication is fused into the same traversal.

---

## Space Complexity

CSR storage requires:

```text
O(NNZ + rows)
```

Where:
- `NNZ` = number of non-zero elements

No additional heap memory is allocated inside the function.

---

# Compilation

Compile using GCC:

```bash
gcc -O3 challenge.c -lm -o run
```

---

# Execution

## Linux / macOS

```bash
./run
```

## Windows

```bash
run.exe
```

---

# Validation

The provided test harness executes 100 randomized test iterations using matrices with varying:
- dimensions
- sparsity levels
- value distributions

The output from `sparse_multiply()` is validated against a dense reference implementation using mixed absolute and relative floating-point tolerance checks.

---

# Notes

This implementation focuses on:
- correct CSR extraction
- memory-safe buffer handling
- sparse traversal logic
- efficient single-pass processing

The current implementation scans a dense input matrix while generating sparse CSR output dynamically. In production sparse HPC systems, matrices are typically already stored in sparse formats to avoid dense scanning overhead.
````


<img width="1038" height="737" alt="Screenshot 2026-05-06 152034" src="https://github.com/user-attachments/assets/c156f865-ea02-43a3-a602-9f3e8d78d1ec" />
