import numpy as np
from numba import cuda

@cuda.jit
def test_kernel(out, y_offset=0):
    i, j_local = cuda.grid(2)
    j = j_local + y_offset
    if i < out.shape[1] and j < out.shape[0]:
        out[j, i] = j * 100 + i

arr = np.zeros((32, 32), dtype=np.int32)
d_arr = cuda.to_device(arr)

# Test 1: call without y_offset
test_kernel[(2, 1), (16, 16)](d_arr)
d_arr.copy_to_host(arr)
print("Test 1 (no y_offset) row 0:", arr[0, :4])
print("Test 1 (no y_offset) row 16:", arr[16, :4])

# Test 2: call with y_offset = 16
test_kernel[(2, 1), (16, 16)](d_arr, 16)
d_arr.copy_to_host(arr)
print("Test 2 (y_offset=16) row 16:", arr[16, :4])
print("SUCCESS!")
