
int MASK_LENGTH = 7;

__kernel void convolution_1d(
    __global int *array,
    __global int *result,
    __local int *s_array,
    __constant int* mask
) 
{
    // Get global thread ID
    int tid = get_global_id(0);

    // r: The number of padded elements on either side
    int r = MASK_LENGTH / 2;

    // d: The total number of padded elements
    int d = 2 * r;

    // Size of padded shared memory array
    int n_padded = get_local_size(0) + d;

    // Offset for second set of loads in shared memory
    int offset = get_local_id(0) + get_local_size(0);

    // Global offset for array in DRAM
    // int g_offset = get_global_offset()
    int g_offset = get_local_size(0) * get_group_id(0) + offset;

    // Load the lower elements first starting at the halo
    // This ensures divergence only once
    s_array[get_local_id(0)] = array[tid];

    // Load in remaining upper elements up till padding
    if (offset < n_padded) {
        s_array[offset] = array[g_offset];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for (int j=0; j < MASK_LENGTH; j++) {
        temp += s_array[get_local_id(0) + j] * mask[j];
    }

    result[tid] = temp;
}   