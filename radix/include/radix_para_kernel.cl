__kernel void parallel_radix_sort(__global ushort *data, __global ushort *output, int num_elements, int bit) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
    
    // Allocate local memory for this work-group
    __local ushort local_data[256];
    
    // Load data into local memory
    int idx = group_id * local_size * 2 + local_id;
    if (idx < num_elements) {
        local_data[local_id] = data[idx];
    } else {
        local_data[local_id] = 0xFFFF; // Sentinel value for unused slots
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform bitwise radix sort on local data
    for (int shift = 0; shift < bit; shift++) {
        int mask = 1 << shift;
        int zero_count = 0;
        int one_count = 0;
        
        // Count zeros and ones
        for (int i = 0; i < local_size; i++) {
            if ((local_data[i] & mask) == 0) {
                zero_count++;
            } else {
                one_count++;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Rearrange based on current bit
        for (int i = 0; i < local_size; i++) {
            int value = local_data[i];
            if ((value & mask) == 0) {
                local_data[zero_count++] = value;
            } else {
                local_data[one_count++] = value;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write sorted data back to global memory
    if (idx < num_elements) {
        output[idx] = local_data[local_id];
    }
}
