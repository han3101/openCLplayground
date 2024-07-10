__kernel void convolution_2d(
    __global int *matrix,
    __global int *result,
    // __local int *s_array,
    __constant int* mask,
    int N,
    int MASK_DIM,
    int MASK_OFFSET
) 
{
    /* get global position in Y direction */
    int row = get_global_id (1);
    /* get global position in X direction */
    int col = get_global_id (0);

    // Starting index for calculation
    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    // Temp value for accumulating result
    int temp = 0;

    // Iterate over all the rows
    for (int i=0; i < MASK_DIM; i++) {
        // Go over column
        for (int j=0 ; j < MASK_DIM; ++j) {
            // Range check for rows
            if ((start_r + i) >= 0 && (start_r + i) < N) {
                // Range check for cols
                if ((start_c + j) >= 0 && (start_c + j) < N) {
                    // Accumulate results
                    temp += matrix[(start_r + i) * N + (start_c + j)] * mask[i * MASK_DIM + j];
                }
            }
        }
    }

    // Write back the result
    result[row * N + col] = temp;
}