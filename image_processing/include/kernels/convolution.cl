#define BYTE_BOUND(value) min(max((value), 0), 255)

__kernel void convolution_0(
    __global uchar *matrix,
    __global uchar *result,
    __constant double* mask,
    int w,
    int h,
    int channels,
    int MASK_DIM,
    int MASK_OFFSET
) 
{
    /* get global position in Y direction */
    int row = get_global_id (1);
    /* get global position in X direction */
    int col = get_global_id (0);


    for (int ch = 0; ch < channels; ++ch) {
        // Starting index for calculation
        int start_r = row - MASK_OFFSET;
        int start_c = col - MASK_OFFSET;

        // Temp value for accumulating result
        double temp = 0;

        // Iterate over all the rows
        for (int i=0; i < MASK_DIM; i++) {
            // Go over column
            for (int j=0 ; j < MASK_DIM; ++j) {
                // Range check for rows
                if ((start_r + i) >= 0 && (start_r + i) < h) {
                    // Range check for cols
                    if ((start_c + j) >= 0 && (start_c + j) < w) {
                        // Accumulate results
                        temp += matrix[((start_r + i) * w + (start_c + j)) * channels + ch] * mask[i * MASK_DIM + j];
                    }
                }
            }
        }
        // Write back the result
        result[(row * w + col) * channels + ch] = (uchar)BYTE_BOUND((int)round(temp));

    }
    
}


__kernel void convolution_border(
    __global uchar *matrix,
    __global uchar *result,
    __constant double* mask,
    int w,
    int h,
    int channels,
    int MASK_DIM,
    int MASK_OFFSET
) 
{
    /* get global position in Y direction */
    int row = get_global_id (1);
    /* get global position in X direction */
    int col = get_global_id (0);


    for (int ch = 0; ch < channels; ++ch) {
        // Starting index for calculation
        int start_r = row - MASK_OFFSET;
        int start_c = col - MASK_OFFSET;

        // Temp value for accumulating result
        double temp = 0;

        // Iterate over all the rows
        for (int i=0; i < MASK_DIM; i++) {
            int r = start_r + i;
            // Range check for rows
            if ((start_r + i) < 0) {
                r = 0;
            } else if ((start_r + i) >= h) {
                r = h-1;
            }

            // Go over column
            for (int j=0 ; j < MASK_DIM; ++j) {
                int c = start_c + j;
                // Range check for cols
                if ((start_c + j) < 0) {
                    c = 0;
                } else if ((start_c + j) >= w) {
                    c = w-1;
                }
                // Accumulate results
                temp += matrix[(r * w + c) * channels + ch] * mask[i * MASK_DIM + j];
            }
        }
        // Write back the result
        result[(row * w + col) * channels + ch] = (uchar)BYTE_BOUND((int)round(temp));

    }

    
}