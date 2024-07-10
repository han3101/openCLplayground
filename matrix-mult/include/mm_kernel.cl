/* widthA=heightB for valid matrix multiplication */
__kernel void simpleMultiply(
    __global float *outputC,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    __global float *inputA,
    __global float *inputB)
{
    /* get global position in Y direction */
    int row = get_global_id (1);
    /* get global position in X direction */
    int col = get_global_id (0);

    float sum = 0.0f;

    /* calculate result of one element of Matrix C */
    for (int i=0; i<widthA; i++) {
        sum += inputA[row*widthA + i] * inputB[i*widthB + col];
    }

    outputC[row*widthB + col] = sum;
}

__kernel void tiledMultiply(
    __global float* outputC,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    __global float* inputA,
    __global float* inputB)
{
    __local float localA[16][16]; // Local memory for tiles of A
    __local float localB[16][16]; // Local memory for tiles of B

    // Get global and local positions
    int globalRow = get_global_id(1);
    int globalCol = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (widthA + 16 - 1) / 16; t++) {
        // Load tiles into local memory
        if (t * 16 + localCol < widthA && globalRow < heightA)
            localA[localRow][localCol] = inputA[globalRow * widthA + t * 16 + localCol];
        else
            localA[localRow][localCol] = 0.0f;

        if (t * 16 + localRow < heightB && globalCol < widthB)
            localB[localRow][localCol] = inputB[(t * 16 + localRow) * widthB + globalCol];
        else
            localB[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial result
        for (int k = 0; k < 16; k++) {
            sum += localA[localRow][k] * localB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (globalRow < heightA && globalCol < widthB) {
        outputC[globalRow * widthB + globalCol] = sum;
    }
}
