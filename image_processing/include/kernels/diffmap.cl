#define BYTE_BOUND(value) min(max((value), 0), 255)

__kernel void diffmap(
    __global uchar* data, 
    __global uchar* data2,
    int w,
    int h,
    int channels,
    int w2,
    int h2,
    int channels2,
    int compare_w,
    int compare_h,
    int compare_ch
) 
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    if (i < compare_h && j < compare_w && k < compare_ch) {
        int index1 = (i * w + j) * channels + k;
        int index2 = (i * w2 + j) * channels2 + k;
        int diff = abs(data[index1] - data2[index2]);
        data[index1] = BYTE_BOUND(diff);
    }

    
}
