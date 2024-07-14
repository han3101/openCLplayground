__kernel void resize_bilinear(
    __global uchar* data, 
    __global uchar* output,
    int nw,
    int nh,
    int w, 
    int h,
    int channels,
    float scaleX,
    float scaleY
) 
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < nh && col < nw) {

        float float_pos_x, float_pos_y, offset_x, offset_y;
        ushort low_x, low_y, high_x, high_y; 


        float_pos_x = col * scaleX;
        float_pos_y = row * scaleY;

        low_x = (ushort)floor(float_pos_x);
        low_y = (ushort)floor(float_pos_y);
        high_x = low_x + 1;
        high_y = low_y + 1;

        if (high_x >= w) {
            high_x = low_x;
        }
        if (high_y >= h) {
            high_y = low_y;
        }

        offset_x = float_pos_x - low_x;
        offset_y = float_pos_y - low_y;

        for (int c = 0; c < channels; ++c) {
            
            float value = (1-offset_x) * (1-offset_y) * data[(low_x + low_y * w) * channels + c] +
                        offset_x * (1-offset_y) * data[(high_x + low_y * w) * channels + c] +
                        (1-offset_x) * offset_y * data[(low_x + high_y * w) * channels + c] +
                        offset_x * offset_y * data[(high_x + high_y * w) * channels + c];

                        //printf("value: %d\n", value);

            output[(col + row * nw) * channels + c] = (uchar) clamp(value, 0.0f, 255.0f);
        }
    }
    
}
