__kernel void mse(read_only image2d_t img1, __const int2 img1dim,
                  read_only image2d_t img2, __const int2 img2dim,
                 write_only image3d_t imgO, __const int2 imgOdim)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    float2 delta = (float2)((img1dim.x - img2dim.x) / 2, (img1dim.y - img2dim.y) / 2);
    int4 pos = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float2 center2 = (float2)(img2dim.x / 2, img2dim.y / 2);

    int2 disp = (int2)(pos.x - (imgOdim.x - 1) / 2, pos.y - (imgOdim.y - 1) / 2);

    float cos_rot = 0.0f;
    float sin_rot = sincos(radians(pos.z * 2.0 - 10), &cos_rot);

    float2 fixed = (float2)(-cos_rot * center2.x - sin_rot * center2.y + center2.x - delta.x - disp.x, 
                             sin_rot * center2.x - cos_rot * center2.y + center2.y - delta.y - disp.y);

    float4 squared_error = (float4)(0.0, 0.0, 0.0, 0.0);
    float mask_sum = 0.0f;
    
    for (int i=0; i < img1dim.x; i++) {
        for (int j=0; j < img1dim.y; j++) {
            float2 pos1 = (float2)(i, j);
            float2 pos2 =((float2)(cos_rot * i + sin_rot * j, 
                                  -sin_rot * i + cos_rot * j)) + fixed;

            float4 val1 = read_imagef(img1, sampler, pos1);
            float4 val2 = read_imagef(img2, sampler, pos2);
            
            float mask = val1.w * val2.w;
            float4 diff = val1 - val2;

            squared_error += diff * diff * (float4)(mask, mask, mask, mask);
            mask_sum += mask;
        }
    }
    
    squared_error /= (float4)(mask_sum, mask_sum, mask_sum, 1);
    write_imagef(imgO, pos, squared_error);
}
