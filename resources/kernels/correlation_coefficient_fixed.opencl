__kernel void avg(read_only image2d_t img1, __const int2 img1dim, 
                  read_only image2d_t img2, __const int2 img2dim,
                 write_only image3d_t img1avg, write_only image3d_t img2avg, 
                 write_only image3d_t fixed_out, __const int2 outdim)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    float2 delta = (float2)((img1dim.x - img2dim.x) / 2, (img1dim.y - img2dim.y) / 2);
    int4 pos = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float2 center2 = (float2)(img2dim.x / 2, img2dim.y / 2);

    int2 disp = (int2)(pos.x - (outdim.x - 1) / 2, pos.y - (outdim.y - 1) / 2);

    float cos_rot = 0.0f;
    float sin_rot = sincos(radians(pos.z * 2.0 - 10), &cos_rot);
    
    float2 fixed = (float2)(-cos_rot * center2.x - sin_rot * center2.y + center2.x - delta.x - disp.x, 
                             sin_rot * center2.x - cos_rot * center2.y + center2.y - delta.y - disp.y);
    
    float4 avg1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 avg2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float mask_sum = 0.0f;
    
    for (int i=0; i < img1dim.x; i++) {
        for (int j=0; j < img1dim.y; j++) {
            float2 pos1 = (float2)(i, j);
            float2 pos2 =((float2)(cos_rot * i + sin_rot * j, 
                                  -sin_rot * i + cos_rot * j)) + fixed.xy;
            
            float4 val1 = read_imagef(img1, sampler, pos1);
            float4 val2 = read_imagef(img2, sampler, pos2);
            
            float mask = val1.w * val2.w;
            float4 mask4 = (float4)(mask, mask, mask, mask);

            avg1 += val1 * mask4;
            avg2 += val2 * mask4;
            mask_sum += mask;
        }
    }
    write_imagef(img1avg, pos, avg1 / mask_sum);
    write_imagef(img2avg, pos, avg2 / mask_sum);
    write_imagef(fixed_out, pos, (float4)(fixed, 0, 0));
}


__kernel void corr(read_only image2d_t img1, __const int2 img1dim, 
                   read_only image2d_t img2, __const int2 img2dim,
                   read_only image3d_t img1avg, read_only image3d_t img2avg,
                   read_only image3d_t fixed_in, write_only image3d_t imgO, __const int2 outdim)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    int4 pos = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

    float2 center2 = (float2)(img2dim.x / 2, img2dim.y / 2);
    
    float4 fixed = read_imagef(fixed_in, pos);
    float sin_rot = fixed.z;
    float cos_rot = fixed.w;

    float4 avg1 = read_imagef(img1avg, pos);
    float4 avg2 = read_imagef(img2avg, pos);
    
    float4 sum1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum3 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i=0; i < img1dim.x; i++) {
        for (int j=0; j < img1dim.y; j++) {
            float2 pos1 = (float2)(i, j);
            float2 pos2 =((float2)(cos_rot * i + sin_rot * j, 
                                  -sin_rot * i + cos_rot * j)) + fixed.xy;
            
            float4 val1 = read_imagef(img1, sampler, pos1);
            float4 val2 = read_imagef(img2, sampler, pos2);
            
            float4 mask1 = (float4)(val1.wwww);
            float4 mask2 = (float4)(val2.wwww);

            float4 diff1 = val1 - avg1;
            float4 diff2 = val2 - avg2;
            
            sum1 += diff1 * diff2 * mask1 * mask2;
            sum2 += diff1 * diff1 * mask1 * mask2;
            sum3 += diff2 * diff2 * mask1 * mask2;
        }
    }
    float4 result = sum1 / (sqrt(sum2) * sqrt(sum3));
    write_imagef(imgO, pos, result);
}
