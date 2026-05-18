#include <api/api.h>

protected const char* ocl_get_source_subtract_images(void)
{
	return
	"__kernel void subtract_images(__global unsigned char* img_a, __global unsigned char* img_b,\n"
	"                              __global unsigned char* result, int width, int height, int stride, int min_val, int max_val)\n"
	"{\n"
	"	int x = get_global_id(0);\n"
	"	int y = get_global_id(1);\n"
	"	if (x < width && y < height)\n"
	"	{\n"
	"		int idx = y * stride + x;\n"
	"		int diff = (int)img_a[idx] - (int)img_b[idx];\n"
	"		if (diff < min_val) diff = min_val;\n"
	"		if (diff > max_val) diff = max_val;\n"
	"		result[idx] = (unsigned char)diff;\n"
	"	}\n"
	"}\n";
}

protected const char* ocl_get_source_histogram(void)
{
	return
	"__kernel void histogram(__global unsigned char* image, __global unsigned int* hist,\n"
	"                        int width, int height, int stride)\n"
	"{\n"
	"	int x = get_global_id(0);\n"
	"	int y = get_global_id(1);\n"
	"	if (x < width && y < height)\n"
	"	{\n"
	"		int idx = y * stride + x;\n"
	"		unsigned char pixel = image[idx];\n"
	"		atomic_inc(&hist[pixel]);\n"
	"	}\n"
	"}\n";
}

protected const char* ocl_get_source_brightest_spot(void)
{
	return
	"float2 subpixel_refine(__global const unsigned char* img, int w, int h, int stride,\n"
	"                       int px, int py, int radius)\n"
	"{\n"
	"	float sx=0, sy=0, sw=0;\n"
	"	int xs=max(px-radius,0), ys=max(py-radius,0);\n"
	"	int xe=min(px+radius,w-1), ye=min(py+radius,h-1);\n"
	"	for(int y=ys; y<=ye; y++)\n"
	"		for(int x=xs; x<=xe; x++)\n"
	"		{\n"
	"			float wt=(float)img[y*stride+x];\n"
	"			sx+=wt*x; sy+=wt*y; sw+=wt;\n"
	"		}\n"
	"	return (sw>0) ? (float2)(sx/sw, sy/sw) : (float2)((float)px, (float)py);\n"
	"}\n"
	"\n"
	"__kernel void brightest_spot(__global const unsigned char* img,\n"
	"                             __global float* res, int w, int h, int stride,\n"
	"                             int cx, int cy, int rw, int rh, int sub_r)\n"
	"{\n"
	"	int hs=rw/2, hh=rh/2;\n"
	"	int xs=cx-hs, ys=cy-hh, xe=cx-hs+rw, ye=cy-hh+rh;\n"
	"	if(rw<=0||rh<=0) { xs=0; ys=0; xe=w; ye=h; }\n"
	"	xs=clamp(xs,0,w); ys=clamp(ys,0,h); xe=clamp(xe,0,w); ye=clamp(ye,0,h);\n"
	"	int bx=xs, by=ys;\n"
	"	unsigned char bv=0;\n"
	"	for(int y=ys; y<ye; y++)\n"
	"		for(int x=xs; x<xe; x++)\n"
	"			if(img[y*stride+x] > bv) { bv=img[y*stride+x]; bx=x; by=y; }\n"
	"	float2 sp = subpixel_refine(img, w, h, stride, bx, by, sub_r);\n"
	"	res[0]=sp.x; res[1]=sp.y; res[2]=(float)bv;\n"
	"}\n";
}

protected const char* ocl_get_source_matvec_bf16(void)
{
	return
	"__kernel void matvec_bf16(__global float *y,\n"
	"                          __global const float *x,\n"
	"                          __global const uint *W,\n"
	"                          int in_dim, int out_dim)\n"
	"{\n"
	"	int o = get_global_id(0);\n"
	"	if (o >= out_dim) return;\n"
	"	const uint *row = W + (size_t)o * in_dim;\n"
	"	float sum = 0.0f;\n"
	"	for (int k = 0; k < in_dim; ++k)\n"
	"	{\n"
	"		float w = as_float(row[k]);\n"
	"		sum = fma(w, x[k], sum);\n"
	"	}\n"
	"	y[o] = sum;\n"
	"}\n";
}

protected const char* ocl_get_source_matvec_bf16_fused(void)
{
	return
	"__kernel void matvec_bf16_fused(__global float *y0, __global float *y1,\n"
	"                                __global const float *x,\n"
	"                                __global const ushort *W0,\n"
	"                                __global const ushort *W1,\n"
	"                                int in_dim, int out_dim)\n"
	"{\n"
	"	int o = get_global_id(0);\n"
	"	if (o >= out_dim) return;\n"
	"	const ushort *row0 = W0 + (size_t)o * in_dim;\n"
	"	const ushort *row1 = W1 + (size_t)o * in_dim;\n"
	"	float sum0 = 0.0f, sum1 = 0.0f;\n"
	"	for (int k = 0; k < in_dim; ++k)\n"
	"	{\n"
	"		uint bits0 = (uint)row0[k] << 16;\n"
	"		uint bits1 = (uint)row1[k] << 16;\n"
	"		sum0 = fma(as_float(bits0), x[k], sum0);\n"
	"		sum1 = fma(as_float(bits1), x[k], sum1);\n"
	"	}\n"
	"	y0[o] = sum0;\n"
	"	y1[o] = sum1;\n"
	"}\n";
}

protected const char* ocl_get_source_matvec_f32(void)
{
	return
	"__kernel void matvec_f32(__global float *y,\n"
	"                          __global const float *x,\n"
	"                          __global const float *W,\n"
	"                          int in_dim, int out_dim)\n"
	"{\n"
	"	int o = get_global_id(0);\n"
	"	if (o >= out_dim) return;\n"
	"	const float *row = W + (size_t)o * in_dim;\n"
	"	float sum = 0.0f;\n"
	"	for (int k = 0; k < in_dim; ++k)\n"
	"		sum = fma(row[k], x[k], sum);\n"
	"	y[o] = sum;\n"
	"}\n";
}
