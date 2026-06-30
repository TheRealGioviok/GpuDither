#include <png.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "ocl_boiler.h"

#define CHANNEL_COUNT 4 
#define FS_W        3
#define FS_SIZE_DW  4

static const float fs_weights[FS_SIZE_DW] = {
    7 / 16.0f,
    3 / 16.0f,
    5 / 16.0f,
    1 / 16.0f
};

typedef struct {
    int width;
    int height;
    unsigned char *pixels;
} image_t;

// Forward decls
static const char *kernel_from_mode(const char *mode);
static image_t read_png(const char *filename);
static void write_png(const char *filename, int width, int height, unsigned char *pixels);
static float *img_to_float(const unsigned char *src, size_t count);
static void float_to_img(const float *f, unsigned char *dst, size_t count);

static void run_floyd_steinberg(const char *kernel_file, const char *mode, const image_t *img, int bits, unsigned char *out, cl_context context, cl_device_id device, cl_command_queue queue);
static void run_parallel_dither(const char *kernel_file, const image_t *img, int bits, unsigned char *out, cl_context context, cl_device_id device, cl_command_queue queue);

static image_t generate_random_image(int side)
{
    srand((unsigned)47); // 47 my beloved
    unsigned char *pixels = malloc(side * side * CHANNEL_COUNT);
    if (!pixels) { perror("random image alloc"); exit(EXIT_FAILURE); }
    for (int i = 0; i < side * side; ++i) {
        pixels[i * CHANNEL_COUNT + 0] = (unsigned char)(rand() & 0xFF); /* R */
        pixels[i * CHANNEL_COUNT + 1] = (unsigned char)(rand() & 0xFF); /* G */
        pixels[i * CHANNEL_COUNT + 2] = (unsigned char)(rand() & 0xFF); /* B */
        pixels[i * CHANNEL_COUNT + 3] = 0xFF;                           /* A */
    }
    return (image_t){ side, side, pixels };
}

int main(int argc, char **argv)
{
    int gen_side = 0;
    int argi = 1;

    if (argc > 2 && !strcmp(argv[argi], "--gen")) {
        gen_side = atoi(argv[argi + 1]);
        if (gen_side < 1) {
            fprintf(stderr, "Error: --gen side must be >= 1\n");
            exit(EXIT_FAILURE);
        }
        argi += 2;
    }

    if (argc - argi < 3) {
        fprintf(stderr, "usage:\n%s [--gen N] in.png out.png mode bits\n", argv[0]);
        fprintf(stderr, "\t\t--gen N   use a random NxN RGBA image instead of reading in.png\n");
        exit(EXIT_FAILURE);
    }

    const char *in_file  = gen_side ? NULL : argv[argi++];
    const char *out_file = argv[argi++];
    const char *mode     = argv[argi++];
    int         bits     = (argi < argc) ? atoi(argv[argi]) : 0;

    const char *kernel_file = kernel_from_mode(mode);

    image_t img = gen_side ? generate_random_image(gen_side)
                           : read_png(in_file);
    if (gen_side)
        printf("generated random %dx%d RGBA image\n", img.width, img.height);

    if (bits < 1 || bits > 8) {
        fprintf(stderr, "Error: bits must be between 1 and 8\n");
        free(img.pixels);
        exit(EXIT_FAILURE);
    }

    size_t n = img.width * img.height * CHANNEL_COUNT;
    unsigned char *out = malloc(n);
    if (!out) {
        perror("Allocation failed for output buffer");
        free(img.pixels);
        exit(EXIT_FAILURE);
    }

    // OpenCL Boilerplate Initialization
    cl_platform_id platform = select_platform();
    cl_device_id device     = select_device(platform);
    cl_context context      = create_context(platform, device);
    cl_command_queue queue  = create_queue(context, device);

    bool is_floyd = (!strcmp(mode, "floyd_steinberg_safe") || 
    !strcmp(mode, "floyd_steinberg_safe_bruno") || 
                     !strcmp(mode, "floyd_steinberg_vec") || 
                     !strcmp(mode, "floyd_steinberg_vec2") || 
                    !strcmp(mode, "floyd_steinberg_vec3"));

    if (is_floyd) {
        run_floyd_steinberg(kernel_file, mode, &img, bits, out, context, device, queue);
    } else {
        run_parallel_dither(kernel_file, &img, bits, out, context, device, queue);
    }

    if (!gen_side){
        write_png(out_file, img.width, img.height, out);
    }
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(img.pixels);
    free(out);

    return 0;
}

static void run_floyd_steinberg(const char *kernel_file, const char *mode, const image_t *img, int bits, unsigned char *out, cl_context context, cl_device_id device, cl_command_queue queue)
{
    cl_int err;
    int levels = 1 << bits;
    float step = 255.0f / (float)(levels - 1);
    size_t n_floats = img->width * img->height * CHANNEL_COUNT;

    float *data_host = img_to_float(img->pixels, n_floats);

    cl_mem data_buf = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        n_floats * sizeof(float), data_host, &err);
    ocl_check(err, "create data buffer");

    cl_mem filter_buf = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        FS_SIZE_DW * sizeof(float), (void *)fs_weights, &err);
    ocl_check(err, "create diff buffer");

    cl_program program = create_program(kernel_file, context, device);
    cl_kernel kernel   = clCreateKernel(program, "dither", &err);
    ocl_check(err, "create kernel");
    
    int block_size;

    size_t max_wgs;
    
    int w_arg       = FS_W;
    int size_dw_arg = FS_SIZE_DW;
    int ch_arg      = CHANNEL_COUNT;

    if (!strcmp(mode, "floyd_steinberg_safe") || !strcmp(mode, "floyd_steinberg_safe_bruno")) {
        err  = clSetKernelArg(kernel, 0, sizeof(data_buf),   &data_buf);
        err |= clSetKernelArg(kernel, 1, sizeof(filter_buf), &filter_buf);
        err |= clSetKernelArg(kernel, 2, sizeof(int),        &img->width);
        err |= clSetKernelArg(kernel, 3, sizeof(int),        &img->height);
        err |= clSetKernelArg(kernel, 4, sizeof(int),        &w_arg);
        err |= clSetKernelArg(kernel, 5, sizeof(int),        &size_dw_arg);
        err |= clSetKernelArg(kernel, 6, sizeof(float),      &step);
        err |= clSetKernelArg(kernel, 7, sizeof(int),    &ch_arg);

        block_size = (img->width + 2) / 3;
    }
    else if (!strcmp(mode, "floyd_steinberg_vec")) {
        err  = clSetKernelArg(kernel, 0, sizeof(data_buf),   &data_buf);
        err |= clSetKernelArg(kernel, 1, sizeof(filter_buf), &filter_buf);
        err |= clSetKernelArg(kernel, 2, sizeof(int),        &img->width);
        err |= clSetKernelArg(kernel, 3, sizeof(int),        &img->height);
        err |= clSetKernelArg(kernel, 4, sizeof(int),        &w_arg);
        err |= clSetKernelArg(kernel, 5, sizeof(int),        &size_dw_arg);
        err |= clSetKernelArg(kernel, 6, sizeof(float),      &step);

        block_size = (img->width + 2) / 3;
    }
    else { // vec 2 and 3 have same args
        err  = clSetKernelArg(kernel, 0, sizeof(data_buf),   &data_buf);
        err |= clSetKernelArg(kernel, 1, sizeof(int),        &img->width);
        err |= clSetKernelArg(kernel, 2, sizeof(int),        &img->height);
        err |= clSetKernelArg(kernel, 3, sizeof(float),      &step);

        block_size = (img->width + 1) / 2;
    }

    if (block_size < 1) block_size = 1;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                            sizeof(max_wgs), &max_wgs, NULL);
    if ((size_t)block_size > max_wgs) {
        block_size = (int)max_wgs;
    }
    printf("Blocksize: %d\n", block_size);

    size_t lws = (size_t)block_size;
    size_t gws = lws;

    ocl_check(err, "set kernel args");

    cl_event evt;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, &lws, 0, NULL, &evt);
    ocl_check(err, "enqueue kernel");

    clWaitForEvents(1, &evt);
    printf("kernel time %.3f ms\n", runtime_ms(evt));

    float *result_f = malloc(n_floats * sizeof(float));
    err = clEnqueueReadBuffer(queue, data_buf, CL_TRUE, 0, n_floats * sizeof(float), result_f, 0, NULL, NULL);
    ocl_check(err, "read result");

    float_to_img(result_f, out, n_floats);

    free(result_f);
    free(data_host);
    clReleaseMemObject(data_buf);
    clReleaseMemObject(filter_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

static void run_parallel_dither(const char *kernel_file, const image_t *img, int bits, unsigned char *out, cl_context context, cl_device_id device, cl_command_queue queue)
{
    cl_int err;
    size_t n = img->width * img->height * CHANNEL_COUNT;

    cl_program program = create_program(kernel_file, context, device);
    cl_kernel kernel   = clCreateKernel(program, "dither", &err);
    ocl_check(err, "create kernel");

    cl_mem srcbuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, img->pixels, &err);
    ocl_check(err, "src buffer");

    cl_mem dstbuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n, NULL, &err);
    ocl_check(err, "dst buffer");

    err  = clSetKernelArg(kernel, 0, sizeof(srcbuf),  &srcbuf);
    err |= clSetKernelArg(kernel, 1, sizeof(dstbuf),  &dstbuf);
    err |= clSetKernelArg(kernel, 2, sizeof(int),     &img->width);
    err |= clSetKernelArg(kernel, 3, sizeof(int),     &bits);
    ocl_check(err, "set kernel args");

    size_t gws[2] = { (size_t)img->width, (size_t)img->height };

    cl_event evt;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, NULL, 0, NULL, &evt);
    ocl_check(err, "enqueue kernel");

    clWaitForEvents(1, &evt);
    printf("kernel time %.3f ms\n", runtime_ms(evt));

    err = clEnqueueReadBuffer(queue, dstbuf, CL_TRUE, 0, n, out, 0, NULL, NULL);
    ocl_check(err, "read result");

    clReleaseMemObject(srcbuf);
    clReleaseMemObject(dstbuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

static const char *kernel_from_mode(const char *mode)
{
    if (!strcmp(mode, "threshold"))                  return "kernels/threshold.ocl";
    if (!strcmp(mode, "random"))                     return "kernels/random.ocl";
    if (!strcmp(mode, "bayer2"))                     return "kernels/bayer2.ocl";
    if (!strcmp(mode, "bayer4"))                     return "kernels/bayer4.ocl";
    if (!strcmp(mode, "bayer8"))                     return "kernels/bayer8.ocl";
    if (!strcmp(mode, "floyd_steinberg_safe"))       return "kernels/floyd_steinberg_safe.ocl";
    if (!strcmp(mode, "floyd_steinberg_safe_bruno"))       return "kernels/floyd_steinberg_safe_bruno.ocl";
    if (!strcmp(mode, "floyd_steinberg_vec")) return "kernels/floyd_steinberg_vec.ocl";
    if (!strcmp(mode, "floyd_steinberg_vec2"))   return "kernels/floyd_steinberg_vec2.ocl";
    if (!strcmp(mode, "floyd_steinberg_vec3"))   return "kernels/floyd_steinberg_vec3.ocl";

    fprintf(stderr, "unknown mode: %s\n", mode);
    exit(EXIT_FAILURE);
}

static image_t read_png(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror(filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info  = png_create_info_struct(png);

    png_init_io(png, fp);
    png_read_info(png, info);

    int width  = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);

    png_byte color = png_get_color_type(png, info);
    png_byte depth = png_get_bit_depth(png, info);

    if (depth == 16)                     png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY || 
        color == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if (color & PNG_COLOR_MASK_ALPHA)    png_set_strip_alpha(png);

    // Dummy alpha channel
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    png_read_update_info(png, info);

    unsigned char *img = malloc(width * height * CHANNEL_COUNT);
    png_bytep *rows    = malloc(height * sizeof(*rows));

    for (int y = 0; y < height; y++) {
        rows[y] = img + y * width * CHANNEL_COUNT;
    }

    png_read_image(png, rows);

    free(rows);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    image_t out = { width, height, img };
    return out;
}

static void write_png(const char *filename, int width, int height, unsigned char *pixels)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror(filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info  = png_create_info_struct(png);

    png_init_io(png, fp);

    png_set_IHDR(
        png, info, width, height, 8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);

    png_bytep *rows = malloc(height * sizeof(*rows));
    for (int y = 0; y < height; y++) {
        rows[y] = pixels + y * width * CHANNEL_COUNT;
    }

    png_write_image(png, rows);
    png_write_end(png, NULL);

    free(rows);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

static float *img_to_float(const unsigned char *src, size_t count)
{
    float *f = malloc(count * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        f[i] = (float)src[i];
    }
    return f;
}

static void float_to_img(const float *f, unsigned char *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        float v = f[i];
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        dst[i] = (unsigned char)v;
    }
}