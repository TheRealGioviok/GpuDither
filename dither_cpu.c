#include <png.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CH 3

typedef struct {
    int width;
    int height;
    unsigned char *pixels; 
} image_t;

static image_t read_png(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror(filename); exit(EXIT_FAILURE); }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop   info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_read_info(png, info);

    int      width  = png_get_image_width(png, info);
    int      height = png_get_image_height(png, info);
    png_byte color  = png_get_color_type(png, info);
    png_byte depth  = png_get_bit_depth(png, info);

    if (depth == 16)                        png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE)    png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY ||
        color == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if (color & PNG_COLOR_MASK_ALPHA)       png_set_strip_alpha(png);
    png_read_update_info(png, info);

    unsigned char *img  = malloc(width * height * CH);
    png_bytep     *rows = malloc(height * sizeof(*rows));
    for (int y = 0; y < height; ++y)
        rows[y] = img + y * width * CH;
    png_read_image(png, rows);

    free(rows);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return (image_t){ width, height, img };
}

static void write_png(const char *filename, int width, int height,
                      unsigned char *pixels)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) { perror(filename); exit(EXIT_FAILURE); }

    png_structp png  = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop   info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8,
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep *rows = malloc(height * sizeof(*rows));
    for (int y = 0; y < height; ++y)
        rows[y] = pixels + y * width * CH;
    png_write_image(png, rows);
    png_write_end(png, NULL);

    free(rows);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

static inline float quantize(float v, float step)
{
    float q = roundf(v / step) * step;
    if (q < 0.0f)   q = 0.0f;
    if (q > 255.0f) q = 255.0f;
    return q;
}

static void dither_threshold(const image_t *src, unsigned char *dst, float step)
{
    int n = src->width * src->height * CH;
    for (int i = 0; i < n; ++i)
        dst[i] = (unsigned char)quantize((float)src->pixels[i], step);
}

static void dither_random(const image_t *src, unsigned char *dst, float step)
{
    srand((unsigned)time(NULL));
    int n = src->width * src->height * CH;
    for (int i = 0; i < n; ++i) {
        float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * step;
        dst[i] = (unsigned char)quantize((float)src->pixels[i] + noise, step);
    }
}

static const float bayer2[2][2] = {
    { 0/4.0f, 2/4.0f },
    { 3/4.0f, 1/4.0f }
};

static const float bayer4[4][4] = {
    {  0/16.0f,  8/16.0f,  2/16.0f, 10/16.0f },
    { 12/16.0f,  4/16.0f, 14/16.0f,  6/16.0f },
    {  3/16.0f, 11/16.0f,  1/16.0f,  9/16.0f },
    { 15/16.0f,  7/16.0f, 13/16.0f,  5/16.0f }
};

static const float bayer8[8][8] = {
    {  0/64.0f, 32/64.0f,  8/64.0f, 40/64.0f,  2/64.0f, 34/64.0f, 10/64.0f, 42/64.0f },
    { 48/64.0f, 16/64.0f, 56/64.0f, 24/64.0f, 50/64.0f, 18/64.0f, 58/64.0f, 26/64.0f },
    { 12/64.0f, 44/64.0f,  4/64.0f, 36/64.0f, 14/64.0f, 46/64.0f,  6/64.0f, 38/64.0f },
    { 60/64.0f, 28/64.0f, 52/64.0f, 20/64.0f, 62/64.0f, 30/64.0f, 54/64.0f, 22/64.0f },
    {  3/64.0f, 35/64.0f, 11/64.0f, 43/64.0f,  1/64.0f, 33/64.0f,  9/64.0f, 41/64.0f },
    { 51/64.0f, 19/64.0f, 59/64.0f, 27/64.0f, 49/64.0f, 17/64.0f, 57/64.0f, 25/64.0f },
    { 15/64.0f, 47/64.0f,  7/64.0f, 39/64.0f, 13/64.0f, 45/64.0f,  5/64.0f, 37/64.0f },
    { 63/64.0f, 31/64.0f, 55/64.0f, 23/64.0f, 61/64.0f, 29/64.0f, 53/64.0f, 21/64.0f }
};

static void dither_bayer(const image_t *src, unsigned char *dst, float step,
                         int size)
{
    for (int y = 0; y < src->height; ++y) {
        for (int x = 0; x < src->width; ++x) {
            float t;
            if      (size == 2) t = bayer2[y & 1][x & 1];
            else if (size == 4) t = bayer4[y & 3][x & 3];
            else                t = bayer8[y & 7][x & 7];

            float bias = (t - 0.5f) * step;
            int base = (y * src->width + x) * CH;
            for (int c = 0; c < CH; ++c)
                dst[base + c] = (unsigned char)quantize((float)src->pixels[base + c] + bias, step);
        }
    }
}

static void dither_floyd_steinberg(const image_t *src, unsigned char *dst,
                                   float step)
{
    int w = src->width;
    int h = src->height;
    int n = w * h * CH;

    float *buf = malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i)
        buf[i] = (float)src->pixels[i];

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int base = (y * w + x) * CH;
            for (int c = 0; c < CH; ++c) {
                float old_v = buf[base + c];
                float new_v = quantize(old_v, step);
                float err   = old_v - new_v;
                buf[base + c] = new_v;

                if (x + 1 < w)
                    buf[(y*w + x+1)*CH + c]       += err * (7.0f/16.0f);
                if (y + 1 < h) {
                    if (x - 1 >= 0)
                        buf[((y+1)*w + x-1)*CH + c] += err * (3.0f/16.0f);
                    buf[((y+1)*w + x  )*CH + c]     += err * (5.0f/16.0f);
                    if (x + 1 < w)
                        buf[((y+1)*w + x+1)*CH + c] += err * (1.0f/16.0f);
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        float v = buf[i];
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        dst[i] = (unsigned char)v;
    }
    free(buf);
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s in.png out.png mode bits\n", argv[0]);
        fprintf(stderr, "modes: threshold random bayer2 bayer4 bayer8 floyd_steinberg\n");
        return EXIT_FAILURE;
    }

    const char *in_file  = argv[1];
    const char *out_file = argv[2];
    const char *mode     = argv[3];
    int         bits     = atoi(argv[4]);

    if (bits < 1 || bits > 8) {
        fprintf(stderr, "Error: bits must be between 1 and 8\n");
        return EXIT_FAILURE;
    }

    float step = 255.0f / (float)((1 << bits) - 1);

    image_t img = read_png(in_file);
    unsigned char *out = malloc(img.width * img.height * CH);
    if (!out) { perror("output alloc"); free(img.pixels); return EXIT_FAILURE; }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if      (!strcmp(mode, "threshold"))       dither_threshold(&img, out, step);
    else if (!strcmp(mode, "random"))          dither_random(&img, out, step);
    else if (!strcmp(mode, "bayer2"))          dither_bayer(&img, out, step, 2);
    else if (!strcmp(mode, "bayer4"))          dither_bayer(&img, out, step, 4);
    else if (!strcmp(mode, "bayer8"))          dither_bayer(&img, out, step, 8);
    else if (!strcmp(mode, "floyd_steinberg")) dither_floyd_steinberg(&img, out, step);
    else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        free(img.pixels); free(out);
        return EXIT_FAILURE;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
    printf("kernel time %.3f ms\n", ms);

    write_png(out_file, img.width, img.height, out);
    printf("wrote %s\n", out_file);

    free(img.pixels);
    free(out);
    return EXIT_SUCCESS;
}