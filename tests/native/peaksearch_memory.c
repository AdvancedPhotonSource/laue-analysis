#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "liblaue.h"

enum {
    IMAGE_WIDTH = 64,
    IMAGE_HEIGHT = 64,
    REPEAT_COUNT = 5
};

static void add_gaussian(unsigned short *pixels, int center_x, int center_y)
{
    int x;
    int y;

    for (y = 0; y < IMAGE_HEIGHT; ++y) {
        for (x = 0; x < IMAGE_WIDTH; ++x) {
            double dx = x - center_x;
            double dy = y - center_y;
            unsigned short signal = (unsigned short)(2000.0 * exp(-(dx * dx + dy * dy) / 8.0));
            pixels[y * IMAGE_WIDTH + x] += signal;
        }
    }
}

int main(void)
{
    unsigned short pixels[IMAGE_WIDTH * IMAGE_HEIGHT];
    laue_peak_params params = {0};
    int iteration;
    int index;

    for (index = 0; index < IMAGE_WIDTH * IMAGE_HEIGHT; ++index) pixels[index] = 10;
    add_gaussian(pixels, 16, 16);
    add_gaussian(pixels, 32, 20);
    add_gaussian(pixels, 46, 44);

    params.boxsize = 6;
    params.max_rfactor = 1.0;
    params.min_size = 2;
    params.min_separation = 5;
    params.threshold = 100.0;
    params.threshold_ratio = 4.0;
    params.peak_shape = 0;
    params.max_peaks = 100;
    params.detect_binning = 1;

    for (iteration = 0; iteration < REPEAT_COUNT; ++iteration) {
        laue_frame_result result = {0};
        int status = laue_find_peaks(pixels, IMAGE_WIDTH, IMAGE_HEIGHT, &params, &result);

        if (status != LAUE_OK) {
            fprintf(stderr, "peak search %d failed: %s\n", iteration, result.message);
            laue_frame_result_free(&result);
            return status;
        }
        laue_frame_result_free(&result);
    }

    return 0;
}
