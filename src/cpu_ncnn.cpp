#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "net.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::fprintf(stderr, "usage: %s model.param model.bin input.png output.png\n", argv[0]);
        return 2;
    }

    int width = 0;
    int height = 0;
    int source_channels = 0;
    unsigned char* pixels = stbi_load(argv[3], &width, &height, &source_channels, 3);
    if (!pixels)
    {
        std::fprintf(stderr, "failed to read %s\n", argv[3]);
        return 3;
    }
    if (width != 128 || height != 128)
    {
        std::fprintf(stderr, "model requires 128x128 input, got %dx%d\n", width, height);
        stbi_image_free(pixels);
        return 4;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 8;

    int ret = net.load_param(argv[1]);
    std::fprintf(stderr, "load_param=%d\n", ret);
    if (ret != 0) return 5;
    ret = net.load_model(argv[2]);
    std::fprintf(stderr, "load_model=%d\n", ret);
    if (ret != 0) return 6;

    ncnn::Mat input(width, height, 3);
    for (int channel = 0; channel < 3; channel++)
    {
        float* channel_data = input.channel(channel);
        for (int i = 0; i < width * height; i++)
            channel_data[i] = pixels[i * 3 + channel] / 255.f;
    }
    stbi_image_free(pixels);
    std::fprintf(stderr, "prepared input w=%d h=%d c=%d\n", input.w, input.h, input.c);

    const char* input_blob = std::getenv("NCNN_INPUT_BLOB");
    if (!input_blob) input_blob = "in0";

    ncnn::Extractor extractor = net.create_extractor();
    ret = extractor.input(input_blob, input);
    std::fprintf(stderr, "input blob=%s ret=%d\n", input_blob, ret);
    if (ret != 0) return 7;

    const char* output_blob = std::getenv("NCNN_OUTPUT_BLOB");
    if (!output_blob) output_blob = "out0";
    ncnn::Mat output;
    ret = extractor.extract(output_blob, output);
    std::fprintf(stderr, "extract blob=%s ret=%d dims=%d w=%d h=%d c=%d total=%zu\n",
                 output_blob, ret, output.dims, output.w, output.h, output.c, output.total());
    if (ret != 0 || output.empty()) return 8;

    float minimum = output[0];
    float maximum = output[0];
    double sum = 0.0;
    size_t finite_count = 0;
    for (int channel = 0; channel < output.c; channel++)
    {
        float* values = output.channel(channel);
        for (int i = 0; i < output.w * output.h; i++)
        {
            const float value = values[i];
            if (std::isfinite(value))
            {
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                sum += value;
                finite_count++;
            }
        }
    }
    std::fprintf(stderr, "finite=%zu/%zu min=%g max=%g mean=%g\n",
                 finite_count, output.total(), minimum, maximum,
                 finite_count ? sum / finite_count : 0.0);

    if (strcmp(output_blob, "out0") != 0 && strcmp(output_blob, "output") != 0)
    {
        std::fprintf(stderr, "intermediate blob requested, skip image write\n");
        return 0;
    }

    if (output.c != 3)
    {
        std::fprintf(stderr, "final output is not 3-channel, got c=%d\n", output.c);
        return 10;
    }

    std::vector<unsigned char> result(output.w * output.h * 3);
    for (int channel = 0; channel < 3; channel++)
    {
        const float* values = output.channel(channel);
        for (int i = 0; i < output.w * output.h; i++)
        {
            const float value = std::max(0.f, std::min(1.f, values[i]));
            result[i * 3 + channel] = static_cast<unsigned char>(value * 255.f + 0.5f);
        }
    }
    if (!stbi_write_png(argv[4], output.w, output.h, 3, result.data(), output.w * 3))
    {
        std::fprintf(stderr, "failed to write %s\n", argv[4]);
        return 9;
    }
    std::fprintf(stderr, "saved %s\n", argv[4]);
    return 0;
}
