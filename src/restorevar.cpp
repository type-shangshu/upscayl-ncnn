#include "restorevar.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{

const int kPatchNums[10] = {1, 2, 3, 4, 6, 9, 13, 18, 24, 32};
const int kRopeDim = 64;
const int kRopeHalfDim = 32;
const int kImageSize = 512;

void copy_tokens(ncnn::Mat& destination, int offset, const ncnn::Mat& source)
{
    for (int row = 0; row < source.h; ++row)
    {
        std::memcpy(
            destination.row(offset + row),
            source.row(row),
            source.w * sizeof(float));
    }
}

} // namespace

RestoreVAR::RestoreVAR(int gpuid)
{
    net.opt.use_vulkan_compute = gpuid >= 0;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 4;
    if (gpuid >= 0)
        net.set_vulkan_device(gpuid);
}

int RestoreVAR::load(const std::string& parampath, const std::string& modelpath)
{
    int ret = net.load_param(parampath.c_str());
    if (ret != 0)
        return ret;
    ret = net.load_model(modelpath.c_str());
    if (ret != 0)
        return ret;
    build_rope();
    std::fprintf(stderr, "restorevar: model loaded\n");
    return 0;
}

void RestoreVAR::build_rope()
{
    int total_tokens = 0;
    for (int stage = 0; stage < 10; ++stage)
        total_tokens += kPatchNums[stage] * kPatchNums[stage];

    rope_cos.create(kRopeHalfDim, total_tokens);
    rope_sin.create(kRopeHalfDim, total_tokens);

    float freqs[kRopeDim / 4];
    for (int index = 0; index < kRopeDim / 4; ++index)
    {
        const float exponent = static_cast<float>(index * 4) / kRopeDim;
        freqs[index] = 1.f / std::pow(10000.f, exponent);
    }

    int row = 0;
    const int grid = kPatchNums[9];
    for (int stage = 0; stage < 10; ++stage)
    {
        const int patch = kPatchNums[stage];
        for (int y = 0; y < patch; ++y)
        {
            for (int x = 0; x < patch; ++x)
            {
                float* cos_row = rope_cos.row(row);
                float* sin_row = rope_sin.row(row);
                const float tx = static_cast<float>(y) / patch * grid;
                const float ty = static_cast<float>(x) / patch * grid;
                for (int index = 0; index < kRopeDim / 4; ++index)
                {
                    const float phase_x = tx * freqs[index];
                    const float phase_y = ty * freqs[index];
                    cos_row[index] = std::cos(phase_x);
                    sin_row[index] = std::sin(phase_x);
                    cos_row[index + kRopeDim / 4] = std::cos(phase_y);
                    sin_row[index + kRopeDim / 4] = std::sin(phase_y);
                }
                row++;
            }
        }
    }
}

bool RestoreVAR::run_prefixed_net(
    const std::string& prefix,
    const std::vector<ncnn::Mat>& inputs,
    int output_count,
    std::vector<ncnn::Mat>& outputs)
{
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(true);
    for (size_t index = 0; index < inputs.size(); ++index)
    {
        const std::string name = prefix + "__in" + std::to_string(index);
        if (extractor.input(name.c_str(), inputs[index]) != 0)
            return false;
    }
    outputs.resize(output_count);
    for (int index = 0; index < output_count; ++index)
    {
        const std::string name = prefix + "__out" + std::to_string(index);
        if (extractor.extract(name.c_str(), outputs[index]) != 0 ||
            outputs[index].empty())
            return false;
    }
    return true;
}

int RestoreVAR::process(const ncnn::Mat& inimage, ncnn::Mat& outimage)
{
    const unsigned char* pixeldata = static_cast<const unsigned char*>(inimage.data);
    const int channels = inimage.elempack;
    if (channels != 3 && channels != 4)
    {
        std::fprintf(stderr, "restorevar only supports RGB/RGBA input, got c=%d\n", channels);
        return -1;
    }

    ncnn::Mat image(kImageSize, kImageSize, 3);
    for (int oy = 0; oy < kImageSize; ++oy)
    {
        const float fy =
            (oy + 0.5f) * static_cast<float>(inimage.h) / kImageSize - 0.5f;
        const int y0 = std::max(0, std::min(inimage.h - 1, static_cast<int>(std::floor(fy))));
        const int y1 = std::max(0, std::min(inimage.h - 1, y0 + 1));
        const float wy = fy - std::floor(fy);
        for (int ox = 0; ox < kImageSize; ++ox)
        {
            const float fx =
                (ox + 0.5f) * static_cast<float>(inimage.w) / kImageSize - 0.5f;
            const int x0 = std::max(0, std::min(inimage.w - 1, static_cast<int>(std::floor(fx))));
            const int x1 = std::max(0, std::min(inimage.w - 1, x0 + 1));
            const float wx = fx - std::floor(fx);
            for (int channel = 0; channel < 3; ++channel)
            {
                const float p00 = pixeldata[(y0 * inimage.w + x0) * channels + channel];
                const float p01 = pixeldata[(y0 * inimage.w + x1) * channels + channel];
                const float p10 = pixeldata[(y1 * inimage.w + x0) * channels + channel];
                const float p11 = pixeldata[(y1 * inimage.w + x1) * channels + channel];
                const float top = p00 + (p01 - p00) * wx;
                const float bottom = p10 + (p11 - p10) * wx;
                image.channel(channel)[oy * kImageSize + ox] =
                    (top + (bottom - top) * wy) / 127.5f - 1.f;
            }
        }
    }

    std::vector<ncnn::Mat> outputs;
    std::fprintf(stderr, "restorevar: running condition encoder\n");
    if (!run_prefixed_net("condition_encoder", {image}, 1, outputs))
        return -2;
    ncnn::Mat encoded = outputs[0];

    if (!run_prefixed_net("initial_tokens", {encoded}, 2, outputs))
        return -3;
    ncnn::Mat x = outputs[0];
    ncnn::Mat condition = outputs[1];

    std::vector<ncnn::Mat> cache_k(16);
    std::vector<ncnn::Mat> cache_v(16);
    for (int index = 0; index < 16; ++index)
    {
        cache_k[index].create(64, 1, 16);
        cache_v[index].create(64, 1, 16);
        cache_k[index].fill(0.f);
        cache_v[index].fill(0.f);
    }

    ncnn::Mat f_hat(32, 32, 32);
    f_hat.fill(0.f);
    ncnn::Mat all_blocks(1024, 2240);
    int sequence_offset = 0;

    for (int stage = 0; stage < 10; ++stage)
    {
        const int patch = kPatchNums[stage];
        const int tokens = patch * patch;
        ncnn::Mat stage_cos = rope_cos.row_range(sequence_offset, tokens);
        ncnn::Mat stage_sin = rope_sin.row_range(sequence_offset, tokens);
        std::fprintf(stderr, "restorevar: stage %d patch=%d tokens=%d\n", stage, patch, tokens);

        for (int block = 0; block < 16; ++block)
        {
            char block_name[32];
            std::snprintf(block_name, sizeof(block_name), "var_block_%02d", block);
            ncnn::Mat bias(cache_k[block].h + tokens);
            bias.fill(0.f);
            bias[0] = -1e30f;
            if (!run_prefixed_net(
                    block_name,
                    {x, condition, cache_k[block], cache_v[block], stage_cos, stage_sin, bias},
                    3,
                    outputs))
                return -4;
            x = outputs[0];
            cache_k[block] = outputs[1];
            cache_v[block] = outputs[2];
        }

        copy_tokens(all_blocks, sequence_offset, x);
        sequence_offset += tokens;

        if (!run_prefixed_net("logits_head", {x}, 1, outputs))
            return -5;
        ncnn::Mat logits = outputs[0];

        ncnn::Mat indices(tokens, static_cast<size_t>(4u), 1);
        int* index_data = reinterpret_cast<int*>(indices.data);
        for (int token = 0; token < tokens; ++token)
        {
            const float* row = logits.row(token);
            index_data[token] =
                static_cast<int>(std::max_element(row, row + 4096) - row);
        }

        char update_name[32];
        std::snprintf(update_name, sizeof(update_name), "token_update_%02d", stage);
        const int update_outputs = stage == 9 ? 1 : 2;
        if (!run_prefixed_net(update_name, {indices, f_hat}, update_outputs, outputs))
            return -6;
        f_hat = outputs[0];
        if (stage != 9)
            x = outputs[1];
    }

    ncnn::Mat latent_tokens(32, 1024);
    for (int token = 0; token < 1024; ++token)
    {
        float* row = latent_tokens.row(token);
        for (int channel = 0; channel < 32; ++channel)
            row[channel] = f_hat.channel(channel)[token];
    }

    std::fprintf(stderr, "restorevar: running refiner\n");
    if (!run_prefixed_net("refiner", {latent_tokens, all_blocks}, 1, outputs))
        return -7;
    latent_tokens = outputs[0];

    for (int token = 0; token < 1024; ++token)
    {
        const float* row = latent_tokens.row(token);
        for (int channel = 0; channel < 32; ++channel)
            f_hat.channel(channel)[token] = row[channel];
    }

    std::fprintf(stderr, "restorevar: running image decoder\n");
    if (!run_prefixed_net("image_decoder", {f_hat}, 1, outputs))
        return -8;

    const ncnn::Mat restored = outputs[0];
    unsigned char* result = static_cast<unsigned char*>(
        std::malloc(restored.w * restored.h * 3));
    if (!result)
        return -9;

    for (int channel = 0; channel < 3; ++channel)
    {
        const float* values = restored.channel(channel);
        for (int index = 0; index < restored.w * restored.h; ++index)
        {
            const float value = std::max(0.f, std::min(1.f, values[index]));
            result[index * 3 + channel] =
                static_cast<unsigned char>(value * 255.f + 0.5f);
        }
    }

    outimage = ncnn::Mat(restored.w, restored.h, result, static_cast<size_t>(3), 3);
    return 0;
}
