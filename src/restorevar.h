#ifndef RESTOREVAR_H
#define RESTOREVAR_H

#include <string>
#include <vector>

#include "net.h"

class RestoreVAR
{
public:
    RestoreVAR(int gpuid);

    int load(const std::string& parampath, const std::string& modelpath);
    int process(const ncnn::Mat& inimage, ncnn::Mat& outimage);

private:
    bool run_prefixed_net(
        const std::string& prefix,
        const std::vector<ncnn::Mat>& inputs,
        int output_count,
        std::vector<ncnn::Mat>& outputs);
    void build_rope();

private:
    ncnn::Net net;
    ncnn::Mat rope_cos;
    ncnn::Mat rope_sin;
};

#endif
