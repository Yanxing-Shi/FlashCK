#pragma once

#include "lightinfer/core/utils/rocm_info.h"

#include "lightinfer/core/module/models/feed_data_map.h"

namespace lightinfer {

class ModelBase {
public:
    ModelBase()
    {
        GetSetDevice(device_id_);
    }

    virtual ~ModelBase() {}

    virtual void BuildGraph() = 0;

    virtual void SetInput(const FeedDataMap& input_data_map) = 0;

    virtual void SetOutput(const FeedDataMap& output_data_map) = 0;

    virtual FeedDataMap GetOutputData() = 0;

    virtual void Forward(hipStream_t stream = nullptr, bool graph_mode = false) = 0;

protected:
    int device_id_ = 0;

    hipEvent_t run_finished_;

    hipGraphExec_t graph_exec_ = nullptr;
    hipStream_t    graph_capture_stream_;
};
}  // namespace lightinfer