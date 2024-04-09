#include "ater/core/module/operations/gemm_universal/gemm_rcr_op.h"

#include "ater/core/module/operations/gemm_universal/gemm_common_utils.h"
#include "ater/core/profiler/alignment.h"
#include "ater/core/utils/enforce.h"

#include "ater/core/graph/node.h"

namespace ater {

template<typename T>
GemmRCROp<T>::GemmRCROp(): GemmCommonOp<T>::GemmCommonOp("gemm_rcr")
{
    GemmCommonOp<T>::op_name_           = "gemm_rcr";
    GemmCommonOp<T>::layout_            = DataLayout::RCR;
    GemmCommonOp<T>::epilogue_op_       = TensorOperation::PassThrough;
    GemmCommonOp<T>::ab_alignment_func_ = [&](int m, int n, int k) { return DefaultAlignAB(k, k, DataType::FLOAT32); };
}

template<typename T>
Shape GemmRCROp<T>::InferShape(Variable* A, Variable* B)
{
    auto              a_shape_vec = A->GetShape().ToVector();
    auto              b_shape_vec = B->GetShape().ToVector();
    std::vector<DDim> m           = std::vector<DDim>(a_shape_vec.begin(), a_shape_vec.end() - 1);
    DDim              n           = B->GetShape().ToVector().back();
    std::vector<DDim> c_infer_shape;
    std::transform(m.begin(), m.end(), std::back_inserter(c_infer_shape), [](const DDim& i) { return i; });
    c_infer_shape.push_back(n);

    return Shape(c_infer_shape);
}

template<typename T>
std::vector<int> GemmRCROp<T>::InvertExecKey(const std::string& key)
{
    return GemmInverseKeyFunc(key);
}

template<typename T>
std::vector<std::string> GemmRCROp<T>::GenProfileCmd(const std::string& profiler_prefix,
                                                     const std::string& profiler_filename,
                                                     const std::string& exec_key)
{
    auto fbuild_cmd = [&](const std::string& exec_key) {
        std::vector<int>         cmd = InvertExecKey(exec_key);
        std::vector<std::string> cmd_str;
        std::transform(cmd.begin(), cmd.end(), std::back_inserter(cmd_str), [](int i) { return std::to_string(i); });
        return cmd_str;
    };

    return GemmCommonOp<T>::GenOpProfileCmd(profiler_prefix, profiler_filename, exec_key, fbuild_cmd);
}

// (M, K) * (N, K) = (M, N)
// profiling always uses 2d * 2d.
template<typename T>
std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> GemmRCROp<T>::ExtractDims(bool for_profiling)
{

    auto a_shape_size = GemmCommonOp<T>::input_var_[0]->GetShape().GetNumDim();

    std::vector<int> dim_idx_0{0};
    std::vector<int> dim_idx_1{1};

    if (for_profiling) {
        return {{"M",
                 {std::make_shared<DimInfo>(TensorSource::Input, 0, dim_idx_0),
                  std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx_0)}},
                {"N",
                 {std::make_shared<DimInfo>(TensorSource::Input, 1, dim_idx_0),
                  std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx_1)}},
                {"K",
                 {std::make_shared<DimInfo>(TensorSource::Input, 0, dim_idx_1),
                  std::make_shared<DimInfo>(TensorSource::Input, 1, dim_idx_1)}}};
    }
    else {
        std::vector<int> dim_idx(a_shape_size - 1);
        std::iota(std::begin(dim_idx_0), std::end(dim_idx_0), 0);
        std::vector<int> dim_idx_shape{a_shape_size - 1};
        return {{"M",
                 {std::make_shared<DimInfo>(TensorSource::Input, 0, dim_idx),
                  std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx)}},
                {"N",
                 {std::make_shared<DimInfo>(TensorSource::Input, 1, dim_idx_0),
                  std::make_shared<DimInfo>(TensorSource::Output, 0, dim_idx_shape)}},
                {"K",
                 {std::make_shared<DimInfo>(TensorSource::Input, 0, dim_idx_shape),
                  std::make_shared<DimInfo>(TensorSource::Input, 1, dim_idx_1)}}};
    }
}

template<typename T>
void GemmRCROp<T>::AlignAB(Variable* A, Variable* B)
{
    auto a_shape = A->GetShape();
    auto b_shape = B->GetShape();

    if (a_shape.GetLastDim() != b_shape.GetLastDim()) {
        ATER_THROW(Unavailable("A/B shape mismatch, A: {}, B: {}", a_shape.ToString(), b_shape.ToString()));
    }
}

template class GemmRCROp<float>;
template class GemmRCROp<_Float16>;

}  // namespace ater