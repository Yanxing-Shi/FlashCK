#pragma once

#include "lightinfer/core/module/operations/gemm_universal_ops/gemm_common_op.h"

namespace lightinfer {

/*
Batch GEMM specialization for A[RowMajor], B[ColMajor], C[RowMajor].

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        Y_pt = torch.bmm(X_pt, W_pt)
*/

template<typename CppType>
class BmmRCROp: public GemmCommonOp<CppType, BmmRCROp<CppType>> {
public:
    BmmRCROp(std::string op_name = "bmm_rcr");

    Shape InferShapeImpl(Variable* a, Variable* b);

    Variable* operator()(Variable* a, Variable* b);

    std::function<std::vector<std::string>(const std::string&)> GenBuildCmd();

    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDimsImpl(bool for_profiling);

    void ForwardImpl();

    DDim GetBatchSize(Variable* a, Variable* b)
    {
        this->SanityCheck(a, b);

        auto a_shape = a->GetShape();
        auto b_shape = b->GetShape();
        if (a_shape.GetNumDim() == 2) {
            return b_shape.GetDim(0);
        }
        else if (b_shape.GetNumDim() == 2) {
            return a_shape.GetDim(0);
        }

        auto a_batch_size = a_shape.GetDim(0);
        auto b_batch_size = b_shape.GetDim(0);
        if (a_batch_size != b_batch_size && a_batch_size != 1 && b_batch_size != 1) {
            LI_THROW(Unavailable(
                "bmm operand A and B should have same batch_size, or batch_size = 1! Current shape A: {} shape B: {} .",
                a_shape.ToString(),
                b_shape.ToString()));
        }

        return a_shape.GetDim(0) != 1 ? a_shape.GetDim(0) : b_shape.GetDim(0);
    }

    void SanityCheck(Variable* a, Variable* b)
    {
        Shape a_shape = a->GetShape();
        Shape b_shape = b->GetShape();

        if (a_shape.GetNumDim() != 2 && a_shape.GetNumDim() != 3) {
            LI_THROW(
                Unavailable("bmm operand A should have 2 or 3 dimensions! Current shape: {}.", a_shape.ToString()));
        }

        if (b_shape.GetNumDim() != 2 && b_shape.GetNumDim() != 3) {
            LI_THROW(
                Unavailable("bmm operand A should have 2 or 3 dimensions! Current shape: {}.", b_shape.ToString()));
        }

        if (a_shape.GetNumDim() == 2 && b_shape.GetNumDim() == 2) {
            LI_THROW(Unavailable(
                "bmm operand A and B both have 2 dimensions! Use gemm instead. Current a_shape: {}, b_shape: {}",
                a_shape.ToString(),
                b_shape.ToString()));
        }

        if (a->GetDtype() != b->GetDtype()) {
            LI_THROW(Unavailable("bmm operand A and B should have the same dtype! Current a_dtype: {}, b_dtype: {}.",
                                 DataTypeToString(a->GetDtype()),
                                 DataTypeToString(b->GetDtype())));
        }
    }

    std::vector<std::shared_ptr<DimInfo>> CreateInputBatchDimInfo(const std::vector<Shape>&   input_shapes,
                                                                  const std::vector<int64_t>& batch_dims,
                                                                  const DDim&                 output_batch)
    {
        LI_ENFORCE_EQ(
            input_shapes.size(),
            batch_dims.size(),
            Unavailable(
                "input_shapes.size should be equal to batch_dims.size, but got input_shapes.size ={}  and batch_dims.size = {}",
                input_shapes.size(),
                batch_dims.size()));

        std::vector<std::shared_ptr<DimInfo>> dim_infos;
        for (int i = 0; i < input_shapes.size(); i++) {
            if (input_shapes[i].GetNumDim() > 2) {
                std::vector<int64_t> dim_idx = {batch_dims[i]};
                dim_infos.push_back(std::make_shared<DimInfo>(
                    TensorSource::Input, i, dim_idx, input_shapes[i].GetDim(batch_dims[i]) != output_batch));
            }
        }

        return dim_infos;
    }
};

}  // namespace lightinfer