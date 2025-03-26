#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "lightinfer/core/profiler/library.h"
#include "lightinfer/core/utils/dtype.h"
#include "lightinfer/core/utils/layout.h"
#include "lightinfer/core/utils/math_utils.h"

namespace lightinfer {

class GemmProblem {
public:
    GemmProblem() = default;

    GemmProblem(GemmOperationKind       operation_kind,
                int64_t                 m,
                int64_t                 n,
                int64_t                 k,
                DataType                a_dtype,
                DataType                b_dtype,
                DataType                c_dtype,
                DataType                acc_dtype,
                std::vector<DataType>   ds_dtype,
                DataType                e_dtype,
                DataLayout              layout,
                std::vector<LayoutType> ds_layout,
                TensorOperation         epilogue_op):
        operation_kind_(operation_kind),
        m_(m),
        n_(n),
        k_(k),
        a_dtype_(a_dtype),
        b_dtype_(b_dtype),
        c_dtype_(c_dtype),
        acc_dtype_(acc_dtype),
        ds_dtype_(ds_dtype),
        e_dtype_(e_dtype),
        layout_(layout),
        ds_layout_(ds_layout),
        epilogue_op_(epilogue_op)
    {
    }

    std::tuple<LayoutType, LayoutType, LayoutType> GetLayout() const
    {
        if (layout_ == DataLayout::RCR)
            return std::make_tuple(LayoutType::RowMajor, LayoutType::ColumnMajor, LayoutType::RowMajor);
        else {
            LI_THROW(Unimplemented("{}", "data layout not supported"));
        }
    }

    // type
    GemmOperationKind operation_kind_;

    // shape
    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;

    // data type
    DataType              a_dtype_   = DataType::UNDEFINED;
    DataType              b_dtype_   = DataType::UNDEFINED;
    DataType              c_dtype_   = DataType::UNDEFINED;
    DataType              acc_dtype_ = DataType::UNDEFINED;
    std::vector<DataType> ds_dtype_  = {};
    DataType              e_dtype_   = DataType::UNDEFINED;

    // layout
    DataLayout              layout_    = DataLayout::UNDEFINED;
    std::vector<LayoutType> ds_layout_ = {};

    // element-wise operation
    TensorOperation epilogue_op_ = TensorOperation::PassThrough;
};

GemmSpecialization GetGemmSpec(const int64_t m,
                               const int64_t n,
                               const int64_t k,
                               const int64_t m_per_block,
                               const int64_t n_per_block,
                               const int64_t k_per_block);

class GemmTileDesc {
public:
    GemmTileDesc() = default;
    GemmTileDesc(int64_t block_size,
                 int64_t m_per_block,
                 int64_t n_per_block,
                 int64_t k_per_block,
                 int64_t ak1,
                 int64_t bk1,
                 int64_t m_per_xdl,
                 int64_t n_per_xdl,
                 int64_t m_xdl_per_wave,
                 int64_t n_xdl_per_wave);

    std::string GetConfigName();

    std::string Emit();

    int64_t block_size_;
    int64_t m_per_block_, n_per_block_, k_per_block_;
    int64_t ak1_, bk1_;
    int64_t m_per_xdl_, n_per_xdl_;
    int64_t m_xdl_per_wave_, n_xdl_per_wave_;
};

class AttnTileDesc {
public:
    AttnTileDesc() = default;
    AttnTileDesc(int64_t block_size,
                 int64_t m_per_block,
                 int64_t n_per_block,
                 int64_t k_per_block,
                 int64_t gemm1_n_per_block,
                 int64_t gemm1_k_per_block,
                 int64_t ak1,
                 int64_t bk1,
                 int64_t b1k1,
                 int64_t m_per_xdl,
                 int64_t n_per_xdl,
                 int64_t m_xdl_per_wave,
                 int64_t n_xdl_per_wave,
                 int64_t gemm1_n_xdl_per_wave);

    std::string GetConfigName();

    std::string Emit();

    int64_t block_size_;
    int64_t m_per_block_, n_per_block_, k_per_block_;
    int64_t gemm1_n_per_block_, gemm1_k_per_block_;
    int64_t ak1_, bk1_, b1k1_;
    int64_t m_per_xdl_, n_per_xdl_;
    int64_t m_xdl_per_wave_, n_xdl_per_wave_;
    int64_t gemm1_n_xdl_per_wave_;
};

class BlockTransferDesc {
public:
    BlockTransferDesc() = default;
    BlockTransferDesc(std::vector<int64_t> thread_cluster_length,
                      std::vector<int64_t> thread_cluster_arrange_order,
                      std::vector<int64_t> src_access_order,
                      int64_t              src_vector_dim,
                      int64_t              src_scalar_per_vector,
                      int64_t              dst_scalar_per_vector,
                      int64_t              add_extra_dim,
                      bool                 add_extra_dim_flag = false);

    std::string GetConfigName();

    std::string Emit();

    std::vector<int64_t>     thread_cluster_length_, thread_cluster_arrange_order_, src_access_order_;
    std::vector<std::string> thread_cluster_length_vec_, thread_cluster_arrange_order_vec_, src_access_order_vec_;

    int64_t src_vector_dim_;
    int64_t src_scalar_per_vector_, dst_scalar_per_vector_;
    int64_t add_extra_dim_;
    bool    add_extra_dim_flag_;
};

class CBlockTransferDesc {
public:
    CBlockTransferDesc() = default;
    CBlockTransferDesc(int64_t              m_xdl_per_wave,
                       int64_t              n_xdl_per_wave,
                       std::vector<int64_t> m_n_block_wave_per_xdl,
                       int64_t              scalar_per_vector);

    std::string GetConfigName();

    std::string Emit();

    int64_t                  m_xdl_per_wave_;
    int64_t                  n_xdl_per_wave_;
    std::vector<int64_t>     m_n_block_wave_per_xdl_;
    std::vector<std::string> m_n_block_wave_per_xdl_vec_;
    int64_t                  scalar_per_vector_;
};

class MaskedCBlockTransferDesc {
public:
    MaskedCBlockTransferDesc() = default;

    MaskedCBlockTransferDesc(int64_t              m_xdl_per_wave,
                             int64_t              n_xdl_per_wave,
                             std::vector<int64_t> m_n_block_wave_per_xdl,
                             int64_t              scalar_per_vector,
                             TensorOperation      causal_mask);

    std::string GetConfigName();

    std::string Emit();

    int64_t                  m_xdl_per_wave_, n_xdl_per_wave_;
    std::vector<int64_t>     m_n_block_wave_per_xdl_;
    std::vector<std::string> m_n_block_wave_per_xdl_vec_;
    int64_t                  scalar_per_vector_;
    TensorOperation          causal_mask_;
};

class GemmOperation {
public:
    GemmOperation() = default;

    GemmOperation(GemmOperationKind                 operation_kind,
                  GemmKernelType                    kernel_type,
                  TensorDesc                        a_tensor_desc,
                  TensorDesc                        b_tensor_desc,
                  TensorDesc                        c_tensor_desc,
                  DataType                          accumulator_type,
                  TensorOperation                   a_element_op,
                  TensorOperation                   b_element_op,
                  TensorOperation                   epilogue_op,
                  GemmSpecialization                gemm_specialization,
                  GemmTileDesc                      tile_desc,
                  BlockTransferDesc                 a_block_transfer,
                  BlockTransferDesc                 b_block_transfer,
                  std::optional<CBlockTransferDesc> c_block_transfer  = std::nullopt,
                  std::optional<BlockTransferDesc>  b1_block_transfer = std::nullopt,
                  std::vector<DataType>             ds_dtype          = {},
                  std::vector<LayoutType>           ds_layout         = {},
                  DataType                          e_dtype           = DataType::UNDEFINED);

    std::string GetConfigName();

    std::string Emit();

    GemmOperationKind operation_kind_;
    GemmKernelType    kernel_type_;

    TensorDesc a_tensor_desc_;
    TensorDesc b_tensor_desc_;
    TensorDesc c_tensor_desc_;
    DataType   accumulator_type_;

    TensorOperation a_element_op_;
    TensorOperation b_element_op_;
    TensorOperation c_element_op_;
    TensorOperation epilogue_op_;

    GemmSpecialization gemm_specialization_;

    GemmTileDesc tile_desc_;
    AttnTileDesc attn_tile_desc_;

    BlockTransferDesc                       a_block_transfer_;
    BlockTransferDesc                       b_block_transfer_;
    std::optional<CBlockTransferDesc>       c_block_transfer_      = std::nullopt;
    std::optional<MaskedCBlockTransferDesc> mask_c_block_transfer_ = std::nullopt;
    std::optional<BlockTransferDesc>        b1_block_transfer_     = std::nullopt;

    std::vector<DataType>   ds_dtype_  = {};
    std::vector<LayoutType> ds_layout_ = {};
    DataType                e_dtype_   = DataType::UNDEFINED;
};

}  // namespace lightinfer