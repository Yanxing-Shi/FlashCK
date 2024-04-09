#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ater/core/profiler/library.h"
#include "ater/core/utils/dtype.h"
#include "ater/core/utils/layout.h"
#include "ater/core/utils/math_utils.h"

namespace ater {

class GemmProblem {
public:
    GemmProblem() = default;

    GemmProblem(int                     m,
                int                     n,
                int                     k,
                DataType                a_dtype,
                DataType                b_dtype,
                DataType                c_dtype,
                DataType                acc_dtype,
                std::vector<DataType>   ds_dtype,
                DataType                e_dtype,
                DataLayout              layout,
                std::vector<LayoutType> ds_layout,
                TensorOperation         epilogue_op):
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
            ATER_THROW(Unimplemented("{}", "data layout not supported"));
        }
    }

    // shape
    int m_ = 0;
    int n_ = 0;
    int k_ = 0;

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

GemmSpecialization
GetGemmSpec(const int m, const int n, const int k, const int m_per_block, const int n_per_block, const int k_per_block);

class TileDesc {
public:
    TileDesc() = default;
    TileDesc(int block_size,
             int m_per_block,
             int n_per_block,
             int k_per_block,
             int ak1,
             int bk1,
             int m_per_xdl,
             int n_per_xdl,
             int m_xdl_per_wave,
             int n_xdl_per_wave);

    std::string GetConfigName();

    std::string Emit();

    int block_size_;
    int m_per_block_, n_per_block_, k_per_block_;
    int ak1_, bk1_;
    int m_per_xdl_, n_per_xdl_;
    int m_xdl_per_wave_, n_xdl_per_wave_;
};

class AttnTileDesc {
public:
    AttnTileDesc() = default;
    AttnTileDesc(int block_size,
                 int m_per_block,
                 int n_per_block,
                 int k_per_block,
                 int gemm1_n_per_block,
                 int gemm1_k_per_block,
                 int ak1,
                 int bk1,
                 int b1k1,
                 int m_per_xdl,
                 int n_per_xdl,
                 int m_xdl_per_wave,
                 int n_xdl_per_wave,
                 int gemm1_n_xdl_per_wave);

    std::string GetConfigName();

    std::string Emit();

    int block_size_;
    int m_per_block_, n_per_block_, k_per_block_;
    int gemm1_n_per_block_, gemm1_k_per_block_;
    int ak1_, bk1_, b1k1_;
    int m_per_xdl_, n_per_xdl_;
    int m_xdl_per_wave_, n_xdl_per_wave_;
    int gemm1_n_xdl_per_wave_;
};

class BlockTransferDesc {
public:
    BlockTransferDesc() = default;
    BlockTransferDesc(std::vector<int> thread_cluster_length,
                      std::vector<int> thread_cluster_arrange_order,
                      std::vector<int> src_access_order,
                      int              src_vector_dim,
                      int              src_scalar_per_vector,
                      int              dst_scalar_per_vector,
                      int              add_extra_dim,
                      bool             add_extra_dim_flag = false);

    std::string GetConfigName();

    std::string Emit();

    std::vector<int>         thread_cluster_length_, thread_cluster_arrange_order_, src_access_order_;
    std::vector<std::string> thread_cluster_length_vec_, thread_cluster_arrange_order_vec_, src_access_order_vec_;

    int  src_vector_dim_;
    int  src_scalar_per_vector_, dst_scalar_per_vector_;
    int  add_extra_dim_;
    bool add_extra_dim_flag_;
};

class CBlockTransferDesc {
public:
    CBlockTransferDesc() = default;
    CBlockTransferDesc(int              m_xdl_per_wave,
                       int              n_xdl_per_wave,
                       std::vector<int> m_n_block_wave_per_xdl,
                       int              scalar_per_vector);

    std::string GetConfigName();

    std::string Emit();

    int                      m_xdl_per_wave_;
    int                      n_xdl_per_wave_;
    std::vector<int>         m_n_block_wave_per_xdl_;
    std::vector<std::string> m_n_block_wave_per_xdl_vec_;
    int                      scalar_per_vector_;
};

class MaskedCBlockTransferDesc {
public:
    MaskedCBlockTransferDesc() = default;

    MaskedCBlockTransferDesc(int              m_xdl_per_wave,
                             int              n_xdl_per_wave,
                             std::vector<int> m_n_block_wave_per_xdl,
                             int              scalar_per_vector,
                             int              causal_mask);

    std::string GetConfigName();

    std::string Emit();

    int                      m_xdl_per_wave_, n_xdl_per_wave_;
    std::vector<int>         m_n_block_wave_per_xdl_;
    std::vector<std::string> m_n_block_wave_per_xdl_vec_;
    int                      scalar_per_vector_;
    int                      causal_mask_;
};

class GemmOperation {
public:
    GemmOperation() = default;

    GemmOperation(OperationKind                     operation_kind,
                  TensorOperation                   extra_kind,
                  KernelType                        kernel_type,
                  TensorDesc                        a_tensor_desc,
                  TensorDesc                        b_tensor_desc,
                  TensorDesc                        c_tensor_desc,
                  DataType                          accumulator_type,
                  TensorOperation                   a_element_op,
                  TensorOperation                   b_element_op,
                  TensorOperation                   epilogue_functor,
                  GemmSpecialization                gemm_specialization,
                  TileDesc                          tile_desc,
                  BlockTransferDesc                 a_block_transfer,
                  BlockTransferDesc                 b_block_transfer,
                  std::optional<CBlockTransferDesc> c_block_transfer  = std::nullopt,
                  std::optional<BlockTransferDesc>  b1_block_transfer = std::nullopt,
                  std::vector<DataType>             ds_dtype          = {},
                  std::vector<LayoutType>           ds_layout         = {},
                  DataType                          e_dtype           = DataType::UNDEFINED);

    std::string GetConfigName();

    std::string Emit();

    OperationKind   operation_kind_;
    TensorOperation extra_kind_;
    KernelType      kernel_type_;

    TensorDesc a_tensor_desc_;
    TensorDesc b_tensor_desc_;
    TensorDesc c_tensor_desc_;
    DataType   accumulator_type_;

    TensorOperation a_element_op_;
    TensorOperation b_element_op_;
    TensorOperation epilogue_functor_;

    GemmSpecialization gemm_specialization_;
    TileDesc           tile_desc_;

    BlockTransferDesc                 a_block_transfer_;
    BlockTransferDesc                 b_block_transfer_;
    std::optional<CBlockTransferDesc> c_block_transfer_  = std::nullopt;
    std::optional<BlockTransferDesc>  b1_block_transfer_ = std::nullopt;

    std::vector<DataType>   ds_dtype_  = {};
    std::vector<LayoutType> ds_layout_ = {};
    DataType                e_dtype_   = DataType::UNDEFINED;
};

}  // namespace ater