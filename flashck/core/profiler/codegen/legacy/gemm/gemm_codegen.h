#pragma once

namespace flashck {

GemmSpecialization GetGemmSpec(const int64_t m,
                               const int64_t n,
                               const int64_t k,
                               const int64_t m_per_block,
                               const int64_t n_per_block,
                               const int64_t k_per_block);

class GemmTileDesc {
public:
    std::string GetConfigName();

    std::string Emit();

    int64_t block_size_;
    int64_t m_per_block_, n_per_block_, k_per_block_;
    int64_t a_k1_, b_k1_;
    int64_t m_per_xdl_, n_per_xdl_;
    int64_t m_xdl_per_wave_, n_xdl_per_wave_;
};

class BlockTransferDesc {
public:
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
    std::string GetConfigName();

    std::string Emit();

    int64_t                  m_xdl_per_wave_;
    int64_t                  n_xdl_per_wave_;
    std::vector<int64_t>     m_n_block_wave_per_xdl_;
    std::vector<std::string> m_n_block_wave_per_xdl_vec_;
    int64_t                  scalar_per_vector_;
};

class GemmCodegen: public CodegenBase {
public:
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

}  // namespace flashck