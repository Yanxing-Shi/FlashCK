#pragma once

#include "flashck/core/profiling/codegen/codegen_base.h"
#include "flashck/core/profiling/codegen/legacy/gemm/gemm_library.h"

namespace flashck {

class GemmTileDesc: public TileDescBase {
public:
    std::string GetConfigName() const override;

    std::string Emit() const override;

    int64_t block_size_;
    int64_t m_per_block_, n_per_block_, k_per_block_;
    int64_t a_k1_, b_k1_;
    int64_t m_per_xdl_, n_per_xdl_;
    int64_t m_xdl_per_wave_, n_xdl_per_wave_;
};

class BlockTransferDesc: public TileDescBase {
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

class CBlockTransferDesc: public TileDescBase {
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
    std::string GetConfigName() const override;

    std::string Emit() const override;

    Layout a_layout_, b_layout_, c_layout_;

    DataType a_dtype_, b_dtype_, c_dtype_;

    GemmTileDesc tile_desc_;

    BlockTransferDesc a_block_transfer_desc_, b_block_transfer_desc_;

    CBlockTransferDesc c_block_transfer_desc_;
};

}  // namespace flashck