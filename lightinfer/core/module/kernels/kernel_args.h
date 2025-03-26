#pragma once

#include <hip/hip_runtime.h>
#include <sstream>
#include <string>

namespace lightinfer {

class EmbeddingKernelArgs {
public:
    void* emb_x0_ptr_;
    void* emb_x1_ptr_;
    void* emb_x2_ptr_;

    void* index_x0_ptr_;
    void* index_x1_ptr_;
    void* index_x2_ptr_;

    void* gamma_ptr_;
    void* beta_ptr_;

    void* y_ptr_;

    int64_t embedding_dims_;
    int64_t num_indices_;

    float epsilon_;

    hipStream_t stream_;
};

class GemmKernelArgs {
public:
    GemmKernelArgs() = default;

    std::string GetDimInfo() const
    {
        std::ostringstream os;
        os << "a_dim0: " << a_dim0_ << ", a_dim1: " << a_dim1_ << ", a_dim2: " << a_dim2_ << ", " << std::endl;
        os << "b_dim0: " << b_dim0_ << ", b_dim1: " << b_dim1_ << ", b_dim2: " << b_dim2_ << ", " << std::endl;
        os << "b1_dim0: " << b1_dim0_ << ", b1_dim1: " << b1_dim1_ << ", b1_dim2: " << b1_dim2_ << ", " << std::endl;
        os << "c_dim0: " << c_dim0_ << ", c_dim1: " << c_dim1_ << ", c_dim2: " << c_dim2_ << ", " << std::endl;
        os << "p_dim0: " << p_dim0_ << ", p_dim1: " << p_dim1_ << ", p_dim2: " << p_dim2_ << ", " << std::endl;

        return os.str();
    }

    GemmKernelArgs(void*       in_ptr,
                   void*       weight_ptr,
                   void*       out_ptr,
                   int64_t     a_dim0,
                   int64_t     a_dim1,
                   int64_t     a_dim2,
                   int64_t     b_dim0,
                   int64_t     b_dim1,
                   int64_t     b_dim2,
                   int64_t     c_dim0,
                   int64_t     c_dim1,
                   int64_t     c_dim2,
                   hipStream_t stream):
        in_ptr_(in_ptr),
        weight_ptr_(weight_ptr),
        out_ptr_(out_ptr),
        a_dim0_(a_dim0),
        a_dim1_(a_dim1),
        a_dim2_(a_dim2),
        b_dim0_(b_dim0),
        b_dim1_(b_dim1),
        b_dim2_(b_dim2),
        c_dim0_(c_dim0),
        c_dim1_(c_dim1),
        c_dim2_(c_dim2),
        stream_(stream)
    {
    }

    // gemm
    void* in_ptr_;
    void* weight_ptr_;
    void* bias_ptr_;
    void* out_ptr_;
    void* d0_ptr_ = nullptr;

    // attn
    void* q_ptr_;
    void* k_ptr_;
    void* v_ptr_;

    int64_t a_dim0_ = -1;
    int64_t a_dim1_ = -1;
    int64_t a_dim2_ = -1;

    int64_t b_dim0_ = -1;
    int64_t b_dim1_ = -1;
    int64_t b_dim2_ = -1;

    int64_t b1_dim0_ = -1;
    int64_t b1_dim1_ = -1;
    int64_t b1_dim2_ = -1;

    int64_t c_dim0_ = -1;
    int64_t c_dim1_ = -1;
    int64_t c_dim2_ = -1;

    int64_t p_dim0_ = -1;
    int64_t p_dim1_ = -1;
    int64_t p_dim2_ = -1;

    int64_t split_k_ = -1;

    hipStream_t stream_;
};

/*----------------------norm-----------------------------------------*/
struct NormKernelArgs {
    void* x_ptr_;
    void* x_residual_ptr_;
    void* smooth_scale_ptr_;
    void* x_bias_ptr_;  // layer_norm
    void* gamma_ptr_;
    void* beta_ptr_;  // layer_norm
    void* y_ptr_;
    void* y_residual_ptr_;
    void* y_scale_ptr_;

    int64_t x_dim_0_ = -1;
    int64_t x_dim_1_ = -1;

    float eps_;

    int64_t x_stride_;
    int64_t xr_stride_;
    int64_t y_stride_;
    int64_t yr_stride_;

    hipStream_t stream_;
};

/*---------------------------fmha----------------------------------*/
struct FmhaFwdKernelArgs {
    // User-provided member variables
    void *                 q_ptr_, *k_ptr_, *v_ptr_, *bias_ptr_, *out_ptr_;
    int64_t *              seqstart_q_ptr_, *seqstart_k_ptr_, *seqlen_k_ptr_;
    int64_t                batch_ = -1, seqlen_q_ = -1, seqlen_k_ = -1;
    int64_t                nhead_q_ = -1, nhead_k_ = -1, hdim_q_ = -1, hdim_v_ = -1;
    int64_t                max_seqlen_q_ = -1;
    float                  scale_;
    std::array<int64_t, 2> window_size_;
    uint32_t               mask_type_;
    hipStream_t            stream_;

    void serialize(std::ostream& os) const
    {
        constexpr int     name_width  = 22;
        constexpr int     value_width = 26;
        const std::string separator(50, '-');

        // Pointer formatting lambda with zero-padded hex
        auto format_ptr = [](auto ptr) -> std::string {
            std::stringstream ss;
            ptr ? ss << "0x" << std::hex << std::setw(16) << std::setfill('0') << reinterpret_cast<uintptr_t>(ptr) :
                  ss << "NULL";
            return ss.str();
        };

        os << "\n"
           << separator << "\n"
           << "FMHA fwd Kernel Configuration\n"
           << separator << "\n";

        // Unified field printer with type detection
        auto print_field = [&](const char* name, auto value, const char* unit = "") {
            os << std::left << std::setw(name_width) << name << std::right << std::setw(value_width);

            if constexpr (std::is_pointer_v<decltype(value)>) {
                os << format_ptr(value);
            }
            else if constexpr (std::is_same_v<decltype(value), std::array<int64_t, 2>>) {
                os << "[" << value[0] << ", " << value[1] << "]";  // Fix array display
            }
            else {
                if (value == -1)
                    os << "N/A";  // Mark uninitialized parameters
                else if constexpr (std::is_same_v<decltype(value), float>) {
                    os << std::fixed << std::setprecision(4) << value;
                }
                else {
                    os << std::dec << value << (unit ? " " : "") << unit;
                }
            }

            os << "\n";
        };

        // Memory pointers section
        print_field("Query Pointer:", q_ptr_);
        print_field("Key Pointer:", k_ptr_);
        print_field("Value Pointer:", v_ptr_);
        print_field("Bias Pointer:", bias_ptr_);
        os << separator << "\n";

        // Sequence metadata section
        print_field("Q Seq Start:", seqstart_q_ptr_);
        print_field("K Seq Start:", seqstart_k_ptr_);
        print_field("K Seq Length:", seqlen_k_ptr_);
        os << separator << "\n";

        // Dimension configuration
        print_field("Batch Size:", batch_);
        print_field("Query Length:", seqlen_q_, "tok");
        print_field("Key Length:", seqlen_k_, "tok");
        print_field("Q Heads:", nhead_q_);
        print_field("K Heads:", nhead_k_);
        print_field("Q Dim:", hdim_q_, "dim");
        print_field("V Dim:", hdim_v_, "dim");
        os << separator << "\n";

        // Advanced parameters
        print_field("Max Q Length:", max_seqlen_q_, "tok");
        print_field("Window Size:", window_size_);
        print_field("Mask Type:", mask_type_);
        print_field("Scale Factor:", scale_);
        print_field("HIP Stream:", stream_);

        os << separator << "\n";
    }

    friend std::ostream& operator<<(std::ostream& os, const FmhaFwdKernelArgs& args)
    {
        args.serialize(os);
        return os;
    }
};

struct FmhaFwdAppendKVKernelArgs {
    void* q_ptr_;
    void* cache_k_ptr_;
    void* cache_v_ptr_;
    void* k_ptr_;
    void* v_ptr_;

    void* cache_seqlen_k_ptr_;

    void* rotary_cos_ptr_;
    void* rotary_sin_ptr_;

    void* block_table_ptr_;

    int64_t* cache_batch_idx_ptr_;

    int64_t batch_    = -1;
    int64_t seqlen_q_ = -1;
    int64_t seqlen_k_ = -1;
    int64_t nhead_q_  = -1;
    int64_t nhead_k_  = -1;
    int64_t hdim_q_   = -1;
    int64_t hdim_v_   = -1;

    int64_t seqlen_knew_ = -1;

    int64_t rotary_dim_;
    int64_t max_num_page_blocks_;
    int64_t paged_block_size_;
    bool    has_mask_;

    hipStream_t stream_;
};

struct FmhaFwdSplitKVKernelArgs {
    void* q_ptr_;
    void* k_ptr_;
    void* v_ptr_;
    void* bias_ptr_;
    void* lse_acc_ptr_;
    void* out_acc_ptr_;

    int64_t* seqstart_q_ptr_;
    int64_t* seqstart_k_ptr_;
    int64_t* seqlen_k_ptr_;

    void*    block_table_ptr_;
    int64_t* cache_batch_idx_ptr_;

    int64_t batch_    = -1;
    int64_t seqlen_q_ = -1;
    int64_t seqlen_k_ = -1;
    int64_t nhead_q_  = -1;
    int64_t nhead_k_  = -1;
    int64_t hdim_q_   = -1;
    int64_t hdim_v_   = -1;

    int64_t max_seqlen_q_ = -1;

    int64_t max_num_page_blocks_;
    int64_t paged_block_size_;

    float scale_;

    std::array<int64_t, 2> window_size_;
    uint32_t               mask_type_;

    hipStream_t stream_;
};

struct FmhaFwdSplitKVCombineKernelArgs {
    void* lse_acc_ptr_;
    void* out_acc_ptr_;
    void* out_ptr_;

    int64_t* seqstart_q_ptr_;

    int64_t batch_    = -1;
    int64_t seqlen_q_ = -1;
    int64_t nhead_q_  = -1;
    int64_t hdim_v_   = -1;

    int64_t max_seqlen_q_ = -1;

    int64_t num_splits_ = -1;

    hipStream_t stream_;
};

}  // namespace lightinfer