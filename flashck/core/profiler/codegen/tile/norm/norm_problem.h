struct NormProblem {
    DataType x_dtype_;
    DataType y_dtype_;
    DataType smooth_scale_dtype_;
    DataType y_scale_dtype_;

    int64_t m_;
    int64_t n_;

    NormOperationKind operation_kind_;
    TensorOperation   epilogue_op_;

    NormBiasEnum   is_add_bias_;
    FusedAddEnum   fused_add_;
    FusedQuantEnum fused_quant_;
};