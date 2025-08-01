#pragma once


namespace flashck{

enum class MoeGemmWeightPermuteEnum
{
    // permute_b_n0_k0_n1_k1_n2_k2 = 0, // 0,1,4,2,5,3,6
    // permute_b_n0_n1_k0_k1_n2_k2 = 1, // 0,1,2,4,5,3,6
    no_permute          = 0,
    b_nr_kr_kw_nw_kv    = 1, // 0,1,3,4,2,5
    b_nr_kr_waveflatten = b_nr_kr_kw_nw_kv,
};






} // namespace flashck