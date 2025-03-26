#include <memory>
#include <vector>

#include "3rdparty/INIReader.h"

#include "lightinfer/core/module/models/bert/bert_model.h"
#include "lightinfer/core/module/models/bert/bert_model_utils.h"
#include "lightinfer/core/module/models/bert/bert_weight.h"

#include "lightinfer/core/utils/dtype.h"
#include "lightinfer/core/utils/flags.h"
#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/memory_utils.h"
#include "lightinfer/core/utils/mpi_utils.h"
#include "lightinfer/core/utils/rocm_info.h"

#include "lightinfer/core/utils/timer.h"

using namespace lightinfer;

template<typename T>
void BertExample(const INIReader reader);

int main(int argc, char* argv[])
{
    // Initialize Google's logging library.
    InitGLOG(argv);
    // parse command line flags
    InitGflags(argc, argv, true);

    // mpi::Initialize(&argc, &argv);

    std::string ini_name = argc >= 2 ? std::string(argv[1]) : "../examples/cpp/bert/bert_config.ini";

    INIReader reader = INIReader(ini_name);
    LI_ENFORCE_GE(reader.ParseError(), 0, Unavailable("can not load ini file: {}", ini_name));

    const std::string data_type = reader.Get("instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        BertExample<float>(reader);
    }
    else if (data_type == "fp16") {
        BertExample<_Float16>(reader);
    }

    // mpi::Finalize();
    return 0;
}

template<typename T>
void BertExample(const INIReader reader)
{
    /*********************************step.1 prepare param****************************************/
    // distributed param
    // RcclParam tensor_para, pipeline_para;
    // int       rank, world_size;
    // std::tie(rank, world_size) = InitMultiProcessing(model_config);
    // InitRccL(model_config, tensor_para, pipeline_para);

    // prepare input and ouput data
    auto model_config   = ReadModelConfig(reader);
    auto request_config = ReadRequestConfig(reader);

    int* d_input_ids      = nullptr;
    int* d_token_type_ids = nullptr;
    int* d_position_ids   = nullptr;

    T* d_bert_out = nullptr;
    DeviceMalloc(&d_input_ids,
                 request_config.request_batch_size_ * request_config.request_seq_len_,
                 true,
                 0,
                 model_config.vocab_size_);
    DeviceMalloc(&d_token_type_ids,
                 request_config.request_batch_size_ * request_config.request_seq_len_,
                 true,
                 0,
                 model_config.type_vocab_size_);
    DeviceMalloc(&d_position_ids,
                 request_config.request_batch_size_ * request_config.request_seq_len_,
                 true,
                 0,
                 model_config.max_position_embeddings_);

    DeviceMalloc(&d_bert_out,
                 request_config.request_batch_size_ * request_config.request_seq_len_ * model_config.hidden_units_,
                 false);

    FeedDataMap input_data_map = FeedDataMap(std::unordered_map<std::string, FeedData>{
        {"input_ids",
         FeedData{BackendType::GPU,
                  DataType::INT32,
                  {(int)request_config.request_batch_size_, (int)request_config.request_seq_len_},
                  d_input_ids}},
        {"token_type_ids",
         FeedData{BackendType::GPU,
                  DataType::INT32,
                  {(int)request_config.request_batch_size_, (int)request_config.request_seq_len_},
                  d_token_type_ids}},
        {"position_ids",
         FeedData{BackendType::GPU,
                  DataType::INT32,
                  {(int)request_config.request_batch_size_, (int)request_config.request_seq_len_},
                  d_position_ids}}});

    FeedDataMap output_data_map =
        FeedDataMap(std::unordered_map<std::string, FeedData>{{"output_hidden_state",
                                                               FeedData{BackendType::GPU,
                                                                        CppTypeToDataType<T>::Type(),
                                                                        {(int)request_config.request_batch_size_,
                                                                         (int)request_config.request_seq_len_,
                                                                         (int)model_config.hidden_units_},
                                                                        d_bert_out}}});

    /*************************************step.2 load weight************************************/

    // unsigned long long random_seed;
    // if (rank == 0) {
    //     random_seed = (unsigned long long)(0);
    // }
    // if (world_size > 1) {
    //     mpi::bcast(&random_seed, 1, mpi::MPI_TYPE_UNSIGNED_LONG_LONG, 0, mpi::COMM_WORLD);
    // }

    // // prepare model

    Bert<T> bert(reader);
    bert.BuildGraph();
    bert.SetInput(input_data_map);
    bert.SetOutput(output_data_map);

    // GetGpuMemoryInfo();

    // // warmup
    int warmup_iter = 1;
    // // // mpi::Barrier();

    for (int i = 0; i < warmup_iter; i++) {
        bert.Forward();
    }

    // // mpi::Barrier();

    // // // if (rank == 0) {

    // // // }

    // // test inference
    // int      test_iter = 5;
    // HipTimer hip_timer;
    // // // mpi::Barrier();
    // hip_timer.Start();
    // for (int i = 0; i < test_iter; i++) {
    //     bert.Forward();
    // }
    // float total_time = hip_timer.Stop();
    // printf("time: %fms\n", total_time / test_iter);
    // mpi::Barrier();

    // RcclParamDestroy(tensor_para);
    // RcclParamDestroy(pipeline_para);

    // DeviceFree(d_input_ids);
    // DeviceFree(d_input_lengths);
    // if (d_input_ids != nullptr) {
    //     DeviceFree(d_input_ids);
    // }
    // if (d_input_lengths != nullptr) {
    //     DeviceFree(d_input_lengths);
    // }
    // if (d_output_ids != nullptr) {
    //     DeviceFree(d_output_ids);
    // }
    // if (d_sequence_lengths != nullptr) {
    //     DeviceFree(d_sequence_lengths);
    // }
    // return;
}