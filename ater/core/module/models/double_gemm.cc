#include "ater/core/module/models/double_gemm.h"

#include "ater/core/utils/flags.h"
#include "ater/core/utils/log.h"

using namespace ater;
int main(int argc, char* argv[])
{
    // Initialize Google's logging library.
    InitGLOG(argv);
    // parse command line flags
    InitGflags(argc, argv, true);

    DoubleGemmModel<_Float16>();
    // DoubleGemmModel<float>();
}
