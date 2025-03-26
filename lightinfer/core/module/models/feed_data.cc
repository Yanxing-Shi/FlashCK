#include "lightinfer/core/module/models/feed_data.h"

#include "lightinfer/core/utils/printf.h"

#include <numeric>

namespace lightinfer {

FeedData::FeedData(const BackendType backend_type,
                   const DataType    dtype,
                   const Shape       shape,
                   const void*       data,
                   const hipStream_t stream):
    backend_type_(backend_type), dtype_(dtype), shape_(shape), data_(data), stream_(stream)
{
}

FeedData::FeedData(const BackendType         backend_type,
                   const DataType            dtype,
                   const Shape               shape,
                   const void*               data,
                   const std::vector<size_t> offsets,
                   const hipStream_t         stream):
    backend_type_(backend_type), dtype_(dtype), shape_(shape), data_(data), offsets_(offsets), stream_(stream)
{
}

size_t FeedData::GetSize() const
{
    if (data_ == nullptr || shape_.GetNumDim() == 0) {
        return 0;
    }
    return std::get<0>(shape_.GetElementSizeTuple());
}

size_t FeedData::GetSizeBytes() const
{
    return GetSize() * SizeOf(dtype_);
}

std::string FeedData::GetBackendTypeStr() const
{
    return BackendTypeToString(backend_type_);
}

std::string FeedData::ToString() const
{
    return Sprintf("FeedData(backend_type={}, dtype={}, shape={}, data={}, stream={})",
                   GetBackendTypeStr(),
                   DataTypeToString(dtype_),
                   shape_.ToString(),
                   data_,
                   static_cast<void*>(stream_));
}

void FeedData::SaveNpy(const std::string& filename) const
{
    // Save tensor to NPY 1.0 format (see https://numpy.org/neps/nep-0001-npy-format.html)
    void*  cpu_data     = (void*)data_;
    bool   is_data_temp = false;
    size_t tensor_size  = GetSize();
    if (backend_type_ == BackendType::GPU) {
        cpu_data     = malloc(tensor_size * SizeOf(dtype_));
        is_data_temp = true;
        hipDeviceSynchronize();
        hipMemcpy(cpu_data, data_, tensor_size * SizeOf(dtype_), hipMemcpyDeviceToHost);
    }

    const char    magic[]   = "\x93"
                              "NUMPY";
    const uint8_t npy_major = 1;
    const uint8_t npy_minor = 0;

    std::stringstream header_stream;
    header_stream << "{'descr': '" << GetNumpyTypeDesc(dtype_) << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape_.GetNumDim(); ++i) {
        header_stream << shape_.GetDim(i).GetValues()[0];
        if (i + 1 < shape_.GetNumDim() || shape_.GetNumDim() == 1) {
            header_stream << ", ";
        }
    }
    header_stream << ")}";
    int base_length = 6 + 4 + header_stream.str().size();
    int pad_length  = 16 * ((base_length + 1 + 15) / 16);  // Take ceiling of base_length + 1 (for '\n' ending)
    for (int i = 0; i < pad_length - base_length; ++i) {
        header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
    }
    std::string    header     = header_stream.str();
    const uint16_t header_len = header.size();

    FILE* f_ptr = fopen(filename.c_str(), "wb");
    LI_ENFORCE_NOT_NULL(f_ptr, Unavailable("Unable to open {} for writing", filename));
    fwrite(magic, sizeof(char), sizeof(magic) - 1, f_ptr);
    fwrite(&npy_major, sizeof(uint8_t), 1, f_ptr);
    fwrite(&npy_minor, sizeof(uint8_t), 1, f_ptr);
    fwrite(&header_len, sizeof(uint16_t), 1, f_ptr);
    fwrite(header.c_str(), sizeof(char), header_len, f_ptr);
    fwrite(cpu_data, SizeOf(dtype_), tensor_size, f_ptr);

    fclose(f_ptr);

    if (is_data_temp) {
        free(cpu_data);
    }
}

FeedData FeedData::LoadNpy(const std::string& npy_file, const BackendType backend_type)
{
    DataType dtype;
    Shape    shape;

    FILE* f_ptr = fopen(npy_file.c_str(), "rb");
    LI_ENFORCE_NOT_NULL(f_ptr, Unavailable("Could not open file {}", npy_file));

    uint32_t header_len, start_data;
    ParseNpyIntro(f_ptr, header_len, start_data);
    ParseNpyHeader(f_ptr, header_len, dtype, shape);

    const size_t size     = std::get<0>(shape.GetElementSizeTuple());
    void*        data_cpu = malloc(size * SizeOf(dtype));
    void*        data     = data_cpu;

    size_t n_elems = fread(data_cpu, SizeOf(dtype), size, f_ptr);
    LI_ENFORCE_EQ(n_elems, size, Unavailable("reading tensor failed"));
    if (backend_type == BackendType::GPU) {
        hipMalloc(&data, size * SizeOf(dtype));
        hipMemcpy(data, data_cpu, size * SizeOf(dtype), hipMemcpyHostToDevice);
        free(data_cpu);
    }

    fclose(f_ptr);
    return FeedData(backend_type, dtype, shape, data);
}

FeedData FeedData::Slice(const Shape& shape, const size_t offset) const
{
    if (this->data_ != nullptr) {
        size_t num_elements        = this->GetSize();
        size_t num_sliced_elements = std::get<0>(shape.GetElementSizeTuple());
        LI_ENFORCE_LE(num_sliced_elements + offset,
                      num_elements,
                      Unavailable("The number {} of elements of sliced tensor exceeds that {} of the original tensor",
                                  num_sliced_elements + offset,
                                  num_elements));
    }

    return FeedData(backend_type_, dtype_, shape_, this->GetPtrWithOffset(offset));
}

void FeedData::ParseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data)
{
    const char magic[]                   = "\x93"
                                           "NUMPY";
    char       magic_test[sizeof(magic)] = "\0";

    size_t n_elems = fread((void*)magic_test, sizeof(char), sizeof(magic) - 1, f_ptr);
    if (n_elems != sizeof(magic) - 1 || std::string(magic) != std::string(magic_test)) {
        LI_THROW(Unavailable("Could read magic token in NPY file"));
    }

    uint8_t npy_major = 0;
    uint8_t npy_minor = 0;
    n_elems           = fread((void*)&npy_major, sizeof(uint8_t), 1, f_ptr);
    n_elems += fread((void*)&npy_minor, sizeof(uint8_t), 1, f_ptr);

    if (npy_major == 1) {
        uint16_t header_len_u16 = 0;
        n_elems                 = fread((void*)&header_len_u16, sizeof(uint16_t), 1, f_ptr);
        header_len              = header_len_u16;
    }
    else if (npy_major == 2) {
        uint32_t header_len_u32 = 0;
        n_elems                 = fread((void*)&header_len_u32, sizeof(uint32_t), 1, f_ptr);
        header_len              = header_len_u32;
    }
    else {
        LI_THROW(Unavailable("Unsupported npy version: {}", npy_major));
    }

    start_data = 8 + 2 * npy_major + header_len;
}

int FeedData::ParseNpyHeader(FILE*& f_ptr, uint32_t header_len, DataType& type, Shape& shape)
{
    char*  header_c = (char*)malloc(header_len * sizeof(char));
    size_t n_elems  = fread((void*)header_c, sizeof(char), header_len, f_ptr);
    if (n_elems != header_len) {
        free(header_c);
        return -1;
    }
    std::string header(header_c, header_len);
    free(header_c);

    size_t start, end;
    start = header.find("'descr'") + 7;
    start = header.find("'", start);
    end   = header.find("'", start + 1);
    type  = GetTypeFromNumpyDesc(header.substr(start + 1, end - start - 1));

    start = header.find("'fortran_order'") + 15;
    start = header.find(":", start);
    end   = header.find(",", start + 1);
    if (header.substr(start + 1, end - start - 1).find("False") == std::string::npos) {
        LI_THROW(Unavailable("Unsupported value for fortran_order while reading npy file"));
    }

    start = header.find("'shape'") + 7;
    start = header.find("(", start);
    end   = header.find(")", start + 1);

    std::istringstream shape_stream(header.substr(start + 1, end - start - 1));
    std::string        token;

    shape.Clear();
    while (std::getline(shape_stream, token, ',')) {
        if (token.find_first_not_of(' ') == std::string::npos) {
            break;
        }

        shape.AppendDim(std::stoul(token));
    }

    return 0;
}

}  // namespace lightinfer