#include "lightinfer/core/module/models/feed_data_map.h"

#include "lightinfer/core/utils/string_utils.h"

#include <dirent.h>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace lightinfer {

FeedDataMap::FeedDataMap(const std::unordered_map<std::string, FeedData>& feed_data)
{
    for (auto& kv : feed_data) {
        if (IsValid(kv.second)) {
            Insert(kv.first, kv.second);
        }
        else {
            LOG(WARNING) << "is not a valid tensor, skipping Insert into TensorMap";
        }
    }
}

FeedDataMap::FeedDataMap(const std::vector<FeedData>& feed_data)
{

    for (size_t i = 0; i < feed_data.size(); i++) {
        if (IsValid(feed_data[i])) {
            Insert(std::to_string(i), feed_data[i]);
        }
        else {
            LOG(WARNING) << "is not a valid tensor, skipping Insert into TensorMap";
        }
    }
}

FeedDataMap::FeedDataMap(std::initializer_list<std::pair<std::string, FeedData>> feed_data)
{
    for (auto& pair : feed_data) {
        if (IsValid(pair.second)) {
            Insert(pair.first, pair.second);
        }
        else {
            LOG(WARNING) << "is not a valid tensor, skipping Insert into TensorMap";
        }
    }
}

FeedDataMap::~FeedDataMap()
{
    feed_data_map_.clear();
}

std::string FeedDataMap::ToString()
{
    std::stringstream ss;
    ss << "{";
    std::vector<std::string> key_names = GetKeyList(feed_data_map_);
    for (size_t i = 0; i < feed_data_map_.size(); ++i) {
        ss << key_names[i] << ": " << feed_data_map_.at(key_names[i]).ToString();
        if (i < feed_data_map_.size() - 1) {
            ss << ", ";
        }
    }
    ss << "}";
    return ss.str();
}

FeedDataMap FeedDataMap::FromNpyFolder(const std::string& base_folder)
{
    DIR* dir_p = opendir(base_folder.c_str());

    ATER_ENFORCE_NOT_NULL(dir_p, InvalidArgument("Could not open folder {}.\n", base_folder));
    struct dirent* dp;

    FeedDataMap ret_tensor;
    while ((dp = readdir(dir_p)) != nullptr) {
        std::string filename(dp->d_name);
        size_t      len = filename.length();
        if (len < 4 || filename.compare(len - 4, 4, ".npy")) {
            continue;
        }

        size_t pos = filename.find('-');

        ATER_ENFORCE_NE(pos, std::string::npos, InvalidArgument("Invalid filename: {}", filename));

        BackendType backend_type;
        if (filename.compare(0, pos, "GPU") == 0) {
            backend_type = BackendType::GPU;
        }
        else if (filename.compare(0, pos, "CPU") == 0) {
            backend_type = BackendType::CPU;
        }
        else if (filename.compare(0, pos, "CPU_PINNED") == 0) {
            backend_type = BackendType::CPU_PINNED;
        }
        else {
            LI_THROW(InvalidArgument("Invalid filename: {}", filename));
        }
        std::string key = filename.substr(pos + 1, len - pos - 5);

        ret_tensor.feed_data_map_.insert({key, FeedData::LoadNpy(base_folder + "/" + filename, backend_type)});
    }

    closedir(dir_p);

    return ret_tensor;
}

void FeedDataMap::SaveNpy(const std::string& base_folder)
{
    mode_t mode_0755 = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
    int    ret       = mkdir(base_folder.c_str(), mode_0755);
    LI_ENFORCE_EQ(ret == 0 || errno == EEXIST, true, InvalidArgument("Could not create folder {}.\n", base_folder));

    for (const auto& item : feed_data_map_) {
        item.second.SaveNpy(base_folder + "/" + item.second.GetBackendTypeStr() + "-" + item.first + ".npy");
    }
}

}  // namespace lightinfer