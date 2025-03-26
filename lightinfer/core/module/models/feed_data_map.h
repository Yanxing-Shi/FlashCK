#pragma once

#include "lightinfer/core/module/models/feed_data.h"

#include "lightinfer/core/utils/enforce.h"

namespace lightinfer {
class FeedDataMap {
public:
    FeedDataMap() = default;
    FeedDataMap(const std::unordered_map<std::string, FeedData>& feed_data);
    FeedDataMap(const std::vector<FeedData>& feed_data);
    FeedDataMap(std::initializer_list<std::pair<std::string, FeedData>> feed_data);
    ~FeedDataMap();

    inline size_t GetSize() const
    {
        return feed_data_map_.size();
    }

    inline bool IsExist(const std::string& key) const
    {
        return feed_data_map_.find(key) != feed_data_map_.end();
    }

    inline void Insert(const std::string& key, const FeedData& value)
    {
        LI_ENFORCE_EQ(IsExist(key), false, Unavailable("Duplicated key {}", key.c_str()));
        LI_ENFORCE_EQ(
            IsValid(value), true, Unavailable("A none tensor or nullptr is not allowed (key is {})", key.c_str()));
        feed_data_map_.insert({key, value});
    }

    inline void InsertIfValid(const std::string& key, const FeedData& value)
    {
        if (IsValid(value)) {
            Insert({key, value});
        }
    }

    inline void Insert(std::pair<std::string, FeedData> p)
    {
        feed_data_map_.insert(p);
    }

    // prevent converting int or size_t to string automatically
    FeedData At(int tmp)    = delete;
    FeedData At(size_t tmp) = delete;

    inline FeedData& At(const std::string& key)
    {
        LI_ENFORCE_EQ(IsExist(key), true, Unavailable("Cannot find a tensor of name {} in the tensor map", key));
        return feed_data_map_.at(key);
    }

    inline FeedData At(const std::string& key) const
    {
        LI_ENFORCE_EQ(IsExist(key), true, Unavailable("Cannot find a tensor of name {} in the tensor map", key));
        return feed_data_map_.at(key);
    }

    inline FeedData& At(const std::string& key, FeedData& default_tensor)
    {
        return IsExist(key) ? feed_data_map_.at(key) : default_tensor;
    }

    inline FeedData At(const std::string& key, FeedData& default_tensor) const
    {
        return IsExist(key) ? feed_data_map_.at(key) : default_tensor;
    }

    inline const FeedData& At(const std::string& key, const FeedData& default_feed_data)
    {
        return IsExist(key) ? feed_data_map_.at(key) : default_feed_data;
    }

    inline FeedData At(const std::string& key, const FeedData& default_feed_data) const
    {
        return IsExist(key) ? feed_data_map_.at(key) : default_feed_data;
    }

    template<typename T>
    inline T GetValue(const std::string& key) const
    {
        LI_ENFORCE_EQ(IsExist(key), true, Unavailable("Cannot find a tensor of name {} in the tensor map", key));
        return feed_data_map_.at(key).GetValue<T>();
    }

    template<typename T>
    inline T GetValue(const std::string& key, T default_value) const
    {
        return IsExist(key) ? feed_data_map_.at(key).GetValue<T>() : default_value;
    }

    template<typename T>
    inline T GetValWithOffset(const std::string& key, size_t index) const
    {
        LI_ENFORCE_EQ(IsExist(key), true, Unavailable("Cannot find a tensor of name {} in the tensor map", key));

        return feed_data_map_.at(key).GetValue<T>(index);
    }

    template<typename T>
    inline T GetValWithOffset(const std::string& key, size_t index, T default_value) const
    {
        return IsExist(key) ? feed_data_map_.at(key).GetValue<T>(index) : default_value;
    }

    template<typename T>
    inline T* GetPtr(const std::string& key) const
    {
        LI_ENFORCE_EQ(IsExist(key), true, Unavailable("Cannot find a tensor of name {} in the tensor map", key));
        return feed_data_map_.at(key).GetPtr<T>();
    }

    template<typename T>
    inline T* GetPtr(const std::string& key, T* default_ptr) const
    {
        return IsExist(key) ? feed_data_map_.at(key).GetPtr<T>() : default_ptr;
    }

    // set ptr
    template<typename T>
    inline void SetPtr(const std::string& key, T* ptr)
    {
        LI_ENFORCE_EQ(IsExist(key), true, Unavailable("Cannot find a tensor of name {} in the tensor map", key));
        feed_data_map_.at(key).SetPtr<T>(ptr);
    }

    template<typename T>
    inline T* GetPtrWithOffset(const std::string& key, size_t index) const
    {
        LI_ENFORCE_EQ(IsExist(key), true, Unavailable("Cannot find a tensor of name {} in the tensor map", key));
        return feed_data_map_.at(key).GetPtrWithOffset<T>(index);
    }

    template<typename T>
    inline T* GetPtrWithOffset(const std::string& key, size_t index, T* default_ptr) const
    {
        return IsExist(key) ? feed_data_map_.at(key).GetPtrWithOffset<T>(index) : default_ptr;
    }

    inline std::unordered_map<std::string, FeedData> GetMap() const
    {
        return feed_data_map_;
    }

    inline std::unordered_map<std::string, FeedData>::iterator Begin()
    {
        return feed_data_map_.begin();
    }

    inline std::unordered_map<std::string, FeedData>::iterator End()
    {
        return feed_data_map_.end();
    }

    std::string        ToString();
    static FeedDataMap FromNpyFolder(const std::string& base_folder);
    void               SaveNpy(const std::string& base_folder);

private:
    std::unordered_map<std::string, FeedData> feed_data_map_;

    inline bool IsValid(const FeedData& feed_data)
    {
        return feed_data.GetSize() > 0 && feed_data.data_ != nullptr;
    }
};
}  // namespace lightinfer