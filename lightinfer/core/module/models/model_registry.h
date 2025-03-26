#pragma once

#include "lightinfer/core/module/models/model_base.h"

// namespace lightinfer {

// typedef ModelBase* (*ModelConstructor)(INIReader);

// class ModelFactory {
// public:
//     static ModelFactory& GetInstance()
//     {
//         static ModelFactory factory;
//         return factory;
//     }

//     inline void ModelRegister(const std::string& model_name, ModelConstructor obj)
//     {
//         if (obj) {
//             object_map_.insert(std::map<std::string, ModelConstructor>::value_type(model_name, obj));
//         }
//         else {
//             LI_THROW(InvalidArgument("Model {} is nullptr.", model_name));
//         }
//     }

//     inline ModelBase* CreateModel(const std::string& model_name, INIReader reader)
//     {
//         std::map<std::string, ModelConstructor>::const_iterator iter = object_map_.find(model_name);
//         if (iter != object_map_.end()) {
//             return iter->second(reader);
//         }
//         else {
//             LI_THROW(Unimplemented("Model {} is not registered.", model_name));
//         }
//     }

// private:
//     std::map<std::string, ModelConstructor> object_map_;
// };

// class ModelRegister {
// public:
//     ModelRegister(std::string model_name, ModelConstructor obj)
//     {
//         ModelFactory::GetInstance().ModelRegister(model_name, obj);
//     }
// };

// }  // namespace lightinfer

// #define ATER_REGISTER_MODEL(model_name)                                                                                \
//     ::lightinfer::ModelBase* CreateModel##model_name(::lightinfer::INIReader reader)                                               \
//     {                                                                                                                  \
//         return new model_name(reader);                                                                                 \
//     }                                                                                                                  \
//     static const ::lightinfer::ModelRegister _reg_model_##model_name(#model_name, CreateModel##model_name);
