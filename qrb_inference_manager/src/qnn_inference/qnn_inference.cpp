// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "qnn_inference/qnn_inference.hpp"

namespace qrb::inference_mgr
{

QnnInference::QnnInference(const std::string & model_path,
    const std::string & backend_option,
    const std::vector<int32_t> & htp_core_ids)
  : model_path_(model_path), backend_option_(backend_option), htp_core_ids_(htp_core_ids)
{
  auto is_bin_model = (std::string::npos != model_path.find(".bin"));

  if (is_bin_model) {
    QRB_INFO("Loading model from binary file: ", model_path);

    load_model_from_binary = true;
    qnn_interface_ = std::make_unique<QnnInterface>(
        backend_option_, &backend_lib_handle, qnn_syslib_path_, &sys_lib_handle_);
  } else {
    qnn_interface_ = std::make_unique<QnnInterface>(
        model_path_, backend_option_, &backend_handle_, &model_handle_);
  }
}

QnnInference::~QnnInference()
{
  free_graphs_info();
  free_context();
  free_device();
  free_backend();

  if (backend_lib_handle) {
    ::dlclose(backend_lib_handle);
    backend_lib_handle = nullptr;
  }

  if (model_handle_) {
    ::dlclose(model_handle_);
    model_handle_ = nullptr;
  }

  if (sys_lib_handle_) {
    ::dlclose(sys_lib_handle_);
    sys_lib_handle_ = nullptr;
  }
}

StatusCode QnnInference::inference_init()
{
  if (initialize_backend() != StatusCode::SUCCESS) {
    return StatusCode::FAILURE;
  }

  if (create_device() != StatusCode::SUCCESS) {
    return StatusCode::FAILURE;
  }

  if (!htp_core_ids_.empty() &&
      backend_option_.find("libQnnHtp") != std::string::npos) {
    if (set_htp_performance_mode() != StatusCode::SUCCESS) {
      QRB_WARNING("set_htp_performance_mode failed, continue without perf binding");
    }
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnInference::inference_graph_init()
{
  if (true == load_model_from_binary) {
    if (init_graph_from_binary() != StatusCode::SUCCESS) {
      return StatusCode::FAILURE;
    }
  } else {
    if (create_context() != StatusCode::SUCCESS) {
      return StatusCode::FAILURE;
    }

    if (compose_graphs() != StatusCode::SUCCESS) {
      return StatusCode::FAILURE;
    }

    if (finalize_graphs() != StatusCode::SUCCESS) {
      return StatusCode::FAILURE;
    }
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnInference::inference_execute(const std::vector<uint8_t> & input_tensor_data)
{
  if (input_tensor_data.size() == 0) {
    QRB_ERROR("Input tensor is NULL!");
    return StatusCode::FAILURE;
  }

  for (uint32_t i = 0; i < graphs_count_; i++) {
    const auto & graphs_info = (*(graphs_info_))[i];

    auto io_tensors =
        QnnTensor(graphs_info.num_of_input_tensors, graphs_info.num_of_output_tensors);

    if (StatusCode::SUCCESS != io_tensors.setup_tensors(io_tensors.inputs,
                                   io_tensors.num_of_input_tensors, graphs_info.input_tensors)) {
      QRB_ERROR("Setup input tensors failed!");
      return StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS != io_tensors.setup_tensors(io_tensors.outputs,
                                   io_tensors.num_of_output_tensors, graphs_info.output_tensors)) {
      QRB_ERROR("Setup output tensors failed!");
      return StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS != io_tensors.write_input_tensors(input_tensor_data)) {
      QRB_ERROR("Write QNN input tensors failed!");
      return StatusCode::FAILURE;
    }

    if (QNN_GRAPH_NO_ERROR != this->qnn_interface_->interface.graphExecute(graphs_info.graph,
                                  io_tensors.inputs, io_tensors.num_of_input_tensors,
                                  io_tensors.outputs, io_tensors.num_of_output_tensors, nullptr,
                                  nullptr)) {
      QRB_ERROR("QNN graphExecute failed!");
      return StatusCode::FAILURE;
    }

#ifndef __hexagon__
    output_tensor_ = io_tensors.read_output_tensors(graphs_info.num_of_output_tensors);
    if (output_tensor_.size() == 0) {
      QRB_ERROR("Get ouput tensors failed!");
    }
#endif
  }

  return StatusCode::SUCCESS;
}

const std::vector<OutputTensor> QnnInference::get_output_tensors()
{
  return std::move(output_tensor_);
}

StatusCode QnnInference::initialize_backend()
{
  auto qnn_status = qnn_interface_->interface.backendCreate(
      nullptr, (const QnnBackend_Config_t **)(nullptr), &(backend_handle_));

  if (QNN_BACKEND_NO_ERROR != qnn_status) {
    QRB_ERROR("Could not initialize backend due to error = %d!", qnn_status);
    return StatusCode::FAILURE;
  }

  QRB_INFO(backend_option_, " initialize successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::create_device()
{
  auto is_device_property_supported = [this] {
    if (nullptr != qnn_interface_->interface.propertyHasCapability) {
      auto qnn_status = qnn_interface_->interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);

      if (QNN_PROPERTY_NOT_SUPPORTED == qnn_status) {
        QRB_WARNING("Device property is not supported!");
      } else if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnn_status) {
        QRB_ERROR("Device property is not known to backend!");
        return StatusCode::FAILURE;
      }
    }
    return StatusCode::SUCCESS;
  };

  if (StatusCode::FAILURE != is_device_property_supported()) {
    if (nullptr != qnn_interface_->interface.deviceCreate) {
      bool use_core_binding = !htp_core_ids_.empty() &&
          (backend_option_.find("libQnnHtp") != std::string::npos);

      if (use_core_binding) {
        auto qnn_status = create_device_with_core_binding();
        if (qnn_status != StatusCode::SUCCESS) {
          return qnn_status;
        }
      } else {
        auto qnn_status =
            qnn_interface_->interface.deviceCreate(nullptr, nullptr, &(device_handle_));
        if (QNN_SUCCESS != qnn_status && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnn_status) {
          QRB_ERROR("Failed to create device!");
          return StatusCode::FAILURE;
        }
      }
    }
    support_device_ = true;
  }

  QRB_INFO("Qnn device initialize successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::create_device_with_core_binding()
{
  // Fixed device_id=0 (hwDevices[0], the multi-core HTP device on this SoC).
  // Select target cores from htp_core_ids_ (indices into hwDevices[0].v1.cores[]).
  // Mirrors qcnode QnnImpl::Initialize: deviceId=GetQnnDeviceId(HTP0)=0, coreIds list.
  const QnnDevice_PlatformInfo_t * platform_info = nullptr;
  if (nullptr != qnn_interface_->interface.deviceGetPlatformInfo) {
    qnn_interface_->interface.deviceGetPlatformInfo(nullptr, &platform_info);
  }

  std::vector<QnnDevice_PlatformInfo_t> device_platform_info_vec;
  std::vector<QnnDevice_Config_t> device_custom_configs;
  std::vector<const QnnDevice_Config_t *> device_config_ptrs;
  QnnDevice_HardwareDeviceInfo_t * hw_device_copy = nullptr;
  QnnDevice_CoreInfo_t * cores_heap = nullptr;

  if (platform_info) {
    // Log all available devices and cores
    for (uint32_t d = 0; d < platform_info->v1.numHwDevices; ++d) {
      const auto & hw_dev = platform_info->v1.hwDevices[d];
      for (uint32_t c = 0; c < hw_dev.v1.numCores; ++c) {
        QRB_INFO("Available: device_id=", hw_dev.v1.deviceId,
            " core_id=", hw_dev.v1.cores[c].v1.coreId,
            " core_type=", hw_dev.v1.cores[c].v1.coreType);
      }
    }

    const uint32_t kDeviceIdx = 0;
    if (kDeviceIdx < platform_info->v1.numHwDevices) {
      hw_device_copy = new QnnDevice_HardwareDeviceInfo_t;
      memcpy(hw_device_copy, &platform_info->v1.hwDevices[kDeviceIdx],
          sizeof(QnnDevice_HardwareDeviceInfo_t));

      // Build selected cores (mirrors qcnode coreIds loop)
      std::vector<QnnDevice_CoreInfo_t> selected_cores;
      const uint32_t num_avail = hw_device_copy->v1.numCores;
      bool valid = true;
      for (int32_t cidx : htp_core_ids_) {
        if (cidx < 0 || static_cast<uint32_t>(cidx) >= num_avail) {
          QRB_ERROR("htp_core_ids entry ", cidx, " out of range, numCores=", num_avail);
          valid = false;
          break;
        }
        selected_cores.push_back(hw_device_copy->v1.cores[cidx]);
      }

      if (valid) {
        cores_heap = new QnnDevice_CoreInfo_t[selected_cores.size()];
        for (size_t i = 0; i < selected_cores.size(); ++i) cores_heap[i] = selected_cores[i];
        hw_device_copy->v1.numCores = static_cast<uint32_t>(selected_cores.size());
        hw_device_copy->v1.cores    = cores_heap;

        device_platform_info_vec.push_back(QnnDevice_PlatformInfo_t());
        QnnDevice_PlatformInfo_t & e = device_platform_info_vec.back();
        e.version         = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
        e.v1.numHwDevices = 1;
        e.v1.hwDevices    = hw_device_copy;

        device_custom_configs.push_back(QnnDevice_Config_t());
        device_custom_configs.back().option       = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
        device_custom_configs.back().hardwareInfo =
            reinterpret_cast<QnnDevice_PlatformInfo_t *>(&device_platform_info_vec[0]);
        device_config_ptrs.push_back(&device_custom_configs.back());
        device_config_ptrs.push_back(nullptr);

        std::string cores_str;
        for (size_t i = 0; i < htp_core_ids_.size(); ++i)
          cores_str += (i ? "," : "") + std::to_string(htp_core_ids_[i]);
        QRB_INFO("HTP device binding: device_id=0 cores=[", cores_str, "]");
      } else {
        delete hw_device_copy; hw_device_copy = nullptr;
      }
    } else {
      QRB_WARNING("hwDevices[0] not found, using default device");
    }
  }

  auto qnn_status = qnn_interface_->interface.deviceCreate(
      nullptr,
      device_config_ptrs.empty() ? nullptr : device_config_ptrs.data(),
      &(device_handle_));

  if (nullptr != qnn_interface_->interface.deviceFreePlatformInfo && platform_info) {
    qnn_interface_->interface.deviceFreePlatformInfo(nullptr, platform_info);
  }
  delete[] cores_heap;  cores_heap = nullptr;
  delete hw_device_copy; hw_device_copy = nullptr;

  if (QNN_SUCCESS != qnn_status && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnn_status) {
    QRB_WARNING("deviceCreate with binding failed (err=", qnn_status,
        "), falling back to default device");
    auto fallback = qnn_interface_->interface.deviceCreate(nullptr, nullptr, &(device_handle_));
    if (QNN_SUCCESS != fallback && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != fallback) {
      QRB_ERROR("Failed to create device (fallback)!");
      return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
  }

  if (!device_config_ptrs.empty()) {
    std::string cs;
    for (size_t i = 0; i < htp_core_ids_.size(); ++i)
      cs += (i ? "," : "") + std::to_string(htp_core_ids_[i]);
    QRB_INFO("HTP device binding device_id=0 cores=[", cs, "] applied successfully");
  }
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::set_htp_performance_mode()
{
  // Mirrors qcnode QnnImpl::SetHtpPerformanceMode():
  // deviceGetInfrastructure -> createPowerConfigId(deviceId, coreId) -> setPowerConfig(BURST)
  if (nullptr == qnn_interface_->interface.deviceGetInfrastructure) {
    QRB_WARNING("deviceGetInfrastructure not supported, skipping perf binding");
    return StatusCode::SUCCESS;
  }

  QnnDevice_Infrastructure_t device_infra = nullptr;
  if (QNN_SUCCESS != qnn_interface_->interface.deviceGetInfrastructure(&device_infra) ||
      nullptr == device_infra) {
    QRB_ERROR("deviceGetInfrastructure failed");
    return StatusCode::FAILURE;
  }

  auto * htp_infra = static_cast<QnnHtpDevice_Infrastructure_t *>(device_infra);
  perf_infra_ = &(htp_infra->perfInfra);

  if (nullptr == perf_infra_->createPowerConfigId ||
      nullptr == perf_infra_->setPowerConfig ||
      nullptr == perf_infra_->destroyPowerConfigId) {
    QRB_WARNING("perfInfra function pointers null, skipping perf binding");
    perf_infra_ = nullptr;
    return StatusCode::SUCCESS;
  }

  // Resolve real deviceId and coreId values from platform info
  // (qcnode: deviceId = hwDevices[GetQnnDeviceId(HTP0)].v1.deviceId = hwDevices[0].v1.deviceId,
  //          coreId  = hwDevices[0].v1.cores[coreIds[i]].v1.coreId)
  const QnnDevice_PlatformInfo_t * platform_info = nullptr;
  if (nullptr == qnn_interface_->interface.deviceGetPlatformInfo ||
      QNN_SUCCESS !=
          qnn_interface_->interface.deviceGetPlatformInfo(nullptr, &platform_info) ||
      nullptr == platform_info) {
    QRB_ERROR("deviceGetPlatformInfo failed in set_htp_performance_mode");
    perf_infra_ = nullptr;
    return StatusCode::FAILURE;
  }

  const uint32_t kDeviceIdx = 0;
  if (kDeviceIdx >= platform_info->v1.numHwDevices) {
    QRB_ERROR("hwDevices[0] unavailable, numHwDevices=", platform_info->v1.numHwDevices);
    qnn_interface_->interface.deviceFreePlatformInfo(nullptr, platform_info);
    perf_infra_ = nullptr;
    return StatusCode::FAILURE;
  }

  const auto & hw_device = platform_info->v1.hwDevices[kDeviceIdx].v1;
  uint32_t real_device_id = hw_device.deviceId;
  bool all_ok = true;

  for (int32_t cidx : htp_core_ids_) {
    if (cidx < 0 || static_cast<uint32_t>(cidx) >= hw_device.numCores) {
      QRB_ERROR("htp_core_ids entry ", cidx, " out of range, numCores=", hw_device.numCores);
      all_ok = false;
      break;
    }
    uint32_t real_core_id = hw_device.cores[cidx].v1.coreId;
    uint32_t power_config_id = UINT32_MAX;
    auto ret = perf_infra_->createPowerConfigId(real_device_id, real_core_id, &power_config_id);
    if (QNN_SUCCESS != ret) {
      QRB_ERROR("createPowerConfigId failed for core ", cidx,
          " (real_core_id=", real_core_id, "), error=", ret);
      all_ok = false;
      break;
    }
    power_config_ids_.push_back(power_config_id);
    QRB_INFO("createPowerConfigId OK: device_id=", real_device_id,
        " core_id=", real_core_id, " -> power_config_id=", power_config_id);
  }

  qnn_interface_->interface.deviceFreePlatformInfo(nullptr, platform_info);

  if (!all_ok) {
    free_perf_config();
    return StatusCode::FAILURE;
  }

  // setPowerConfig: PERFORMANCE_MODE / BURST (matches qcnode BURST profile for HTP0 on 8797)
  QnnHtpPerfInfrastructure_PowerConfig_t power_config;
  memset(&power_config, 0, sizeof(power_config));
  power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  power_config.dcvsV3Config.dcvsEnable             = 0;
  power_config.dcvsV3Config.setDcvsEnable          = 1;
  power_config.dcvsV3Config.powerMode              =
      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  power_config.dcvsV3Config.setSleepLatency        = 1;
  power_config.dcvsV3Config.sleepLatency           = 40;  // sg_lowerLatency (V66+)
  power_config.dcvsV3Config.setBusParams           = 1;
  power_config.dcvsV3Config.setCoreParams          = 1;
  power_config.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  power_config.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  power_config.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  power_config.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  power_config.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  power_config.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

  for (uint32_t pcid : power_config_ids_) {
    power_config.dcvsV3Config.contextId = pcid;
    const QnnHtpPerfInfrastructure_PowerConfig_t * configs[] = { &power_config, nullptr };
    auto ret = perf_infra_->setPowerConfig(pcid, configs);
    if (QNN_SUCCESS != ret) {
      QRB_WARNING("setPowerConfig failed for power_config_id=", pcid, " error=", ret);
    }
  }

  std::string cores_str;
  for (size_t i = 0; i < htp_core_ids_.size(); ++i)
    cores_str += (i ? "," : "") + std::to_string(htp_core_ids_[i]);
  QRB_INFO("HTP performance mode bound to device_id=0 cores=[", cores_str, "]");
  return StatusCode::SUCCESS;
}

void QnnInference::free_perf_config()
{
  if (nullptr != perf_infra_) {
    for (uint32_t pcid : power_config_ids_) {
      perf_infra_->destroyPowerConfigId(pcid);
    }
    perf_infra_ = nullptr;
  }
  power_config_ids_.clear();
}

StatusCode QnnInference::create_context()
{
  if (QNN_CONTEXT_NO_ERROR != qnn_interface_->interface.contextCreate(
                                  backend_handle_, device_handle_, nullptr, &(context_))) {
    QRB_ERROR("Could not create context!");
    return StatusCode::FAILURE;
  }
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::compose_graphs()
{
  if (ModelError::MODEL_NO_ERROR !=
      qnn_interface_->compose_graphs(backend_handle_, qnn_interface_->interface, context_, nullptr,
          0, &(graphs_info_), &(graphs_count_), false, nullptr, QNN_LOG_LEVEL_MAX)) {
    QRB_ERROR("Failed in composeGraphs()!");
    return StatusCode::FAILURE;
  }
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::finalize_graphs()
{
  for (size_t i = 0; i < graphs_count_; i++) {
    if (QNN_GRAPH_NO_ERROR !=
        qnn_interface_->interface.graphFinalize((*graphs_info_)[i].graph, nullptr, nullptr)) {
      return StatusCode::FAILURE;
    }
  }

  return StatusCode::SUCCESS;
}

void QnnInference::free_graphs_info()
{
  if (graphs_info_ == nullptr) {
    return;
  }

  for (uint32_t i = 0; i < graphs_count_; i++) {
    auto & graph_info = graphs_info_[i];
    free(graph_info->graph_name);
    graph_info->graph_name = nullptr;

    QnnTensor tensor_ops;
    tensor_ops.free_qnn_tensors(graph_info->input_tensors, graph_info->num_of_input_tensors);
    tensor_ops.free_qnn_tensors(graph_info->output_tensors, graph_info->num_of_output_tensors);
  }

  free(*graphs_info_);
  *graphs_info_ = nullptr;
  free(graphs_info_);
  graphs_info_ = nullptr;
}

void QnnInference::free_context()
{
  if (QNN_CONTEXT_NO_ERROR != qnn_interface_->interface.contextFree(context_, nullptr)) {
    QRB_ERROR("Failed to free context!");
  }

  context_ = nullptr;
}

void QnnInference::free_device()
{
  // destroyPowerConfigId before deviceFree (mirrors qcnode DeInitialize order)
  free_perf_config();

  if (true == support_device_) {
    if (nullptr != qnn_interface_->interface.deviceFree) {
      auto qnn_status = qnn_interface_->interface.deviceFree(device_handle_);
      if (QNN_SUCCESS != qnn_status && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnn_status) {
        QRB_ERROR("Failed to free device!");
      }
    }
  }

  device_handle_ = nullptr;
}

void QnnInference::free_backend()
{
  if (qnn_interface_->interface.backendFree != nullptr) {
    if (QNN_BACKEND_NO_ERROR != qnn_interface_->interface.backendFree(backend_handle_)) {
      QRB_ERROR("Could not free backend!");
    }
  }

  backend_handle_ = nullptr;
}

StatusCode QnnInference::init_graph_from_binary()
{
  auto [model_buf, model_buf_size] = read_binary_model();
  if (nullptr == model_buf) {
    return StatusCode::FAILURE;
  }

  if (StatusCode::SUCCESS != get_and_set_graph_info_from_binary(model_buf, model_buf_size)) {
    return StatusCode::FAILURE;
  }

  if (StatusCode::SUCCESS != create_context_from_binary(model_buf, model_buf_size)) {
    return StatusCode::FAILURE;
  }

  QRB_INFO("Initialize Qnn graph from binary file successfully");
  return StatusCode::SUCCESS;
}

std::tuple<std::shared_ptr<uint8_t[]>, uint64_t> QnnInference::read_binary_model()
{
  uint64_t buf_size = 0;
  std::ifstream model_binary(model_path_, std::ifstream::binary);
  if (!model_binary.is_open() || model_binary.fail()) {
    QRB_ERROR("Fail to open model file: ", model_path_);
    return { nullptr, 0 };
  }

  model_binary.seekg(0, model_binary.end);
  buf_size = model_binary.tellg();
  model_binary.seekg(0, model_binary.beg);
  if (0 == buf_size) {
    QRB_ERROR("Received path to an empty file. Nothing to deserialize.");
    return { nullptr, 0 };
  }

  std::shared_ptr<uint8_t[]> model_buf(new uint8_t[buf_size]);
  if (!model_binary.read(reinterpret_cast<char *>(model_buf.get()), buf_size)) {
    QRB_ERROR("Fail to read model file: ", model_path_);
    return { nullptr, 0 };
  }

  return { model_buf, buf_size };
}

StatusCode QnnInference::get_and_set_graph_info_from_binary(
    const std::shared_ptr<uint8_t[]> model_buf,
    const uint64_t model_buf_size)
{
  QnnSystemContext_Handle_t sys_context_headle = nullptr;
  if (QNN_SUCCESS !=
      qnn_interface_->qnn_system_interface.systemContextCreate(&sys_context_headle)) {
    QRB_ERROR("Could not create system handle.");
    return StatusCode::FAILURE;
  }

  const QnnSystemContext_BinaryInfo_t * binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size = 0;
  if (QNN_SUCCESS !=
      qnn_interface_->qnn_system_interface.systemContextGetBinaryInfo(sys_context_headle,
          static_cast<void *>(model_buf.get()), model_buf_size, &binary_info, &binary_info_size)) {
    QRB_ERROR("Failed to get context binary info.");
    return StatusCode::FAILURE;
  }

  if (StatusCode::SUCCESS != set_up_graph_info(binary_info)) {
    return StatusCode::FAILURE;
  }

  qnn_interface_->qnn_system_interface.systemContextFree(sys_context_headle);
  sys_context_headle = nullptr;

  return StatusCode::SUCCESS;
}

template <typename T>
StatusCode QnnInference::copy_graph_info(T graph_info_from_binary)
{
  graphs_info_ = (GraphInfo **)calloc(graphs_count_, sizeof(GraphInfo *));
  if (nullptr == graphs_info_) {
    return StatusCode::FAILURE;
  }

  for (uint32_t i = 0; i < graphs_count_; i++) {
    auto graph_info_dst = (GraphInfo *)calloc(1, sizeof(GraphInfo));
    if (nullptr == graph_info_dst) {
      return StatusCode::FAILURE;
    }

    graph_info_dst->graph_name =
        (char *)malloc(sizeof(char) * (strlen(graph_info_from_binary.graphName) + 1));
    if (nullptr == graph_info_dst->graph_name) {
      return StatusCode::FAILURE;
    }
    memcpy(graph_info_dst->graph_name, graph_info_from_binary.graphName,
        strlen(graph_info_from_binary.graphName));
    graph_info_dst->graph_name[strlen(graph_info_from_binary.graphName)] = '\0';

    auto set_up_tensors_info = [](const Qnn_Tensor_t * src, const uint32_t cnt,
                                   Qnn_Tensor_t *& dst) {
      dst = (Qnn_Tensor_t *)calloc(cnt, sizeof(Qnn_Tensor_t));
      if (nullptr == dst) {
        return StatusCode::FAILURE;
      }

      QnnTensor tensor_ops;

      for (size_t i = 0; i < cnt; i++) {
        dst[i] = QNN_TENSOR_INIT;
        if (StatusCode::SUCCESS != tensor_ops.tensor_info_deep_copy(&dst[i], &src[i])) {
          return StatusCode::FAILURE;
        }
      }

      return StatusCode::SUCCESS;
    };

    graph_info_dst->num_of_input_tensors = graph_info_from_binary.numGraphInputs;
    if (StatusCode::SUCCESS != set_up_tensors_info(graph_info_from_binary.graphInputs,
                                   graph_info_from_binary.numGraphInputs,
                                   graph_info_dst->input_tensors)) {
      return StatusCode::FAILURE;
    }

    graph_info_dst->num_of_output_tensors = graph_info_from_binary.numGraphOutputs;
    if (StatusCode::SUCCESS != set_up_tensors_info(graph_info_from_binary.graphOutputs,
                                   graph_info_from_binary.numGraphOutputs,
                                   graph_info_dst->output_tensors)) {
      return StatusCode::FAILURE;
    }

    graphs_info_[i] = graph_info_dst;
  }

  return StatusCode::SUCCESS;
}

/// @brief set up graphs_info_ based on binary_info
/// @param binary_info
/// @return SUCCESS or FAILURE
StatusCode QnnInference::set_up_graph_info(const QnnSystemContext_BinaryInfo_t * binary_info)
{
  if (nullptr == binary_info) {
    return StatusCode::FAILURE;
  }

  switch (binary_info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
      graphs_count_ = binary_info->contextBinaryInfoV1.numGraphs;
      if (StatusCode::SUCCESS !=
          copy_graph_info(binary_info->contextBinaryInfoV1.graphs->graphInfoV1)) {
        return StatusCode::FAILURE;
      }
      break;
    }
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
      graphs_count_ = binary_info->contextBinaryInfoV2.numGraphs;
      if (StatusCode::SUCCESS !=
          copy_graph_info(binary_info->contextBinaryInfoV2.graphs->graphInfoV2)) {
        return StatusCode::FAILURE;
      }
      break;
    }
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
      graphs_count_ = binary_info->contextBinaryInfoV3.numGraphs;
      if (StatusCode::SUCCESS !=
          copy_graph_info(binary_info->contextBinaryInfoV3.graphs->graphInfoV3)) {
        return StatusCode::FAILURE;
      }
      break;
    }
    default: {
      QRB_ERROR("Unrecognized system context binary info version.");
      return StatusCode::FAILURE;
    }
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnInference::create_context_from_binary(const std::shared_ptr<uint8_t[]> model_buf,
    const uint64_t model_buf_size)
{
  if (qnn_interface_->interface.contextCreateFromBinary(backend_handle_, device_handle_, nullptr,
          static_cast<void *>(model_buf.get()), model_buf_size, &context_, nullptr)) {
    QRB_ERROR("Could not create context from binary!");
    return StatusCode::FAILURE;
  }

  for (uint32_t i = 0; i < graphs_count_; i++) {
    if (QNN_SUCCESS != qnn_interface_->interface.graphRetrieve(
                           context_, (*graphs_info_)[i].graph_name, &((*graphs_info_)[i].graph))) {
      QRB_ERROR("Unable to retrieve graph handle for the graph");
      return StatusCode::FAILURE;
    }
  }
  return StatusCode::SUCCESS;
}

}  // namespace qrb::inference_mgr
