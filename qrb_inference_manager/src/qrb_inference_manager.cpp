// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear
#include "qrb_inference_manager.hpp"

#include <stdexcept>

#include "qnn_inference/qnn_inference.hpp"

namespace qrb::inference_mgr
{

/**
 * \brief initialize qrb_inference_ to QnnInference
 * \param model_path path of model
 * \param backend_option backend lib of QNN
 * \param htp_core_ids cores inside hwDevices[0] to bind to (empty = no binding)
 * \throw std::logic_error, if param not meet requirement
 */
QrbInferenceManager::QrbInferenceManager(const std::string & model_path,
    const std::string & backend_option,
    const std::vector<int32_t> & htp_core_ids)
{
  auto is_so_model = (std::string::npos != model_path.find(".so"));
  auto is_bin_model = (std::string::npos != model_path.find(".bin"));

  if (!is_so_model && !is_bin_model) {
    throw std::logic_error("ERROR: Model format NOT support!");
  }

  qrb_inference_ = std::make_unique<QnnInference>(model_path, backend_option, htp_core_ids);

  if (qrb_inference_->inference_init() != StatusCode::SUCCESS) {
    throw std::logic_error("ERROR: Inference init fail!");
  }

  if (qrb_inference_->inference_graph_init() != StatusCode::SUCCESS) {
    throw std::logic_error("ERROR: Inference graph init fail!");
  }
}

bool QrbInferenceManager::inference_execute(const std::vector<uint8_t> & input_tensor_data)
{
  if (qrb_inference_->inference_execute(input_tensor_data) == StatusCode::SUCCESS) {
    return true;
  }
  return false;
}

std::vector<OutputTensor> QrbInferenceManager::get_output_tensors()
{
  return this->qrb_inference_->get_output_tensors();
}

}  // namespace qrb::inference_mgr
