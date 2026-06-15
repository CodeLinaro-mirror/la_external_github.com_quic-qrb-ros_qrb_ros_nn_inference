<div align="center">
  <h1>QRB ROS NN Inference</h1>
  <p align="center">
   <img src="./docs/assets/qrb_ros_nn_inference.jpg">
  </p>
  <p>ROS2 package for performing neural network model</p>
  <a href="https://ubuntu.com/download/qualcomm-iot" target="_blank"><img src="https://img.shields.io/badge/Qualcomm%20Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white" alt="Qualcomm Ubuntu"></a>
  <a href="https://docs.ros.org/en/jazzy/" target="_blank"><img src="https://img.shields.io/badge/ROS%20Jazzy-1c428a?style=for-the-badge&logo=ros&logoColor=white" alt="Jazzy"></a>
</div>

---

## 👋 Overview

**qrb_ros_nn_inference** is a ROS2 package for performing neural network model, providing 🤖AI-based perception for robotics applications. It provides:
- ✨model inference API which supports three kinds of model format: **.so**, **.bin**
- 🚀model inference acceleration based on Qualcomm platforms

<div align="center">
  <img src="./docs/assets/architecture.jpg" alt="architecture">
</div>

<br>

**qrb_ros_nn_inference** is a ROS2 package based on [qrb_inference_manager](./qrb_inference_manager/README.md) which is a C++ library encapsulates the APIs of [Qualcomm AI Engine Direct](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html) and [QNN Delegate for TensorFlow Lite](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/overview.html).

**qrb_ros_nn_inference** receives data from a specific topic, then without any processing, directly uses the received data for model inference. Afterwards, the results of the model inference are sent out directly through another specific topic.

---

## 🔎 Table of Contents
  * [APIs](#-apis)
    * [`qrb_ros_nn_inference` APIs](#-qrb_ros_nn_inference-apis)
    * [`qrb_inference_manager` APIs](#-qrb_inference_manager-apis)
  * [Usage](#-usage)
  * [Build from Source](#-build-from-source)
  * [Contributing](#-contributing)
  * [License](#-license)

---

## ⚓ APIs

### 🔹 `qrb_ros_nn_inference` APIs

#### ROS node parameters

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default Value</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>backend_option</td>
    <td>string</td>
    <td>""</td>
    <td>Hardware acceleration option for model inference, vaild values are listed <a href="https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference/blob/main/qrb_inference_manager/Documentation.md#2.1" target="_blank">here</a></td>
  </tr>
  <tr>
    <td>model_path</td>
    <td>string</td>
    <td>""</td>
    <td>Path of model file</td>
  </tr>
</table>

#### ROS topics

<table>
  <tr>
    <th>Topic Name</th>
    <th>Message Type</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>qrb_inference_input_tensor</td>
    <td><a href="https://github.com/qualcomm-qrb-ros/qrb_ros_interfaces/blob/main/qrb_ros_tensor_list_msgs/msg/TensorList.msg" target="_blank">TensorList</a></td>
    <td>Subscribed topic</td>
  </tr>
  <tr>
    <td>qrb_inference_output_tensor</td>
    <td><a href="https://github.com/qualcomm-qrb-ros/qrb_ros_interfaces/blob/main/qrb_ros_tensor_list_msgs/msg/TensorList.msg" target="_blank">TensorList</a></td>
    <td>Published topic</td>
  </tr>
</table>

### 🔹 `qrb_inference_manager` APIs

Please see [qrb_inference_manager APIs](./qrb_inference_manager/Documentation.md).

---

## 🚀 Usage

1. install the qrb_ros_nn_inference.

2. prepare the pre-process node and post-process node for model inference

  ```bash
    # qrb_ros_nn_inference/test includes the pre-process node and post-process node
    mkdir -p ~/ros-ws/src && cd ~/ros-ws/src && \
    git clone https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference
  ```

3. test qrb_ros_nn_inference with YOLOv8 detection model

    3.1 download [YOLOv8-Detection](https://aihub.qualcomm.com/iot/models/yolov8_det).

    3.2 download the test image for object detecion

    ```bash
    wget -P ~/ros-ws/src/qrb_ros_nn_inference/test/qrb_ros_pre_process/image/ \
    https://ultralytics.com/images/bus.jpg && \
    python3 ~/ros-ws/src/qrb_ros_nn_inference/test/qrb_ros_post_process/scripts/yolov8_input_pre_process.py
    ```

    3.3 point out the raw image path and model path in `~/ros-ws/src/qrb_ros_nn_inference/test/qrb_ros_post_process/launch/nn_node_test.launch.py`

    ```python
    pre_process_node = ComposableNode(
       package = "qrb_ros_pre_process",
       plugin = "qrb_ros::pre_process::QrbRosPreProcessNode",
       name = "pre_process_node",
       parameters=[
         {
           "image_path": os.environ['HOME'] + "/ros-ws/src/qrb_ros_nn_inference/test/qrb_ros_pre_process/image/bus.raw"
         }
       ]
    )

    nn_inference_node = ComposableNode(
       package = "qrb_ros_nn_inference",
       plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
       name = "nn_inference_node",
       parameters=[
         {
           "backend_option": "",
           "model_path": "/path/to/model"
         }
       ]
    )
    ```

    3.4 build the pre-process node and post-process node

    ```bash
      source /opt/ros/jazzy/setup.bash && \
      cd ~/ros-ws && \
      rm ./src/qrb_ros_nn_inference/test/qrb_ros_post_process/COLCON_IGNORE && \
      rm ./src/qrb_ros_nn_inference/test/qrb_ros_pre_process/COLCON_IGNORE && \
      colcon build --packages-select qrb_ros_pre_process qrb_ros_post_process
    ```

    3.5 execute the inference

    ```bash
      cd ~/ros-ws && \
      source install/local_setup.bash && \
      ros2 launch qrb_ros_post_process nn_node_test.launch.py
    ```

    3.6 visualize the detection result

    ```bash
      python3 ~/ros-ws/src/qrb_ros_nn_inference/test/qrb_ros_post_process/scripts/qrb_ros_yolo_detection_visualizer.py \
      --original_image ~/ros-ws/src/qrb_ros_nn_inference/test/qrb_ros_pre_process/image/bus.jpg
    ```

    reulst image will be stroed in `~/ros-ws/src/qrb_ros_nn_inference/test/qrb_ros_post_process/inference_result`

---

## 👨‍💻 Build from Source

Source is located at sources/quic-qrb-ros/qrb_ros_nn_inference in the workspace.

```
cd build-utils/ubuntu/
python3 build.py --gen-debians --package ros-jazzy-qrb-ros-nn-inference
```

Built .deb files are output to:

```
<workspace>/debian_packages/oss/ros-jazzy-qrb-ros-nn-inference
```

---

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).
Feel free to create an issue for bug report, feature requests or any discussion.

---

## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](./LICENSE) for the full license text.
