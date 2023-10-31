\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* <br>
INSTALLATION AND DEPLOYMENT GUIDELINE YOLOV7's PRE-PRETRAINED MODEL ON
JETSON PLATFORM
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* <br>

1\. Requirements - To deploy yolov7's pre-trained model on jetson
platform, we need to install:  + JetPack  + NDIVIA DeepStream SDK  +
DeepStream-Yolo

\- Depend on the version of DeepStream (5.1, 6.0/6.0.1, 6.1, 6.1.1, 6.2,
6.3), we need to install corresponding version of JetPack and DeepStream
SDK. Detail information is following:

1.1. DeepStream 6.3 on Jetson platform  - JetPack 5.1.2  - NVIDIA
DeepStream SDK 6.3  - DeepStream-Yolo

1.2. DeepStream 6.2 on Jetson platform  - JetPack 5.1.2 / 5.1.1 / 5.1  -
NVIDIA DeepStream SDK 6.2  - DeepStream-Yolo

1.3. DeepStream 6.1.1 on Jetson platform  - JetPack 5.0.2  - NVIDIA
DeepStream SDK 6.1.1  - DeepStream-Yolo

1.4. DeepStream 6.1 on Jetson platform  - JetPack 5.0.1 DP  - NVIDIA
DeepStream SDK 6.1  - DeepStream-Yolo

1.5. DeepStream 6.0.1 / 6.0 on Jetson platform  - JetPack 4.6.4  -
NVIDIA DeepStream SDK 6.0.1 / 6.0  - DeepStream-Yolo

1.6. DeepStream 5.1 on Jetson platform  - JetPack 4.5.1  - NVIDIA
DeepStream SDK 5.1  - DeepStream-Yolo

(\*) Note: This guideline we use DeepStream 6.2 and JetPack 5.1.1 for
stability

2\. Deployment Yolov7's pre-trained model on jetson platform

2.1 Install DeepStream

2.1.1 Install Dependencies  - Enter the following commands to install
the prerequisite packages: """ $ sudo apt install \\ libssl1.1 \\
libgstreamer1.0-0 \\ gstreamer1.0-tools \\ gstreamer1.0-plugins-good \\
gstreamer1.0-plugins-bad \\ gstreamer1.0-plugins-ugly \\
gstreamer1.0-libav \\ libgstreamer-plugins-base1.0-dev \\
libgstrtspserver-1.0-0 \\ libjansson4 \\ libyaml-cpp-dev """

 - Install librdkafka (to enable Kafka protocol adaptor for message
broker)  + Clone the librdkafka repository from GitHub: """ $ git clone
https://github.com/edenhill/librdkafka.git """  + Configure and build
the library: """ $ cd librdkafka $ git reset --hard
7101c2310341ab3f4675fc565f64f0967e135a6a ./configure $ make $ sudo make
install """  + Copy the generated libraries to the deepstream directory:
""" $ sudo mkdir -p /opt/nvidia/deepstream/deepstream-6.2/lib $ sudo cp
/usr/local/lib/librdkafka\* /opt/nvidia/deepstream/deepstream-6.2/lib
"""

2.1.2 Install the DeepStream SDK  - Method 1: Using the DeepStream tar
package:
https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived
(You need a ndivia account to down load, so please register ndivia.)  +
Download the DeepStream 6.2 Jetson tar package
deepstream_sdk_v6.2.0_jetson.tbz2 to the Jetson device.  + Enter the
following commands to extract and install the DeepStream SDK: """ $ sudo
tar -xvf deepstream_sdk_v6.2.0_jetson.tbz2 -C / $ cd
/opt/nvidia/deepstream/deepstream-6.2 $ sudo ./install.sh $ sudo
ldconfig """  - Method 2: Using the DeepStream Debian package:
https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived
 + Download the DeepStream 6.3 Jetson Debian package
deepstream-6.2_6.2.0-1_arm64.deb to the Jetson device. Enter the
following command: """ $ sudo apt-get install
./deepstream-6.2_6.2.0-1_arm64.deb """

2.2 Install PyTorch and Torchvision  - We cannot install PyTorch and
Torchvision from pip because they are not compatible to run on Jetson
platform which is based on ARM aarch64 architecture. Therefore we need
to manually install pre-built PyTorch pip wheel and compile/ install
Torchvision from source.  - Visit this link:
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 to access
all the PyTorch and Torchvision links. In this guideline, we use PyTorch
v1.11 - torchvision v0.12.0  - Download file
'torch-1.11.0-cp38-cp38-linux_aarch64.whl' from
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048  -
Install torch v1.11, run the following commanda: """ $ sudo apt-get
install -y libopenblas-base libopenmpi-dev $ pip3 install
torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl """  - To
install torchvision, run the following commands: """ $ sudo apt install
-y libjpeg-dev zlib1g-dev $ git clone --branch v0.12.0
https://github.com/pytorch/vision torchvision $ cd torchvision $ python3
setup.py install --user """

2.3 Convert Yolov7's pre-trained model to onnx model

2.3.1 Download the YOLOv7 official repository and install the
requirements  - Yolov7's Repository:
https://github.com/WongKinYiu/yolov7  - Run the following commands: """
$ git clone https://github.com/WongKinYiu/yolov7.git $ cd yolov7 """  -
Open the file 'requirements.txt', delete the line that includes
'tensorboard' lib  - Run the following commands: """ $ pip3 install -r
requirements.txt $ pip3 install onnx onnxruntime """

2.3.2 Copy conversor and Convert model  - Download the DeepStream-Yolo
repository ( https://github.com/marcoslucianops/DeepStream-Yolo )  -
Copy the 'export_yoloV7.py' file from 'DeepStream-Yolo/utils' directory
to the 'yolov7' root folder.  - Copy yolov7's pre-trained model to the
'yolov7' root folder also. We used Pytorch framework to training Yolov7.
 - Generate the ONNX model file: """ $ python3 export_yoloV7.py -w
yolov7.pt --dynamic """ NOTE: To use dynamic batch-size (DeepStream \>=
6.1)

2.3.3 Copy generated files  - Copy the generated ONNX model file and
'labels.txt' file (if generated) to the 'DeepStream-Yolo' root folder.

2.4 Compile the lib

 - Open the 'DeepStream-Yolo' root folder and compile the lib:  - Run
the following command for DeepStream 6.3/6.2.6.1.1/6.1: """ $
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo """  - For DeepStream
6.0.1 / 6.0 / 5.1, run the following command: """ $ CUDA_VER=10.2 make
-C nvdsinfer_custom_impl_Yolo """

2.5 Edit the 'config_infer_primary_yoloV7.txt' and
'deepstream_app_config.txt' file on 'DeepStream-Yolo' root folder  -
Edit the 'config_infer_primary_yoloV7.txt' file according to your model:
""" \[property\] ... onnx-file=yolov7.onnx \# onnx model ...
num-detected-classes=4 \# number of class of dataset ...
parse-bbox-func-name=NvDsInferParseYolo ... """  - Edit the
'deepstream_app_config.txt' file: """ ... \[primary-gie\] ...
config-file=config_infer_primary_yoloV7.txt """

3\. Run deployed model  - Open the 'DeepStream-Yolo' root folder and
compile the lib: """ $ deepstream-app -c deepstream_app_config.txt """
NOTE: In the first run, program will generate TensorRT engine file. The
TensorRT engine file may take a very long time to generate (sometimes
more than 10 minutes). If it can not generate TensorRT engine file,
maybe some things is wrong. You need install again.
