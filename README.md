# fast-sam-nvidia-jeston

Step 1: setup jetson (refer to this guide)


step 2: clone this repo and install requirements 
```
pip install -r requirements.txt
```

step 3:
```
pip install git+https://github.com/openai/CLIP.git
```

step 4:
Download model weights in PyTorch format from here : https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints

step 5:
convert to TensorRT. Create a new Python script and enter the following code:
```
from ultralytics import YOLO

model = YOLO('FastSAM-s.pt')  # load a custom trained
# TensorRT FP32 export
# model.export(format='engine', device='0', imgsz=640)

# TensorRT FP16 export
model.export(format='engine', device='0', imgsz=640, half=True)
```

Save and execute the file.


step 6: Run (using webcam)
```
python3 Inference_video.py --model_path FastSAM-s.engine --img_path /dev/video2 --imgsz 640
```
"/dev/video2" is the path of webcam in Linux.

