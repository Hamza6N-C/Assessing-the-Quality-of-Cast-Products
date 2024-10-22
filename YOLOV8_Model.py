from ultralytics import YOLO

#This line of code initializes a YOLO model for image classification by loading the pre-trained weights from the yolov8n-cls.pt file. 
model = YOLO('yolov8n-cls.pt') 
# Train the model
results = model.train(data='casting_data', epochs=100, imgsz=512)

#yolov8: This refers to version 8 of the YOLO model.
#The file extension .pt indicates that it is a PyTorch model file. PyTorch is a popular deep learning framework.
#n: This could indicate the 'nano' version, which is a smaller and faster variant of the YOLOv8 model, optimized for performance on devices with limited computational power.
#cls: stands for "classification," indicating that this particular model is trained for classification tasks rather than detection or segmentation.
