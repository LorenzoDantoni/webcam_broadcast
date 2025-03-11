# Video Broadcaster

Real-time segmentation using YOLOv8 to add effects to your webcam (remove background, add new background, add blur). 

The original frame from the webcam is modified based on the masks given by the YOLO model, and the new frame is sent to a virtual cam (OBS virtual cam) that handles the continuous streaming of frames (like a valid Webcam).

