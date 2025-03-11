import cv2
import pyvirtualcam
import torch
from engine import CustomSegmentationWithYOLO



class Streaming(CustomSegmentationWithYOLO):
    def __init__(
        self, 
        in_source=None, 
        out_source=None, 
        fps=None, 
        blur_strength=None,
        cam_fps=30,
        background="none"
    ):
        super().__init__(erode_size=4, erode_intensity=2)

        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.original_fps = cam_fps
        self.background = background
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        print(f"Device selected/found for inference: {self.device}")
    
    def update_streaming_config(
        self, 
        in_source=None, 
        out_source=None, 
        fps=None, 
        blur_strength=None, 
        background="none"
    ):
        """Update streaming configuration parameters."""
        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.background = background

        return self

    def update_running_status(self, running_status=False):
        """Update the running status of the stream."""
        self.running = running_status

    # def update_cam_fps(self, fps):
    #     self.original_fps = fps

    def list_available_devices(self):
        """List all available camera devices."""
        devices = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append({"id": i, "name": f"Camera {i}"})
                cap.release()

        if not devices:
            print("Warning: No camera devices found")

        return devices

    def stream_video(self):
        """Stream video from input source to virtual camera with segmentation."""
        self.running = True
        print(f"Retrieving  feed from source ({self.input_source}), FPS : {self.fps}, Blur Strength : {self.blur_strength}")

        cap = cv2.VideoCapture(int(self.input_source))
        frame_idx = 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            detected_fps = cap.get(cv2.CAP_PROP_FPS)

            if detected_fps > 1:
                self.original_fps = int(detected_fps)
                print(f"Detected camera FPS: {self.original_fps}")
            else:
                print(f"Invalid camera FPS detected: {detected_fps}, using default: {self.original_fps}")
        except Exception as e:
            print(f"Error detecting camera FPS: {e}. \nUsing default: {self.original_fps}")

        if self.fps:
            if self.fps > self.original_fps:
                print(f"Requested FPS ({self.fps}) is higher than camera FPS ({self.original_fps}). Limiting to camera FPS.")
                self.fps = self.original_fps

            frame_interval = max(1, int(self.original_fps / self.fps))
        else:
            self.fps = self.original_fps
            frame_interval = 1

        with pyvirtualcam.Camera(width=width, height=height, fps=self.fps) as cam:
            print(f"Virtual camera running using '{cam.backend}' backend at {width}x{height} {self.fps}fps")

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                result_frame = frame.copy()

                if frame_idx % frame_interval == 0:
                    results = self.model.predict(
                        source=frame,
                        save=False,
                        save_txt=False,
                        stream=True,
                        retina_masks=True,
                        verbose=False,
                        device=self.device
                    )
                    mask = self.generate_mask_from_result(results)

                    if mask is not None:
                        if self.background == "blur":
                            result_frame = self.apply_blur_with_mask(frame, mask, blur_strength=self.blur_strength)
                        elif self.background == "none":
                            result_frame = self.apply_black_background(frame, mask)
                        elif self.background == "default":
                            result_frame = self.apply_custom_background(frame, mask)

                frame_idx += 1

                cam.send(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()

        cap.release()


    if __name__ == "__main__":
        print(list_available_devices())
