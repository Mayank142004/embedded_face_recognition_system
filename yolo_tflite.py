import cv2 as cv
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class YOLOTFLite:
    def __init__(self, model_path, input_size=320, conf_thres=0.5, iou_thres=0.4):
        # P2 FIX: Use all 4 CPU cores for ~3x speedup on Pi 3/4
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def predict(self, frame):
        orig_h, orig_w = frame.shape[:2]
        
        # Simple resize (no letterbox to keep it fast)
        img = cv.resize(frame, (self.input_size, self.input_size))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # (1, 320, 320, 3)
        
        # YOLOv8 export to tflite usually expects NCHW (1, 3, 320, 320) or NHWC
        # Let's check model input shape
        input_shape = self.input_details[0]['shape']
        if input_shape[1] == 3: # NCHW
            img = np.transpose(img, (0, 3, 1, 2))
            
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Output is (1, 5, 2100)
        predictions = np.squeeze(output).T # (2100, 5)
        
        scores = predictions[:, 4]
        mask = scores > self.conf_thres
        valid_preds = predictions[mask]
        
        boxes = []
        confidences = []
        
        for pred in valid_preds:
            cx, cy, w, h = pred[:4]
            # Scale back to original image (outputs are normalized 0-1)
            cx = cx * orig_w
            cy = cy * orig_h
            w = w * orig_w
            h = h * orig_h
            
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(pred[4]))
            
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append([x, y, x + w, y + h, confidences[i]])
                
        return results

