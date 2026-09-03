import logging
import time

import cv2 as cv
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from config import TFLITE_NUM_THREADS

logger = logging.getLogger(__name__)


class YOLOTFLite:
    def __init__(self, model_path, input_size=320, conf_thres=0.5, iou_thres=0.4,
                 num_threads=TFLITE_NUM_THREADS):
        # One thread fewer than the core count: capture, JPEG encode, MQTT and
        # networking all need CPU too, and saturating every core on a passively
        # cooled Pi 3 costs more in thermal throttling than it gains.
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self._in_index = self.input_details[0]["index"]
        self._out_index = self.output_details[0]["index"]

        # Layout is fixed at export time, so decide once instead of
        # re-inspecting input_details on every single inference.
        self._nchw = bool(self.input_details[0]["shape"][1] == 3)

        # Preallocated scratch buffers — the old code allocated a resized
        # frame, an RGB copy and a float32 copy on every call.
        s = self.input_size
        self._resized = np.empty((s, s, 3), dtype=np.uint8)
        self._rgb = np.empty((s, s, 3), dtype=np.uint8)
        self._inp = np.zeros((1, s, s, 3), dtype=np.float32)

        # Timing for the perf log (step 9 instrumentation).
        self.last_ms = 0.0

        logger.info(
            "YOLO TFLite loaded: %s | threads=%d | input=%s | output=%s",
            model_path.split("/")[-1], num_threads,
            list(self.input_details[0]["shape"]),
            list(self.output_details[0]["shape"]),
        )

    def predict(self, frame):
        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        # Simple resize (no letterbox to keep it fast)
        cv.resize(frame, (self.input_size, self.input_size),
                  dst=self._resized, interpolation=cv.INTER_LINEAR)
        cv.cvtColor(self._resized, cv.COLOR_BGR2RGB, dst=self._rgb)
        np.multiply(self._rgb, np.float32(1.0 / 255.0),
                    out=self._inp[0], casting="unsafe")

        img = np.transpose(self._inp, (0, 3, 1, 2)) if self._nchw else self._inp

        self.interpreter.set_tensor(self._in_index, img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self._out_index)

        # Output is (1, 5, 2100)
        predictions = np.squeeze(output).T  # (2100, 5)

        scores = predictions[:, 4]
        valid = predictions[scores > self.conf_thres]

        results = []
        if valid.shape[0]:
            # Vectorised box decode — outputs are normalised 0-1.
            # int() truncates toward zero and so does astype(int32), so this
            # matches the previous per-detection Python loop exactly.
            cx = valid[:, 0] * orig_w
            cy = valid[:, 1] * orig_h
            w = valid[:, 2] * orig_w
            h = valid[:, 3] * orig_h

            x = (cx - w / 2).astype(np.int32)
            y = (cy - h / 2).astype(np.int32)
            wi = w.astype(np.int32)
            hi = h.astype(np.int32)

            boxes = np.stack([x, y, wi, hi], axis=1).tolist()
            confidences = valid[:, 4].astype(np.float64).tolist()

            indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)
            if len(indices) > 0:
                for i in np.asarray(indices).flatten():
                    bx, by, bw, bh = boxes[i]
                    results.append([bx, by, bx + bw, by + bh, confidences[i]])

        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return results
