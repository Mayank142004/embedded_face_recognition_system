# Performance Analysis

## The Frame Skipping Paradigm
Running YOLOv8n and FaceNet sequentially on a Raspberry Pi 3/4 CPU takes roughly 200-400ms per frame. Running this at 15 FPS results in a massive CPU backlog, thermal throttling, and undervoltage warnings.

To solve this, the pipeline utilizes **Frame Skipping**. The heavy ML pipeline executes exactly **once every 5 frames**. 
*   **Performance Gain:** Drops CPU usage by 80%.
*   **Visual Continuity:** For the 4 skipped frames, the previous bounding boxes are frozen on the screen. Because humans walking through a door move relatively slowly, a 300ms visual freeze on the bounding box is barely perceptible, yet saves massive compute power.

## Memory Optimizations
*   **TFLite FP16:** FaceNet was quantized to Float16 to halve its RAM footprint.
*   **Static Batch Size:** The `tflite_runtime` on 32-bit ARM OS has a known bug where dynamic batch dimensions (`None`) cause integer overflows when allocating memory for large CNNs. FaceNet was explicitly re-exported with a fixed `[1, 160, 160, 3]` input tensor.
*   **Zero-Copy Routing:** WebSockets on the server use `defaultdict(set)` to fan-out the exact same binary JPEG bytes to all UI clients without re-encoding.

