# Appendix B: API Reference

> [!NOTE]
> The Face Recognition System currently operates as a local monolithic script processing video files/streams. It does **not** expose HTTP/REST APIs.

If the system is later migrated to a microservices or web-based architecture, API endpoints (such as `POST /video/frame` or `GET /attendance/today`) would be documented here.

### System "Entry Points" (Local)

While there are no HTTP APIs, the system accepts the following local inputs:

| Method | Endpoint | Handler | Request (Input) | Response (Output) |
| ------ | -------- | ------- | --------------- | ----------------- |
| `EXEC` | `python main.py` | `main.py` | `test_datas/testing_video.mp4` | Writes CSV to `marked_attendance/` and video to `result_datas/` |
| `EXEC` | `python training.py` | `training.py` | Images in `facenet_files/dataset2/` | Writes `.pkl` SVM model to `facenet_models/` |
