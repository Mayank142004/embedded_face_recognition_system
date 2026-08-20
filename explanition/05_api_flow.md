# 05. API Flow

> `UNKNOWN / NOT CONFIRMED FROM CODE`

The current Face Recognition System does not expose any web HTTP endpoints or REST APIs. It operates strictly as a local processing script (`main.py`) that reads a video file and writes a video file.

If a web API were implemented, it would likely follow this structure:

```text
HTTP Method: POST (Hypothetical)
Endpoint: /api/recognize
File: N/A
Function: N/A
Purpose: Process an uploaded image to recognize faces.

Request:
  Headers: Content-Type: multipart/form-data
  Body: image (file)

Response:
  Status Code: 200 OK
  Response Body:
  {
      "faces": [
          {
              "name": "Employee A",
              "confidence": 0.95
          }
      ]
  }
```

Since this does not exist in the codebase, the primary flow is detailed in [04_application_flow.md](04_application_flow.md) and [06_function_call_flow.md](06_function_call_flow.md).
