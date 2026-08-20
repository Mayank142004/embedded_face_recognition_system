# Environment Reference

There is no `.env` file used in this project. Most values are hardcoded.

| Variable | Purpose | Used By | Required | Example |
| -------- | ------- | ------- | -------- | ------- |
| `TF_CPP_MIN_LOG_LEVEL` | Suppresses TensorFlow warnings to keep the console clean. | `training.py`, `facent_svm_rec_passing.py` | No | `'2'` |

*(If external APIs or databases are integrated in the future, variables like `<DB_CONNECTION_STRING>` or `<API_KEY>` should be documented here).*
