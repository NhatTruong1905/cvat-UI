# YOLO evaluation backend

This isolated Django app implements phases 1 and 2 of the evaluation workflow.

- `POST /api/yolo-evaluation/tasks/{id}/dataset` samples frames, creates the native CVAT Ground Truth job, and stores condition metadata.
- `POST /api/yolo-evaluation/tasks/{id}/runs` exports the task's Ground Truth job, runs
  Ultralytics validation/inference on those GT frames, and persists metrics and predictions.
- `GET /api/yolo-evaluation/tasks/{id}/dataset` returns dataset configuration and run history.
- `GET /api/yolo-evaluation/runs/{id}/predictions` returns saved per-image predictions.

Install the isolated dependency file with `pip install -r cvat/apps/yolo_evaluation/requirements.txt`.
The model path must be under `CVAT_YOLO_EVALUATION_ROOT` (defaults to
`/opt/cvat/evaluation-data`). The Ground Truth dataset is exported automatically by the backend;
clients do not provide a dataset path.
