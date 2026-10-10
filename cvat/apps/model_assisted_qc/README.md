# Model-assisted QC backend

This isolated Django app implements phases 1, 2, and 3 of the evaluation workflow.

Implementation coverage and the six-phase gap analysis are documented in
[`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).

- `POST /api/model-assisted-qc/tasks/{id}/dataset` samples frames, creates the native CVAT Ground Truth job, and stores condition metadata.
- `POST /api/model-assisted-qc/tasks/{id}/runs` exports the task's Ground Truth job, runs
  Ultralytics validation/inference on those GT frames, and persists metrics and predictions.
- `GET /api/model-assisted-qc/tasks/{id}/dataset` returns dataset configuration and run history.
- `GET /api/model-assisted-qc/runs/{id}/predictions` returns saved per-image predictions.
- `POST /api/model-assisted-qc/tasks/{id}/threshold-analyses` runs a bounded Confidence/IoU
  grid, attaches the dataset condition metadata, and selects the best threshold pair.
- `GET /api/model-assisted-qc/tasks/{id}/threshold-analyses` returns analysis history.
- `GET /api/model-assisted-qc/condition-summary` averages the best completed results by
  environment metadata across the current user's tasks.
- `POST /api/model-assisted-qc/tasks/{id}/alerts` creates or reuses a risk-ranked QC queue from a
  completed evaluation run; `GET` lists its alerts.
- `POST /api/model-assisted-qc/alerts/{id}/review` records `START_REVIEW`, `CONFIRMED`, `REJECTED`,
  `NEEDS_MORE_INFO`, or `RESOLVED` decisions in an append-only audit trail.
- `POST /api/model-assisted-qc/tasks/{id}/benchmarks` compares random, unranked, and risk-ranked
  review order against an explicit verified-error manifest.

The backend is organized by workflow phase:

- `phase1_dataset_ground_truth/`: sampling, Ground Truth job creation, and GT validation.
- `phase2_yolo_evaluation/`: Ultralytics validation, inference, metrics, and predictions.
- `phase3_condition_threshold/`: condition aggregation and Confidence/IoU experiments.

Database models, migrations, shared permissions, and URL routing remain at the app root because
all three phases use them.

The P0 QC detector is implemented as pure functions in
`phase3_condition_threshold/qc_detection.py`. It only produces review candidates and never writes
CVAT annotations. Prediction reuse is keyed by Ground Truth job, model SHA-256, confidence, and NMS
IoU; changing any of these inputs creates a different inference cache key.

Phases 4 through 6 are organized separately:

- `phase4_risk_review/`: deterministic filtering, explainable risk scoring, queue generation.
- `phase5_human_review/`: guarded review state transitions and audit records.
- `phase6_benchmark/`: QC Precision, QC Recall, Precision@K, type slices, and baseline comparison.

The benchmark manifest must contain independently verified errors. Its metrics are not model mAP,
and should not be reported as production QC performance when the input only contains synthetic
errors. No review endpoint writes, deletes, or patches CVAT annotations.

Each phase 3 request supports up to five
Confidence and five IoU values (25 combinations). Available optimization objectives are
`map50_95`, `map50`, `precision`, `recall`, and `f1`.

Install the isolated dependency file with `pip install -r cvat/apps/model_assisted_qc/requirements.txt`.
The model path must be under `CVAT_YOLO_EVALUATION_ROOT` (defaults to
`/opt/cvat/evaluation-data`). The Ground Truth dataset is exported automatically by the backend;
clients do not provide a dataset path.
