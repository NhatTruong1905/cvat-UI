# Model-assisted QC implementation status

This assessment is scoped to the custom `model_assisted_qc` app, its CVAT integration points,
and reusable CVAT quality-control infrastructure. Existing CVAT engine, dataset export, IAM,
job creation, and UI routing remain the foundation and are not reimplemented.

## Six-phase gap analysis

| Phase | Existing | Added in P0-1..P0-3 | Remaining |
|---|---|---|---|
| 1. Dataset and verified GT | Native GT job creation, random/manual sampling, condition JSON, technical GT validation | BDD100K loader and normalized records, explicit taxonomy mapping, deterministic condition/rare-class sampling, sequence-safe calibration/test split, copied synthetic corruption with audit manifest | Persist sampling manifests and explicit reviewer verification/version state through an API; import native BDD100K condition metadata automatically |
| 2. Inference and reliability | Ultralytics export/validation, mAP, precision, recall, confusion matrix, saved predictions | Model/config provenance and deterministic completed-run cache key; per-image prediction provenance | Include an annotation-content/version fingerprint; per-class metrics and explicit `UNMAPPED` counts; move long inference to RQ |
| 3. QC error detection | Threshold grid and condition summaries (this is P1 analysis, not the QC detector) | Pure box validation, IoU, prediction dedup, class-agnostic Hungarian matching, missing/wrong-class rules | Persist QC runs/alerts and connect prediction class IDs through the taxonomy mapper |
| 4. Filtering and ranking | Confidence threshold and basic prediction storage only | Dedup/minimum-confidence filtering, transparent risk breakdown, deterministic ordering, idempotent persisted queue and frame links | Per-class/condition calibration and explicit frame-level aggregation policy |
| 5. Human review | Links to GT frames and validation issue display | Guarded Pending/In-review/Confirmed/Rejected/Resolved workflow, append-only reviewer audit and minimal UI; no annotation mutation | Optional reviewer-authorized CVAT correction flow and conflict/retry handling |
| 6. Benchmark and reporting | Detection metrics and threshold tables | QC Precision/Recall, Precision@K, missing/wrong-class slices and random/unranked/risk-ranked baselines persisted from verified manifests | CSV/CLI export, reviewer-time/return-round KPI and confidence intervals |

## File-level change plan

- `phase1_dataset_ground_truth/bdd100k.py`: data loading, class mapping, split, sampling, synthetic benchmark generation.
- `phase2_yolo_evaluation/services.py`: model fingerprint, inference provenance, prediction metadata.
- `phase2_yolo_evaluation/views.py`: reuse completed runs with the same cache key.
- `models.py` and migration `0004`: persist provenance and cache key without replacing existing run storage.
- `phase3_condition_threshold/qc_detection.py`: pure matching and candidate detection logic.
- `phase4_risk_review/`: queue creation, filtering and explainable deterministic ranking.
- `phase5_human_review/`: review state machine, endpoint and immutable audit decisions.
- `phase6_benchmark/`: verified-manifest benchmark metrics and persisted reports.
- `tests/test_bdd100k_data.py`: P0-1 determinism, split isolation, mapping, immutable corruption tests.
- `tests/test_inference_provenance.py`: P0-2 cache identity tests.
- `tests/test_qc_detection.py`: P0-3 mandatory edge cases.

## Safety boundary

The P0 detector only returns candidate alert dictionaries. It does not call CVAT annotation write
APIs. A later review workflow must store the human decision separately and must not modify a CVAT
annotation unless a reviewer explicitly confirms the change.

## Validation commands

```powershell
docker exec cvat_server python manage.py test `
  cvat.apps.model_assisted_qc.tests.test_bdd100k_data `
  cvat.apps.model_assisted_qc.tests.test_inference_provenance `
  cvat.apps.model_assisted_qc.tests.test_qc_detection `
  cvat.apps.model_assisted_qc.tests.test_validate_ground_truth `
  cvat.apps.model_assisted_qc.tests.test_condition_threshold_analysis --keepdb

docker exec cvat_server python manage.py makemigrations --check --dry-run yolo_evaluation
```
