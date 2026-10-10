# BDD100K example — 50 images

This test fixture contains 50 BDD100K validation images and 1,050 bounding boxes in the
10-class BDD100K detection taxonomy. It is split into five environment groups with 10 images
each:

- `clear_day_city`
- `clear_night_city`
- `overcast_day_city`
- `rainy_day_city`
- `snowy_day_city`

## Recommended test for phases 1–3

Create one CVAT Task per directory under `condition_splits/`. For each Task:

1. Paste `labels.json` into the Task label editor in Raw mode.
2. From the single `images/` directory, upload the 10 image names listed in
   `<group>/files.txt`.
3. Open **Dataset & Evaluation** and create a random Ground Truth dataset with 10 frames.
4. Copy the JSON from `<group>/conditions.json` into Environment metadata.
5. Open the Ground Truth Job and upload `<group>/annotations_coco.zip` as `COCO 1.0`.
6. Save, run **Validate Ground Truth**, then run phases 2 and 3.

Using one Task per condition is important: phase 3 aggregates completed analyses by the
dataset-level environment metadata. After all five Tasks have an analysis, the condition table
compares clear, night, overcast, rainy, and snowy performance.

## Single mixed Task

For a quick pipeline test, upload `images/`, create a 50-frame Ground Truth Job, and upload
`annotations_coco.zip`. Use this metadata:

```json
{"weather":"mixed","timeofday":"mixed","scene":"city street"}
```

## Provenance

- Images: BDD100K validation split.
- Bounding boxes: BDD100K detection labels distributed by the Hirundo validation mirror.
- Conditions: parsed from the per-image descriptions in `Roger618/Bdd100k_desc` and recorded in
  `manifest.json`.
- Builder: `utils/dataset_manifest/build_bdd100k_example.py`.

BDD100K data is subject to the BDD100K license. Use this fixture for permitted educational,
research, and evaluation purposes and retain dataset attribution.
