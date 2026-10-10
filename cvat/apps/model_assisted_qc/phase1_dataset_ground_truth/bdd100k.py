import copy
import json
import random
from collections import defaultdict
from pathlib import Path


BDD100K_CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

BDD100K_TO_COCO = {
    "pedestrian": "person",
    "rider": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "stop sign",
}


def load_bdd100k_records(source):
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as stream:
            rows = json.load(stream)
    else:
        rows = source
    if not isinstance(rows, list):
        raise ValueError("BDD100K labels must be a JSON array")
    records = []
    for row in rows:
        image_id = row.get("name")
        if not image_id:
            raise ValueError("Every BDD100K record must have a name")
        attributes = row.get("attributes") or {}
        annotations = []
        for label in row.get("labels") or []:
            category = label.get("category")
            box = label.get("box2d")
            if category not in BDD100K_CLASSES or not box:
                continue
            xyxy = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                raise ValueError(f"Invalid box in {image_id}: {xyxy}")
            annotations.append(
                {
                    "annotation_id": str(label.get("id", f"{image_id}:{len(annotations)}")),
                    "class_name": category,
                    "bbox_xyxy": xyxy,
                    "source": "bdd100k",
                    "version": 1,
                }
            )
        records.append(
            {
                "image_id": image_id,
                "source_video_id": row.get("videoName") or image_id.split("-")[0],
                "frame_id": row.get("frameIndex"),
                "weather": attributes.get("weather", "unknown"),
                "timeofday": attributes.get("timeofday", "unknown"),
                "scene": attributes.get("scene", "unknown"),
                "annotations": annotations,
            }
        )
    return records


def map_bdd100k_class(class_name, mapping=None):
    mapping = mapping or BDD100K_TO_COCO
    return mapping.get(class_name, "UNMAPPED")


def split_by_sequence(records, calibration_ratio=0.5, seed=42):
    if not 0 < calibration_ratio < 1:
        raise ValueError("calibration_ratio must be between 0 and 1")
    groups = defaultdict(list)
    for record in records:
        groups[record["source_video_id"]].append(record)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    target = round(len(records) * calibration_ratio)
    calibration_groups = set()
    count = 0
    for group_id in group_ids:
        if count >= target and calibration_groups:
            break
        calibration_groups.add(group_id)
        count += len(groups[group_id])
    calibration = [record for record in records if record["source_video_id"] in calibration_groups]
    held_out = [record for record in records if record["source_video_id"] not in calibration_groups]
    return calibration, held_out


def stratified_sample(records, count, seed=42):
    if count < 1:
        raise ValueError("count must be positive")
    if count >= len(records):
        return sorted(records, key=lambda record: record["image_id"])
    strata = defaultdict(list)
    class_frequency = defaultdict(int)
    for record in records:
        for class_name in {item["class_name"] for item in record["annotations"]}:
            class_frequency[class_name] += 1
    for record in records:
        key = (record["weather"], record["timeofday"], record["scene"])
        strata[key].append(record)
    randomizer = random.Random(seed)
    for values in strata.values():
        values.sort(key=lambda record: record["image_id"])
        randomizer.shuffle(values)
        values.sort(
            key=lambda record: min(
                (class_frequency[item["class_name"]] for item in record["annotations"]),
                default=len(records) + 1,
            ),
            reverse=True,
        )
    selected = []
    keys = sorted(strata)
    while len(selected) < count:
        available = [key for key in keys if strata[key]]
        if not available:
            break
        for key in available:
            selected.append(strata[key].pop())
            if len(selected) == count:
                break
    return sorted(selected, key=lambda record: record["image_id"])


def corrupt_annotations(records, error_count, seed=42):
    corrupted = copy.deepcopy(records)
    candidates = [
        (record, annotation)
        for record in corrupted
        for annotation in record["annotations"]
    ]
    if error_count > len(candidates):
        raise ValueError("error_count exceeds the number of annotations")
    randomizer = random.Random(seed)
    selected = randomizer.sample(candidates, error_count)
    manifest = []
    for index, (record, annotation) in enumerate(selected, start=1):
        before = copy.deepcopy(annotation)
        if index % 2:
            record["annotations"].remove(annotation)
            error_type = "MISSING_OBJECT"
            after = None
        else:
            choices = [name for name in BDD100K_CLASSES if name != annotation["class_name"]]
            annotation["class_name"] = randomizer.choice(choices)
            error_type = "WRONG_CLASS"
            after = copy.deepcopy(annotation)
        manifest.append(
            {
                "error_id": f"synthetic-{seed}-{index}",
                "image_id": record["image_id"],
                "error_type": error_type,
                "annotation_before": before,
                "annotation_after": after,
                "seed": seed,
            }
        )
    return corrupted, manifest
