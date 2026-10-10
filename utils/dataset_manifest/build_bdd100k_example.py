import argparse
import csv
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


BDD100K_CATEGORIES = [
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
]

CONDITION_GROUPS = {
    "clear_day_city": ("clear", "daytime", "city street"),
    "clear_night_city": ("clear", "night", "city street"),
    "overcast_day_city": ("overcast", "daytime", "city street"),
    "rainy_day_city": ("rainy", "daytime", "city street"),
    "snowy_day_city": ("snowy", "daytime", "city street"),
}


def infer_conditions(description):
    text = description.lower()
    weather = next(
        (value for value in ("partly cloudy", "overcast", "rainy", "snowy", "foggy", "clear") if value in text),
        "undefined",
    )
    if "night" in text:
        timeofday = "night"
    elif "dawn" in text or "dusk" in text:
        timeofday = "dawn/dusk"
    elif "daytime" in text or "daylight" in text:
        timeofday = "daytime"
    else:
        timeofday = "undefined"
    scene = next(
        (value for value in ("parking lot", "gas stations", "city street", "residential", "highway", "tunnel") if value in text),
        "undefined",
    )
    return weather, timeofday, scene


def load_boxes(csv_path):
    boxes = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["class_name"] in BDD100K_CATEGORIES:
                boxes[Path(row["image_path"]).name].append(row)
    return boxes


def select_images(descriptions_path, boxes, images_per_group):
    selected = defaultdict(list)
    targets = {conditions: name for name, conditions in CONDITION_GROUPS.items()}
    with descriptions_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            filename = row["image_name"]
            group = targets.get(infer_conditions(row["description"]))
            if group and len(selected[group]) < images_per_group and boxes.get(filename):
                selected[group].append({
                    "filename": filename,
                    "description": row["description"],
                    "conditions": dict(zip(("weather", "timeofday", "scene"), CONDITION_GROUPS[group])),
                })
    missing = {group: images_per_group - len(images) for group, images in selected.items() if len(images) < images_per_group}
    if missing or len(selected) != len(CONDITION_GROUPS):
        raise RuntimeError(f"Not enough images for condition groups: {missing}")
    return selected


def coco_document(records, boxes):
    category_ids = {name: index + 1 for index, name in enumerate(BDD100K_CATEGORIES)}
    images = []
    annotations = []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        images.append({"id": image_id, "file_name": record["filename"], "width": 1280, "height": 720})
        for box in boxes[record["filename"]]:
            x_min, y_min = float(box["xmin"]), float(box["ymin"])
            width = float(box["xmax"]) - x_min
            height = float(box["ymax"]) - y_min
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[box["class_name"]],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [],
            })
            annotation_id += 1
    return {
        "info": {"description": "50-image BDD100K validation example for CVAT evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name, "supercategory": "object"}
            for name, category_id in category_ids.items()
        ],
    }


def write_annotation_zip(output_path, document):
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "annotations/instances_default.json")
    json_path.unlink()


def build(archive_path, labels_path, descriptions_path, output_dir, images_per_group):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    boxes = load_boxes(labels_path)
    groups = select_images(descriptions_path, boxes, images_per_group)
    all_records = [record for group in CONDITION_GROUPS for record in groups[group]]
    combined_images = output_dir / "images"
    combined_images.mkdir()
    with zipfile.ZipFile(archive_path) as source:
        for record in all_records:
            member = f"bdd100k/images/100k/val/{record['filename']}"
            with source.open(member) as input_stream, (combined_images / record["filename"]).open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
    write_annotation_zip(output_dir / "annotations_coco.zip", coco_document(all_records, boxes))
    split_root = output_dir / "condition_splits"
    for group, records in groups.items():
        group_dir = split_root / group
        group_dir.mkdir(parents=True)
        write_annotation_zip(group_dir / "annotations_coco.zip", coco_document(records, boxes))
        (group_dir / "conditions.json").write_text(
            json.dumps(records[0]["conditions"], indent=2), encoding="utf-8"
        )
        (group_dir / "files.txt").write_text(
            "\n".join(record["filename"] for record in records) + "\n", encoding="utf-8"
        )
    manifest = {
        "source": "BDD100K validation split",
        "image_count": len(all_records),
        "annotation_count": sum(len(boxes[record["filename"]]) for record in all_records),
        "groups": groups,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-per-group", type=int, default=10)
    arguments = parser.parse_args()
    build(arguments.archive, arguments.labels, arguments.descriptions, arguments.output, arguments.images_per_group)


if __name__ == "__main__":
    main()
