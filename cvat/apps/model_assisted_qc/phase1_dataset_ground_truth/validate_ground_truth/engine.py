from collections import Counter, defaultdict
from math import hypot


BDD100K_LABELS = {
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
}


def bbox_iou(first, second):
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def bbox_distance(first, second):
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    return hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def _issue(frame, severity, rule, message, shape_id=None):
    return {
        "frame": frame,
        "severity": severity,
        "rule": rule,
        "message": message,
        "shape_id": shape_id,
    }


def _rectangles(annotation_data, labels):
    rectangles = []
    for shape in annotation_data.get("shapes", []):
        if shape.get("outside") or shape.get("type") != "rectangle":
            continue
        rectangles.append((shape, labels.get(shape.get("label_id"), "unknown")))
    for track in annotation_data.get("tracks", []):
        label = labels.get(track.get("label_id"), "unknown")
        for shape in track.get("shapes", []):
            if shape.get("outside") or shape.get("type") != "rectangle":
                continue
            rectangles.append((shape, label))
    return rectangles


def validate_ground_truth(annotation_data, labels, image_sizes, sampled_frames):
    issues = []
    boxes_by_frame = defaultdict(list)
    sampled_frames = set(sampled_frames)

    for shape, label in _rectangles(annotation_data, labels):
        frame = int(shape["frame"])
        if frame not in sampled_frames:
            continue
        points = [float(value) for value in shape.get("points", [])]
        if len(points) != 4:
            issues.append(_issue(frame, "blocker", "invalid_geometry", "Rectangle must have four coordinates", shape.get("id")))
            continue
        x1, y1, x2, y2 = points
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        boxes_by_frame[frame].append((box, label, shape.get("id")))
        if label not in BDD100K_LABELS:
            issues.append(_issue(frame, "blocker", "invalid_label", f"Label '{label}' is not in the BDD100K taxonomy", shape.get("id")))
        image_size = image_sizes.get(frame)
        if image_size is None:
            issues.append(_issue(frame, "blocker", "missing_image", "Image dimensions are unavailable", shape.get("id")))
        elif box[0] < 0 or box[1] < 0 or box[2] > image_size[0] or box[3] > image_size[1]:
            issues.append(_issue(frame, "blocker", "out_of_bounds", f"Box exceeds image bounds {image_size[0]}x{image_size[1]}", shape.get("id")))
        if box[2] - box[0] < 15 or box[3] - box[1] < 15:
            issues.append(_issue(frame, "warning", "small_box", "Box width and height should be at least 15 px", shape.get("id")))

    for frame in sorted(sampled_frames):
        boxes = boxes_by_frame[frame]
        if not boxes:
            issues.append(_issue(frame, "warning", "empty_frame", "Frame has no Ground Truth boxes"))
        if len(boxes) > 120:
            issues.append(_issue(frame, "warning", "too_many_boxes", "Frame contains more than 120 boxes"))
        for index, (box, label, shape_id) in enumerate(boxes):
            for other_box, other_label, other_id in boxes[index + 1 :]:
                if label == other_label and bbox_iou(box, other_box) >= 0.95:
                    issues.append(_issue(frame, "warning", "duplicate_box", f"Likely duplicate '{label}' boxes ({shape_id}, {other_id})", shape_id))
            if label == "rider" and not any(
                other_label in {"bicycle", "motorcycle"} and bbox_distance(box, other_box) <= 60
                for other_box, other_label, _ in boxes
            ):
                issues.append(_issue(frame, "warning", "rider_association", "Rider has no bicycle or motorcycle within 60 px", shape_id))

    severity_counts = Counter(issue["severity"] for issue in issues)
    rule_counts = Counter(issue["rule"] for issue in issues)
    if severity_counts["blocker"]:
        validation_status = "blocked"
    elif severity_counts["warning"]:
        validation_status = "warning"
    else:
        validation_status = "passed"
    return {
        "status": validation_status,
        "summary": {
            "frames_checked": len(sampled_frames),
            "boxes_checked": sum(len(boxes) for boxes in boxes_by_frame.values()),
            "blockers": severity_counts["blocker"],
            "warnings": severity_counts["warning"],
            "rules": dict(rule_counts),
        },
        "issues": issues,
    }
