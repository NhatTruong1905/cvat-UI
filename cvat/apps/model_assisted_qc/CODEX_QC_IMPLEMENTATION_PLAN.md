# CODEX IMPLEMENTATION PLAN — Model-assisted Annotation QC (N2-04D)

> Ngôn ngữ: tiếng Việt. Tài liệu giao việc cho coding agent (Codex).  
> Trọng tâm: **phát hiện annotation thiếu vật thể (missing object) và sai class (wrong class)** trên BDD100K, sử dụng YOLO pretrained để hỗ trợ reviewer trong CVAT.  
> Không coi prediction của model là Ground Truth và **không tự sửa nhãn** khi chưa có xác nhận của con người.

## 0. Hướng dẫn Codex trước khi sửa dự án

1. Đọc README, cấu trúc repository, dependency, test, cấu hình CVAT, API và pipeline đang tồn tại. **Không giả định dự án trống.**
2. Lập bảng `existing -> missing -> change` cho từng giai đoạn bên dưới; tận dụng code sẵn có, tránh viết lại không cần thiết.
3. Triển khai theo thứ tự P0 rồi P1; thay đổi nhỏ, chạy test sau mỗi bước; giữ tương thích API hiện có nếu có thể.
4. Không tải BDD100K hoặc model lớn nếu chưa được cấu hình/cấp quyền; cho phép cấu hình đường dẫn local và model weights qua environment/config.
5. Không ghi đè nhãn gốc CVAT. Mọi cảnh báo và kết quả review phải truy vết được theo dataset, task, frame, annotation/prediction, model version, config version.
6. Nếu thiếu thông tin về API CVAT hoặc schema thực tế, kiểm tra phiên bản đang sử dụng và nêu rõ giả định; không tự tạo endpoint không có thật.
7. Trước khi code, tạo kế hoạch triển khai ngắn gọn, liệt kê file cần sửa. Sau khi code, cung cấp lệnh chạy, test, hạn chế và kết quả thực tế; không tuyên bố KPI khi chưa đo.

## 1. Mục tiêu, phạm vi và KPI

**Yêu cầu đề tài:** so sánh annotation do người gán với prediction của model pretrained; nếu model thấy vật thể chưa được gán hoặc khác class thì đưa frame nghi vấn lên đầu danh sách review. Dataset: BDD100K.

**In scope (MVP/P0):**
- BDD100K detection subset; import/đọc annotation CVAT; chuẩn hóa box và class.
- Chạy YOLO pretrained, lưu prediction, confidence và provenance.
- Ghép box theo IoU, **class-agnostic**, một-một; phát hiện `POSSIBLE_MISSING_OBJECT`, `POSSIBLE_WRONG_CLASS`.
- Lọc trùng, xếp hạng cảnh báo, danh sách review; reviewer xác nhận/bác bỏ/sửa; log đầy đủ.
- Benchmark QC trên tập có lỗi đã xác minh, tách khỏi tập hiệu chỉnh.

**Out of scope MVP:** tự động chỉnh annotation production, tự động fine-tune YOLO, VLM/Qwen3-VL, pipeline active learning hoàn chỉnh, giao diện CVAT tùy biến sâu nếu API chưa hỗ trợ.

**KPI chính:**
- `Missing Recall = confirmed missing errors detected / all verified missing errors`.
- `Wrong-class Recall = confirmed wrong-class errors detected / all verified wrong-class errors`.
- `QC Precision = true QC alerts / all reviewed QC alerts` (ghi rõ mẫu số; nếu chưa review hết thì không gọi là precision toàn bộ).
- `Precision@K = confirmed true alerts in top K / K` (đo trên top K đã review đầy đủ).
- `Errors found per reviewer-hour`, `review time per frame`.
- `Return rounds`: số vòng labeler sửa và QC kiểm lại trong workflow thật hoặc mô phỏng có ghi rõ; không suy ra từ ảnh BDD100K tĩnh.

## 2. Kiến trúc và dữ liệu

```text
BDD100K / CVAT annotations ──> Normalization & class mapping ──┐
                                                               ├─> IoU matrix
YOLO pretrained ──> inference + NMS + confidence filtering ───┘       │
                                                                       v
                                             Class-agnostic one-to-one matching
                                                                       │
                                      ┌────────────────────────────────┴───────────┐
                                      v                                            v
                             Missing candidates                           Wrong-class candidates
                                      └────────────────┬───────────────────────────┘
                                                       v
                                            Dedup + risk ranking
                                                       v
                                           Review queue / CVAT link
                                                       v
                                        Human decision + audit log
                                                       v
                                      QC benchmark + feedback report
```

**Dữ liệu đề xuất (điều chỉnh theo repository):**
- `ImageRecord`: `image_id`, `dataset_split`, `source_video_id` (nếu có), `frame_id`, `weather`, `timeofday`, `scene`.
- `Annotation`: `annotation_id`, `image_id`, `bbox_xyxy`, `class_id`, `source`, `version`.
- `Prediction`: `prediction_id`, `image_id`, `bbox_xyxy`, `class_id`, `confidence`, `model_id`, `inference_config_id`.
- `QCAlert`: `alert_id`, `image_id`, `type`, `prediction_id`, `annotation_id|null`, `iou|null`, `risk_score`, `status`, `reason_codes`, `created_at`, `run_id`.
- `ReviewDecision`: `alert_id`, `reviewer_id`, `decision` (`CONFIRMED`, `REJECTED`, `NEEDS_MORE_INFO`), `corrected_annotation_id|null`, `notes`, `timestamp`.

## 3. Giai đoạn 1 — Dataset Sampling & Verified Ground Truth (P0)

**Hiện trạng ý tưởng:** chọn ngẫu nhiên ảnh rồi nhờ chuyên gia labeling.

**Cải tiến cần làm:**
1. Chọn đúng tập BDD100K có nhãn bounding box; chuẩn hóa tọa độ, class, image IDs.
2. Kết hợp random sampling và sampling có phân tầng theo `weather`, `timeofday`, `scene`, class hiếm. Một ảnh có thể mang nhiều điều kiện.
3. Phân biệt **annotation cần QC** (có thể sai) với **verified GT** (đã được reviewer độc lập kiểm chứng).
4. Tách `calibration` và `held-out QC test` theo video/sequence khi có thể, tránh frame gần nhau nằm ở hai tập.
5. Tạo benchmark lỗi bằng cách **sao chép GT** rồi xóa box/đổi class có kiểm soát; lưu manifest lỗi. Giữ thêm tập lỗi tự nhiên để đánh giá tính thực tế.

**Deliverables:** loader BDD100K, class taxonomy mapping, sampling manifest, GT versioning, synthetic corruption generator (seed cố định).

**Acceptance tests:** cùng seed sinh cùng manifest; không overlap split theo group; không chỉnh GT gốc; lỗi tạo ra có `error_id`, loại lỗi, annotation trước/sau.

## 4. Giai đoạn 2 — YOLO Reliability Evaluation (P1 hỗ trợ P0)

**Hiện trạng ý tưởng:** tính mAP và confidence trên GT.

**Cải tiến cần làm:**
1. Chạy inference một lần, cache prediction có `model_version`, input resolution, NMS IoU, confidence threshold.
2. Mapping taxonomy rõ ràng (ví dụ BDD100K `motor` và COCO `motorcycle`; `bike` và `bicycle`), class không tương thích phải đánh dấu `UNMAPPED`, không gắn cờ wrong class tùy tiện.
3. Đo `mAP@0.5`, `mAP@0.5:0.95`, precision/recall theo class và điều kiện; lưu confusion matrix.
4. Phân tích chất lượng confidence trên tập calibration; không coi confidence 0.9 là xác suất đúng 90% nếu chưa calibration.
5. Phân biệt **IoU để đánh giá detection**, **IoU để matching QC**, **NMS IoU**: ba tham số khác mục đích.

**Deliverables:** inference adapter, prediction cache, evaluation report, condition slices.

**Acceptance tests:** class mapping nhất quán; chạy lại cùng config cho output tái lập trong giới hạn backend; test ảnh không có detection.

## 5. Giai đoạn 3 — QC Error Detection Engine (P0, phần lõi)

**Thuật toán khuyến nghị:** class-agnostic IoU + Hungarian one-to-one assignment + rule-based classification. So sánh với greedy IoU làm baseline.

**Quy trình:**
1. Đọc annotation và prediction trên cùng ảnh, cùng hệ tọa độ; validate box (`x2>x1`, `y2>y1`).
2. Chuẩn hóa class, lọc các class ngoài scope; áp dụng NMS/dedup trước matching.
3. Tính ma trận IoU `P x A`.
4. Ghép một-một **không ràng buộc class**, chỉ cho phép cặp IoU >= `qc_match_iou`. Với Hungarian, dùng dummy unmatched nodes hoặc objective ưu tiên số cặp hợp lệ trước rồi tối ưu tổng IoU; **không ép ghép các cặp dưới ngưỡng**.
5. Cặp hợp lệ, cùng class: không cảnh báo. Cặp hợp lệ, khác class: `POSSIBLE_WRONG_CLASS` (chỉ khi class mapping hợp lệ).
6. Prediction không ghép: `POSSIBLE_MISSING_OBJECT` nếu qua bộ lọc tin cậy. Annotation không ghép **không tự động là lỗi nhãn**; có thể YOLO bỏ sót.
7. Xử lý đặc biệt crowded scenes, overlapping objects, near-duplicate predictions, partial occlusion; thêm reason code `LOW_IOU`, `CLASS_MISMATCH`, `UNMATCHED_PREDICTION`, `UNMAPPED_CLASS`.

**Pseudocode:**
```python
preds = preprocess_predictions(raw_predictions, conf_config, nms_config)
anns = normalize_annotations(cvat_annotations, class_mapping)
valid_pairs = iou_matrix(preds, anns) >= qc_match_iou
matches = hungarian_with_unmatched(preds, anns, valid_pairs, iou_matrix)
for pred, ann in matches:
    if pred.class_id != ann.class_id:
        emit('POSSIBLE_WRONG_CLASS', pred, ann)
for pred in unmatched_predictions(matches, preds):
    if passes_qc_filter(pred):
        emit('POSSIBLE_MISSING_OBJECT', pred, None)
```

**Deliverables:** pure functions `box_iou`, `match_boxes`, `detect_qc_errors`, structured alerts; unit tests và benchmark greedy vs Hungarian.

**Acceptance tests (bắt buộc):**
- 1 pred/1 annotation cùng class IoU cao -> không cảnh báo.
- 1 pred/1 annotation khác class IoU cao -> wrong class, **không đồng thời missing**.
- 1 pred không annotation -> missing candidate.
- 1 annotation không pred -> không tự kết luận annotation sai.
- 2 pred trùng một annotation -> không tạo cảnh báo missing giả do duplicate (sau NMS/dedup).
- Box không giao nhau -> không ghép; class khác nhau nhưng IoU thấp -> không kết luận wrong class.
- Ảnh không có annotation, không có prediction, class unmapped, invalid boxes, crowded frames.

## 6. Giai đoạn 4 — Alert Filtering, Risk Scoring & Prioritization (P0 đơn giản, P1 nâng cao)

**Phân biệt:** `filtering` loại cảnh báo nhiễu rõ ràng; `ranking` đưa **cảnh báo có khả năng là lỗi annotation thật** lên trước, không phải cảnh báo sai do model.

**MVP P0:**
- Rule-based filter: class hợp lệ, confidence tối thiểu theo class (mặc định cấu hình), dedup, vùng/box quá nhỏ nếu đã kiểm chứng, suppression logic cho box trùng.
- Risk score đơn giản, minh bạch, có `reason_codes` và breakdown; score không được gọi là xác suất lỗi khi chưa hiệu chuẩn.
- Sắp xếp giảm dần theo risk; gom cảnh báo theo frame (ví dụ `frame_priority = max(alert_risk)`; cấu hình được); cho reviewer thấy loại lỗi và box so sánh.

**P1:**
- Hiệu chỉnh ngưỡng riêng theo class và điều kiện môi trường trên calibration set.
- Dùng reviewer-confirmed labels để đánh giá và hiệu chỉnh ranking; tối ưu `Precision@20/50`, `Recall@review_budget`, không tối ưu riêng mAP.
- Không lọc bỏ toàn bộ cảnh báo low-confidence nếu điều đó làm QC Recall giảm quá mạnh; báo cáo trade-off.

**Deliverables:** configurable ranking/filter module, deterministic sorting, score explanation, threshold experiment script.

**Acceptance tests:** cảnh báo hợp lệ xếp theo score; cùng input/config cho cùng ranking; đo top-K trên tập test độc lập; không sử dụng ground truth test để chỉnh threshold.

## 7. Giai đoạn 5 — Human Review & CVAT Integration (P0)

**Cải tiến cần làm:**
1. Tạo queue theo frame/alert với trạng thái `PENDING`, `IN_REVIEW`, `CONFIRMED`, `REJECTED`, `RESOLVED` (tách review decision và fix status nếu cần).
2. Hiển thị box annotation và prediction, class, confidence, IoU, reason codes, risk score; link trực tiếp đến CVAT task/frame nếu phiên bản CVAT hỗ trợ.
3. Reviewer xác nhận/bác bỏ; nếu sửa nhãn, tạo bản cập nhật qua luồng được hỗ trợ, **không sửa tự động từ prediction**.
4. Lưu audit trail, người thao tác, timestamp, version; chống trùng cảnh báo qua `run_id`/stable alert key; hỗ trợ rerun idempotent.
5. Nếu API CVAT không hỗ trợ UI review queue theo cách dự kiến, triển khai dashboard ngoài (Streamlit/FastAPI) liên kết đến CVAT thay vì giả định API có sẵn.

**Deliverables:** CVAT adapter, queue storage, review endpoints/UI tối thiểu, audit log.

**Acceptance tests:** tạo/review/reject/resolve alert; không mất lịch sử; rerun không nhân đôi; không ghi đè annotation gốc; lỗi kết nối CVAT có retry và thông báo.

## 8. Giai đoạn 6 — QC Benchmark & Reporting (P0)

**Thiết kế thực nghiệm:**
- Baseline A: review ngẫu nhiên hoặc theo thứ tự frame.
- Baseline B: cảnh báo YOLO không ranking.
- Proposed: cảnh báo + risk ranking.
- Cùng ngân sách `K` frame hoặc thời gian, cùng điều kiện dữ liệu; tránh leakage và reviewer bias.
- Báo cáo riêng missing/wrong-class, class và condition slices; kèm số mẫu, khoảng tin cậy nếu đủ dữ liệu.
- Đo số vòng bị trả lại bằng workflow label -> QC -> sửa -> QC lại; không tự suy ra từ BDD100K.

**Deliverables:** benchmark CLI, metrics JSON/CSV, dashboard/report, reproducible experiment config.

**Acceptance tests:** kiểm tra thủ công TP/FP/FN trên fixture nhỏ; tránh chia cho 0; metric khớp manifest lỗi; top-K không vượt số item; báo cáo nêu rõ lỗi synthetic và lỗi tự nhiên.

## 9. Phân kỳ triển khai

| Ưu tiên | Hạng mục | Điều kiện hoàn thành |
|---|---|---|
| P0-1 | BDD100K loader, class mapping, GT/split | Test dữ liệu và manifest pass |
| P0-2 | Inference adapter + cache | Prediction có provenance |
| P0-3 | IoU/Hungarian + QC rules | Unit tests các edge case pass |
| P0-4 | Filter/rank đơn giản + queue | Reviewer thấy top alerts và lý do |
| P0-5 | Review workflow + audit | Confirm/reject/resolve hoạt động |
| P0-6 | QC benchmark | Có QC Recall, Precision@K, baseline |
| P1-1 | Phân tích theo condition | Có bảng theo ngày/đêm/mưa/class |
| P1-2 | Threshold theo class/condition | Có kết quả so sánh trên held-out test |
| P1-3 | Dashboard và return-round workflow | Có đo thời gian/vòng sửa thực tế |
| P2 | VLM/active learning/fine-tune | Chỉ làm khi P0/P1 đã được chứng minh |

## 10. Gợi ý tổ chức code (chỉ tạo nếu phù hợp repository)

```text
src/qc/
  data/              # BDD100K/CVAT loader, split, mapping
  inference/         # YOLO adapter, cache
  matching/          # IoU, Hungarian, greedy baseline
  detection/         # missing/wrong-class rules
  ranking/           # filtering, scoring, top-K
  review/            # queue, CVAT adapter, audit
  evaluation/        # QC metrics, baselines, condition slices
configs/
  qc.yaml
scripts/
  run_qc.py
  benchmark_qc.py
tests/
  test_matching.py
  test_error_detection.py
  test_ranking.py
  test_qc_metrics.py
```

**Config mẫu (tham số chỉ là giá trị khởi tạo để thử nghiệm, không phải ngưỡng tối ưu):**
```yaml
model:
  weights: ${YOLO_WEIGHTS}
  inference_conf: 0.20
  nms_iou: 0.70
qc:
  match_iou: 0.50
  min_confidence: 0.50
  class_agnostic_matching: true
  auto_fix_annotations: false
ranking:
  strategy: rule_based
  frame_aggregation: max
review:
  backend: cvat_or_external_queue
  require_human_confirmation: true
experiment:
  seed: 42
  top_k: [20, 50]
```

## 11. Definition of Done — điều kiện nghiệm thu toàn dự án

- [ ] Chạy được end-to-end trên subset BDD100K với annotation CVAT hoặc file tương đương.
- [ ] Phát hiện và phân biệt đúng **candidate missing** và **candidate wrong class** theo fixture đã biết.
- [ ] Không đánh đồng prediction với Ground Truth; không tự sửa nhãn.
- [ ] Class mapping và IoU matching xử lý được mismatch taxonomy, trùng box, no-match.
- [ ] Có review queue sắp xếp theo risk, có trạng thái và audit log.
- [ ] Có benchmark so sánh baseline vs proposed trên held-out test, công bố đầy đủ QC Precision, Missing Recall, Wrong-class Recall, Precision@K.
- [ ] Có README hướng dẫn setup, config, chạy inference, QC, review, benchmark, test.
- [ ] Có test tự động cho logic lõi; lệnh chạy và kết quả thực tế được Codex báo cáo.

## 12. Yêu cầu Codex bắt đầu thực hiện

**Bước đầu tiên:** đọc repository và trả về (1) kiến trúc hiện có, (2) bảng gap so với tài liệu này, (3) thứ tự thay đổi theo P0, (4) danh sách file sẽ chỉnh. Sau đó thực hiện P0-1 đến P0-3 trước, chạy test và báo cáo. Không làm tất cả giai đoạn trong một commit lớn; mỗi bước cần có test và hướng dẫn chạy. Nếu gặp phụ thuộc bên ngoài chưa cấu hình (BDD100K, CVAT, weights), cung cấp mock/fixture cho unit test và ghi rõ integration test nào chưa chạy được.
