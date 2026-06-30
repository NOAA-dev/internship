# #!/usr/bin/env python3
# """
# inference.py — OAK-D YOLOv8 Traffic Sign Detector (ROS2 Node)

# Publishes per-frame results on three topics:

#   ~/scores  (std_msgs/Float32MultiArray)
#       Length = NUM_CLASSES.
#       scores[i] = max confidence seen for class i across all NMS detections.
#       0.0 if class i was not detected this frame.

#   ~/mask    (std_msgs/Int8MultiArray)
#       Length = NUM_CLASSES.
#       mask[i] = 1 if scores[i] > mask_threshold, else 0.

#   ~/labels  (std_msgs/String)
#       JSON-encoded list of length NUM_CLASSES.
#       labels[i] = class label string if mask[i] == 1, else "".

# ROS Parameters
# --------------
#   mask_threshold  (float, default 0.7)  — threshold for the binary mask
# """

# import cv2
# import json
# import numpy as np
# import depthai as dai
# import torchvision
# import torch
# import time

# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray, Int8MultiArray, String


# BLOB_PATH = "/home/car-02/Music/Ams_stack/bno_test_pkg/bno_test_pkg/result/best_openvino_2022.1_6shave.blob"
# JSON_PATH = "/home/car-02/Music/Ams_stack/bno_test_pkg/bno_test_pkg/result/best.json"



# class InferenceNode(Node):

#     def __init__(self):
#         super().__init__('inference_node')

#         # ROS parameter: binary mask threshold
#         self.declare_parameter('mask_threshold', 0.7)
#         self.mask_threshold = (
#             self.get_parameter('mask_threshold').get_parameter_value().double_value
#         )

#         # Publishers
#         self.pub_scores = self.create_publisher(Float32MultiArray, '~/scores', 10)
#         self.pub_mask   = self.create_publisher(Int8MultiArray,    '~/mask',   10)
#         self.pub_labels = self.create_publisher(String,            '~/labels', 10)

#         self.get_logger().info(
#             f'InferenceNode started — mask_threshold={self.mask_threshold}'
#         )

#     def publish_results(self, score_vec, mask_vec, active_labels):
#         """Publish the three detection topics."""
#         scores_msg      = Float32MultiArray()
#         scores_msg.data = score_vec.tolist()
#         self.pub_scores.publish(scores_msg)

#         mask_msg      = Int8MultiArray()
#         mask_msg.data = mask_vec.tolist()
#         self.pub_mask.publish(mask_msg)

#         labels_msg      = String()
#         labels_msg.data = json.dumps(active_labels)
#         self.pub_labels.publish(labels_msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = InferenceNode()

#     # Load Config
#     with open(JSON_PATH, 'r') as f:
#         config = json.load(f)

#     metadata = config.get("NN_specific_metadata", {})
#     NUM_CLASSES = metadata.get("classes", 26)
#     LABELS = config.get("mappings", {}).get("labels", [f"class_{i}" for i in range(NUM_CLASSES)])
#     CONF_THRESHOLD = metadata.get("confidence_threshold", 0.5)
#     IOU_THRESHOLD = metadata.get("iou_threshold", 0.5)
#     INPUT_SIZE = int(config.get("nn_config", {}).get("input_size", "416x416").split("x")[0])

#     node.get_logger().info(f"Loaded Config: {NUM_CLASSES} classes, {INPUT_SIZE}x{INPUT_SIZE} input")

#     # Build Pipeline using DepthAI v3 Context Manager
#     with dai.Pipeline() as pipeline:

#         cam = pipeline.create(dai.node.ColorCamera)
#         cam.setPreviewSize(INPUT_SIZE, INPUT_SIZE)
#         cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#         cam.setInterleaved(False)
#         cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#         cam.setFps(30)

#         nn = pipeline.create(dai.node.NeuralNetwork)
#         nn.setBlobPath(BLOB_PATH)
#         nn.input.setBlocking(False)
#         nn.input.setMaxSize(1)

#         cam.preview.link(nn.input)

#         q_rgb = nn.passthrough.createOutputQueue(maxSize=1, blocking=False)
#         q_nn = nn.out.createOutputQueue(maxSize=1, blocking=False)

#         node.get_logger().info("Starting pipeline...")
#         pipeline.start()
#         node.get_logger().info("Connected to OAK-D!")

#         while pipeline.isRunning() and rclpy.ok():

#             in_nn = q_nn.get()
#             in_rgb = q_rgb.tryGet()

#             if in_rgb is not None:
#                 frame = in_rgb.getCvFrame()
#             else:
#                 continue

#             H, W = INPUT_SIZE, INPUT_SIZE
#             layer_names = in_nn.getAllLayerNames()
#             raw_outputs = []

#             for name in layer_names:
#                 for method_name in ['getLayerFp16', 'getLayer', 'getTensor']:
#                     if hasattr(in_nn, method_name):
#                         try:
#                             t = np.array(getattr(in_nn, method_name)(name))
#                             if t.size > 0:
#                                 raw_outputs.append(t)
#                                 break
#                         except Exception:
#                             pass

#             if not raw_outputs:
#                 continue

#             raw_outputs.sort(key=lambda x: len(x), reverse=True)

#             z = []
#             shapes = [(52, 52), (26, 26), (13, 13)]
#             strides = [8.0, 16.0, 32.0]
#             for i in range(min(3, len(raw_outputs))):
#                 ny, nx = shapes[i]
#                 channels = raw_outputs[i].size // (ny * nx)
#                 x = raw_outputs[i].reshape(channels, ny, nx).transpose(1, 2, 0)

#                 # Definitively identified as YOLOv8 [l, t, r, b] format!
#                 l = x[..., 0]
#                 t = x[..., 1]
#                 r = x[..., 2]
#                 b = x[..., 3]

#                 yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
#                 grid_x = xv.astype(np.float32)
#                 grid_y = yv.astype(np.float32)

#                 stride = strides[i]

#                 # Decode to absolute pixel coordinates
#                 x1 = (grid_x + 0.5 - l) * stride
#                 y1 = (grid_y + 0.5 - t) * stride
#                 x2 = (grid_x + 0.5 + r) * stride
#                 y2 = (grid_y + 0.5 + b) * stride

#                 # Convert to [cx, cy, w, h] to match downstream logic
#                 cx = (x1 + x2) / 2
#                 cy = (y1 + y2) / 2
#                 w = x2 - x1
#                 h = y2 - y1

#                 # Apply sigmoid ONLY if they are raw logits
#                 probs = x[..., 4:]
#                 if np.min(probs) < 0.0 or np.max(probs) > 1.0:
#                     probs = 1 / (1 + np.exp(-probs))

#                 decoded = np.concatenate([
#                     cx[..., None], cy[..., None], w[..., None], h[..., None],
#                     probs
#                 ], axis=-1)

#                 z.append(decoded.reshape(-1, channels))

#             if not z:
#                 continue

#             res = np.concatenate(z, axis=0)
#             boxes = res[:, :4]

#             if res.shape[1] == 31:
#                 obj_conf = res[:, 4:5]
#                 cls_probs = res[:, 5:]
#                 scores = obj_conf * cls_probs
#             else:
#                 scores = res[:, 4:]

#             max_scores = np.max(scores, axis=1)
#             class_ids = np.argmax(scores, axis=1)

#             mask = max_scores > CONF_THRESHOLD
#             boxes = boxes[mask]
#             class_ids = class_ids[mask]
#             max_scores = max_scores[mask]

#             detections = []
#             for i in range(len(boxes)):
#                 x1 = int(boxes[i, 0] - boxes[i, 2] / 2)
#                 y1 = int(boxes[i, 1] - boxes[i, 3] / 2)
#                 x2 = int(boxes[i, 0] + boxes[i, 2] / 2)
#                 y2 = int(boxes[i, 1] + boxes[i, 3] / 2)

#                 x1 = max(0, min(x1, W))
#                 y1 = max(0, min(y1, H))
#                 x2 = max(0, min(x2, W))
#                 y2 = max(0, min(y2, H))

#                 detections.append({
#                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                     "conf": float(max_scores[i]),
#                     "label": int(class_ids[i])
#                 })

#             # NMS Filter
#             if len(detections) > 1:
#                 nms_boxes = torch.tensor([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=torch.float32)
#                 nms_scores = torch.tensor([d["conf"] for d in detections], dtype=torch.float32)
#                 keep_idx = torchvision.ops.nms(nms_boxes, nms_scores, IOU_THRESHOLD)
#                 detections = [detections[i] for i in keep_idx.numpy()]

#             # ---------------------------------------------------------------- #
#             # Build per-class confidence score vector
#             # score_vec[i] = max confidence for class i across all NMS detections
#             # ---------------------------------------------------------------- #
#             score_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
#             for d in detections:
#                 cls = d["label"]
#                 if d["conf"] > score_vec[cls]:
#                     score_vec[cls] = d["conf"]

#             # Binary mask: 1 if score > mask_threshold, else 0
#             mask_vec = (score_vec > node.mask_threshold).astype(np.int8)

#             # Masked label list: label name where mask=1, "" elsewhere
#             active_labels = [
#                 LABELS[i] if mask_vec[i] else ""
#                 for i in range(NUM_CLASSES)
#             ]

#             # Publish topics
#             node.publish_results(score_vec, mask_vec, active_labels)

#             # Visualization
#             for d in detections:
#                 label_text = f"{LABELS[d['label']]} {d['conf']:.2f}"
#                 cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 255, 0), 2)
#                 cv2.putText(frame, label_text, (d["x1"], d["y1"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#             cv2.imshow("OAK-D YOLOv8 Inference", frame)

#             if cv2.waitKey(1) == ord('q'):
#                 break

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()#!/usr/bin/env python3
"""
inference.py — OAK-D YOLOv8 Traffic Sign Detector (ROS2 Node)

Publishes per-frame results on three topics:

  ~/scores  (std_msgs/Float32MultiArray)
      Length = NUM_CLASSES.
      scores[i] = max confidence seen for class i across all NMS detections.
      0.0 if class i was not detected this frame.

  ~/mask    (std_msgs/Int8MultiArray)
      Length = NUM_CLASSES.
      mask[i] = 1 if scores[i] > mask_threshold, else 0.

  ~/labels  (std_msgs/String)
      JSON-encoded list of length NUM_CLASSES.
      labels[i] = class label string if mask[i] == 1, else "".

ROS Parameters
--------------
  mask_threshold  (float, default 0.7)  — threshold for the binary mask
"""

import cv2
import json
import numpy as np
import depthai as dai
import torchvision
import torch
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int8MultiArray, String



BLOB_PATH = "/home/car-02/Music/Ams_stack/bno_test_pkg/bno_test_pkg/result/best_openvino_2022.1_6shave.blob"
JSON_PATH = "/home/car-02/Music/Ams_stack/bno_test_pkg/bno_test_pkg/result/best.json"


class InferenceNode(Node):

    def __init__(self):
        super().__init__('inference_node')

        # ROS parameter: binary mask threshold
        self.declare_parameter('mask_threshold', 0.7)
        self.mask_threshold = (
            self.get_parameter('mask_threshold').get_parameter_value().double_value
        )

        # Publishers
        self.pub_scores = self.create_publisher(Float32MultiArray, '~/scores', 10)
        self.pub_mask   = self.create_publisher(Int8MultiArray,    '~/mask',   10)
        self.pub_labels = self.create_publisher(String,            '~/labels', 10)

        self.get_logger().info(
            f'InferenceNode started — mask_threshold={self.mask_threshold}'
        )

    def publish_results(self, score_vec, mask_vec, active_labels):
        """Publish the three detection topics."""
        scores_msg      = Float32MultiArray()
        scores_msg.data = score_vec.tolist()
        self.pub_scores.publish(scores_msg)

        mask_msg      = Int8MultiArray()
        mask_msg.data = mask_vec.tolist()
        self.pub_mask.publish(mask_msg)

        labels_msg      = String()
        labels_msg.data = json.dumps(active_labels)
        self.pub_labels.publish(labels_msg)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()

    # Load Config
    with open(JSON_PATH, 'r') as f:
        config = json.load(f)

    metadata = config.get("NN_specific_metadata", {})
    NUM_CLASSES = metadata.get("classes", 26)
    LABELS = config.get("mappings", {}).get("labels", [f"class_{i}" for i in range(NUM_CLASSES)])
    CONF_THRESHOLD = metadata.get("confidence_threshold", 0.5)
    IOU_THRESHOLD = metadata.get("iou_threshold", 0.5)
    INPUT_SIZE = int(config.get("nn_config", {}).get("input_size", "416x416").split("x")[0])

    node.get_logger().info(f"Loaded Config: {NUM_CLASSES} classes, {INPUT_SIZE}x{INPUT_SIZE} input")

    # Build Pipeline using DepthAI v3 Context Manager
    with dai.Pipeline() as pipeline:

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(INPUT_SIZE, INPUT_SIZE)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(30)

        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(BLOB_PATH)
        nn.input.setBlocking(False)
        nn.input.setMaxSize(1)

        cam.preview.link(nn.input)

        q_rgb = nn.passthrough.createOutputQueue(maxSize=1, blocking=False)
        q_nn = nn.out.createOutputQueue(maxSize=1, blocking=False)

        node.get_logger().info("Starting pipeline...")
        pipeline.start()
        node.get_logger().info("Connected to OAK-D!")

        while pipeline.isRunning() and rclpy.ok():

            in_nn = q_nn.get()
            in_rgb = q_rgb.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
            else:
                continue

            H, W = INPUT_SIZE, INPUT_SIZE
            layer_names = in_nn.getAllLayerNames()
            raw_outputs = []

            for name in layer_names:
                for method_name in ['getLayerFp16', 'getLayer', 'getTensor']:
                    if hasattr(in_nn, method_name):
                        try:
                            t = np.array(getattr(in_nn, method_name)(name))
                            if t.size > 0:
                                raw_outputs.append(t)
                                break
                        except Exception:
                            pass

            if not raw_outputs:
                continue

            raw_outputs.sort(key=lambda x: len(x), reverse=True)

            z = []
            shapes = [(52, 52), (26, 26), (13, 13)]
            strides = [8.0, 16.0, 32.0]
            for i in range(min(3, len(raw_outputs))):
                ny, nx = shapes[i]
                channels = raw_outputs[i].size // (ny * nx)
                x = raw_outputs[i].reshape(channels, ny, nx).transpose(1, 2, 0)

                # Definitively identified as YOLOv8 [l, t, r, b] format!
                l = x[..., 0]
                t = x[..., 1]
                r = x[..., 2]
                b = x[..., 3]

                yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
                grid_x = xv.astype(np.float32)
                grid_y = yv.astype(np.float32)

                stride = strides[i]

                # Decode to absolute pixel coordinates
                x1 = (grid_x + 0.5 - l) * stride
                y1 = (grid_y + 0.5 - t) * stride
                x2 = (grid_x + 0.5 + r) * stride
                y2 = (grid_y + 0.5 + b) * stride

                # Convert to [cx, cy, w, h] to match downstream logic
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                # Apply sigmoid ONLY if they are raw logits
                probs = x[..., 4:]
                if np.min(probs) < 0.0 or np.max(probs) > 1.0:
                    probs = 1 / (1 + np.exp(-probs))

                decoded = np.concatenate([
                    cx[..., None], cy[..., None], w[..., None], h[..., None],
                    probs
                ], axis=-1)

                z.append(decoded.reshape(-1, channels))

            if not z:
                continue

            res = np.concatenate(z, axis=0)
            boxes = res[:, :4]

            if res.shape[1] == 31:
                obj_conf = res[:, 4:5]
                cls_probs = res[:, 5:]
                scores = obj_conf * cls_probs
            else:
                scores = res[:, 4:]

            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)

            mask = max_scores > CONF_THRESHOLD
            boxes = boxes[mask]
            class_ids = class_ids[mask]
            max_scores = max_scores[mask]

            detections = []
            for i in range(len(boxes)):
                x1 = int(boxes[i, 0] - boxes[i, 2] / 2)
                y1 = int(boxes[i, 1] - boxes[i, 3] / 2)
                x2 = int(boxes[i, 0] + boxes[i, 2] / 2)
                y2 = int(boxes[i, 1] + boxes[i, 3] / 2)

                x1 = max(0, min(x1, W))
                y1 = max(0, min(y1, H))
                x2 = max(0, min(x2, W))
                y2 = max(0, min(y2, H))

                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(max_scores[i]),
                    "label": int(class_ids[i])
                })

            # NMS Filter
            if len(detections) > 1:
                nms_boxes = torch.tensor([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=torch.float32)
                nms_scores = torch.tensor([d["conf"] for d in detections], dtype=torch.float32)
                keep_idx = torchvision.ops.nms(nms_boxes, nms_scores, IOU_THRESHOLD)
                detections = [detections[i] for i in keep_idx.numpy()]

            # ---------------------------------------------------------------- #
            # Build per-class confidence score vector
            # score_vec[i] = max confidence for class i across all NMS detections
            # ---------------------------------------------------------------- #
            score_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            for d in detections:
                cls = d["label"]
                if d["conf"] > score_vec[cls]:
                    score_vec[cls] = d["conf"]

            # Binary mask: 1 if score > mask_threshold, else 0
            mask_vec = (score_vec > node.mask_threshold).astype(np.int8)

            # Masked label list: label name where mask=1, "" elsewhere
            active_labels = [
                LABELS[i] if mask_vec[i] else ""
                for i in range(NUM_CLASSES)
            ]

            # Publish topics
            node.publish_results(score_vec, mask_vec, active_labels)

            # Visualization
            for d in detections:
                label_text = f"{LABELS[d['label']]} {d['conf']:.2f}"
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (d["x1"], d["y1"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow("OAK-D YOLOv8 Inference", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
