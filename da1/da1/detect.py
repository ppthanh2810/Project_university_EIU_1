#!/usr/bin/env python3
import os

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = im.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w0, new_h / h0)
    w1, h1 = int(round(w0 * r)), int(round(h0 * r))

    im_resized = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_LINEAR)

    dw = (new_w - w1) / 2.0
    dh = (new_h - h1) / 2.0

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im_out = cv2.copyMakeBorder(
        im_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return im_out, r, left, top


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thres]

    return keep


class YoloPersonPercentNode(Node):
    def __init__(self):
        super().__init__("yolo_person_percent_node")

        # Detect params
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("model", "/home/yolov8n.onnx")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)

        # GPU / ORT params
        self.declare_parameter("provider", "auto")   # auto | tensorrt | cuda | cpu
        self.declare_parameter("gpu_device_id", 0)
        self.declare_parameter("gpu_mem_limit_mb", 2048)
        self.declare_parameter("trt_fp16", True)
        self.declare_parameter("trt_workspace_mb", 2048)
        self.declare_parameter("trt_engine_cache_enable", True)
        self.declare_parameter("trt_engine_cache_path", "/tmp/ort_trt_cache")
        self.declare_parameter("intra_op_num_threads", 1)
        self.declare_parameter("inter_op_num_threads", 1)

        # Output params
        self.declare_parameter("detections_topic", "/person_detections")
        self.declare_parameter("percent_topic", "/PT/percent")
        self.declare_parameter("no_person_percent", 0.0)

        # Debug params
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("debug_image_topic", "/person_debug_image")
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_name", "Person Detection + Max Percent")

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.model_path = str(self.get_parameter("model").value)
        self.conf_thres = float(self.get_parameter("conf").value)
        self.iou_thres = float(self.get_parameter("iou").value)

        self.provider_mode = str(self.get_parameter("provider").value).strip().lower()
        self.gpu_device_id = int(self.get_parameter("gpu_device_id").value)
        self.gpu_mem_limit_mb = int(self.get_parameter("gpu_mem_limit_mb").value)
        self.trt_fp16 = bool(self.get_parameter("trt_fp16").value)
        self.trt_workspace_mb = int(self.get_parameter("trt_workspace_mb").value)
        self.trt_engine_cache_enable = bool(self.get_parameter("trt_engine_cache_enable").value)
        self.trt_engine_cache_path = str(self.get_parameter("trt_engine_cache_path").value)
        self.intra_threads = int(self.get_parameter("intra_op_num_threads").value)
        self.inter_threads = int(self.get_parameter("inter_op_num_threads").value)

        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.percent_topic = str(self.get_parameter("percent_topic").value)
        self.no_person_percent = float(self.get_parameter("no_person_percent").value)

        self.publish_debug = bool(self.get_parameter("publish_debug_image").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)

        self.show_window = bool(self.get_parameter("show_window").value)
        self.window_name = str(self.get_parameter("window_name").value)

        self.bridge = CvBridge()
        self._logged_output_shape = False
        self.outdata = 0.0

        if not os.path.isfile(self.model_path):
            self.get_logger().fatal(f"Model not found: {self.model_path}")
            raise FileNotFoundError(self.model_path)

        self.session = self._build_ort_session()

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape

        self.in_h = int(shape[2]) if isinstance(shape[2], int) else 640
        self.in_w = int(shape[3]) if isinstance(shape[3], int) else 640

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.pub_det = self.create_publisher(
            Detection2DArray,
            self.detections_topic,
            10,
        )

        self.pub_percent = self.create_publisher(
            Float32,
            self.percent_topic,
            10,
        )

        self.pub_dbg = None
        if self.publish_debug:
            self.pub_dbg = self.create_publisher(
                Image,
                self.debug_image_topic,
                10,
            )

        if self.show_window:
            if os.environ.get("DISPLAY", "") == "":
                self.get_logger().warn("show_window=true nhưng DISPLAY chưa có, tự tắt window.")
                self.show_window = False
            else:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.create_timer(0.2, self.print_percent)

        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Input size: {self.in_w}x{self.in_h}")
        self.get_logger().info(f"Available providers: {ort.get_available_providers()}")
        self.get_logger().info(f"Using providers: {self.session.get_providers()}")
        self.get_logger().info(f"Provider mode: {self.provider_mode}")
        self.get_logger().info(f"Thresholds: conf={self.conf_thres}, iou={self.iou_thres}")
        self.get_logger().info(f"Publishing detections: {self.detections_topic}")
        self.get_logger().info(f"Publishing percent: {self.percent_topic}")
        if self.publish_debug:
            self.get_logger().info(f"Publishing debug image: {self.debug_image_topic}")
        if self.show_window:
            self.get_logger().info(f"Show window: {self.window_name}")

    def _build_ort_session(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, self.intra_threads)
        sess_options.inter_op_num_threads = max(1, self.inter_threads)
        sess_options.enable_cpu_mem_arena = True
        sess_options.log_severity_level = 3

        available = ort.get_available_providers()
        providers = []

        if self.provider_mode not in ("auto", "tensorrt", "cuda", "cpu"):
            self.get_logger().warn(
                f"provider={self.provider_mode} không hợp lệ -> dùng auto"
            )
            self.provider_mode = "auto"

        if self.provider_mode != "cpu":
            if self.provider_mode in ("auto", "tensorrt") and "TensorrtExecutionProvider" in available:
                if self.trt_engine_cache_enable:
                    os.makedirs(self.trt_engine_cache_path, exist_ok=True)

                providers.append((
                    "TensorrtExecutionProvider",
                    {
                        "device_id": self.gpu_device_id,
                        "trt_fp16_enable": self.trt_fp16,
                        "trt_max_workspace_size": self.trt_workspace_mb * 1024 * 1024,
                        "trt_engine_cache_enable": self.trt_engine_cache_enable,
                        "trt_engine_cache_path": self.trt_engine_cache_path,
                    }
                ))

            if self.provider_mode in ("auto", "cuda", "tensorrt") and "CUDAExecutionProvider" in available:
                providers.append((
                    "CUDAExecutionProvider",
                    {
                        "device_id": self.gpu_device_id,
                        "gpu_mem_limit": self.gpu_mem_limit_mb * 1024 * 1024,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    }
                ))

        providers.append("CPUExecutionProvider")

        try:
            session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as e:
            self.get_logger().warn(
                f"Không tạo được GPU session ({e}) -> fallback CPU."
            )
            session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

        return session

    def print_percent(self):
        self.get_logger().info(f"{self.outdata}")

    def preprocess(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_lb, r, pad_x, pad_y = letterbox(img_rgb, (self.in_w, self.in_h))
        img = img_lb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img, r, pad_x, pad_y

    def postprocess(self, outputs, orig_w, orig_h, r, pad_x, pad_y):
        if not outputs:
            return []

        out = np.array(outputs[0])

        while out.ndim > 2 and out.shape[0] == 1:
            out = out[0]

        if not self._logged_output_shape:
            self.get_logger().info(f"ONNX output shape: {tuple(out.shape)}")
            self._logged_output_shape = True

        dets = []

        if out.ndim == 2 and (out.shape[0] in [84, 85] or out.shape[1] in [84, 85]):
            pred = out.T if out.shape[0] in [84, 85] else out

            if pred.shape[1] < 6:
                return []

            boxes_xywh = pred[:, :4].astype(np.float32, copy=False)

            if pred.shape[1] == 84:
                cls_scores = pred[:, 4:].astype(np.float32, copy=False)
                cls_ids = np.argmax(cls_scores, axis=1)
                scores = cls_scores[np.arange(len(cls_scores)), cls_ids]
            elif pred.shape[1] == 85:
                obj = pred[:, 4].astype(np.float32, copy=False)
                cls_scores = pred[:, 5:].astype(np.float32, copy=False)
                cls_ids = np.argmax(cls_scores, axis=1)
                cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
                scores = obj * cls_conf
            else:
                return []

            keep = (cls_ids == 0) & (scores >= self.conf_thres) & np.isfinite(scores)
            if not np.any(keep):
                return []

            boxes_xywh = boxes_xywh[keep]
            scores = scores[keep]

            cx = boxes_xywh[:, 0]
            cy = boxes_xywh[:, 1]
            w = boxes_xywh[:, 2]
            h = boxes_xywh[:, 3]

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            x1 = (x1 - pad_x) / r
            y1 = (y1 - pad_y) / r
            x2 = (x2 - pad_x) / r
            y2 = (y2 - pad_y) / r

            x1 = np.clip(x1, 0, orig_w - 1)
            y1 = np.clip(y1, 0, orig_h - 1)
            x2 = np.clip(x2, 0, orig_w - 1)
            y2 = np.clip(y2, 0, orig_h - 1)

            valid = (
                (x2 - x1 > 2.0)
                & (y2 - y1 > 2.0)
                & np.isfinite(x1)
                & np.isfinite(y1)
                & np.isfinite(x2)
                & np.isfinite(y2)
            )
            if not np.any(valid):
                return []

            boxes = np.stack([x1, y1, x2, y2], axis=1)[valid]
            scores = scores[valid]

            keep_idx = nms(boxes, scores, self.iou_thres)
            for i in keep_idx:
                bx1, by1, bx2, by2 = boxes[i]
                dets.append((
                    float(bx1),
                    float(by1),
                    float(bx2),
                    float(by2),
                    float(scores[i]),
                ))
            return dets

        if out.ndim == 2 and 6 <= out.shape[1] <= 7:
            for row in out:
                x1, y1, x2, y2, score, cls_id = row[:6]

                if int(cls_id) != 0 or float(score) < self.conf_thres:
                    continue

                x1 = (x1 - pad_x) / r
                y1 = (y1 - pad_y) / r
                x2 = (x2 - pad_x) / r
                y2 = (y2 - pad_y) / r

                x1 = float(np.clip(x1, 0, orig_w - 1))
                y1 = float(np.clip(y1, 0, orig_h - 1))
                x2 = float(np.clip(x2, 0, orig_w - 1))
                y2 = float(np.clip(y2, 0, orig_h - 1))

                if (x2 - x1) > 2.0 and (y2 - y1) > 2.0:
                    dets.append((x1, y1, x2, y2, float(score)))

            return dets

        return []

    def make_detection(self, header, x1, y1, x2, y2, score):
        det = Detection2D()
        det.header = header

        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        bbox = BoundingBox2D()
        bbox.center.position.x = cx
        bbox.center.position.y = cy
        bbox.center.theta = 0.0
        bbox.size_x = w
        bbox.size_y = h
        det.bbox = bbox

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "person"
        hyp.hypothesis.score = float(score)
        det.results.append(hyp)

        return det

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = np.ascontiguousarray(frame)
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        orig_h, orig_w = frame.shape[:2]
        img_area = float(orig_w * orig_h) if orig_w > 0 and orig_h > 0 else 0.0

        try:
            inp, r, pad_x, pad_y = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: inp})
            dets = self.postprocess(outputs, orig_w, orig_h, r, pad_x, pad_y)
        except Exception as e:
            self.get_logger().error(f"inference error: {e}")
            return

        det_array = Detection2DArray()
        det_array.header = msg.header

        debug_frame = frame.copy()
        max_area_percent = None

        for x1, y1, x2, y2, score in dets:
            det_array.detections.append(
                self.make_detection(msg.header, x1, y1, x2, y2, score)
            )

            bw = max(0.0, float(x2 - x1))
            bh = max(0.0, float(y2 - y1))
            bbox_area = bw * bh

            area_percent = 0.0
            if img_area > 0.0:
                area_percent = (bbox_area / img_area) * 100.0
                area_percent = max(0.0, min(100.0, area_percent))

            if max_area_percent is None or area_percent > max_area_percent:
                max_area_percent = area_percent

            if self.show_window or self.publish_debug:
                ix1 = max(0, min(int(x1), orig_w - 1))
                iy1 = max(0, min(int(y1), orig_h - 1))
                ix2 = max(0, min(int(x2), orig_w - 1))
                iy2 = max(0, min(int(y2), orig_h - 1))

                cv2.rectangle(
                    debug_frame,
                    (ix1, iy1),
                    (ix2, iy2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    debug_frame,
                    f"person {score:.2f} | {area_percent:.2f}%",
                    (ix1, max(0, iy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        self.pub_det.publish(det_array)

        out = Float32()
        out.data = float(self.no_person_percent) if max_area_percent is None else float(max_area_percent)
        self.pub_percent.publish(out)
        self.outdata = out.data

        if self.show_window or self.publish_debug:
            cv2.putText(
                debug_frame,
                f"Max Percent: {out.data:.2f}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        if self.publish_debug and self.pub_dbg is not None:
            try:
                dbg_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
                dbg_msg.header = msg.header
                self.pub_dbg.publish(dbg_msg)
            except Exception as e:
                self.get_logger().error(f"debug publish error: {e}")

        if self.show_window:
            cv2.imshow(self.window_name, debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info("Pressed 'q' -> shutting down node")
                rclpy.shutdown()

    def destroy_node(self):
        if self.show_window:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloPersonPercentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
