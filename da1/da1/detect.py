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


class YoloV8PersonDetector(Node):
    def __init__(self):
        super().__init__("yolov8_person_detector")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("model", "/home/yolov8n.onnx")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("publish_debug_image", False)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.model_path = str(self.get_parameter("model").value)
        self.conf_thres = float(self.get_parameter("conf").value)
        self.iou_thres = float(self.get_parameter("iou").value)
        self.publish_debug = bool(self.get_parameter("publish_debug_image").value)

        self.bridge = CvBridge()
        self._logged_output_shape = False

        if not os.path.isfile(self.model_path):
            self.get_logger().fatal(f"Model not found: {self.model_path}")
            raise FileNotFoundError(self.model_path)

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )

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
            "/person_detections",
            10,
        )

        self.pub_dbg = None
        if self.publish_debug:
            self.pub_dbg = self.create_publisher(
                Image,
                "/person_debug_image",
                10,
            )

        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Input size: {self.in_w}x{self.in_h}")
        self.get_logger().info(f"Providers: {self.session.get_providers()}")
        self.get_logger().info(
            f"Thresholds: conf={self.conf_thres}, iou={self.iou_thres}"
        )
        self.get_logger().info("Publishing: /person_detections")
        if self.publish_debug:
            self.get_logger().info("Publishing: /person_debug_image")

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

        for x1, y1, x2, y2, score in dets:
            det_array.detections.append(
                self.make_detection(msg.header, x1, y1, x2, y2, score)
            )

            if self.publish_debug:
                cv2.rectangle(
                    debug_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    debug_frame,
                    f"person {score:.2f}",
                    (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        self.pub_det.publish(det_array)

        if self.publish_debug and self.pub_dbg is not None:
            try:
                dbg_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
                dbg_msg.header = msg.header
                self.pub_dbg.publish(dbg_msg)
            except Exception as e:
                self.get_logger().error(f"debug publish error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloV8PersonDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()