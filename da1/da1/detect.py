#!/usr/bin/env -S PYTHONNOUSERSITE=1 python3
import os
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize + pad (letterbox) giống Ultralytics.
    new_shape = (new_w, new_h)
    Return:
      - im_lb: ảnh đã letterbox
      - r: scale ratio
      - (dw, dh): padding (left/top) tính theo pixel
    """
    h0, w0 = im.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w0, new_h / h0)
    w1, h1 = int(round(w0 * r)), int(round(h0 * r))

    im_resized = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_LINEAR)

    dw = new_w - w1
    dh = new_h - h1
    dw /= 2
    dh /= 2

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im_lb = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return im_lb, r, (left, top)


def nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float):
    """
    Pure numpy NMS, tránh cv2.dnn.NMSBoxes (có thể abort).
    boxes_xyxy: (N,4) float32, [x1,y1,x2,y2]
    scores: (N,) float32
    return: list indices keep
    """
    if boxes_xyxy.size == 0:
        return []

    boxes = boxes_xyxy.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    areas = w * h

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

        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih

        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union

        inds = np.where(iou <= iou_thres)[0]
        order = rest[inds]

    return keep


class YoloV8OnnxPersonDetector(Node):
    def __init__(self):
        super().__init__('yolo_person_detector_onnx')

        # Params
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('model', 'yolov8n.onnx')
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('publish_debug_image', True)

        # show window (docker thường để False)
        self.declare_parameter('show_window', False)
        self.declare_parameter('window_name', 'YOLOv8 ONNX Person Debug')

        # onnxruntime providers
        self.declare_parameter('providers', ['CPUExecutionProvider'])

        # tốc độ / an toàn
        self.declare_parameter('topk', 1000)         # lấy topk score trước NMS
        self.declare_parameter('min_box_size', 2.0)  # lọc box nhỏ

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model').value
        self.conf_thres = float(self.get_parameter('conf').value)
        self.iou_thres = float(self.get_parameter('iou').value)
        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.providers = list(self.get_parameter('providers').value)

        self.topk = int(self.get_parameter('topk').value)
        self.min_box_size = float(self.get_parameter('min_box_size').value)

        self.bridge = CvBridge()
        self._logged_output_shape = False

        # Load ONNX
        try:
            sess_opt = ort.SessionOptions()
            sess_opt.log_severity_level = 3
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_opt,
                providers=self.providers
            )
        except Exception as e:
            self.get_logger().fatal(f"Cannot load ONNX model: {self.model_path} | err: {e}")
            raise

        # Input info
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape  # [1,3,H,W] hoặc dynamic
        self.in_h = int(in_shape[2]) if isinstance(in_shape[2], int) else 640
        self.in_w = int(in_shape[3]) if isinstance(in_shape[3], int) else 640

        # Sub / Pub
        self.sub = self.create_subscription(Image, self.image_topic, self.cb_image, 10)
        self.pub_det = self.create_publisher(Detection2DArray, 'person_detections', 10)
        self.pub_dbg = self.create_publisher(Image, 'person_debug_image', 10) if self.publish_debug else None

        self.get_logger().info(f"Subscribing: {self.image_topic}")
        self.get_logger().info(f"ONNX model: {self.model_path} | input={self.in_w}x{self.in_h} | conf>={self.conf_thres} | iou={self.iou_thres}")
        self.get_logger().info(f"ONNXRuntime providers: {self.session.get_providers()}")
        self.get_logger().info("Publishing: /person_detections (vision_msgs/Detection2DArray)")
        if self.publish_debug:
            self.get_logger().info("Publishing: /person_debug_image (sensor_msgs/Image)")

        if self.show_window:
            if os.environ.get("DISPLAY", "") == "":
                self.get_logger().warn("show_window=true but DISPLAY is not set. Disable window.")
                self.show_window = False
            else:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.get_logger().info(f"Showing OpenCV window: {self.window_name} (press 'q' to quit node)")

    def preprocess(self, frame_bgr: np.ndarray):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_lb, r, (pad_x, pad_y) = letterbox(img, (self.in_w, self.in_h))
        img_lb = img_lb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_lb, (2, 0, 1))[None, ...]  # (1,3,H,W)
        return img_chw, r, pad_x, pad_y

    def postprocess(self, outputs, orig_w, orig_h, r, pad_x, pad_y):
        """
        Return list dets: (x1,y1,x2,y2,score,cls_id)
        Hỗ trợ:
          - (N,6) / (1,N,6): [x1,y1,x2,y2,score,cls]
          - raw YOLOv8: (1,84,N) hoặc (1,N,84) hoặc (84,N) hoặc (N,84)
            + có thể gặp 85 (có obj)
        """
        if not outputs:
            return []

        out0 = outputs[0]
        out0 = np.array(out0)

        # squeeze batch dims
        while out0.ndim > 2 and out0.shape[0] == 1:
            out0 = out0[0]

        if not self._logged_output_shape:
            self.get_logger().info(f"ONNX output[0] shape: {tuple(out0.shape)} dtype={out0.dtype}")
            self._logged_output_shape = True

        dets = []

        # Case: already NMS output (N,6) or (N,>=6)
        if out0.ndim == 2 and out0.shape[1] >= 6:
            arr = out0
            # assume first 6 = x1 y1 x2 y2 score cls
            for row in arr:
                x1, y1, x2, y2, score, cls_id = row[:6].tolist()
                cls_id = int(cls_id)
                score = float(score)
                if cls_id != 0 or score < self.conf_thres:
                    continue

                x1 = (x1 - pad_x) / r
                y1 = (y1 - pad_y) / r
                x2 = (x2 - pad_x) / r
                y2 = (y2 - pad_y) / r

                x1 = float(np.clip(x1, 0, orig_w - 1))
                y1 = float(np.clip(y1, 0, orig_h - 1))
                x2 = float(np.clip(x2, 0, orig_w - 1))
                y2 = float(np.clip(y2, 0, orig_h - 1))

                if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
                    continue

                dets.append((x1, y1, x2, y2, score, cls_id))
            return dets

        # Raw YOLO output: make pred shape = (N, C)
        if out0.ndim != 2:
            return []

        # Heuristic: if first dim is small (channels) and second is big (N)
        if out0.shape[0] <= 100 and out0.shape[1] > out0.shape[0]:
            pred = out0.T
        else:
            pred = out0

        if pred.shape[1] < 6:  # at least x,y,w,h + something
            return []

        C = pred.shape[1]

        # Decide format: 84(no obj) / 85(with obj) / other
        has_obj = False
        if C == 84:
            has_obj = False
        elif C == 85:
            has_obj = True
        else:
            # fallback: nếu C-5 >= 1 và trông giống YOLO (có obj)
            # ưu tiên coi có obj nếu C >= 7
            has_obj = (C >= 7)

        boxes_xywh = pred[:, 0:4].astype(np.float32, copy=False)

        if has_obj and C >= 6:
            obj = pred[:, 4].astype(np.float32, copy=False)
            cls_probs = pred[:, 5:].astype(np.float32, copy=False)
            if cls_probs.shape[1] == 0:
                return []
            cls_ids = np.argmax(cls_probs, axis=1)
            cls_conf = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]
            confs = obj * cls_conf
        else:
            cls_probs = pred[:, 4:].astype(np.float32, copy=False)
            if cls_probs.shape[1] == 0:
                return []
            cls_ids = np.argmax(cls_probs, axis=1)
            confs = cls_probs[np.arange(cls_probs.shape[0]), cls_ids].astype(np.float32, copy=False)

        # filter person + conf
        keep = (cls_ids == 0) & (confs >= self.conf_thres) & np.isfinite(confs)
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        confs = confs[keep]

        # limit topk for speed/stability
        if self.topk > 0 and confs.shape[0] > self.topk:
            order = confs.argsort()[::-1][:self.topk]
            boxes_xywh = boxes_xywh[order]
            confs = confs[order]

        # xywh -> xyxy (model-space)
        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # undo letterbox to original image
        x1 = (x1 - pad_x) / r
        y1 = (y1 - pad_y) / r
        x2 = (x2 - pad_x) / r
        y2 = (y2 - pad_y) / r

        # clip & filter invalid
        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        bw = x2 - x1
        bh = y2 - y1
        valid = (bw >= self.min_box_size) & (bh >= self.min_box_size) & np.isfinite(bw) & np.isfinite(bh)
        if not np.any(valid):
            return []

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32, copy=False)[valid]
        confs = confs[valid]

        keep_idx = nms_xyxy(boxes_xyxy, confs, self.iou_thres)
        for i in keep_idx:
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            score = float(confs[i])
            dets.append((x1, y1, x2, y2, score, 0))

        return dets

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = np.ascontiguousarray(frame)
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        orig_h, orig_w = frame.shape[:2]

        try:
            inp, r, pad_x, pad_y = self.preprocess(frame)
        except Exception as e:
            self.get_logger().error(f"preprocess error: {e}")
            return

        try:
            outputs = self.session.run(None, {self.input_name: inp})
        except Exception as e:
            self.get_logger().error(f"onnxruntime inference error: {e}")
            return

        dets = self.postprocess(outputs, orig_w, orig_h, r, pad_x, pad_y)

        det_array = Detection2DArray()
        det_array.header = msg.header

        need_draw = (self.publish_debug and self.pub_dbg is not None) or self.show_window
        if need_draw and len(dets) > 0:
            for (x1, y1, x2, y2, score, cls_id) in dets:
                # message
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                det = Detection2D()
                det.header = msg.header

                bbox = BoundingBox2D()
                bbox.center.position.x = cx
                bbox.center.position.y = cy
                bbox.center.theta = 0.0
                bbox.size_x = w
                bbox.size_y = h
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(cls_id)
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)

                det_array.detections.append(det)

                # draw
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"person {score:.2f}",
                            (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # still publish detections even if no draw
            for (x1, y1, x2, y2, score, cls_id) in dets:
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                det = Detection2D()
                det.header = msg.header

                bbox = BoundingBox2D()
                bbox.center.position.x = cx
                bbox.center.position.y = cy
                bbox.center.theta = 0.0
                bbox.size_x = w
                bbox.size_y = h
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(cls_id)
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)

                det_array.detections.append(det)

        self.pub_det.publish(det_array)

        if self.publish_debug and self.pub_dbg is not None:
            dbg_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            dbg_msg.header = msg.header
            self.pub_dbg.publish(dbg_msg)

        if self.show_window:
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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
    node = YoloV8OnnxPersonDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()