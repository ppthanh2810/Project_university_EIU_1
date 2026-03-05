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

    im_lb = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=color)
    return im_lb, r, (left, top)


class YoloV8OnnxPersonDetector(Node):
    """
    Sub:
      - image_topic (sensor_msgs/Image)

    Pub:
      - /person_detections (vision_msgs/Detection2DArray)
      - /person_debug_image (sensor_msgs/Image) [optional]
    """

    def __init__(self):
        super().__init__('yolo_person_detector_onnx')

        # Params
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('model', 'yolov8n.onnx')     # <-- ONNX
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('iou', 0.45)                # NMS IoU
        self.declare_parameter('publish_debug_image', True)

        # show window
        self.declare_parameter('show_window', False)       # trong docker thường để False
        self.declare_parameter('window_name', 'YOLOv8 ONNX Person Debug')

        # onnxruntime providers
        self.declare_parameter('providers', ['CPUExecutionProvider'])

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model').value
        self.conf_thres = float(self.get_parameter('conf').value)
        self.iou_thres = float(self.get_parameter('iou').value)
        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.providers = list(self.get_parameter('providers').value)

        self.bridge = CvBridge()

        # Load ONNX
        try:
            sess_opt = ort.SessionOptions()
            sess_opt.log_severity_level = 3
            self.session = ort.InferenceSession(self.model_path, sess_options=sess_opt, providers=self.providers)
        except Exception as e:
            self.get_logger().fatal(f"Cannot load ONNX model: {self.model_path} | err: {e}")
            raise

        # Input info
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape  # [1,3,H,W] hoặc dynamic
        # fallback nếu dynamic
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
                self.get_logger().warn("show_window=true but DISPLAY is not set. Window will not open (no GUI).")
                self.show_window = False
            else:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.get_logger().info(f"Showing OpenCV window: {self.window_name} (press 'q' to quit node)")

    def preprocess(self, frame_bgr: np.ndarray):
        # BGR -> RGB
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # letterbox to model size
        img_lb, r, (pad_x, pad_y) = letterbox(img, (self.in_w, self.in_h))

        # normalize + CHW
        img_lb = img_lb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_lb, (2, 0, 1))[None, ...]  # (1,3,H,W)
        return img_chw, r, pad_x, pad_y

    def postprocess(self, outputs, orig_w, orig_h, r, pad_x, pad_y):
        """
        Trả về list detections dạng (x1,y1,x2,y2,score,class_id)
        Hỗ trợ:
          - Raw YOLOv8: (1,84,N) hoặc (1,N,84) -> decode + NMS
          - NMS-ready: (N,6) hoặc (1,N,6): [x1,y1,x2,y2,score,cls]
        """
        out0 = outputs[0]

        # squeeze batch nếu cần
        if out0.ndim == 3:
            out0 = out0[0]

        dets = []

        # Case: already NMS output (N,6)
        if out0.ndim == 2 and out0.shape[1] == 6:
            arr = out0
            for row in arr:
                x1, y1, x2, y2, score, cls_id = row.tolist()
                cls_id = int(cls_id)
                if cls_id != 0:
                    continue
                if score < self.conf_thres:
                    continue

                # scale/clip (giả định output theo input size)
                x1 = (x1 - pad_x) / r
                y1 = (y1 - pad_y) / r
                x2 = (x2 - pad_x) / r
                y2 = (y2 - pad_y) / r

                x1 = float(np.clip(x1, 0, orig_w - 1))
                y1 = float(np.clip(y1, 0, orig_h - 1))
                x2 = float(np.clip(x2, 0, orig_w - 1))
                y2 = float(np.clip(y2, 0, orig_h - 1))
                dets.append((x1, y1, x2, y2, float(score), cls_id))
            return dets

        # Case: raw YOLOv8 output
        # shape có thể là (84,N) hoặc (N,84)
        if out0.ndim == 2:
            if out0.shape[0] in (84, 85):          # (C,N)
                pred = out0.transpose(1, 0)         # -> (N,C)
            else:
                pred = out0                          # (N,C)
        else:
            return dets

        if pred.shape[1] < 5:
            return dets

        # YOLOv8 raw: [x,y,w,h, cls0..cls79]
        boxes = pred[:, 0:4]
        scores_all = pred[:, 4:]

        cls_ids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(scores_all.shape[0]), cls_ids]

        # filter person + conf
        keep = (cls_ids == 0) & (confs >= self.conf_thres)
        boxes = boxes[keep]
        confs = confs[keep]

        if boxes.shape[0] == 0:
            return dets

        # xywh -> xyxy (theo input-size)
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # undo letterbox
        x1 = (x1 - pad_x) / r
        y1 = (y1 - pad_y) / r
        x2 = (x2 - pad_x) / r
        y2 = (y2 - pad_y) / r

        # clip
        x1 = np.clip(x1, 0, orig_w - 1).astype(np.float32)
        y1 = np.clip(y1, 0, orig_h - 1).astype(np.float32)
        x2 = np.clip(x2, 0, orig_w - 1).astype(np.float32)
        y2 = np.clip(y2, 0, orig_h - 1).astype(np.float32)

        # NMSBoxes cần (x,y,w,h)
        nms_boxes = []
        for i in range(len(confs)):
            bx = float(x1[i])
            by = float(y1[i])
            bw = float(max(0.0, x2[i] - x1[i]))
            bh = float(max(0.0, y2[i] - y1[i]))
            nms_boxes.append([bx, by, bw, bh])

        idxs = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), self.conf_thres, self.iou_thres)
        if len(idxs) == 0:
            return dets

        # idxs có thể là [[i],[j]] hoặc [i,j]
        idxs = np.array(idxs).reshape(-1).tolist()
        for i in idxs:
            dets.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(confs[i]), 0))

        return dets

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        orig_h, orig_w = frame.shape[:2]

        # preprocess
        try:
            inp, r, pad_x, pad_y = self.preprocess(frame)
        except Exception as e:
            self.get_logger().error(f"preprocess error: {e}")
            return

        # inference
        try:
            outputs = self.session.run(None, {self.input_name: inp})
        except Exception as e:
            self.get_logger().error(f"onnxruntime inference error: {e}")
            return

        # postprocess
        dets = self.postprocess(outputs, orig_w, orig_h, r, pad_x, pad_y)

        det_array = Detection2DArray()
        det_array.header = msg.header

        # fill message + draw
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
            hyp.hypothesis.class_id = str(cls_id)  # "0"
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)

            det_array.detections.append(det)

            if self.publish_debug or self.show_window:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"person {score:.2f}",
                            (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # publish detections
        self.pub_det.publish(det_array)

        # publish debug topic
        if self.publish_debug and self.pub_dbg is not None:
            dbg_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            dbg_msg.header = msg.header
            self.pub_dbg.publish(dbg_msg)

        # show window
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