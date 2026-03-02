#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

import cv2
from ultralytics import YOLO


class YoloPersonDetector(Node):
    """
    Sub:
      - image_topic (sensor_msgs/Image)

    Pub:
      - /person_detections (vision_msgs/Detection2DArray)
      - /person_debug_image (sensor_msgs/Image) [optional]
    """
    def __init__(self):
        super().__init__('yolo_person_detector')

        # Params
        self.declare_parameter('image_topic', 'YOLO_show')
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('publish_debug_image', True)

        # show window
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'YOLO Person Debug')

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model').value
        self.conf = float(self.get_parameter('conf').value)
        self.device = str(self.get_parameter('device').value)
        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        # Sub / Pub
        self.sub = self.create_subscription(Image, self.image_topic, self.cb_image, 10)

        # giữ đúng như bạn đang publish
        self.pub_det = self.create_publisher(Detection2DArray, 'person_detections', 10)
        self.pub_dbg = self.create_publisher(Image, 'person_debug_image', 10) if self.publish_debug else None

        self.get_logger().info(f"Subscribing: {self.image_topic}")
        self.get_logger().info(f"Model: {self.model_path} | conf>={self.conf} | device={self.device}")
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

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        img_h, img_w = frame.shape[:2]

        # YOLO predict
        try:
            results = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)
        except Exception as e:
            self.get_logger().error(f"YOLO predict error: {e}")
            return

        det_array = Detection2DArray()
        det_array.header = msg.header

        if results:
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls = int(b.cls[0])
                    if cls != 0:  # person
                        continue

                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    score = float(b.conf[0])

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
                    hyp.hypothesis.class_id = "0"  # person class
                    hyp.hypothesis.score = score
                    det.results.append(hyp)

                    det_array.detections.append(det)

                    # draw debug
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
    node = YoloPersonDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
