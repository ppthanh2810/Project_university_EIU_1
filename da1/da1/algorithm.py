#!/usr/bin/env python3
import os

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge


class BBoxPercentNode(Node):
    """
    Sub:
      - image_topic (sensor_msgs/Image)
      - detections_topic (vision_msgs/Detection2DArray)

    Pub:
      - percent_topic (std_msgs/Float32)
    """
    def __init__(self):
        super().__init__('bbox_percent_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('detections_topic', '/person_detections')
        self.declare_parameter('percent_topic', '/PT/percent')
        self.declare_parameter('no_person_percent', 0.0)

        self.declare_parameter('queue_size', 10)
        self.declare_parameter('slop', 0.10)

        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'BBox Percent Debug')

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.detections_topic = str(self.get_parameter('detections_topic').value)
        self.percent_topic = str(self.get_parameter('percent_topic').value)
        self.no_person_percent = float(self.get_parameter('no_person_percent').value)

        queue_size = int(self.get_parameter('queue_size').value)
        slop = float(self.get_parameter('slop').value)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)

        self.pub_percent = self.create_publisher(Float32, self.percent_topic, 10)

        self.bridge = CvBridge()

        self.sub_img = Subscriber(
            self,
            Image,
            self.image_topic,
            qos_profile=qos_profile_sensor_data
        )
        self.sub_det = Subscriber(
            self,
            Detection2DArray,
            self.detections_topic,
            qos_profile=qos_profile_sensor_data
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_det],
            queue_size=queue_size,
            slop=slop
        )
        self.sync.registerCallback(self.cb_sync)

        if self.show_window:
            if os.environ.get('DISPLAY', '') == '':
                self.get_logger().warn('show_window=true nhưng DISPLAY chưa có, tự tắt window.')
                self.show_window = False
            else:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("BBoxPercentNode is running.")
        self.get_logger().info(f"Subscribing: {self.image_topic} (Image)")
        self.get_logger().info(f"Subscribing: {self.detections_topic} (Detection2DArray)")
        self.get_logger().info(f"Publishing: {self.percent_topic} (Float32)")

    def cb_sync(self, img_msg: Image, det_msg: Detection2DArray):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        h, w = frame.shape[:2]
        img_area = float(w * h) if (w > 0 and h > 0) else 0.0

        max_area_percent = None

        if det_msg is not None and len(det_msg.detections) > 0 and img_area > 0.0:
            for d in det_msg.detections:
                bw = float(d.bbox.size_x)
                bh = float(d.bbox.size_y)

                # Quan trọng: với BoundingBox2D của bạn phải đọc như này
                cx = float(d.bbox.center.position.x)
                cy = float(d.bbox.center.position.y)

                bbox_area = max(0.0, bw) * max(0.0, bh)
                area_percent = (bbox_area / img_area) * 100.0
                area_percent = max(0.0, min(100.0, area_percent))

                if max_area_percent is None or area_percent > max_area_percent:
                    max_area_percent = area_percent

                if self.show_window:
                    x1 = int(cx - bw / 2.0)
                    y1 = int(cy - bh / 2.0)
                    x2 = int(cx + bw / 2.0)
                    y2 = int(cy + bh / 2.0)

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{area_percent:.2f}%",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        out = Float32()
        out.data = float(self.no_person_percent) if max_area_percent is None else float(max_area_percent)
        self.pub_percent.publish(out)

        if self.show_window:
            cv2.putText(
                frame,
                f"Max Percent: {out.data:.2f}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

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
    node = BBoxPercentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()