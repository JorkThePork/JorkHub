#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import threading, queue
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from PIL import Image
from self_driving.model import ImageOnlySteerNet
import time


class SteeringNode(Node):
    def __init__(self):
        super().__init__('steering_inference_node')

        # ---- Parameters ----
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('model_path', 'model.pt')
        self.declare_parameter('linear_vel', 0.01)
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('flip_sign', False)

        self.device = torch.device(
            self.get_parameter('device').get_parameter_value().string_value
        )
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.linear_vel = float(self.get_parameter('linear_vel').get_parameter_value().double_value)
        self.rate_hz = float(self.get_parameter('rate_hz').get_parameter_value().double_value)
        self.flip_sign = bool(self.get_parameter('flip_sign').get_parameter_value().bool_value)

        # ---- Load model ----
        self.get_logger().info(f"Loading ImageOnlySteerNet from {model_path} on {self.device}")
        ckpt = torch.load(model_path, map_location=self.device)
        self.mu = float(ckpt.get("mu", 0.0))
        self.sigma = float(ckpt.get("sigma", 1.0))
        self.omega_sign = ckpt.get("omega_sign", 1.0)
        if self.flip_sign:
            self.omega_sign *= -1.0

        self.model = ImageOnlySteerNet(out_len=1, pretrained=False).to(self.device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.get_logger().info("Model loaded and ready for inference.")

        # ---- ROS Interfaces ----
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.bridge = CvBridge()
        self.sub = self.create_subscription(CompressedImage, '/image/compressed', self.image_callback, qos) # create a subscriber that subscribes to /image/compressed topic
        self.pub = self.create_publisher(Twist, '/cmd_vel', qos) # create a publisher that publishes to /cmd_vel topic

        # ---- Timer-driven publisher ----
        self.latest_omega = 0.0
        self.timer = self.create_timer(1.0 / self.rate_hz, self.publish_cmd)

        # ---- Preprocessing (matches training) ----
        self.tf = T.Compose([
            # TODO: Convert the image (NumPy array) to PIL Image,
            lambda img : cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            # TODO: Crop the top 20% of the image (use a lambda)
            lambda pil : pil.crop((0, int(0.2 * pil.height), pil.width, pil.height)),
            # TODO: Resize and center crop to 224×224 for ResNet
            T.Resize((224,224)),
            # TODO: Convert to a normalized tensor
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # ---- Runtime smoothing ----
        self.omega_f = 0.0
        self.omega_prev = 0.0

        # ---- Frame queue + background thread ----
        self.frame_q = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(f"Ready — publishing /cmd_vel at {self.rate_hz:.1f} Hz")

    # ---------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        """Store latest image without blocking."""
        if self.frame_q.full():
            try:
                self.frame_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_q.put_nowait(msg)
        except queue.Full:
            pass

    # ---------------------------------------------------
    def worker_loop(self):
        """Background thread: runs inference on latest frame."""
        while rclpy.ok():
            try:
                msg = self.frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # TODO: Decode the incoming compressed image into a CV2 image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                continue

            # TODO: Preprocess the image into a PyTorch tensor
            img_tensor = self.tf(cv_image)

            # Inference
            with torch.inference_mode():
                y_std = self.model(img_tensor)
                y_raw = y_std * self.sigma + self.mu
                omega = float(self.omega_sign * y_raw.item())

            omega_cmd = omega
            # Optional gain scaling (tunable)
            self.latest_omega = float(1.2* omega_cmd)

            now = time.time()
            if not hasattr(self, "_last_log_t") or now - self._last_log_t > 1.0:
                self._last_log_t = now
                self.get_logger().info(f"ω_cmd={omega_cmd:+.3f} rad/s  (raw={omega:+.3f})")


    # ---------------------------------------------------
    def publish_cmd(self):
        """Timer callback: publish latest steering command at fixed rate."""
        vel_cmd = Twist()
        vel_cmd.linear.x = self.linear_vel
        vel_cmd.angular.z = self.latest_omega
        self.pub.publish(vel_cmd)
        # TODO: Create a new Twist message
        # TODO: Set the forward linear velocity (x-axis)
        # TODO: Set the angular velocity for steering (z-axis)
        # TODO: Publish the velocity command


# -------------------------------------------------------
from rclpy.executors import MultiThreadedExecutor

def main():
    rclpy.init()
    node = SteeringNode()
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

