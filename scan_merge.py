import math
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from laser_geometry import LaserProjection
import tf2_ros
from tf2_ros import TransformException
import sensor_msgs_py.point_cloud2 as pc2

class PauseAndCapture(Node):
    def __init__(self):
        super().__init__('pause_and_capture')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.laser_projector = LaserProjection()


        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )        # HINT: Publish on the '/scan' topic
        self.subscription = self.create_subscription(LaserScan,
                                                     '/scan', self.scan_callback, qos_profile)

        # Create a publisher for PointCloud2 messages
        # HINT: Publish on the '/accumulated_cloud' topic
        self.pc_pub = self.create_publisher(PointCloud2,'/accumulated_cloud', 10)
        self.accumulated_points = []

        self.capture_enabled = False
        self.latest_scan = None
        self.delay_timer = None

        self.input_thread = threading.Thread(target=self.key_press_listener, daemon=True)
        self.input_thread.start()

        self.get_logger().info("PauseAndCapture node started. Press Enter to capture a scan.")

    def key_press_listener(self):
        """key input"""
        while True:
            input(">> Press Enter to capture scan: ")
            self.capture_enabled = True

    def scan_callback(self, scan_msg):
        """scan callback"""
        if not self.capture_enabled:
            return

        self.latest_scan = scan_msg
        self.capture_enabled = False

        # Add slight delay to allow TF to catch up (avoids extrapolation error)
        if self.delay_timer:
            self.delay_timer.cancel()
        self.delay_timer = self.create_timer(0.1, self.delayed_transform_lookup)

    def delayed_transform_lookup(self):
        """delay transform lookup"""
        self.delay_timer.cancel()
        scan_msg = self.latest_scan
        self.latest_scan = None



        try:
            cloud_in_laser = self.laser_projector.projectLaser(scan_msg)

            # Perform a lookup to transform
            # the point cloud from its original frame to the 'odom' frame
            transform = self.tf_buffer.lookup_transform(
                'odom', # Target frame
                scan_msg.header.frame_id,# Source
                #frame (the point cloud's original frame)
                # ^ should be lidar, not sure if it is yet
                rclpy.time.Time.from_msg(scan_msg.header.stamp),  # Timestamp of
                #the scan message to ensure proper time synchronization
                timeout=rclpy.duration.Duration(seconds=0.5)  # Timeout of 0.5
                #seconds to wait for the transform
            )

            transformed_points = self.transform_pointcloud2(cloud_in_laser, transform)
            self.accumulated_points.extend(transformed_points)
            self.publish_accumulated_cloud(scan_msg.header.stamp)
            self.get_logger().info(f"Captured and transformed {len(transformed_points)} points.")

        except TransformException as ex:
            self.get_logger().warn(f"Transform failed after delay: {str(ex)}")

    def transform_pointcloud2(self, cloud_msg, transform):
        """Transform a point cloud using Euler
          angles from a given quaternion."""

        # Helper function to convert
        # quaternion to Euler angles (roll, pitch, yaw)
        def quaternion_to_euler(q):
            """Convert quaternion to Euler angles (roll, pitch, yaw)."""
            x, y, z, w = q.x, q.y, q.z, q.w
            roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            pitch = math.asin(2.0 * (w * y - z * x))
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            return [roll, pitch, yaw]

        # Given a point in 3D and its rotation angles, construct a series
        # of rotation matrices and apply them to the point
        # HINT: Yaw @ Pitch @ Roll
        def rotate_point_euler(x_value, y_value, z_value, roll, pitch, yaw) -> tuple[int, int, int]:
            roll_array = np.array([[1, 0, 0], [0, np.cos(roll),
                                               -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
            pitch_array = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                    [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
            yaw_array = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                  [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            return tuple(yaw_array @ pitch_array @ roll_array @ (np.array([x_value, y_value, z_value])))


        # Extract translation and rotation (quaternion) from the transform
        t_vec, q_vec = transform.transform.translation, transform.transform.rotation
        quat = [q_vec.x, q_vec.y, q_vec.z, q_vec.w]
        trans_vector = np.array([t_vec.x,t_vec.y,t_vec.z])
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        euler_angles = quaternion_to_euler(quat)

        # Transform the point cloud using Euler rotation
        transformed_points = []
        for pt in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = pt

            # Apply rotation to the point using Euler angles use the rotate point euler function
            rotated_point = rotate_point_euler(x, y, z,
                                               euler_angles[0], euler_angles[1], euler_angles[2])

            # Apply translation to the rotated point using the variavble t
            translated_point = rotated_point + trans_vector
            # Append transformed point
            transformed_points.append(translated_point)

        return transformed_points

    def publish_accumulated_cloud(self, stamp):
        """publish cloud"""
        header = Header()
        header.stamp = stamp
        header.frame_id = "odom"

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, self.accumulated_points)
        self.pc_pub.publish(cloud_msg)
        self.get_logger().info("Published accumulated cloud.")

def main(args=None):
    """main func"""
    rclpy.init(args=args)
    node = PauseAndCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
