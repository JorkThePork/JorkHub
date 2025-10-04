import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from laser_geometry import LaserProjection
import tf2_ros
from tf2_ros import TransformException, TransformStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
import sensor_msgs_py.point_cloud2 as pc2
#import math
import numpy as np
import threading
from scipy.spatial import cKDTree
#from geometry_msgs.msg import Quaternion
from tf_transformations import euler_from_quaternion

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
        )        # HINT: Subscribe on the '/scan' topic
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )

        # Create a publisher for PointCloud2 messages
        # HINT: Publish on the '/accumulated_cloud' topic
        self.pc_pub = self.create_publisher(
            PointCloud2,
            '/accumulated_cloud',
            qos_profile
        )

        # Create a publisher for ICP merged cloud
        # HINT: Publish on the '/icp_merged_cloud' topic
        self.icp_pub = self.create_publisher(
            PointCloud2,
            '/icp_merged_cloud',
            qos_profile
        )

        self.accumulated_points = []
        self.icp_accumulated_points = []

        self.capture_enabled = False
        self.latest_scan = None
        self.delay_timer = None

        self.input_thread = threading.Thread(target=self.key_press_listener, daemon=True)
        self.input_thread.start()

        self.get_logger().info("PauseAndCapture node started. Press Enter to capture a scan.")

    def key_press_listener(self):
        while True:
            input(">> Press Enter to capture scan: ")
            self.capture_enabled = True

    def scan_callback(self, scan_msg: LaserScan):
        if not self.capture_enabled:
            return

        self.latest_scan = scan_msg
        self.capture_enabled = False

        if self.delay_timer:
            self.delay_timer.cancel()
        self.delay_timer = self.create_timer(0.1, self.delayed_transform_lookup)

    def delayed_transform_lookup(self):
        self.delay_timer.cancel()
        scan_msg = self.latest_scan
        self.latest_scan = None
        try:
            cloud_in_laser = self.laser_projector.projectLaser(scan_msg)


            # Perform a lookup to transform the point cloud
            # from its original frame to the 'odom' frame
            # transform = self.tf_buffer.lookup_transform(
            #     'odom'
            #     cloud_in_laser.header.frame_id,
            #     rclpy.time.Time.from_msg(scan_msg.header.stamp)

            # )
            transform = self.tf_buffer.lookup_transform(
                'odom', # Target frame
                scan_msg.header.frame_id,# Source frame (the point cloud's original frame)
                # ^ should be lidar, not sure if it is yet
                rclpy.time.Time.from_msg(scan_msg.header.stamp),
                # Timestamp of the scan message to ensure proper time synchronization
                timeout=rclpy.duration.Duration(seconds=0.5)
                # Timeout of 0.5 seconds to wait for the transform
            )

            transformed_points = self.transform_pointcloud2(cloud_in_laser, transform)
            # Transform the point cloud with the transform_pointclod2 function

            if self.icp_accumulated_points:
                icp_aligned = self.perform_icp(self.icp_accumulated_points, transformed_points )
                self.icp_accumulated_points.extend(icp_aligned)
                self.publish_icp_merged_cloud(scan_msg.header.stamp)
                self.get_logger().info(f"ICP-aligned and merged {len(icp_aligned)} points.")
            else:
                self.icp_accumulated_points.extend(transformed_points)
                self.publish_icp_merged_cloud(scan_msg.header.stamp)
                self.get_logger().info(f"Initialized ICP merged cloud with {len(transformed_points)} points.")

            self.accumulated_points.extend(transformed_points)
            self.publish_accumulated_cloud(scan_msg.header.stamp)
            self.get_logger().info(f"Captured and transformed {len(transformed_points)} points.")

        except TransformException as ex:
            self.get_logger().warn(f"Transform failed after delay: {str(ex)}")




    # Complete the rotate_point_euler in transform_pointcloud2 functios
    #Note that ros iherently processes point clouds
    # in 3d even though the robot's point cloud is in 2d.

    def transform_pointcloud2(self, cloud_msg: PointCloud2, transform: TransformStamped) -> list[tuple[int, int, int]]:
        """Transform a point cloud using Euler angles from a given quaternion."""

        def rotate_point_euler(x, y, z, roll, pitch, yaw) -> tuple[int, int, int]:
            """Rotate a point (x, y, z) using Euler angles (roll, pitch, yaw)."""

            #using the roll,pitch and yaw construct the Rx , Ry, Rz matrix
            roll_array = np.array([[1, 0, 0], [0, np.cos(roll),
                                               -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
            pitch_array = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                    [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
            yaw_array = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                  [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])


            # Combined rotation matrix
            combined_rotation_matrix = yaw_array @ pitch_array @ roll_array

            # Apply the rotation to the point

            return tuple(combined_rotation_matrix @ np.array([x, y, z]))


        # Extract translation and rotation (quaternion) from the transform method
        t, q = transform.transform.translation, transform.transform.rotation
        # if t is not an np array we'll need to do np.array([[t.x], [t.y], [t.z]])
        quat = [q.x,q.y,q.z,q.w]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Hint: Use the euler_from_quaternion
        euler_angles = euler_from_quaternion(quat)
        trans_vector = np.array([t.x, t.y, t.z])
        # Transform the point cloud using Euler rotation
        transformed_points = []
        for pt in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            # Get values of pt
            x, y, z = pt


            # Apply rotation to the point using Euler angles use the rotate point euler function
            rotated_point = rotate_point_euler(x, y, z,
                                               euler_angles[0], euler_angles[1], euler_angles[2])
            translated_point = rotated_point + trans_vector
            # Append transformed point
            transformed_points.append((translated_point))

        return transformed_points

    def perform_icp(self, previous_points, current_points, max_iterations=20, tolerance=1e-4):
        src = np.array(current_points)
        tgt = np.array(previous_points)

        def svd_estimation(previous_points, current_points):
            target_centroid = np.mean(previous_points, axis=0)
            source_centroid = np.mean(current_points, axis=0)

            # Center both
            p_ = current_points - source_centroid
            q_ = previous_points - target_centroid

            # Cross-covariance
            h_ = p_.T @ q_
            u_, _, vt_matrix = np.linalg.svd(h_)
            r_opt = vt_matrix.T @ u_.T

            # Fix reflection case
            if np.linalg.det(r_opt) < 0:
                vt_matrix[2, :] *= -1
                r_opt = vt_matrix.T @ u_

            t_opt = target_centroid - (r_opt @ source_centroid)
            return r_opt, t_opt

        r_total = np.eye(3)
        t_total = np.zeros(3)
        prev_error = float("inf")

        for _ in range(max_iterations):
            # Find nearest neighbors
            tree = cKDTree(tgt)
            _, indices = tree.query(src)
            nearest_neighbors = tgt[indices]

            # Compute transform
            r_iter, t_iter = svd_estimation(nearest_neighbors, src)

            # Apply transform to source (row-vector form)
            src = (src @ r_iter.T) + t_iter

            # Accumulate global transform
            r_total = r_iter @ r_total
            t_total = r_iter @ t_total + t_iter

            # Mean squared distance
            mse = np.mean(np.sum((nearest_neighbors - src)**2, axis=1))
            if abs(prev_error - mse) < tolerance:
                break
            prev_error = mse

        # Apply accumulated transform to original current_points
        final_aligned = (np.array(current_points) @ r_total.T) + t_total
        return final_aligned.tolist()



    def publish_accumulated_cloud(self, stamp):
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

    def publish_icp_merged_cloud(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = "odom"

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, self.icp_accumulated_points)
        self.icp_pub.publish(cloud_msg)
        self.get_logger().info("Published ICP merged cloud.")

def main(args=None):
    rclpy.init(args=args)
    node = PauseAndCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
