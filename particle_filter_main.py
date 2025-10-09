# Particale filter (mcl) implementation with static, auto latch, and manual selection
import numpy as np
import cv2
import math
import random
from PIL import Image
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Quaternion, PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros

# ************************************************
#           IMPORTANT DETAILS BELOW             #
# ************************************************
"""
This starter code provides code base to implement particle filter based localization,
 also known as MonteCarlo Localization (MCL)
Parts of this assignment depends on your HW5 solutions

--- TBD --- and ... indicates the section to fill out. There are 9 TBD sections to fill out

... indicates the specific place to type the code

Additional notes:

Using MCL to Localize the Robot in the Arena

Monte Carlo Localization (MCL) is a powerful but complex algorithm that often requires extensive tuning and
reliable convergence detection. In our setup, we provide a standard deviation-based auto-anchoring mechanism
to automatically localize (i.e., align the robot’s base frame to the global map frame). However, due to the
presence of repetitive structures in the arena, MCL can struggle to uniquely localize the robot.

To address this, we offer a manual initialization function that allows you to provide a rough pose estimate
if the automatic localization fails to converge. We strongly recommend using the manual method first to get
your MCL node functioning correctly before enabling auto-anchoring.

use the flag " self.use_auto_anchor" to enable/ disable this feature

In summary:

Auto vs Manual Anchoring in MCL

# When self.use_auto_anchor = True

The MCL node will attempt to automatically localize the robot without any external input. Over time, the particles
will converge to the robot’s true pose, assuming enough motion and distinctive features in the environment.

    - You should see the estimated MCL pose (yellow arrows in RViz) aligning with the true robot location and orientation.
    - The live LiDAR scan should accurately overlap with the map, indicating successful convergence.

# When self.use_auto_anchor = False

Once all nodes are running, you must use the “2D Pose Estimate” button in RViz to manually provide an approximate
initial pose of the robot.

    - The code will sample particles around the provided pose, allowing for quicker and more reliable convergence of MCL.
    - This is the recommended approach, especially in environments with symmetry or repeated structures,
      where auto-localization may fail or take longer.
"""


class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight


class AutoAnchorManager:
    """
    No TBD Items in this helper class
    """

    def __init__(self, node: Node, pf_node):
        self.node = node
        self.pf = pf_node
        self.anchor_attempted = False

    def check_and_anchor(self):
        """
        constantly checks for mcl convergence if this feature is enabled
        """
        if not self.pf.use_auto_anchor:
            self.node.get_logger().warn("Auto anchoring is disabled ...", once=True)
            return
        if self.anchor_attempted or not self.pf.last_odom or not self.pf.particles:
            return
        if not self.pf.has_converged():
            return

        x, y, theta = self.pf.estimate_pose()
        self.pf.set_map_to_odom_transform(x, y, theta)
        self.anchor_attempted = True
        self.node.get_logger().info("Anchored map to odom from converged estimate.")


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')

# --------------- TBD ---------------
        self.num_particles = 100 # try a value between 30 -120
# --------------- TBD -END  ---------------
        self.particles = []
        self.map = None
        self.map_info = None
        self.last_odom = None
        self.distance_map = None
        self.T_map_odom = None
        self.map_to_odom_set = False
        self.publish_tf_enabled = False
        self.use_auto_anchor = False  # use this to enable/ disable auto anchoring
        self.visualize_dt = False     # visualize distance transform

        self.anchor_manager = AutoAnchorManager(self, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ------------ TBD -------------------
        # write the appropriate QoS profile for the publishing MAP. this requires additional durability policy
        # refer to the lecture for more details
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.SYSTEM_DEFAULT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1

        )
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.SYSTEM_DEFAULT,
            depth=10
        )
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.SYSTEM_DEFAULT,
            depth=10
        )

        # Create four subsribers, subscribing to
        # 1) /map, 2) /odom, 3) /scan, 4) /initialpose,
        # with the appropriate QoS and callbacks for each
        self.mab_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, map_qos)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, odom_qos)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, sensor_qos)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initialpose_callback, odom_qos)
        # Create a PoseArray publisher on the particle_cloud topic
        self.particle_pub = self.create_publisher(PoseArray, "particle_cloud", sensor_qos) #make sure this is right qos
        self.create_timer(0.05, self.publish_tf)
        self.create_timer(1.0, self.anchor_manager.check_and_anchor)
        # ------------ TBD - END -------------------
    def map_callback(self, msg: OccupancyGrid):
        """
        subscribes to occupancy grid map
        """
        self.map = msg.data
        self.map_info = msg.info
        self.compute_distance_map()
        self.init_particles()

    def compute_distance_map(self):
        """
        uses cv2 distance transform function to compute distance transform
        """
        width = self.map_info.width
        height = self.map_info.height
        occ_grid = np.array(self.map, dtype=np.uint8).reshape((height, width))
        binary_occupancy = np.where(occ_grid >= 65, 1, 0).astype(np.uint8)  # this is the expected input format
        dist_map = cv2.distanceTransform(1 - binary_occupancy, cv2.DIST_L2, 5)
        if self.visualize_dt:
            plt.imshow(dist_map)
            plt.show()
        self.distance_map = dist_map * self.map_info.resolution
        self.get_logger().info("Distance map computed.")

    def init_particles(self):
        """
        mcl has more chance of convergence when the particles are properly initialized. There are sections of map
        that are either free, unexplored or obstacles. This function initializes the particles randomly but
        only in the free region using the value of the pree pixels
        """
        self.particles.clear()
        res = self.map_info.resolution
        ox, oy = self.map_info.origin.position.x, self.map_info.origin.position.y
        width, height = self.map_info.width, self.map_info.height
        # -------------------- TBD --------------------------
        """
        provide location to your map file. it directly reads from the source.
        the map server doesn't load image data
        """

        # Experiment with this path, as the absolute path might be the easiest to do from your root directory or may have to
        # use the relative path from the current directory you are running the node from or the file itself
        map_image = Image.open('map.pgm').convert('L')
        map_data = map_image.load()

        valid_cells = [(ox + (x + 0.5) * res, oy + (y + 0.5) * res)
                       for y in range(height) for x in range(width)
                       if map_data[x, height - y - 1] >= 250]   # write the pixel value for "free" grid. refer to lecture notes
        # -------------------- TBD -END ---------------------

        for _ in range(self.num_particles):
            x, y = random.choice(valid_cells)
            theta = random.uniform(-math.pi, math.pi)
            self.particles.append(Particle(x, y, theta, 1.0 / self.num_particles))
        self.get_logger().info("Particles initialized from map")

    def odom_callback(self, msg: Odometry):
        """
        This function subscribes to the /odom topic for latest odometry. This is where we compute the prior
        using the motion model
        x't ∼ p(xt ∣ ut,x't−1)

        Computes the relative motion between the previous and current odometry poses using
        SE(2) homogeneous transformation matrices.

        Let T_prev and T_curr be 3x3 matrices representing the poses:

            T_prev = [ R(theta1) | [x1, y1]^T ]
                      [   0     |     1     ]

            T_curr = [ R(theta2) | [x2, y2]^T ]
                      [   0     |     1     ]

        Then the relative motion in the local frame of T_prev is:

            T_rel = inv(T_prev) @ T_curr

        This gives us dx, dy, and dtheta in the robot's local frame, which we
        apply as the motion update to all particles.
        The change in heading should be normalized to [-pi, pi].
        """

        if self.last_odom is None:
            self.last_odom = msg
            return
        #---------------------- TBD ----------------------------
        '''
        Extract x1, y1, th1, x2, y2, th2 from the last and recent odom message respectively
        Make the SE(2) homogenous matrices representing their movement
        Compute the relative motion model from above
        Extract dx, dy from T_rel
        dtheta = math.atan2(sin, cos) from T_rel
        '''

        x1 = self.last_odom.pose.pose.position.x
        y1 = self.last_odom.pose.pose.position.y
        th1 = self.get_yaw(self.last_odom.pose.pose.orientation)

        x2 = msg.pose.pose.position.x
        y2 = msg.pose.pose.position.y
        th2 = self.get_yaw(msg.pose.pose.orientation)
        T_prev = np.array([[np.cos(th1), -np.sin(th1), x1], [np.sin(th1), np.cos(th1), y1], [0, 0, 1]])
        T_curr = np.array([[np.cos(th2), -np.sin(th2), x2], [np.sin(th2), np.cos(th2), y2], [0, 0, 1]])
        T_rel = np.linalg.inv(T_prev) @ T_curr
        dx = T_rel[0, 2]
        dy = T_rel[1, 2]
        dtheta = math.atan2(T_rel[1, 0], T_rel[0, 0])

        # ---------------------- TBD ----------------------------

        self.motion_update((dx, dy, dtheta))
        self.last_odom = msg


    def motion_update(self, u):
        """
        Applies the given motion update (dx, dy, dtheta) to each particle. i.e. the priors
        The motion is in the local frame of the robot. Use homogeneous
        transformation matrices to apply this motion to each particle.

        For each particle with pose T = [R(theta), t], we compute:
            T_new = T @ T_delta

        where T_delta is the motion transform built from (dx, dy, dtheta)
        with added Gaussian noise.
        """
        # ---------------------- TBD -----------------------
        dx, dy, dtheta = u
        for p in self.particles:
            ndx = dx + random.gauss(0, 0.01)  # add random noise (unit is meters)
            ndy = dy + random.gauss(0, 0.01)
            ndtheta = dtheta + random.gauss(0, 0.01)

            T_particle = np.array([
                [np.cos(p.theta), -np.sin(p.theta), p.x],
                [np.sin(p.theta), np.cos(p.theta), p.y],
                [0, 0, 1]
            ])
            T_delta = np.array([
                [np.cos(ndtheta), -np.sin(ndtheta), ndx],
                [np.sin(ndtheta), np.cos(ndtheta), ndy],
                [0, 0, 1]
            ])

            # T_new = T @ T_delta
            T_new = T_particle @ T_delta

            p.x = T_new[0, 2]
            p.y = T_new[1, 2]
            p.theta = math.atan2(T_new[1, 0], T_new[0, 0])
        #------------------------- TBD -END --------------------

    def scan_callback(self, msg: LaserScan):
        """
        This function computes the likelihood
        """
        if self.distance_map is None:
            return

        for p in self.particles:
            p.weight = self.measurement_likelihood(p, msg)

        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total
        else:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

        self.resample()
        self.publish_particles()


    def measurement_likelihood(self, p, scan_msg, sigma=0.01):

        """
        Compute likelihood here.
        Compute weights for each particle here
        refer to lecture notes for the equations but something like
        wt[i]=p(zt ∣xt[i])

        Steps:
        1. Initialize total weight w = 1.0
        2. For each valid beam:
           a. Project beam endpoint in world frame using p.x, p.y, p.theta
           b. Convert world coords to map indices (mx, my)
           c. Query distance transform map to get d = distance to nearest obstacle
           d. Compute likelihood using: likelihood = exp(-d^2 / (2 * sigma^2))
           e. Clamp likelihood using max(likelihood, small value ike 0.00001) to prevent collapse
           f. Multiply into total weight: w *= likelihood
        3. Return total weight w for this particle

        Notes:
        - If endpoint is out of bounds, use penalty likelihood (precomputed for d = 0.5 - 1.0)
        - sigma defines how sharp the sensor model is. default is 0.1 adjust between 0.1 and 0.5
        """
        w = 1.0
        angle = scan_msg.angle_min # this is the no. of lidar beams. in total around 220. Set it between range
                                   # 1 and 10. 1 = consider all beams. more beams = more compute time
        #------------------------ TBD --------------------------
        for r in scan_msg.ranges[::1]:
            if scan_msg.range_min < r < scan_msg.range_max:
                T_p = np.array([
                [np.cos(p.theta), -np.sin(p.theta), p.x],
                [np.sin(p.theta), np.cos(p.theta), p.y],
                [0, 0, 1]
                ])
                x = r * np.cos(angle) # r is the distance reported by each scan, so we want to extract the x and y components using the angle
                y = r * np.sin(angle) # note that we need to use the particle's x and y position to transform this to world coordinates
                global_x = (T_p @ np.array([x, y, 1]))[0]
                global_y = (T_p @ np.array([x, y, 1]))[1]
                mx = int((global_x -self.map_info.origin.position.x) / 0.01) # it is in pixel unit so divide by resolution and should be int type
                my = int((global_y-self.map_info.origin.position.y) / 0.01) # similar as above
                # check if it falls within the map or not
                if 0 <= mx < self.distance_map.shape[1] and 0 <= my < self.distance_map.shape[0]:
                    d = self.distance_map[my, mx] #my is the row and mx is the col (think coordinate plane)
                    likelihood = np.exp(-np.square(d) / (2 * np.square(sigma)))
                else:
                    # if not some other value. try values between 0.5 and 1
                    likelihood = np.exp(-np.square(0.5) / (2 * np.square(sigma)))# similar expression to 2.d but d is fixed to a number
                w *= max(likelihood, 0.00001)  # a small value here... just so that the weights are not too small and collapse
            angle += scan_msg.angle_increment
        # ------------------------ TBD-END --------------------------
        return w

    def resample(self):
        """
        Performs systematic resampling to generate a new particle set
        based on the current weights.

        Steps:
        1. Normalize particle weights so they sum to 1
        2. Compute cumulative sum of weights (CDF)
        3. Generate N evenly spaced positions in [0, 1) with random offset
        4. Use CDF to select which particles to duplicate based on positions
        5. Create new particles by copying selected ones (with reset weights)

           Why use CDF in practice?
            The CDF converts discrete particle weights into intervals on the [0, 1] line.
            This allows us to map random numbers to particles proportionally to their weights.
            Sampling from the CDF ensures that higher-weight particles are selected more often,
            and lower-weight ones may be skipped — as required by importance sampling.

            example: if weights = [0.2, 0.5, 0.3]
                            CDF = [0.2, 0.7, 1.0]
                            CDF segment of the probability owned
                            0.2 for particle 0 with w = 0.2  = [0 to 0.2] range
                            0.7 for particle 1 with w = 0.7  = [0.2 to 0.7] range
                            0.3 for particle 2 with w = 0.2 = [0.7 to 1.0] range

                            since the largest weight is 0.5 it also covers the widest range = more chances of selection

                            Now, say we draw a random number 0.65.
                            It falls between 0.2 and 0.7 → we select Particle 1 to ge replicated

        This ensures that high-weight particles are likely duplicated
        multiple times and low-weight ones may disappear.
        """

        #------------------------- TBD ------------------------------
        weights = np.array([p.weight for p in self.particles])
        weights /= np.sum(weights)
        N = self.num_particles
        positions = (np.arange(N) + random.random()) / N
        cumulative_sum = np.cumsum(weights)
        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = [self.particles[i] for i in indexes] # make sure this works


        #------------------------- TBD-END ------------------------------

    def estimate_pose(self):

        """
        Computes the expected robot pose (x, y, theta) as a weighted average
        over all particles. For theta, we compute a weighted average of sine and
        cosine components to properly handle angular wraparound.

        Note: all particles are available under self.particles

        Let T_i = [x_i, y_i, theta_i] be the pose of particle i.
        The estimated mean pose is:

            x = sum(w_i * x_i)
            y = sum(w_i * y_i)
            theta = atan2(sum(w_i * sin(theta_i)), sum(w_i * cos(theta_i)))

        where w_i are the normalized particle weights.

        For the following lines of code
        step 1: find total weights
        step 2: Normalize the weight
        Step 3-6: we're assuming vectorized code to compute x, y, sin_sum, cos_sum for faster processing
                  however, you may use break it to multiple lines of code
        Step 7: atan2 between sin_sum, and cos_sum for proper angle wrapup
        """

        #----------------------- TBD -----------------------
        weights = np.array([p.weight for p in self.particles]) # use self.particles
        weights /= np.sum(weights)
        x = sum(weights[i] * self.particles[i].x for i in range(len(self.particles)))
        y = sum(weights[i] * self.particles[i].y for i in range(len(self.particles)))
        sin_sum = sum(weights[i] * np.sin(self.particles[i].theta) for i in range(len(self.particles)))
        cos_sum = sum(weights[i] * np.cos(self.particles[i].theta) for i in range(len(self.particles)))
        theta = math.atan2(sin_sum, cos_sum)
        return x, y, theta

        # ----------------------- TBD-END -----------------------

    def has_converged(self, threshold=0.05):
        """
        This is part of auto anchor to check for mcl convergence
        """
        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        thetas = np.array([p.theta for p in self.particles])
        return (np.std(xs) < threshold and np.std(ys) < threshold and np.std(thetas) < (math.pi / 6))

    def set_map_to_odom_transform(self, x, y, theta):
        """
        Transform odom to map

        In HW5, we used a static tf to publish 0,0,0,0,0,0 (x,y,x, roll, pitch,yaw) angle between /odom and /map
        This was necessary because while building the map, we didn't have a tf between map and odom and all maps are
        published to map frame.

        In robotics, it is usually a SLAM/ localization node that publishes the tf between /odom and /map frames.
        /map will be fixed to the origin of the map
        /odom will be where the robot starts/ powers for the first time
        /base_link is the moving frame of the robot w.r.t the /map as a more stable frame

        This is an important concept

        """

        # ----------------------- TBD -----------------------
        ox = self.last_odom.pose.pose.position.x
        oy = self.last_odom.pose.pose.position.y

        otheta = self.get_yaw(self.last_odom.pose.pose.orientation)

        # for the homogeneous matrix using the theta or otheta angle
        T_map_base = np.array([[np.cos(theta), -np.sin(theta), x],  # x or ox?
                               [np.sin(theta), np.cos(theta), y],   # y or oy?
                               [0, 0, 1]])
        T_odom_base = np.array([[np.cos(otheta), -np.sin(otheta), ox], # x or ox?
                                [np.sin(otheta), np.cos(otheta), oy],  # y or oy?
                                [0, 0, 1]])
        # T_map_odom =T_map_base @ T_base_odom or T_map_odom =T_map_base @ inverse(T_odom_base)
        self.T_map_odom = T_map_base @ np.linalg.inv(T_odom_base)

        self.map_to_odom_set = True
        # ----------------------- TBD-END -----------------------

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        """
        This uses the "2D Pose estimate" tool from Rivz to give better initial estimat for the mcl
        """
        self.publish_tf_enabled = True
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.get_yaw(msg.pose.pose.orientation)
        # adding random noise to the given pose
        # ------------------ TBD --------------------
        self.particles = [Particle(random.gauss(x, 0.2),  # try same value on all (0.1 to 0.3)
                                   random.gauss(y, 0.2),  # same here
                                   random.gauss(theta, 0.2),
                                   1.0 / self.num_particles)
                          for _ in range(self.num_particles)]
        # ------------------ TBD-END --------------------
        if self.last_odom:
            self.set_map_to_odom_transform(x, y, theta)

    def publish_particles(self):

        """
        published the particle for visualization
        """
        msg = PoseArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for p in self.particles:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.orientation = self.yaw_to_quaternion(p.theta)
            msg.poses.append(pose)
        self.particle_pub.publish(msg)

    def publish_tf(self):
        """
        As indicated under self.set_map_to_odom_transform()
        This is where and how we publish tf between /odom and /map

        """
        if not self.map_to_odom_set:
            return
        dx = self.T_map_odom[0, 2]
        dy = self.T_map_odom[1, 2]
        dtheta = math.atan2(self.T_map_odom[1, 0], self.T_map_odom[0, 0])
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = dx
        t.transform.translation.y = dy
        q = quaternion_from_euler(0, 0, dtheta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def get_yaw(self, orientation):
        """
        helper function to convert quaternion orientation to euler yaw
        """
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        return euler_from_quaternion(q)[2]

    def yaw_to_quaternion(self, yaw):
        """
        helper function to convert  euler yaw to quaternion
        """
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.z = math.sin(yaw / 2.0)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
