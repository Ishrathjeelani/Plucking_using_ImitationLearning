"""ROS Node for Teleoperation using a Glove."""
import rospy
import hydra
import numpy as np
from omegaconf import DictConfig
from std_msgs.msg import String
import os
from cv_bridge import CvBridge
from cetiglove.robot.allegroHand import Allegro
from cetiglove.robot.arm.wrapper import XArmAPI
from sensor_msgs.msg import Image
import torch
from ultralytics import YOLO 
import cv2
import sys
import builtins
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

class Teleoperation:
    """Interface for teleoperating a Robot using a glove."""

    def __init__(self, cfg):
        """
        Initalizing the required
        :param cfg: Configuration files containing relevant robot and glove parameters.
        """
        self.device = cfg.devices.device
        self.bridge = CvBridge()

        # Object variables
        self.line = None
        self.obj_x = None
        self.obj_y = None
        self.obj_z = None
        self.bb_x1 = None
        self.bb_y1 = None
        self.bb_x2 = None
        self.bb_y2 = None
        # Map back to original coordinates
        self.left_pad = None
        self.left_pad = None
        self.top_pad = None
        self.top_pad = None
        self.status = "Approach"

        # Set joint modes
        self.allegro = Allegro(hand_topic_prefix="allegroHand_0")
        self.allegro.set_joint_mode(0, "position")
        self.allegro.set_joint_mode(1, "torque")
        self.allegro.set_joint_mode(2, "torque")
        self.allegro.set_joint_mode(3, "position")
        self.allegro.set_joint_mode(4, "position")
        self.allegro.set_joint_mode(5, "torque")
        self.allegro.set_joint_mode(6, "torque")
        self.allegro.set_joint_mode(7, "position")
        self.allegro.set_joint_mode(8, "position")
        self.allegro.set_joint_mode(9, "torque")
        self.allegro.set_joint_mode(10, "torque")
        self.allegro.set_joint_mode(11, "position")
        self.allegro.set_joint_mode(12, "position")
        self.allegro.set_joint_mode(13, "position")
        self.allegro.set_joint_mode(14, "torque")
        self.allegro.set_joint_mode(15, "position")

        self.control_freq = cfg.control.control_freq
        self.hand_target = torch.from_numpy(np.array(cfg.control.hand_target_peace)).to(
            self.device
        )
        self.hand_lower = torch.from_numpy(np.array(cfg.control.hand_lower)).to(
            self.device
        )
        self.hand_upper = torch.from_numpy(np.array(cfg.control.hand_upper)).to(
            self.device
        )
        self.hand_lower_torq = torch.from_numpy(np.array(cfg.control.hand_lower_torq)).to(
            self.device
        )
        self.hand_upper_torq = torch.from_numpy(np.array(cfg.control.hand_upper_torq)).to(
            self.device
        )
        self.glove_max_value = np.array(cfg.control.glove_max_position)
        self.glove_min_value = np.array(cfg.control.glove_min_position)
        if cfg.control.arm_and_hand:
            self.arm = XArmAPI(cfg.robot_arm.xarm_ip)
            self.arm.motion_enable(enable=True)
            self.arm.clean_error()
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            self.arm.set_tcp_load(
                weight=cfg.robot_arm.xarm_weight,
                center_of_gravity=cfg.robot_arm.xarm_tcp,
                wait=True,
            )

    def glove_listener(self):
        """Initialize ROS subscriber for glove data."""
        rospy.Subscriber("/glove/RawData", String, self.glove_callback)

    def glove_callback(self, data):
        """Callback function to process incoming glove data."""
        self.line = data.data

    def low_pass_filter(self,new_value, previous_filtered_value):
        alpha = 0.1
        return alpha * new_value + (1 - alpha) * previous_filtered_value
    
    def butter_lowpass_filter(self, cutoff, fs, order=4):
        """Butterworth filter to filter raw sensor data from glove"""
        nyquist = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def map_yaw(self, xarm_new_pose):
        """Check and set yaw within bounds"""
        if xarm_new_pose[2]<-13:
            xarm_new_pose[2]=-13
        if xarm_new_pose[2]>78:
            xarm_new_pose[2]=78
        new_yaw = 50+(xarm_new_pose[2]+13)*100/(78+13)
        xarm_new_pose[2]=new_yaw
        return xarm_new_pose

    def set_arm_pose(self, rotx0, roty0, rotz0, filtRotx, filtRoty, filtRotz):
        """Set the new arm pose using previous previous pose and filtered sensor data"""
        _, xarm_current_position = self.arm.get_position()
        xarm_new_pose = [filtRoty, filtRotx, filtRotz] 
        xarm_new_pose = self.map_yaw(xarm_new_pose)
        xarm_new_pose = self.check_arm_pose(xarm_current_position[3:6], xarm_new_pose)
        self.arm.set_position( yaw=xarm_new_pose[2],speed=100, mvacc=50)
        rotz0=filtRotz
        return rotx0, roty0, rotz0

    def get_xArm_states(self):
        """Fetch the xArm states."""
        _, position = self.arm.get_position()
        _, joint_angles = self.arm.get_servo_angle()
        velocity = self.arm.realtime_joint_speeds#get_servo_angle_speed()
        _, torques = self.arm.get_joints_torque()
        endForce = self.arm.ft_ext_force
        # _, error_code = self.arm.get_error_code()

        return {
            "position_x": position[0],
            "position_y": position[1],
            "position_z": position[2],
            "roll": position[3],
            "pitch": position[4],
            "yaw": position[5],
            "joint_1_angle": joint_angles[0],
            "joint_2_angle": joint_angles[1],
            "joint_3_angle": joint_angles[2],
            "joint_4_angle": joint_angles[3],
            "joint_5_angle": joint_angles[4],
            "joint_6_angle": joint_angles[5],
            "joint_7_angle": joint_angles[6],
            "joint_1_velocity": velocity[0],
            "joint_2_velocity": velocity[1],
            "joint_3_velocity": velocity[2],
            "joint_4_velocity": velocity[3],
            "joint_5_velocity": velocity[4],
            "joint_6_velocity": velocity[5],
            "joint_7_velocity": velocity[6],
            "joint_1_torque": torques[0],
            "joint_2_torque": torques[1],
            "joint_3_torque": torques[2],
            "joint_4_torque": torques[3],
            "joint_5_torque": torques[4],
            "joint_6_torque": torques[5],
            "joint_7_torque": torques[6],
            "Fx": endForce[0],
            "Fy": endForce[1],
            "Fz": endForce[2]
        }

    def get_average_depth(self, depth_image):
        """
        Calculate the average depth of the pixels within the bounding box.
        """
        # Clip bounding box to image size (in case it's outside the image frame)
        x1, y1, x2, y2 = max(int(self.bb_x1), 0), max(int(self.bb_y1), 0), min(int(self.bb_x2), depth_image.shape[1]), min(int(self.bb_y2), depth_image.shape[0])
        
        # Extract the region of interest (ROI) based on the bounding box
        roi_depth = depth_image[y1:y2, x1:x2]

        # Avoid division by zero if the ROI is empty
        if roi_depth.size == 0:
            return float('nan')  # Return NaN for invalid depth

        # Calculate the average depth of the pixels in the ROI
        avg_depth = np.mean(roi_depth)
        
        # Return the depth in meters (assuming depth is in millimeters; convert to meters)
        return avg_depth  # Convert mm to meters

    def resize_and_pad(self,image, size=(640, 640)):
        """Resize ItenRealSense camera stream images"""
        h, w, _ = image.shape

        # Calculate padding
        self.top_pad = (size[1] - h) // 2
        self.bottom_pad = size[1] - h - self.top_pad
        self.left_pad = (size[0] - w) // 2
        self.right_pad = size[0] - w - self.left_pad

        # Add padding
        pad_image = cv2.copyMakeBorder(image, self.top_pad, self.bottom_pad, self.left_pad, self.right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return pad_image

    def detect_red_circle(self,hsv):
        """Detect Tomato and return x,y coordinates"""
        # Red mask (2 ranges for red)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        # Combine both red masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Noise removal
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        # Detect the largest circle
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            # print(radius)
            if radius > 25:  # Ignore small noise
                return float(x), float(y)

        return None, None
    
    def callback_colorImage(self, msg):#, model
        """Callback for Tomato detection and getting it's x,y coordinates"""
        CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for filtering detections
        CLASSES = ["b_fully_ripened", "b_half_ripened", "b_green", "l_fully_ripened", "l_half_ripened", "l_green"]
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")  # Convert ROS image to OpenCV image
            # Convert to HSV for better red detection
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            # # YOLO inference
            # sys.stdout = open(os.devnull, 'w')
            # sys.stderr = open(os.devnull, 'w')
            # original_print = builtins.print
            # builtins.print = lambda *args, **kwargs: None
            x, y = self.detect_red_circle(cv_image)
            self.obj_x = x
            self.obj_y = y

            # cv_image = cv2.resize(cv_image, (640, 640))

            # cv_image = self.resize_and_pad(cv_image)
        #     results = model(cv_image)  # Perform inference
        #     sys.stdout = sys.__stdout__ 
        #     sys.stderr = sys.__stderr__
        #     builtins.print = original_print
        #     detections = results[0].boxes.data.cpu().numpy()  # Extract detections
        #     for detection in detections:
        #         x1, y1, x2, y2, conf, cls = detection  # Bounding box and class
        #         if conf < CONFIDENCE_THRESHOLD:
        #             continue
        #         # Map back to original coordinates
        #         # x1 -= self.left_pad
        #         # x2 -= self.left_pad
        #         # y1 -= self.top_pad
        #         # y2 -= self.top_pad
        #         self.obj_x = (x1+x2)/2
        #         self.obj_y = (y1+y2)/2
        #         self.bb_x1 = x1
        #         self.bb_y1 = y1
        #         self.bb_x2 = x2
        #         self.bb_y2 = y2
        except Exception as e:
            rospy.logerr(f"Error saving image from color frame: {e}")

    def callback_depthImage(self, msg):
        """Callback for image messages."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")  # Convert ROS image to OpenCV image
            # Get the average depth of the detected object
            avg_depth = self.get_average_depth(cv_image)
            self.obj_z = avg_depth
        except Exception as e:
            rospy.logerr(f"Error saving image from depth frame: {e}")

    def operate(self, cfg):
        """Main teleoperation loop."""
        rospy.init_node("main_teleoperation_node")
        rospy.sleep(0.5)
        rospy.loginfo("Starting teleoperation...")
        rospy.Subscriber("/glove/RawData", String, self.glove_callback)
        rate = rospy.Rate(50)
        rospy.sleep(0.5)
        # Publishers for allegro hand and arm
        xArm_pub = rospy.Publisher("/xArm/States", String, queue_size=3)
        print("xArm publisher created")

        # Default values of initial finger positons
        pos_array0 = [0, 0, 0, 0]

        # End-effector positions of the xArm
        _,prevCurPosition = self.arm.get_position()
        prevCurPos = np.array(prevCurPosition[:3])

        # Default initialisation of glove readings
        pos_array=np.zeros(11)
        if not self.line==None:
            pos_array = self.string_to_array(self.line)

        # For pose and position of End-effector
        rotx0, roty0, rotz0 = pos_array[5], pos_array[6], pos_array[7]
        filteredX, filteredY, filteredZ = [], [], []
        rawX, rawY, rawZ = [], [], []

        # Butterworth filter params
        b, a = self.butter_lowpass_filter(0.05,1)
        # Initialize filter state (zi) for each axis
        zi_pos0 = np.zeros(len(b) - 1)
        zi_pos1 = np.zeros(len(b) - 1)
        zi_pos2 = np.zeros(len(b) - 1)
        zi_pos3 = np.zeros(len(b) - 1)
        zi_x = np.zeros(len(b) - 1)
        zi_y = np.zeros(len(b) - 1)
        zi_z = np.zeros(len(b) - 1)
        zi_rotx = np.zeros(len(b) - 1)
        zi_roty = np.zeros(len(b) - 1)
        zi_rotz = np.zeros(len(b) - 1)
        

        # # Approach phase block
        # # Move to home position
        # teleoperator.arm.set_servo_angle(angle=[1.6,-20.6,-3.7,88,-3.2,107.3,-46.5],wait=True) #[94.2,-2.1,-102.4,88.7,176.4,-82,127.4]
        
        # # # Load YoLo model
        # # print("Loading YOLO model")
        # # MODEL_PATH = "/home/bonggeeun/team_project/ceti-glove-main/cetiglove/tomatoModel.pt"  # Replace with your custom .pt model file
        # # model = YOLO(MODEL_PATH, verbose=False)
        # # Subscribe end-effector camera once (updates obj's x y and z)
        # rospy.Subscriber("/yolo/color_image", Image, teleoperator.callback_colorImage)#, callback_args=model
        # rospy.sleep(0.5)
        # print("Getting x and y")
        # # # Move to x,y 
        # # while None==teleoperator.obj_x:
        # #     try:
        # #         continue
        # #     except Exception as e:
        # #         rospy.logerr(f"Error during teleoperation: {e}")
        #     # print("Getting frame info")
        # moveX, moveY =0, 0
        # # Take mean of atleast 5 x and y
        # x_arr, y_arr = [],[]
        # for j in range(0,10):
        #     x_arr.append(teleoperator.obj_x)
        #     y_arr.append(teleoperator.obj_y)
        # print(np.mean(x_arr))
        # print(np.mean(y_arr))
        # moveX = 334 - (np.mean(y_arr)-211)*0.87 + 60
        # moveY = 196 - (np.mean(x_arr)-41)*0.76 + 40
        # objPos = [moveX, moveY]
        # print("x:{} y:{}".format(objPos[0],objPos[1]))

        # X = objPos[0]
        # Y = objPos[1]
        
        # # Set the depth 
        # teleoperator.arm.set_position(z=Z+100,speed=100, mvacc=10, wait=True) #update values
        # # Set predefined x, y
        # teleoperator.arm.set_position(x=objPos[0], y=objPos[1], speed=100, mvacc=10, wait=True) #update values
        # # Set predefined pose
        # teleoperator.arm.set_position(roll=100, pitch=-46, yaw=70,speed=50, mvacc=5, wait=True)

        # # Set the depth 
        # teleoperator.arm.set_position(z=Z,speed=100, mvacc=10, wait=True)

        user_input = input("Set the desired grasp pose, start subscriber and press any key to continue")

        rospy.loginfo("Taking Approch phase data")
        thumb, index, middle, ring = [], [], [], []
        # Update config
        ts=0
        try:
            while ts<5000:
                xArm_states = self.get_xArm_states()
                # Flatten states into a CSV-ready string
                xArm_state_values = [str(value) for value in xArm_states.values()]
                xArm_state_values.append(self.status)
                xArm_csv_line = ",".join(xArm_state_values)
                # Publish the CSV-ready string
                xArm_state_msg = String()
                xArm_state_msg.data = xArm_csv_line
                xArm_pub.publish(xArm_state_msg)
                ts+=1
                # rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("Configuration update interrupted by user.")
        finally:
            rospy.loginfo("Published approch phase data.")

        user_input = input("Start the simple Allegro script and press any key to continue")

        print("Start of Teleop, press ctrl+c when plucking is done")
        self.status="Plucking"
        stepFlag = "Pose"
        pullFlag=False
        pos_target0 = None
        while not rospy.is_shutdown():
            try:
                # print(self.line)
                if self.line:
                    pos_array = self.string_to_array(self.line)
                    # rospy.loginfo(f"Glove data: {pos_array}")

                # Get the coordinates for arm
                accn = pos_array[8:11]
                accnx, zi_x = lfilter(b, a, [accn[0]], zi=zi_x) #self.butter_lowpass_filter(accnx, 0.05, 1)#
                accny, zi_y = lfilter(b, a, [accn[1]], zi=zi_y)
                accnz, zi_z = lfilter(b, a, [accn[2]], zi=zi_z)
                accnNew = [accnx, accny, accnz]
                # print(accnz)

                # Get the end-effector orientation
                rotx, roty, rotz = pos_array[5], pos_array[6], pos_array[7]
                filtRotx, zi_rotx = lfilter(b, a, [rotx], zi=zi_rotx) #self.low_pass_filter(rotx00,rotx)
                filtRoty, zi_roty = lfilter(b, a, [roty], zi=zi_roty)
                filtRotz, zi_rotz = lfilter(b, a, [rotz], zi=zi_rotz)

                # Get current xArm positions
                _, xarm_current_position = self.arm.get_position()
                rotx0, roty0, rotz0=self.set_arm_pose(rotx0, roty0, rotz0, filtRotx, filtRoty, filtRotz)
                stepFlag="Pose"

                # Publishing arm states
                xArm_states = self.get_xArm_states()
                # Flatten states into a CSV-ready string
                xArm_state_values = [str(value) for value in xArm_states.values()]
                xArm_state_values.append(self.status)
                xArm_csv_line = ",".join(xArm_state_values)
                # Publish the CSV-ready string
                xArm_state_msg = String()
                xArm_state_msg.data = xArm_csv_line
                xArm_pub.publish(xArm_state_msg)

                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error during teleoperation: {e}")
                break

        rospy.loginfo("Teleoperation ended.")
        self.status="Retreat"
        self.arm.set_position(z=271,speed=100, mvacc=10, wait=True)
        self.arm.set_position(x=493,y=340,speed=500, mvacc=100, wait=True)
        # Publishing arm states
        xArm_states = self.get_xArm_states()
        # Flatten states into a CSV-ready string
        xArm_state_values = [str(value) for value in xArm_states.values()]
        xArm_state_values.append(self.status)
        xArm_csv_line = ",".join(xArm_state_values)
        # Publish the CSV-ready string
        xArm_state_msg = String()
        xArm_state_msg.data = xArm_csv_line
        xArm_pub.publish(xArm_state_msg)

        self.arm.disconnect()
        self.allegro.disconnect()

    @staticmethod
    def string_to_array(serial_string):
        """Convert string of sensor values to a numeric array."""
        res=[]
        for val in serial_string.split(":"):
            if not val.strip()=="":
                res.append(float(val.strip()))
        return res

    def get_arm_coordinates(self, curr, accn):
        """Get X, Y and Z positions using acceleration
        
        :param curr: Current position
        :param accn: sensor acceleration values
        :return: New positions
        """
        accn_lower_limit = 0.02#8
        accn_upper_limit = 10#30
        scale_f = 1
        posIncr = 20
        newPos = curr
        # accn = [accn[1], accn[0], accn[2]]
        # To get the xyz-coordinates
        for i in range(0,3):
            if accn[i]>=accn_lower_limit and accn[i]<=accn_upper_limit:
                newPos[i] = curr[i]+posIncr*scale_f
            elif accn[i]<=-accn_lower_limit and accn[i]>=-accn_upper_limit:
                newPos[i] = curr[i]+posIncr*scale_f #-
            else:
                # print("Noise ---- Position not commanded")
                newPos[i] = curr[i]
        return newPos

    def check_arm_coordinates(self, curPos,pos):
        """Check X, Y and Z bounds
        
        :param curPos: Current position
        :param pos: Next position
        :return: Position within bounds
        """
        x_min, x_max = 333,550
        y_min, y_max = -140,150
        z_min, z_max = 200, 704

        min_vals = [x_min, y_min, z_min]
        max_vals = [x_max, y_max, z_max]

        newPos=curPos.copy()
        for i in range(0,3):
            if pos[i]>min_vals[i] and pos[i]<max_vals[i]: #and 0!=flag:
                # print("Point within workspace")
                newPos[i]=pos[i]
            else:
                newPos[i]=curPos[i]
        return newPos

    def check_arm_pose(self, curPos,pos):
        """Check roll, pitch and yaw bounds
        
        :param curPos: Current pose
        :param pos: Next pose
        :return: Pose within bounds
        """
        roll_min, roll_max = 0, 100#-100, -60#-90, 10
        pitch_min, pitch_max = -60, 60
        yaw_min, yaw_max = 20, 150 #-100, -60

        min_vals = [roll_min, pitch_min, yaw_min]
        max_vals = [roll_max, pitch_max, yaw_max]

        for i in range(0,3):
            if pos[i]>min_vals[i] and pos[i]<max_vals[i]: #and 0!=flag:
                # print("Point within workspace")
                pos[i]=pos[i]
            elif pos[i]>max_vals[i]:
                pos[i]=max_vals[i]
            elif pos[i]<min_vals[i]:
                pos[i]=min_vals[i]
            else:
                pos[i]=curPos[i]

        return pos

    def pinchGrasp_finger_values(self, sensor_data):
        """
        Convert sensor values to joint data.

        :param sensor_data: Sensor data from the glove.
        :return: Torque values for each joint.
        """
        #  Format of sensor_data: [index, middle, annulary, thumb]

        min_values = np.array(str(self.glove_min_value).strip("[]").split(","))
        max_values = np.array(str(self.glove_max_value).strip("[]").split(","))
        min_values = min_values.astype(np.float64)
        max_values = max_values.astype(np.float64)

        hand_lowerPos_value = self.hand_lower
        hand_upperPos_value = self.hand_upper
        hand_lower_value = self.hand_lower_torq
        hand_upper_value = self.hand_upper_torq
        final_target = [
        -0.3, 1.5, 0.05, 0.05,  # Finger 1
        0.3, 1.5, 0.05, 0.05,  # Finger 2
        0.3, 1.5, 0.05, 0.05,  # Finger 3
        1.5, 0.05, 0.05, 0.05   # Thumb
    ]
        
        j1Ratio, j2Ratio = 0.5, 0.5
        newFormat = sensor_data
        sensor_dataN = [newFormat[1], newFormat[2], newFormat[3], newFormat[0]]
        for i in range(4):
            finger_value = sensor_dataN[i]
            idx = i*4+1
            final_target[idx] = (hand_lower_value[idx] + (hand_upper_value[idx] - hand_lower_value[idx]) * (finger_value - min_values[i]) / (max_values[i] - min_values[i]))*j1Ratio
            idx = i*4+2
            final_target[idx] = (hand_lower_value[idx] + (hand_upper_value[idx] - hand_lower_value[idx]) * (finger_value - min_values[i]) / (max_values[i] - min_values[i]))*j2Ratio
            
        # Fixing certain joints of the thumb
        final_target[12] = 1.5
        final_target[13] = 0.0
        return final_target


def update_config(cfg):
    """Update glove configuration based on live data."""
    rospy.loginfo("Updating configuration. Flex fingers at least 5 times...")
    thumb, index, middle, ring = [], [], [], []
    line = None

    def callback(data):
        nonlocal line
        line = data.data

    rospy.Subscriber("/glove/RawData", String, callback)
    rate = rospy.Rate(50)

    try:
        while not rospy.is_shutdown():
            if line:
                arrLine = line.split(":")
                thumb.append(int(arrLine[0].strip()))
                index.append(int(arrLine[1].strip()))
                middle.append(int(arrLine[2].strip()))
                ring.append(int(arrLine[3].strip()))
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("Configuration update interrupted by user.")
    finally:
        thumb_min, thumb_max = np.min(thumb), np.max(thumb)
        index_min, index_max = np.min(index), np.max(index)
        middle_min, middle_max = np.min(middle), np.max(middle)
        ring_min, ring_max = np.min(ring), np.max(ring)
        min_arr = [index_min, middle_min, ring_min, thumb_min]
        max_arr = [index_max, middle_max, ring_max, thumb_max]
        cfg.control.glove_min_position = str(min_arr)
        cfg.control.glove_max_position = str(max_arr)
        rospy.loginfo(f"Updated configuration: {cfg.control.glove_min_position}")

    return cfg


@hydra.main(config_name="teleoperation", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    """Main entry point for the combined teleoperation ROS node."""
    rospy.loginfo("Teleoperation node initialized.")

    # Start teleoperation
    teleoperator = Teleoperation(cfg)
    teleoperator.operate(cfg)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
