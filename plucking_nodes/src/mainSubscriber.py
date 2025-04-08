import rospy
import cv2
import os
import csv
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from datetime import datetime

class DataSaver:
    def __init__(self, csv_file, image_base_folder):
        # Prepare the CSV file for writing
        self.csv_file = open(csv_file, "a")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
    "timestamp",
    # xArm7 Joint States (7 joints: positions, velocities, and torques)
    "x_pos", "y_pos", "z_pos", "roll", "pitch", "yaw",
    "xarm_joint_1_position", "xarm_joint_2_position", "xarm_joint_3_position", "xarm_joint_4_position", 
    "xarm_joint_5_position", "xarm_joint_6_position", "xarm_joint_7_position",
    "xarm_joint_1_velocity", "xarm_joint_2_velocity", "xarm_joint_3_velocity", "xarm_joint_4_velocity", 
    "xarm_joint_5_velocity", "xarm_joint_6_velocity", "xarm_joint_7_velocity",
    "xarm_joint_1_torque", "xarm_joint_2_torque", "xarm_joint_3_torque", "xarm_joint_4_torque", 
    "xarm_joint_5_torque", "xarm_joint_6_torque", "xarm_joint_7_torque","Fx", "Fy", "Fz", "Stage",

    # Allegro Hand Finger States (16 joints: positions, velocities, and torques for 4 fingers and thumb)
    "thumb_1_position", "thumb_2_position", "thumb_3_position", "thumb_4_position",
    "index_1_position", "index_2_position", "index_3_position", "index_4_position",
    "middle_1_position", "middle_2_position", "middle_3_position", "middle_4_position",
    "ring_1_position", "ring_2_position", "ring_3_position", "ring_4_position",

    # Finger velocities
    "thumb_1_velocity", "thumb_2_velocity", "thumb_3_velocity", "thumb_4_velocity",
    "index_1_velocity", "index_2_velocity", "index_3_velocity", "index_4_velocity",
    "middle_1_velocity", "middle_2_velocity", "middle_3_velocity", "middle_4_velocity",
    "ring_1_velocity", "ring_2_velocity", "ring_3_velocity", "ring_4_velocity",

    # Finger torques
    "thumb_1_torque", "thumb_2_torque", "thumb_3_torque", "thumb_4_torque",
    "index_1_torque", "index_2_torque", "index_3_torque", "index_4_torque",
    "middle_1_torque", "middle_2_torque", "middle_3_torque", "middle_4_torque",
    "ring_1_torque", "ring_2_torque", "ring_3_torque", "ring_4_torque"
]
)  # CSV header

        self.image_base_folder = image_base_folder
        if not os.path.exists(image_base_folder):
            os.makedirs(image_base_folder)  # Create the folder if it does not exist

        self.bridge = CvBridge()  # For converting ROS images to OpenCV
        self.xarm_data = None  # Placeholder for xArm7 data
        self.allegro_data = None  # Placeholder for Allegro Hand data

    def callback_state(self, msg, source):
        """Callback for string messages (joint states)."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
        state_values = msg.data.split(",")  # Assuming CSV format in the message
        state_values.insert(0, timestamp)  # Insert timestamp at the start of the list

        # Store the data based on source (xArm or Allegro Hand)
        if source == "xarm":
            self.xarm_data = state_values
        elif source == "allegro":
            self.allegro_data = state_values

        # If both xArm and Allegro data are received, combine and save to CSV
        if self.xarm_data and self.allegro_data:
            combined_data = self.xarm_data + self.allegro_data[1:]  # Exclude the timestamp of Allegro (as it's already added)
            self.csv_writer.writerow(combined_data)  # Write the data to the CSV file
            rospy.loginfo(f"Saved combined data to CSV: {combined_data}")
            self.xarm_data = None  # Reset data after saving
            self.allegro_data = None  # Reset data after saving

    def callback_image(self, msg, camera_name):
        """Callback for image messages."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Current timestamp for filename
        camera_folder = os.path.join(self.image_base_folder, camera_name)
        if not os.path.exists(camera_folder):
            os.makedirs(camera_folder)  # Create a folder for each camera if it does not exist

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")  # Convert ROS image to OpenCV image
            image_filename = os.path.join(camera_folder, f"{camera_name}_image_{timestamp}.jpg")
            cv2.imwrite(image_filename, cv_image)  # Save the image
            rospy.loginfo(f"Saved image from {camera_name} to {image_filename}")
        except Exception as e:
            rospy.logerr(f"Error saving image from {camera_name}: {e}")

    def joint_states_callback(self,msg):
        global joint_states_data
        # Extract positions, velocities, and efforts
        positions = msg.position
        velocities = msg.velocity
        efforts = msg.effort
        timestamp = msg.header.stamp.to_sec()

        # Create a row with the timestamp, positions, velocities, and efforts
        row = [timestamp] + list(positions) + list(velocities) + list(efforts)
        self.allegro_data = row

        # Log the data
        rospy.loginfo(f"Received joint states: {row}")

    def start(self):
        """Start the subscriber node."""
        rospy.init_node("data_recorder_node")
        rospy.sleep(0.5)
        # xArm and allegro hand string msgs
        rospy.Subscriber("/xArm/States", String, self.callback_state, callback_args="xarm")
        # rospy.Subscriber("/allegro/States", String, self.callback_state, callback_args="allegro")
        rospy.Subscriber("/allegroHand_0/joint_states", JointState, self.joint_states_callback)
        rospy.sleep(0.5)
        # IntelRealsense camera streams
        rospy.Subscriber("/cameraEOF/color_image", Image, self.callback_image, callback_args="EOF_color")
        rospy.Subscriber("/cameraEOF/depth_image", Image, self.callback_image, callback_args="EOF_depth")
        rospy.Subscriber("/cameraWorld/color_image", Image, self.callback_image, callback_args="World_color")
        rospy.Subscriber("/cameraWorld/depth_image", Image, self.callback_image, callback_args="World_depth")
        rospy.sleep(0.5)
        # Digit streams
        rospy.Subscriber("/DigitThumb/color_image", Image, self.callback_image, callback_args="Digit_thumb")
        # rospy.Subscriber("/DigitMiddle/color_image", Image, self.callback_image, callback_args="Digit_middle")
        rospy.Subscriber("/DigitIndex/color_image", Image, self.callback_image, callback_args="Digit_index")
        # rospy.Subscriber("/DigitRing/color_image", Image, self.callback_image, callback_args="Digit_ring")
        rospy.sleep(0.5)
        rospy.loginfo("Data saver node started. Saving data to CSV and images to folder.")
        rospy.spin()  # Keep the node running


if __name__ == "__main__":
    try:
        # Define the CSV file and image folder paths
        csv_file = r"/home/bonggeeun/Ishrath/Experiment/Exp1.csv"  # Path to save CSV file
        image_folder = r"/home/bonggeeun/Ishrath/Experiment/imagesExp1"  # Folder to save images

        # Start the DataSaver node
        data_saver = DataSaver(csv_file, image_folder)
        data_saver.start()
    except rospy.ROSInterruptException:
        rospy.loginfo("Data saver node interrupted.")
