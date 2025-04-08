#!/usr/bin/env python3
import rospy
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

serial="244222073574"#"244222073574"

def main():
    # Initialize the ROS node
    rospy.init_node("EOF_camera_node")
    rate = rospy.Rate(200)
    # Publishers for color and depth images
    color_pub = rospy.Publisher("/yolo/color_image", Image, queue_size=1)
    depth_pub = rospy.Publisher("/yolo/depth_image", Image, queue_size=1)

    # OpenCV to ROS Image bridge
    bridge = CvBridge()

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        # Start the RealSense pipeline
        pipeline.start(config)
        rospy.loginfo("RealSense pipeline for YOLO started successfully.")

        # Main loop
        while not rospy.is_shutdown():
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                rospy.logwarn("Incomplete frames received. Skipping...")
                continue

            # Convert frames to OpenCV format
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert depth to colormap for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Publish dept# Publish the images
            color_pub.publish(bridge.cv2_to_imgmsg(color_image, encoding="bgr8"))
            depth_pub.publish(bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8"))

            # rospy.loginfo("Published eye color and depth images.")
            rate.sleep()

    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")

    finally:
        # Stop the pipeline and clean up
        pipeline.stop()
        rospy.loginfo("yolo pipeline stopped.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


# import threading

# # Define camera serial numbers
# CAMERA_SERIALS = ["244222073574", "243522071155"]  # Replace with your camera serial numbers

# # Define the topic names for each camera
# TOPICS = {
#     "244222073574": {"color": "/cameraEOF/color_image", "depth": "/cameraEOF/depth_image"},
#     "243522071155": {"color": "/cameraWorld/color_image", "depth": "/cameraWorld/depth_image"},
# }

# # A function to handle a single RealSense camera
# def handle_camera(serial, topics):
#     rospy.loginfo(f"Starting camera {serial}")
    
#     # Create a pipeline for this camera
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_device(serial)  # Bind the pipeline to this specific camera
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
#     # Start the pipeline
#     pipeline.start(config)

#     # Create a ROS publisher and CvBridge for this camera
#     color_pub = rospy.Publisher(topics["color"], Image, queue_size=10)
#     depth_pub = rospy.Publisher(topics["depth"], Image, queue_size=10)
#     bridge = CvBridge()

#     try:
#         rate = rospy.Rate(10)  # 10 Hz
#         while not rospy.is_shutdown():
#             # Get frames
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()

#             if not color_frame or not depth_frame:
#                 rospy.logwarn(f"Incomplete frame received from camera {serial}. Skipping...")
#                 continue

#             # Convert frames to OpenCV format
#             color_image = np.asanyarray(color_frame.get_data())
#             depth_image = np.asanyarray(depth_frame.get_data())

#             # Convert depth to colormap for visualization
#             depth_colormap = cv2.applyColorMap(
#                 cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
#             )

#             # Publish the images
#             color_pub.publish(bridge.cv2_to_imgmsg(color_image, encoding="bgr8"))
#             depth_pub.publish(bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8"))
            
#             rospy.loginfo(f"Published images from camera {serial}")

#             rate.sleep()

#     except Exception as e:
#         rospy.logerr(f"Error with camera {serial}: {e}")

#     finally:
#         rospy.loginfo(f"Stopping camera {serial}")
#         pipeline.stop()

# def main():
#     rospy.init_node("multi_realsense_publisher")
    
#     # Create threads for each camera
#     threads = []
#     for serial in CAMERA_SERIALS:
#         t = threading.Thread(target=handle_camera, args=(serial, TOPICS[serial]))
#         t.start()
#         threads.append(t)
    
#     # Wait for threads to finish
#     for t in threads:
#         t.join()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass


