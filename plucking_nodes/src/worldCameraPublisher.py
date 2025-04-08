#!/usr/bin/env python3
import rospy
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# serial="243522071155"
serial="244222076079"

def main():
    # Initialize the ROS node
    rospy.init_node("World_camera_node")
    rate = rospy.Rate(200)

    # Publishers for color and depth images
    color_pub = rospy.Publisher("/cameraWorld/color_image", Image, queue_size=3)
    depth_pub = rospy.Publisher("/cameraWorld/depth_image", Image, queue_size=3)

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
        rospy.loginfo("Published world color and depth images.")

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

            rate.sleep()

    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")

    finally:
        # Stop the pipeline and clean up
        pipeline.stop()
        rospy.loginfo("RealSense pipeline stopped.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


