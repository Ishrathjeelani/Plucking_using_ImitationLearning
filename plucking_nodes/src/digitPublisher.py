#!/usr/bin/env python3
from digit import Digit
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# Digit serial numbers
thumb_serial = "D20781"
ring_serial = "D20778"
index_serial = "D20784"
middle_serial = "D20814"

def main():
    # Initialize the ROS node
    rospy.init_node("Digit_node")

    # Publishers for each Digit sensor
    thumb_pub = rospy.Publisher("/DigitThumb/color_image", Image, queue_size=1)
    # middle_pub = rospy.Publisher("/DigitMiddle/color_image", Image, queue_size=1)
    index_pub = rospy.Publisher("/DigitIndex/color_image", Image, queue_size=1)
    # ring_pub = rospy.Publisher("/DigitRing/color_image", Image, queue_size=1)

    # OpenCV to ROS Image bridge
    bridge = CvBridge()

    try:
        rospy.loginfo("Tactile sensing started")
        rate = rospy.Rate(200)

        #Connect to digit
        digit_thumb = Digit(thumb_serial)
        # digit_middle = Digit(middle_serial)
        digit_index = Digit(index_serial)
        # digit_ring = Digit(ring_serial)
        digit_thumb.connect()
        # digit_middle.connect()
        digit_index.connect()
        # digit_ring.connect()

        rospy.loginfo("Published digit images.")
        while not rospy.is_shutdown():  # Exit loop if ROS is shutting down
            try:
                # Wait for frames 
                frame_thumb = digit_thumb.get_frame()
                # frame_middle = digit_middle.get_frame()
                frame_index = digit_index.get_frame()
                # frame_ring = digit_ring.get_frame()

                # # Convert frames to OpenCV format
                # thumb_image = np.asanyarray(frame_thumb.get_data())
                # middle_image = np.asanyarray(frame_middle.get_data())
                # index_image = np.asanyarray(frame_index.get_data())
                # ring_image = np.asanyarray(frame_ring.get_data())

                # # Display the color and depth images
                # cv2.namedWindow("Thumb Image", cv2.WINDOW_AUTOSIZE)
                # cv2.imshow("thumb", frame_thumb)

                # cv2.namedWindow("Middle Image", cv2.WINDOW_AUTOSIZE)
                # cv2.imshow("Middle", frame_middle)

                # cv2.namedWindow("Index Image", cv2.WINDOW_AUTOSIZE)
                # cv2.imshow("index", frame_index)

                # Publish dept# Publish the images
                thumb_pub.publish(bridge.cv2_to_imgmsg(frame_thumb, encoding="bgr8"))
                # middle_pub.publish(bridge.cv2_to_imgmsg(frame_middle, encoding="bgr8"))
                index_pub.publish(bridge.cv2_to_imgmsg(frame_index, encoding="bgr8"))
                # ring_pub.publish(bridge.cv2_to_imgmsg(frame_ring, encoding="bgr8"))

                # # Exit on ESC key
                # if cv2.waitKey(1) == 27:
                #     rospy.loginfo("ESC pressed. Exiting loop...")
                #     break

                # rospy.loginfo("Published digit images.")
                rate.sleep()

            except Exception as e:
                rospy.logerr(f"Error during frame processing: {e}")
                break

    except Exception as e:
        rospy.logerr(f"Failed to start Digits: {e}")

    finally:
        cv2.destroyAllWindows()
        digit_thumb.disconnect()
        # digit_middle.disconnect()
        digit_index.disconnect()
        # digit_ring.disconnect()
        rospy.loginfo("Exiting...")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
