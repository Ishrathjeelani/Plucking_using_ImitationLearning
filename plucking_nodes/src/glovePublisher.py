#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import numpy as np
import serial

def main():
    # Initialize the ROS node
    rospy.init_node("Glove_node")
    # Publishers for raw data
    glove_pub = rospy.Publisher("/glove/RawData", String, queue_size=10)
    rospy.loginfo("Published raw glove data")
    try:
        # Initialise serial communication
        ser = serial.Serial("/dev/ttyUSB0", 230400)
        rate = rospy.Rate(50)
        rospy.loginfo("Ceti Glove started successfully.")

        # Main loop
        while not rospy.is_shutdown():
            # Read the line
            line = ser.readline()
            strLine = line.decode('utf-8').strip('b\n')

            # Publish string msg
            glove_pub.publish(strLine)

            # rospy.loginfo("Published raw glove data")
            rate.sleep()

    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")

    finally:
        # Stop the pipeline and clean up
        ser.close()
        rospy.loginfo("Stopped recording glove data")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


