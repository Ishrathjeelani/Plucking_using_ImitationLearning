#!/usr/bin/env python3
import torch
import numpy as np
from std_msgs.msg import String
import rospy
from scipy.signal import butter, lfilter
from cetiglove.robot.allegroHand import Allegro 

line = None
glove_min_value = None
glove_max_value = None
hand_lower_torq = torch.from_numpy(np.array([-0.4, -0.4, -0.4, -0.4,
     -0.4, -0.4, -0.4, -0.4,
     -0.4, -0.4, -0.4, -0.4,
     -0.4, -0.4, -0.4, -0.4])).to("cpu")
hand_upper_torq = torch.from_numpy(np.array([0.45, 0.45, 0.45, 0.45,
     0.45, 0.45, 0.45, 0.45,
     0.45, 0.45, 0.45, 0.45,
     0.45, 0.45, 0.45, 0.45])).to("cpu")

def butter_lowpass_filter(cutoff, fs, order=3):
    """Butterworth filter to filter raw sensor data from glove"""
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def string_to_array(serial_string):
        """Convert string of sensor values to a numeric array."""
        res=[]
        for val in serial_string.split(":"):
            if not val.strip()=="":
                res.append(float(val.strip()))
        return res

def pinchGrasp_finger_values(sensor_data):
        """
        Convert sensor values to joint data.

        :param sensor_data: Sensor data from the glove.
        :return: Torque values for each joint.
        """
        #  Format of sensor_data: [index, middle, annulary, thumb]
        global glove_max_value, glove_min_value, hand_lower_torq, hand_upper_torq
        min_values = np.array(str(glove_min_value).strip("[]").split(","))
        max_values = np.array(str(glove_max_value).strip("[]").split(","))
        min_values = min_values.astype(np.float64)
        max_values = max_values.astype(np.float64)
        hand_lower_value = hand_lower_torq
        hand_upper_value = hand_upper_torq
        final_target = np.zeros(
            [
                16,
            ],
            dtype=np.float64,
        )
        
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

def glove_callback(data):
        """Callback function to process incoming glove data."""
        global line
        line = data.data

def main():
    global line, glove_min_value, glove_max_value, hand_lower_torq, hand_upper_torq
    rospy.init_node('simple_allegro')
    rospy.sleep(0.5)
    # Initialize the Allegro hand client
    hand = Allegro(hand_topic_prefix="allegroHand_0")
    rospy.Subscriber("/glove/RawData", String, glove_callback)
    # Set joint modes
    hand.set_joint_mode(0, "torque")#position
    hand.set_joint_mode(1, "torque")
    hand.set_joint_mode(2, "torque")
    hand.set_joint_mode(3, "torque")#position
    hand.set_joint_mode(4, "position")
    hand.set_joint_mode(5, "torque")
    hand.set_joint_mode(6, "torque")
    hand.set_joint_mode(7, "position")
    hand.set_joint_mode(8, "position")
    hand.set_joint_mode(9, "torque")
    hand.set_joint_mode(10, "torque")
    hand.set_joint_mode(11, "position")
    hand.set_joint_mode(12, "torque")#position
    hand.set_joint_mode(13, "position")
    hand.set_joint_mode(14, "torque")
    hand.set_joint_mode(15, "position")
    # Configure the Min and Max glove values
    print("Updating configuration. Flex fingers at least 5 times...")
    thumb, index, middle, ring = [], [], [], []
    ts=0
    try:
        while ts<10000000:
            if line:
                arrLine = line.split(":")
                thumb.append(int(arrLine[0].strip()))
                index.append(int(arrLine[1].strip()))
                middle.append(int(arrLine[2].strip()))
                ring.append(int(arrLine[3].strip()))
            ts+=1
            # rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("Configuration update interrupted by user.")
    finally:
        thumb_min, thumb_max = np.min(thumb), np.max(thumb)
        index_min, index_max = np.min(index), np.max(index)
        middle_min, middle_max = np.min(middle), np.max(middle)
        ring_min, ring_max = np.min(ring), np.max(ring)
        min_arr = [index_min, middle_min, ring_min, thumb_min]
        max_arr = [index_max, middle_max, ring_max, thumb_max]
        glove_min_value = str(min_arr)
        glove_max_value = str(max_arr)
        # print(glove_max_value)

    # Range of values
    span = index_max-index_min
    
    # Butterworth filter params
    b, a = butter_lowpass_filter(0.01,1)
    # Initialize filter state (zi) for each axis
    zi_pos0 = np.zeros(len(b) - 1)
    zi_pos1 = np.zeros(len(b) - 1)
    zi_pos2 = np.zeros(len(b) - 1)
    zi_pos3 = np.zeros(len(b) - 1)

    k=1
    flag=True
    t1=np.zeros(
            [
                16,
            ],
            dtype=np.float64,
        )
    
    pos0,pos=t1,t1
    t11,t14,t21,t22,t24=0,0,0,0,0
    while not rospy.is_shutdown():

        pos_array = string_to_array(line)
        # Compute desired joint torque
        if pos_array[1]>=0 and pos_array[1]<index_min+span*1/10:
            val=0.1+0.1*0.4
        elif pos_array[1]>=index_min+span*1/10 and pos_array[1]<index_min+span*2/10:
            val=0.1+0.2*0.4
        elif pos_array[1]>=index_min+span*2/10 and pos_array[1]<index_min+span*3/10:
            val=0.1+0.3*0.4     
        elif pos_array[1]>=index_min+span*3/10 and pos_array[1]<index_min+span*4/10:
            val=0.1+0.4*0.4 
        elif pos_array[1]>=index_min+span*4/10 and pos_array[1]<index_min+span*5/10:
            val=0.1+0.5*0.4 
        elif pos_array[1]>=index_min+span*5/10 and pos_array[1]<index_min+span*6/10:
            val=0.1+0.6*0.4 
        elif pos_array[1]>=index_min+span*6/10 and pos_array[1]<index_min+span*7/10:
            val=0.1+0.7*0.4
        elif pos_array[1]>=index_min+span*7/10 and pos_array[1]<index_min+span*8/10:
            val=0.1+0.8*0.4 
        elif pos_array[1]>=index_min+span*8/10 and pos_array[1]<index_min+span*9/10:
            val=0.1+0.9*0.4 
        else:
            val=0.5

        # Define the target joint positions (example configuration) 0.15 to 0.4
        target_positions = [
            t11, val, val, t14,  # Index
            0.3, 0.0, 0.0, 0.0,  # Middle
            0.0, 0.0, 0.0, 0.0,  # Ring
            t21, t22, val, t24   # Thumb
        ]
        
        cur=hand._joint_state
        if not cur==None and flag:
            cur0=cur
            pos0 = cur.position
            flag=False
        if not cur==None:    
            pos = cur.position

        # Calculate the torques
        t11 = k*(pos0[0]-pos[0])
        t14 = k*(pos0[3]-pos[3])
        t21 = k*(pos0[12]-pos[12])
        t22 = k*(pos0[13]-pos[13])
        t24 = k*(pos0[15]-pos[15])


        # Command the joint positions
        success = hand.command_joint_torques(target_positions)
        # success = hand.command_joint_position(target_positions)
        print(target_positions)
        # print(line)
        if success:
            print("Successfully commanded joint positions.")
            rospy.loginfo("Successfully commanded joint positions.")
        else:
            print("Failed to commanded joint positions.")
            rospy.logwarn("Failed to command joint positions.")

    rospy.sleep(2)  # Keep the node alive to allow time for the command to take effect
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        hand.disconnect()
        pass
