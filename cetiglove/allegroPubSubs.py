#!/usr/bin/env python3
import torch
import numpy as np
from std_msgs.msg import Float32MultiArray, String
import rospy
from scipy.signal import butter, lfilter
from cetiglove.robot.allegroHand import Allegro  # Assuming the class is saved in allegro_class.py

command_pos=[]
command_torq=[]

lower_limit = 0.15
upper_limit = 0.4
default_value = 0

def states_callback(msg):
    global command_pos, command_torq
    data = msg.data
    command_pos = data[3:11]
    command_torq = data[11:19]

def main():
    global command_pos, command_torq
    rospy.init_node('allegro_node')
    rospy.sleep(0.5)
    # Initialize the Allegro hand client
    hand = Allegro(hand_topic_prefix="allegroHand_0")
    rospy.Subscriber('/predicted_states', Float32MultiArray, states_callback)
    rospy.sleep(0.5)

    init_pos=hand._joint_state
    # print(command_pos)
    # Predefined hand pose
    target_positions = [0.0, 1.3, 0.3, 0.0,
                        0.0, 1.4, 0.0, 0.0,
                        0.0, 1.4, 0.0, 0.0, 
                        1.36, 0.0, 0.12, 0.0]
    # Set initial grasp config
    success = hand.command_joint_position(target_positions)
    print("Commanded Positions")
    print(target_positions)

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
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        # print(command_torq)
        # Define the target joint positions (example configuration) 0.15 to 0.4
        if command_torq!=[]:
            command_torq = np.array(command_torq, dtype=float)
            # Set the values outside the range to the default value
            command_torq[(command_torq < lower_limit)] = upper_limit
            command_torq[(command_torq > upper_limit)] = upper_limit
            target_positions = [
                t11, np.abs(command_torq[5]), np.abs(command_torq[6]), t14,  # Finger 1 -0.3, val, val, 0.0
                0.3, 0.0, 0.0, 0.0,  # Finger 2
                0.0, 0.0, 0.0, 0.0,  # Finger 3
                t21, t22, np.abs(command_torq[2]), t24   # Thumb 1.5, 0.0, val, 0.15
            ]
            
            cur=hand._joint_state
            if not cur==None and flag:
                cur0=cur
                pos0 = cur.position
                flag=False
            if not cur==None:    
                pos = cur.position

            # Calculate the torques (for ensure stable grasp pose)
            t11 = k*(pos0[0]-pos[0])
            t14 = k*(pos0[3]-pos[3])
            t21 = k*(pos0[12]-pos[12])
            t22 = k*(pos0[13]-pos[13])
            t24 = k*(pos0[15]-pos[15])

            # Convert list to a NumPy array for element-wise operations
            target_positions = np.array(target_positions, dtype=float)
            # Command the joint positions
            success = hand.command_joint_torques(target_positions)
            print(target_positions)

        rate.sleep()

    rospy.sleep(0.1)  # Keep the node alive to allow time for the command to take effect
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        # hand.disconnect()
        pass
