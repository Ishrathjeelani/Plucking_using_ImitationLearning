import numpy as np
from std_msgs.msg import Float32MultiArray, String
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO 
import hydra
from omegaconf import DictConfig
from mainTeleoperate import Teleoperation
import csv
from datetime import datetime


X, Y, Z=509, -196, 177 #Home position
csv_file = "/home/bonggeeun/Ishrath/Validation Demonstrations/Toy_Tomato/TactileVisionProprioACT_4c.csv"
command_states, predictions = [],[]
status = 0.2
# Callback function for predictions
def states_callback(msg):
    global command_states, status, predictions
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S-%f') 
    data = msg.data
    predictions = [str(timestamp)] + list(np.array(data, dtype=str))
    command_states = np.array(data[:3])
    status = data[-1]

@hydra.main(config_name="teleoperation", config_path="config", version_base="1.2")
def main(cfg: DictConfig):
    global csv_file, command_states, predictions, status, X, Y
    # Initialize the ROS node
    rospy.init_node("xArm_node", anonymous=False)

    # Publishers for xArm states
    xArm_pub = rospy.Publisher("/xArm/states", Float32MultiArray, queue_size=3)
    rospy.sleep(0.5)
    rospy.Subscriber('/predicted_states', Float32MultiArray, states_callback)

    teleoperator = Teleoperation(cfg)

    _,prevCurPosition = teleoperator.arm.get_position()
    prevCurPos = np.array(prevCurPosition[:3])

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
    # Add offset to compensate for Allegro hand
    # moveX = 334 - (np.mean(y_arr)-211)*0.87 + 60
    # moveY = 196 - (np.mean(x_arr)-41)*0.76 + 40
    # objPos = [moveX, moveY]
    # print("x:{} y:{}".format(objPos[0],objPos[1]))
    # # teleoperator.arm.set_position(x=437,y=-160,speed=100, mvacc=50, wait=True)
    
    # moveZ = Z#prevCurPos[2] - self.obj_z*5

    # X = objPos[0]
    # Y = objPos[1]
    
    # # Set the depth 
    # teleoperator.arm.set_position(z=Z+100,speed=100, mvacc=10, wait=True) #update values
    # # Set predefined x, y
    # teleoperator.arm.set_position(x=objPos[0], y=objPos[1], speed=100, mvacc=10, wait=True) #update values
    # # Set predefined pose
    # teleoperator.arm.set_position(roll=100, pitch=-46, yaw=70,speed=50, mvacc=5, wait=True)

    # # Set the depth 
    # teleoperator.arm.set_position(z=Z,speed=100, mvacc=10, wait=True) #update values

    # Log predictions
    csv_file = open(csv_file, "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp","roll", "pitch", "yaw", "thumb_1_position", "thumb_2_position", "thumb_3_position",
                "thumb_4_position", "index_1_position", "index_2_position", "index_3_position",
                "index_4_position", "thumb_1_torque", "thumb_2_torque", "thumb_3_torque", 
                "thumb_4_torque", "index_1_torque", "index_2_torque", "index_3_torque", 
                "index_4_torque", "task_status"])

    rospy.loginfo("xArm publisher and subscriber initialised")
    # Periodic check for inference (every 0.02 seconds)
    rate = rospy.Rate(50)  # 50Hz (0.02 seconds)
    iter=0
    while not rospy.is_shutdown():
        # Publishng states to get inference
        xArm_states = teleoperator.get_xArm_states()
        # Flatten states into a CSV-ready string
        xArm_state_values = [float(value) for value in xArm_states.values()]
        xArm_state_msg = Float32MultiArray()
        xArm_state_msg.data = np.concatenate([xArm_state_values[:13],xArm_state_values[20:27]])
        xArm_pub.publish(xArm_state_msg)
        # print(xArm_state_values)
        print(command_states)
        if command_states!=[]: #status<0.7 and 
            # Setting roll, pitch, Yaw from inference
            _, xarm_current_position = teleoperator.arm.get_position()
            xarm_new_pose = teleoperator.check_arm_pose(xarm_current_position[3:6], command_states)
            print(xarm_new_pose)
            teleoperator.arm.set_position( yaw=xarm_new_pose[2],speed=100, mvacc=50) #roll=xarm_new_pose[0], pitch=xarm_new_pose[1], , wait=True
            csv_writer.writerow(predictions)
            iter+=1
        print(status)
        # Check task completion
        if status>=70 and iter>500:
            print("Plucking complete")
            break
        if status <= 70 and iter>2000:
            print("Plucking failed")
            break
        rate.sleep()
    
    csv_file.close()
    print("File closed")
    teleoperator.arm.set_position(z=271,speed=100, mvacc=10, wait=True)
    teleoperator.arm.set_position(x=493,y=340,speed=500, mvacc=100, wait=True)
    teleoperator.arm.disconnect()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")