import rospy  # type: ignore
from sensor_msgs.msg import JointState  # type: ignore
from std_msgs.msg import Float32, String  # type: ignore

class Allegro:
    def __init__(self, hand_topic_prefix="allegroHand_0", num_joints=16):
        hand_topic_prefix = hand_topic_prefix.rstrip("/")
        topic_grasp_command = "{}/lib_cmd".format(hand_topic_prefix)
        topic_joint_command = "{}/joint_cmd".format(hand_topic_prefix)
        topic_joint_state = "{}/joint_states".format(hand_topic_prefix)
        topic_envelop_torque = "{}/envelop_torque".format(hand_topic_prefix)

        # Publishers for above topics.
        self.pub_grasp = rospy.Publisher(topic_grasp_command, String, queue_size=10)
        self.pub_joint = rospy.Publisher(topic_joint_command, JointState, queue_size=10)
        self.pub_envelop_torque = rospy.Publisher(
            topic_envelop_torque, Float32, queue_size=1
        )
        rospy.Subscriber(topic_joint_state, JointState, self._joint_state_callback)
        self._joint_state = None

        self._num_joints = num_joints

        rospy.loginfo(
            "Allegro Client start with hand topic: {}".format(hand_topic_prefix)
        )

        # "Named" grasps are those provided by the bhand library. These can be
        # commanded directly and the hand will execute them. The keys are more
        # human-friendly names, the values are the expected names from the
        # allegro controller side. Multiple strings mapping to the same value
        # are allowed.
        self._named_grasps_mappings = {
            "home": "home",
            "ready": "ready",
            "three_finger_grasp": "grasp_3",
            "three finger grasp": "grasp_3",
            "four_finger_grasp": "grasp_4",
            "four finger grasp": "grasp_4",
            "index_pinch": "pinch_it",
            "index pinch": "pinch_it",
            "middle_pinch": "pinch_mt",
            "middle pinch": "pinch_mt",
            "envelop": "envelop",
            "off": "off",
            "gravity_compensation": "gravcomp",
            "gravity compensation": "gravcomp",
            "gravity": "gravcomp",
        }
        # Add a new dictionary to store joint operation modes
        self.joint_modes = ["position"] * num_joints  # Default: all joints in position mode
    
    def disconnect(self):
        """
        Disconnect the allegro client from the hand by sending the 'off'
        command. This is principally a convenience binding.

        Note that we don't actually 'disconnect', so you could technically
        continue sending other commands after this.
        """
        self.command_hand_configuration("off")

    def _joint_state_callback(self, data):
        self._joint_state = data

    def set_joint_mode(self, joint_index, mode):
        """
        Set the control mode for a specific joint.

        :param joint_index: Index of the joint (0 to num_joints-1).
        :param mode: The desired control mode ('position' or 'torque').
        """
        if joint_index < 0 or joint_index >= self._num_joints:
            rospy.logwarn(f"Invalid joint index: {joint_index}")
            return False
        if mode not in ["position", "torque"]:
            rospy.logwarn(f"Invalid mode: {mode}. Use 'position' or 'torque'.")
            return False

        self.joint_modes[joint_index] = mode
        rospy.loginfo(f"Set joint {joint_index} to {mode} mode.")
        return True
    
    def command_joint_position(self, desired_pose):
        if (
            not hasattr(desired_pose, "__len__")
            or len(desired_pose) != self._num_joints
        ):
            rospy.logwarn(f"Desired pose must be a {self._num_joints}-d array: got {desired_pose}.")
            return False

        msg = JointState()
        msg.position = [0.0] * self._num_joints  # Default positions

        for i in range(self._num_joints):
            if self.joint_modes[i] == "position":
                msg.position[i] = desired_pose[i]
            else:
                rospy.logdebug(f"Skipping joint {i}, not in position mode.")

        self.pub_joint.publish(msg)
        rospy.logdebug("Published desired pose.")
        return True

    def command_joint_torques(self, desired_torques):
        if (
            not hasattr(desired_torques, "__len__")
            or len(desired_torques) != self._num_joints
        ):
            rospy.logwarn(f"Desired torques must be a {self._num_joints}-d array: got {desired_torques}.")
            return False

        msg = JointState()
        msg.effort = [0.0] * self._num_joints  # Default torques

        for i in range(self._num_joints):
            if self.joint_modes[i] == "torque":
                msg.effort[i] = desired_torques[i]
            else:
                rospy.logdebug(f"Skipping joint {i}, not in torque mode.")

        self.pub_joint.publish(msg)
        rospy.logdebug("Published desired torques.")
        return True

    def poll_joint_position(self, wait=False):
        """Get the current joint positions of the hand.

        :param wait: If true, waits for a 'fresh' state reading.
        :return: Joint positions, or None if none have been received.
        """
        if wait:  # Clear joint state and wait for the next reading.
            self._joint_state = None
            while not self._joint_state:
                rospy.sleep(0.001)

        if self._joint_state:
            return (self._joint_state.position, self._joint_state.effort)
        else:
            return None

    def command_hand_configuration(self, hand_config):
        """
        Command a named hand configuration (e.g., pinch_index, envelop,
        gravity_compensation).

        The internal hand configuration names are defined in the
        AllegroNodeGrasp controller file. More human-friendly names are used
        by defining them as 'shortcuts' in the _named_grasps_mapping variable.
        Multiple strings can map to the same commanded configuration.

        :param hand_config: A human-friendly string of the desired
        configuration.
        :return: True if the grasp was known and commanded, false otherwise.
        """

        # Only use known named grasps.
        if hand_config in self._named_grasps_mappings:
            # Look up conversion of string -> msg
            msg = String(self._named_grasps_mappings[hand_config])
            rospy.logdebug("Commanding grasp: {}".format(msg.data))
            self.pub_grasp.publish(msg)
            return True
        else:
            rospy.logwarn("Unable to command unknown grasp {}".format(hand_config))
            return False

    def list_hand_configurations(self):
        """
        :return: List of valid strings for named hand configurations (including
        duplicates).
        """
        return self._named_grasps_mappings.keys()

    def set_envelop_torque(self, torque):
        """
        Command a specific envelop grasping torque.

        This only applies for the envelop named hand command. You can set the
        envelop torque before or after commanding the envelop grasp.

        :param torque: Desired torque, between 0 and 1. Values outside this
        range are clamped.
        :return: True.
        """

        torque = max(0.0, min(1.0, torque))  # Clamp within [0, 1]
        msg = Float32(torque)
        self.pub_envelop_torque.publish(msg)
        return True


