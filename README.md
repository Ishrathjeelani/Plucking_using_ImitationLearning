# Plucking_using_ImitationLearning
Prerequisites
1. Setup lasr-robot environment - https://github.com/lasr-lab/lasr-robot
2. Setup sparsh environment - https://github.com/facebookresearch/sparsh
3. Download mae_vit base checkpoint file and place it in checkpoints folder

For teleoperating the arm pose and allegro hand [Data Collection]
1. Launch Allegro hand node (rossallegro env)
2. Launch Plucking_nodes node (rossallegro env)
3. Run cetiglove/mainTeleoperate.py (test env)
4. Run cetiglove/simpleAllegro.py (test env)
5. Run cetiglove/mainSubscriber.py (test env)
Note - Glove publisher should be uncommented in Plucking_nodes launch file

For training model [Imitation Learning]
1. Activate tactile_ssl env (from Sparsh setup)
2. Run each class script placed in the sparsh folder, uncommenting the main function using the command shown below in terminal
python modelxxxxx.py +experiment=mae_vit

For Realtime inference and deployment [Demonstrations]
1. Launch Allegro hand node (rossallegro env)
2. Launch Plucking_nodes node (rossallegro env)
3. Run sparsh/pluckingModelInference.py +experiment=mae_vit (tactile_ssl env)
4. Run cetiglove/allegroPubsSubs.py (test env)
5. Run cetiglove/xArmPubsSubs.py (test env)
Note - Glove publisher should be commented and Yolo camera node uncommented in Plucking_nodes launch file
