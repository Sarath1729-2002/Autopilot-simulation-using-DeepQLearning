import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from temp import CarEnv, MEMORY_FRACTION


MODEL_PATH = "G:\Autopilot_simulation\PythonAPI\examples\Project\models\MobileNetV2__-195.00max_-197.50avg_-200.00min__1677994549.h5"


if __name__ == '__main__':

    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    

    
    model = load_model(MODEL_PATH)


    env = CarEnv()


    fps_counter = deque(maxlen=60)

    
    model.predict(np.ones((1, env.im_height, env.im_width, 3)))

    
    while True:

        print('Restarting episode')
        current_state = env.reset()
        env.collision_hist = []
        done = False

        while True:

            
            step_start = time.time()
            cv2.imshow(f'Agent - preview', current_state)
            cv2.waitKey(1)
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            action = np.argmax(qs)
            new_state, reward, done, _ = env.step(action)
            current_state = new_state
            if done:
                break

            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')


        for actor in env.actor_list:
            actor.destroy()