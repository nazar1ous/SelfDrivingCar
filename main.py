from agent_structures import CarEnv
from predictors import make_prediction
import time
import cv2

# Initialization
carEnv = CarEnv()
carEnv.reset()

# self.client = carla.Client('localhost', 2000)
# print(carEnv.client.get_available_maps())

for i in range(100):
    data = carEnv.front_camera
    result = make_prediction(data)
    carEnv.make_step(*result)
    # carEnv.make_step(5, 0)
    cv2.imwrite(f'./data/{i}.png', data)
    time.sleep(0.1)
    # pass
    # data = carEnv.get_data()

    print(i)