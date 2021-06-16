from agent_structures import CarEnv
from predictors import make_prediction
import time
import cv2

FPS = 60
PICTURE_TAKE_FREQ = 2
TAKE_PICTURES = True
MAX_PICTURES = 100
PICTURES_DIR = './data/'
cap = cv2.VideoCapture(0)


# Initialization of carEnv
carEnv = CarEnv()
carEnv.reset()


i = 0
n_pictures = 0
while True:
    data = carEnv.front_camera
    result = make_prediction(data)
    carEnv.make_step(*result)
    cv2.imshow('frame', data)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if TAKE_PICTURES and i % PICTURE_TAKE_FREQ == 0 and n_pictures < MAX_PICTURES:
        cv2.imwrite(PICTURES_DIR + f'{n_pictures}.png', data)
        n_pictures += 1
    # It is not quite realistic, because of the code above, but still
    # we can somehow max limit the FPS
    time.sleep(1/FPS)
    i = (i+1) % PICTURE_TAKE_FREQ

carEnv.clean_up()

cap.release()
cv2.destroyAllWindows()