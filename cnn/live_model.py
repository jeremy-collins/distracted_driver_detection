import torch
import numpy as np
import cv2
import argparse
import os
import time
from camera import Camera
from my_resnet import MyResNet18
from PIL import Image

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--index', '-i', type=int, default=0)
args = parser.parse_args()

gt_dict = {
            0: 'Driving Safely',
            1: 'Texting',
            2: 'Talking on Phone',
            3: 'Texting',
            4: 'Talking on Phone',
            5: 'Operating Radio',
            6: 'Drinking',
            7: 'Reaching Back',
            8: 'Driving Safely',
            9: 'Talking to Passenger'
            }

model_path = 'trained_model_rn18_74.pt'
model = MyResNet18()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
model.eval()
with torch.no_grad():
    feed = Camera(view=True, resource=args.index)
    frame = feed.get_frame()

    while feed.ret:
        frame = feed.get_frame()
        # horizontally flipping the frame 
        # frame = cv2.flip(frame, 1)
        # frame = cv2.imread('/media/jer/data2/state-farm-distracted-driver-detection/imgs/train/c0/img_800.jpg')
        # time.sleep(0.1)
            
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        # cutting frame size in half
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        frame = np.array(frame)

    
        # taking 224x224 center crop
        frame = frame[(frame.shape[0] - 224) // 2 : (frame.shape[0] + 224) // 2, (frame.shape[1] - 224) // 2 : (frame.shape[1] + 224) // 2, :]
        print(frame.shape)
        
        if type(frame) == np.ndarray:
            frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1)
        frame = frame.float()/255.0
        frame = frame.unsqueeze(0)

        output = model(frame)
        output = output.detach().numpy().squeeze()
        # applying softmax to get probabilities
        output = np.exp(output) / np.sum(np.exp(output))
        pred = np.argmax(output)
        prob = output[pred]
        prob = float(prob)
        prob = 100 * round(prob, 3)

        img = cv2.putText(img, "Prediction: " + gt_dict[pred], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 10), 2)
        img = cv2.putText(img, "Confidence: " + str(prob)[:4] + '%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 10), 2)

        if feed.view:
            cv2.namedWindow("Distracted Driver Detection: Live Model", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Distracted Driver Detection: Live Model",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Distracted Driver Detection: Live Model", img)
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        
        print("Prediction: ", output)
        # print('Average FPS', feed.frame_count / (time.time() - feed.first_frame_time))
        # print(feed.frame_count, ' frames captured')