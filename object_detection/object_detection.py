

#importing libraries
import torch 
from torch.autograd import Variable
import cv2 
import imageio
from ssd import build_ssd
from data import BaseTransform , VOC_CLASSES as lable_map

#define a detect fucntion 
def detect(frame,nn,transform):
    height,width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))
    y = nn(x)
    detections = y.data
    scale = torch.Tensor([width,height,width,height])
    
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6 :
            pt = (detections[0,i,j,1:]*scale).numpy()
            cv2.rectangle(frame,(pt[0],pt[1]),(pt[2],pt[3]),(255,0,0),2)
            cv2.putText(frame,lable_map[i-1],(pt[0],pt[1]),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),2)
            j += 1
        
    return frame

#create a SSD Neural Network
net = build_ssd('test')
net.load_state_dict(torch.load('trained_model.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (trainde_model.pth).

# create a transform model
transform = BaseTransform(net.size,(104/256.0,117/256.0,123/256.0))

# doing object detection on a video
reader = imageio.get_reader('horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output2.mp4',fps=fps)

# iterate loop for each frame in video
for i , frame in enumerate(reader):
    frame = detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)# for checking how many frames of video will recognize

writer.close()

