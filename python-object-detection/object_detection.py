import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import time
import onnxruntime as rt

ONNX = True

VIDEO_DIM = (640, 480)
MODEL_DIM = (512, 512)

ANCHOR_NUM = 3
CLASS_NUM = 80
ANCHOR_WIDTH = [[116, 156, 373], [30, 62, 59], [10, 16, 33]]
ANCHOR_HEIGHT = [[90, 198, 326], [61, 45, 119], [13, 30, 23]]
LABEL = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

class BoundingBox:
    def __init__(self, class_index, score, area, x, y, w, h):
        self.class_index = class_index
        self.score = score
        self.area = area
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def getModelDims(model):
    '''
    Returns model dimensions

    # Arguments: 
        model: loaded model

    # Returns:
        width, height, channel
    '''
    return model.input.shape[1], model.input.shape[2], model.input.shape[3]


def setFrameDim(vid, video_dims):
    '''
    Sets video frame to desired dimensions

    # Arguments:
        vid: video capture object
        video_dims: desired dimensions

    # Returns:
        none
    '''
    vid.set(cv.CAP_PROP_FRAME_WIDTH, video_dims[0])
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, video_dims[1])


def getFrameDim(vid):
    '''
    Gets video frame dimensions

    # Arguments:
        vid: video capture object

    # Returns:
        width, height
    '''
    return int(vid.get(cv.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))


def resizeFrame(frame, padding, dims):
    '''
    Make frame into a square by adding border
    Resize frame to desired dimensions

    # Arguments:
        frame: video frame
        padding: padding pixels to make the frame square
        dims: desired square size
        i: counter for while loop

    # Returns:
        desired resized frame
    '''
    resized_frame = cv.copyMakeBorder(frame, 0, padding, 0, 0, cv.BORDER_CONSTANT)
    resized_frame = cv.resize(resized_frame, dims)

    return resized_frame


def normalizeFrame(frame):
    '''
    Normalize frame from 0 ~ 255 to 0 ~ 1
    Expand dimension of frame to include batch

    # Arguments:
        frame: video frame

    # Returns:
        normalized frame contains [batch, width, height, channel]
    '''
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)

    return frame


def sigmoid(num):
    '''
    Non-linear way of making values from 0 ~ 1

    # Arguments:
        num: number that wants be be normalized 

    # Returns:
        Non-linear value between 0 ~ 1 
    '''
    return 1 / (1 + np.exp(-num))

def intClip(value, min, max):
    if value < min:
        return int(min)
    elif value > max:
        return int(max)
    else:
        return int(value)


def calcIOU(bounding_box_0, bounding_box_1):
    '''
    Calculates the intersection over union of two bounding boxes

    # Arguments:
        bounding_box_0: first bounding box
        bounding_box_1: second bounding box

    # Returns:
        intersection over union of two bounding boxes
    '''
    x_left = max(bounding_box_0.x, bounding_box_1.x)
    x_right = min(bounding_box_0.x + bounding_box_0.w, bounding_box_1.x + bounding_box_1.w)
    y_top = max(bounding_box_0.y, bounding_box_1.y)
    y_bottom = min(bounding_box_0.y + bounding_box_0.h, bounding_box_1.y + bounding_box_1.h)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    return intersection / (bounding_box_0.area + bounding_box_1.area - intersection)


def drawBoundingBoxes(frame, bounding_boxes, color):
    '''
    Draws the bounding boxes and labels onto the frame

    # Arguments: 
        frame: video frame
        bounding_boxes: all the bounding boxes
        color: color of the bounding boxes

    # Returns:
        none
    '''
    # if no bounding boxes, then return regular image
    if len(bounding_boxes) == 0:
        return frame
    
    for bb in bounding_boxes:
        # label created in format 'classname score'
        label = '{} {:.2f}'.format(LABEL[bb.class_index], bb.score)
        print(f'* {label}')

        # determining position of label
        if bb.y > 20:
            label_coords = (bb.x, bb.y)
            label_type = 'LABEL_TOP_OUTSIDE'
        elif bb.y <= 20 and bb.y + bb.h <= frame.shape[2] - 20:
            label_coords = (bb.x, bb.y + bb.h)
            label_type = 'LABEL_BOTTOM_OUTSIDE'
        elif bb.y + bb.h > frame.shape[2] - 20:
            label_coords = (bb.x, bb.y)
            label_type = 'LABEL_TOP_INSIDE'

        cv.rectangle(frame, (bb.x, bb.y), (bb.x + bb.w, bb.y + bb.h), color, 1, cv.LINE_AA)

        # creating label
        font = cv.FONT_HERSHEY_PLAIN
        font_scale = 1.
        (label_width, label_height) = cv.getTextSize(label, font, fontScale=font_scale, thickness=1)[0]

        # adjustments for rectangle behind label
        padding = 5
        rect_width = label_width + padding * 2
        rect_height = label_height + padding * 2

        x, y = label_coords

        if label_type == 'LABEL_TOP_OUTSIDE':
            cv.rectangle(frame, label_coords, (x + rect_width, y - rect_height), color, cv.FILLED)
            cv.putText(frame, label, (x + padding, y - label_height + padding), font,
                        fontScale=font_scale, color=(255, 255, 255), lineType=cv.LINE_AA)
        else:
            cv.rectangle(frame, label_coords, (x + rect_width, y + rect_height), color, cv.FILLED)
            cv.putText(frame, label, (x + padding, y + label_height + padding), font,
                        fontScale=font_scale, color=(255, 255, 255), lineType=cv.LINE_AA)

    return frame


def getAllBoundingBox(tensors, score_threshold, resized_frame_dim):
    '''
    Gets all bounding boxes from tensors

    # Arguments: 
        tensors: tensors provided after calling the model
        score_threshold: threshold to determine making a bounding box
    
    # Returns:
        A list of all bounding boxes
    '''
    all_bounding_box = []

    for t in range(len(tensors)):
        b, h, w, c = tensors[t].shape
        if ONNX: tensor_data = tensors[t].flatten()
        else: tensor_data = tensors[t].numpy().flatten()
        data_position = 0

        for y in range(h):
            for x in range(w):
                for a in range(ANCHOR_NUM):
                    # objectness score of that area
                    # objectness_score = sigmoid(tensors[t][0, y, x, a * (CLASS_NUM + 5) + 4])
                    objectness_score = sigmoid(tensor_data[data_position + 4])
                    if objectness_score < score_threshold:
                        data_position += CLASS_NUM + 5 
                        continue

                    for c in range(5, 85):
                        # score per class
                        # class_score = sigmoid(tensors[t][0, y, x, a * (CLASS_NUM + 5) + c])
                        class_score = sigmoid(tensor_data[data_position + c])
                        score = objectness_score * class_score
                        
                        # make bounding box only if score is greater than threshold
                        if (score > score_threshold):
                            # x0 = (sigmoid(tensors[t][0, y, x, a * (CLASS_NUM + 5) + 0]) + x) / w
                            # y0 = (sigmoid(tensors[t][0, y, x, a * (CLASS_NUM + 5) + 1]) + y) / h
                            # x1 = x0 + np.exp(tensors[t][0, y, x, a * (CLASS_NUM + 5) + 2]) * ANCHOR_WIDTH[t][a] / resized_frame_dim[0]
                            # y1 = y0 + np.exp(tensors[t][0, y, x, a * (CLASS_NUM + 5) + 3]) * ANCHOR_HEIGHT[t][a] / resized_frame_dim[1]
                            x0 = (sigmoid(tensor_data[data_position + 0]) + x) / w
                            y0 = (sigmoid(tensor_data[data_position + 1]) + y) / h
                            w0 = np.exp(tensor_data[data_position + 2]) * ANCHOR_WIDTH[t][a] / resized_frame_dim[0]
                            h0 = np.exp(tensor_data[data_position + 3]) * ANCHOR_HEIGHT[t][a] / resized_frame_dim[1]
                            bounding_box = BoundingBox(c - 5, score, 0, x0, y0, w0, h0)
                            all_bounding_box.append(bounding_box)
                    
                    data_position += CLASS_NUM + 5 

    return all_bounding_box

def convertBoundingBoxToImageDim(all_bounding_box, resized_frame_dim, frame_dim):
    '''
    Converts all bounding box dimensions into image dimensions

    # Arguments:
        all_bounding_box: all the bounding boxes
        resize_frame_dim: dimensions of the resized frame
        frame_dim: dimensions of the original frame

    # Returns:
        none
    '''
    # finding scale to go back to frame dimension from resized frame dimension
    scale = max(frame_dim[0] / resized_frame_dim[0], frame_dim[1] / resized_frame_dim[1])

    for bounding_box in all_bounding_box:
        # scaling back to frame dimension
        x = bounding_box.x * resized_frame_dim[0] * scale 
        y = bounding_box.y * resized_frame_dim[1] * scale 
        w = bounding_box.w * resized_frame_dim[0] * scale
        h = bounding_box.h * resized_frame_dim[1] * scale

        x = x - (w) / 2
        y = y - (h) / 2

        # ensuring positions are in frame
        x = intClip(x, 0, frame_dim[0] - 1)
        y = intClip(y, 0, frame_dim[1] - 1)
        w = intClip(w, 0, frame_dim[0] - 1)
        h = intClip(h, 0, frame_dim[1] - 1)

        # reassigning bounding box coordinates & area
        bounding_box.x = x
        bounding_box.y = y
        bounding_box.w = w
        bounding_box.h = h
        if (w < 1 or h < 1):
            bounding_box.score = 0
        else:
            bounding_box.area = w * h

    sorted(all_bounding_box, reverse=True, key=lambda bounding_box: bounding_box.score)


def nonMaximumSuppression(all_bounding_box, iou_threshold):
    '''
    Using Non Maximum Suppression to eliminate overlapping bounding boxes

    # Arguments:
        all_bounding_box: all the bounding boxes
        iou_threshold: threshold of intersection over union 

    # Returns:
        revised list of bounding boxes
    '''
    revised_bounding_box = []
    excluded_bounding_box = [False for i in range(len(all_bounding_box))]

    for i in range(len(all_bounding_box)):
        # if the bounding box is excluded, then go to next
        if excluded_bounding_box[i] == True:
            continue

        bounding_box_0 = all_bounding_box[i]
        # if score is 0, then skip this bounding_box
        if bounding_box_0.score == 0:
            continue

        # add the bounding box to the revised list
        revised_bounding_box.append(bounding_box_0)
        class_index_0 = bounding_box_0.class_index

        # loop through next few bounding_boxes
        for j in range(i + 1, len(all_bounding_box)):
            # if the bounding box is excluded, then go to next
            if excluded_bounding_box[j] == True:
                continue

            bounding_box_1 = all_bounding_box[j]
            # if the class index isn't same or area is 0, then go to next
            if bounding_box_1.class_index != class_index_0:
                continue
            elif bounding_box_1.area == 0:
                continue
            else:
                # calculate to see if iou is more than threshold
                if calcIOU(bounding_box_0, bounding_box_1) > iou_threshold:
                    excluded_bounding_box[j] = True
    
    return revised_bounding_box


def getBoundingBoxes(tensors, resized_frame_dim, frame_dim, i, 
                        score_threshold = 0.6, iou_threshold = 0.6):
    '''
    Gets all bounding boxes from tensors

    # Arguments:
        tensors: tensors provided after calling the model
        resize_frame_dim: dimensions of the resized frame
        frame_dim: dimensions of the original frame
        i: counter for while loop
        score_threshold: threshold to determine making a bounding box
        iou_threshold: threshold of intersection over union

    # Returns:
        bounding boxes from the tensors
    '''
    
    all_bounding_box = getAllBoundingBox(tensors, score_threshold, resized_frame_dim)
    print(f'All bounding box num     = {len(all_bounding_box)}')

    convertBoundingBoxToImageDim(all_bounding_box, resized_frame_dim, frame_dim)

    revised_bounding_box = nonMaximumSuppression(all_bounding_box, iou_threshold)
    print(f'Revised bounding box num = {len(revised_bounding_box)}')

    return revised_bounding_box


def runModel(model_path: str):
    if ONNX:
        # loading the model
        session = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_names = [x.name for x in session.get_inputs()]
        output_names = [x.name for x in session.get_outputs()]
    else:
        # loading the model
        model = tf.keras.models.load_model(model_path, compile=False)
        net_h, net_w, net_c = getModelDims(model)
        # model.save('C:\Kyle_Work\python-object-detection')
        print(f'\nModel input dimensions   = {net_h, net_w, net_c}')

    # getting video
    vid = cv.VideoCapture(0)
    if not vid.isOpened():
        raise Exception('Could not open camera')

    # setting frame dimension & getting padding
    setFrameDim(vid, VIDEO_DIM)
    width, height = getFrameDim(vid)
    padding = abs(width - height)
    print(f'Video dimensions         = {width, height, padding}')

    # video loop
    i = 0
    while(True):
        # get frame from video
        ret, frame = vid.read()
        if ret == False:
            continue
        if i == 0:
            print(f'Frame dimensions         = {frame.shape}')
        frame = cv.flip(frame, 1)

        start_time = time.time()
        resized_frame = resizeFrame(frame, padding, MODEL_DIM)
        if i == 0:
            print(f'Resized frame dimensions = {resized_frame.shape}')

        resized_frame= cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
        frame_data = normalizeFrame(resized_frame)
        if i == 0:
            print(f'Net input dimensions     = {frame_data.shape}')

        if ONNX:

            frame_data = frame_data.transpose(0, 3, 1, 2)
            onnx_inputs = {input_names[0]: frame_data}
            onnx_outputs = session.run(output_names, onnx_inputs)

            tensors = list()
            for i in range(len(onnx_outputs)):
                tensors.append(onnx_outputs[i].transpose(0, 2, 3, 1))
        else:
            tensors = model(frame_data)

        if i == 0:
            for j in range(len(tensors)):
                print(f'Tensors {j}                = {tensors[j].shape}')
            print()

        bounding_boxes = getBoundingBoxes(tensors, resized_frame.shape[0:2], frame.shape[0:2], i)

        new_frame = drawBoundingBoxes(frame, bounding_boxes, (0, 0, 0))

        # show frame
        end_time = time.time()
        print(f'\nFrames per second        = {1 / (end_time - start_time)}')
        cv.imshow('frame', new_frame)

        # wait for 1 millisecond 
        k = cv.waitKey(1)
        if k == ord('q'):
            break
    
        i += 1

    # release video & delete windows
    vid.release()
    cv.destroyAllWindows()

    
if __name__ == '__main__':
    # file path yolo3: /Users/kylepan/Documents/keras-YOLOv3-model-set/weights/yolov3.h5
    # model_path = input('Enter Model File Path:\n')
    if ONNX:
        model_path = 'C:/Kyle_Work/models/yolov3.onnx'
    else:
        model_path = 'C:/work/third_party/keras-YOLOv3-model-set/weights/yolov3.h5'

    if not os.path.exists(model_path):
        print(f"model file path {model_path} doesn't exist")
    else:
        runModel(model_path)