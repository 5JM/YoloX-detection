import onnxruntime as rt
import os
import argparse
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from time import monotonic
import json
import copy

parser = argparse.ArgumentParser(
    description='detect_video')

# parser.add_argument('--input_path', default=None, type=str,)

parser.add_argument('--input_path', default="./datasets/test" , type=str,)
# parser.add_argument('--input_path', default="./datasets/smoke_test_data_well_jpg" , type=str,)
# parser.add_argument('--input_path', default="./datasets/_inference_data_002" , type=str,)
# parser.add_argument('--input_path', default="datasets", type=str,)
parser.add_argument('--save_path', default="onnx_test_rear_view_well", type=str,)
parser.add_argument('--save_name', default="smoke_1004_3", type=str,)
parser.add_argument("--conf", default=0.4, type=float, help="test conf")
parser.add_argument("--onnx_path", default="exported_onnx/sim_smoke_detect_resize_1004_3_nms.onnx", type=str, help="onnx path")
parser.add_argument("--plot_img", default=True, type=bool, help="plot image")
parser.add_argument("-n", "--name", type=str, default=None, help="model name")
parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

def compute_iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def nms(bounding_boxes, iou_threshold):
    # Ensure bounding boxes are sorted by confidence
    bounding_boxes = bounding_boxes[bounding_boxes[:, 4].argsort()[::-1]]

    # List to hold boxes that make it through the NMS process
    selected_boxes = []

    # Iterate through boxes, comparing each with every other box
    while len(bounding_boxes) > 0:
        # Select the box with the highest confidence and add it to the final list
        current_box = bounding_boxes[0]
        selected_boxes.append(current_box)

        # If there's only one box left, break out of the loop
        if len(bounding_boxes) == 1:
            break

        # Compare the current box with the rest
        rest_boxes = bounding_boxes[1:]
        to_keep = []

        for box in rest_boxes:
            if compute_iou(current_box, box) <= iou_threshold or current_box[5] != box[5]:
                to_keep.append(box)

        # Keep only boxes that were not suppressed
        bounding_boxes = np.array(to_keep)

    return np.array(selected_boxes)

def preproc(img, input_size, swap=(2, 0, 1)):

    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones((input_size[0], input_size[1], 1), dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    
    if len(img.shape) != 3:
        resized_img = np.expand_dims(resized_img, -1)
    
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

class Detector:

    def __init__(self, onnx_path, threshold=0.8, nmsthreshold=0.3):

        self.input_shape = (1, 1, 320, 320)
        self.sess = rt.InferenceSession(onnx_path)
        self.conf = threshold
        self.nmsthreshold = nmsthreshold

    def __call__(self, input):
        return self.infer(input)

# Load the serialized TensorRT engine from file

    def decode_outputs_np(self, outputs):

        strides = []
        grids = []
        predefined_strides = [8, 16, 32]
        predefined_hw = [[40, 40], [20, 20], [10, 10]]
        for (hsize, wsize), stride in zip(predefined_hw, predefined_strides):
            xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(np.full((shape[0], shape[1], 1), stride))

        grids = np.concatenate(grids, 1)
        strides = np.concatenate(strides, 1)
        outputs = np.concatenate([
            (outputs[..., 0:2] + grids) * strides,
            np.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], -1)
        return outputs
    
    def infer(self, image):

        # height, width = image.shape[:2]
        input_data = np.zeros(self.input_shape, dtype=np.float32)
        ratio = min(320 / image.shape[0], 320 / image.shape[1])
        
        input_data[0] = preproc(image, (320,320))

        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        # print(input_data)
        # Perform inference using the ONNX model
        outputs = self.sess.run([output_name], {input_name: input_data})

        outputs = self.decode_outputs_np(outputs[0])
        
        output = outputs[0]
        
        box_corner = np.zeros_like(output)
        
        box_corner[:, 0] = output[:, 0] - output[:, 2] / 2
        box_corner[:, 1] = output[:, 1] - output[:, 3] / 2
        box_corner[:, 2] = output[:, 0] + output[:, 2] / 2
        box_corner[:, 3] = output[:, 1] + output[:, 3] / 2
        output[:, :4] = box_corner[:, :4]

        categories = np.argmax(output[:, 5:], axis=1)
        conf = output[:, 4] * np.max(output[:, 5:], axis=1)

        conf_mask = conf > self.conf
        conf_masked = conf[conf_mask]

        if not conf_masked.shape[0] == 0:
            
            categories_masked = categories[conf_mask]
            bboxes = output[conf_mask, :4]
            bboxes /= ratio
            output = np.concatenate([bboxes, conf_masked.reshape(-1, 1), categories_masked.reshape(-1, 1)], axis=1)
            bounding_box = nms(output, self.nmsthreshold)
            output_box = bounding_box[:, :4]
            output_category = bounding_box[:, 5]
            output_conf = bounding_box[:, 4]
        
        else:
            
            output_box = np.array([])
            output_category = np.array([])
            output_conf = np.array([])

        return output_box, output_category, output_conf

def main(args):

    object_detector = Detector(onnx_path=args.onnx_path,
                               threshold=args.conf,
                               nmsthreshold=args.nms)

    imgs_name = [i for i in os.listdir(args.input_path) if i.endswith(".jpg") or i.endswith(".png")]

    result_dict = {}

    for img_name in tqdm(imgs_name):
        
        img_path = os.path.join(args.input_path, img_name)
        result_dict[img_name] = {}
        
        input_image = cv2.imread(img_path)
        
        input_h, input_w = input_image.shape[:2]
        
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        input_image = input_gray
            
        bboxes, categories, confes = object_detector(input_image)
        
        result_dict[img_name] = []
        
        for box, category, conf in zip(bboxes, categories, confes):
            
            x1, y1, x2, y2 = box
            
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(input_w, x2))
            y2 = int(min(input_h, y2))
            category = int(category)
            
            result_dict[img_name].append([category+1, x1, y1, x2, y2, conf])

            if args.plot_img:

                cv2.rectangle(input_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(input_image, f"{conf:.2f}", (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        if args.plot_img:
            cv2.imwrite(f"{args.save_path}/{img_name}", input_image)

    with open(f"{args.save_path}/{args.save_name}.json", 'w') as f:
        json.dump(result_dict, f)
    

if __name__ == "__main__":
    main(args)   
    
    

    




