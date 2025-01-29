import torch
import argparse
from yolox.exp import get_exp
import os
import onnx
from onnxsim import simplify
import numpy as np
import cv2
from torch import nn
import glob

parser = argparse.ArgumentParser(
    description='exporting torch model to onnx')

parser.add_argument("-v", "--version", default=11, type=int)

# exp file
parser.add_argument("-e", "--exp_file", default="exps/smoke_ti.py", type=str)
parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/vaping_1004_3/best_ckpt.pth", type=str, help="ckpt for eval")
parser.add_argument("-d", "--save_dir", default="exported_onnx", type=str, help="dir for saved onnx file")
parser.add_argument("-n", "--save_name", default="smoke_detect_resize_1004_3", type=str, help="dir for saved onnx file")
parser.add_argument("-s", "--shape", default=[320,320], type=list, help="inference shape")
args = parser.parse_args()

exp = get_exp(args.exp_file, None)

os.makedirs(args.save_dir, exist_ok=True)

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
        # padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        padded_img = np.ones((3, input_size[0], input_size[1]), dtype=np.uint8) * 114
    else:
        # padded_img = np.ones((input_size[0], input_size[1], 1), dtype=np.uint8) * 114
        padded_img = np.ones((1, input_size[0], input_size[1]), dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    resized_img = np.squeeze(resized_img)

    # if len(img.shape) != 3:
    #     resized_img = np.expand_dims(resized_img, -1)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

class DetectorWithNMS(nn.Module):
    def __init__(self, threshold=0.5, nmsthreshold=0.3) -> None:
        super().__init__()
        self.input_shape = (1, 1, 320, 320)

        self.conf = threshold
        self.nmsthreshold = nmsthreshold

        self.model = exp.get_model()

        ckpt = torch.load(args.ckpt, map_location="cpu")

        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        self.model.head.decode_in_inference = False
        self.model.cuda().float()

        self.model.eval()
        
    def forward(self, x):
        return self.__infer(x)

    def __decode_outputs_np(self, outputs):
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
    
    def __infer(self, image):
        ratio = min(320 / image.shape[1], 320 / image.shape[2])
        outputs = self.model(image)

        outputs = self.__decode_outputs_np(outputs.cpu().detach().numpy())
        
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
    

if __name__ == '__main__':
    if exp.grayscale:
        in_channels = 1
    else:
        in_channels = 3

    # Convert the PyTorch model to an ONNX model
    input_data = torch.randn( 1, in_channels, args.shape[0], args.shape[1])

    model = exp.get_model()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    model.head.decode_in_inference = False

    onnx_model_path = f"{args.save_name}_nms.onnx"
    torch.onnx.export(model, 
                    input_data, 
                    os.path.join(args.save_dir, onnx_model_path),
                    opset_version=args.version, 
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'])

    print('Converting is done!!')

    onnx_model = onnx.load(os.path.join(args.save_dir, onnx_model_path))
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, os.path.join(args.save_dir, "sim_"+onnx_model_path))
    print('Simplified model is saved!!')


    #################################

    # input_data = np.ones(( 1, in_channels, args.shape[0], args.shape[1]))


    # os.makedirs('datasets/seat_crop', exist_ok=True)

    # for path in glob.glob('datasets/seat_coco_320_wo_histeq/test/*.jpg'):
    #     file_name = path.split('/')[0]
    #     crop_save_path = os.path.join('datasets/seat_crop',file_name)

    #     origin_data = cv2.imread(path,0)

    #     hist_eq_img = cv2.equalizeHist(origin_data)
    #     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #     # origin_data = clahe.apply(origin_data)

    #     H,W = origin_data.shape[:2]
    #     input_data = torch.tensor(hist_eq_img, dtype=torch.float32)
    #     input_data = input_data.unsqueeze(dim=0).unsqueeze(dim=0).cuda()

    #     core = DetectorWithNMS(threshold=0.5)
    #     out = core(input_data)
        
        
    #     bboxes, cls, conf = out
    #     margin = 15
    #     for bbox in bboxes:
    #         top_left = int(bbox[0])-margin
    #         top_left = 0 if top_left < 0 else top_left

    #         top_right = int(bbox[1])-margin
    #         top_right = 0 if top_right < 0 else top_right

    #         bot_left = int(bbox[2])+margin
    #         bot_left = W-1 if bot_left >= W  else bot_left

    #         bot_right = int(bbox[3])+margin
    #         bot_right = H-1 if bot_right >= H  else bot_right
    #         # print(conf)
    #         cv2.rectangle(origin_data, (top_left, top_right), (bot_left, bot_right), (255,255,255), 3)
    #         # cropped_image = origin_data[int(bbox[0]):int(bbox[1]), int(bbox[2]):int(bbox[2])]

    #     cv2.imshow('tt', origin_data)
        
    #     key = cv2.waitKey()
        
    #     if key == 27:
    #         break
        
    #     cv2.destroyAllWindows()
