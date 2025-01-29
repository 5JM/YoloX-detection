import os, cv2
from glob import glob
from performance_inference_onnx import Detector
from sklearn.metrics import precision_score, recall_score

def calc_classification_performance_v2(
    genuine_path, 
    imposter_path, 
    smoke_onnx_path,
    smoke_bbox_conf_thr = 0.7, 
    smoke_nms_thr = 0.3,
):
    '''
    smoke detect와 담배 연기가 있는 데이터(genuine path), 담배 연기가 없는 데이터(imposter path)를 이용해
    classification 성능 계산
    
    params:
        genuine_path: 담배 연기가 있는 데이터 path
        imposter_path: 담배 연기가 없는 데이터 path
        smoke_onnx_path: smoke detector onnx 파일 path
        smoke_bbox_conf_thr (default 0.7):
        smoke_nms_thr (default 0.3): 
    
    return:
        None
    '''

    # load smoke detector
    object_detector = Detector(
        onnx_path= smoke_onnx_path,
        threshold= smoke_bbox_conf_thr,
        nmsthreshold= smoke_nms_thr
    )

    # load image path and make label (smoke 1, no smoke 0)
    img_path_list, label_list = [], []

    for img_path in sorted(glob(os.path.join(genuine_path,'*.jpg'))):
        img_path_list.append(img_path)
        label_list.append(1)

    for img_path in sorted(glob(os.path.join(imposter_path,'*.jpg'))):
        img_path_list.append(img_path)
        label_list.append(0)

    ####################
    # Calc performance #
    ####################

    pred_list = []
    gt_list = []

    for img_path, label in zip(img_path_list, label_list):
        img = cv2.imread(img_path, 0)

        bboxes, _, _ = object_detector(img)

        # if no detect smoke
        if len(bboxes) < 1:
            pred_list.append(0)
            gt_list.append(label)
            
        else:
            pred_list.append(1)
            gt_list.append(label)

    assert len(pred_list) == len(gt_list), 'Not match length pred & label'

    # Precision 계산
    precision = precision_score(gt_list, pred_list)

    # Recall 계산
    recall = recall_score(gt_list, pred_list)

    # 결과 출력
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == '__main__':
    calc_classification_performance_v2(
        genuine_path = './datasets/smoke_test_data_classification/genuine', 
        imposter_path = './datasets/smoke_test_data_classification/imposter', 
        smoke_onnx_path = 'exported_onnx/smoke_detect_resize_1004_3_nms.onnx',
        smoke_bbox_conf_thr = 0.6, 
        smoke_nms_thr = 0.5,
    )