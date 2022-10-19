import cv2
import numpy as np

from app.fmr_utils.utils.anchor_generator import generate_anchors
from app.fmr_utils.utils.anchor_decode import decode_bbox
from app.fmr_utils.utils.nms import single_class_non_max_suppression
from app.fmr_utils.load_model.keras_loader import load_keras_model, keras_inference

# TODO: Path para checar 
model = load_keras_model('fmr_utils/models/face_mask_detection.json', 'fmr_utils/models/face_mask_detection.hdf5')


# anchor config 
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# expand dim for anchors
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {
    0: "Mask",
    1: "NoMask",
}

def evaluate_fmr(image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160)):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :return:
    '''
    output_info = []
    
    img = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  
    image_exp = np.expand_dims(image_np, axis=0)

    y_bboxes_output, y_cls_output = keras_inference(model, image_exp)
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]

    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh,)

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]

        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    return output_info

    # if len(output_info)>1:
    #     # Mas de una persona detectada
    #     return {
    #         "status": 1,
    #         "description": "More than one person detected",
    #     }
    # else:
    #     try:
    #         if output_info[0][0] == 1:
    #             # Sin mascarilla
    #             return {
    #                 "status": 0,
    #                 "mask": False,
    #                 "description": "Foto correcta",
    #             }
    #         else: 
    #             # Con mascarilla 
    #             return {
    #                 "status": 0,
    #                 "mask": True,
    #                 "description": "Mascarilla detectada",
    #             }
    #     except IndexError:
    #         return{
    #             "status": 2,
    #             "description": "No face detected"
    #         }