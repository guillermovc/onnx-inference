import json
import cv2
import numpy as np
import numpy as np
import onnxruntime as ort

from util import *

IMG_SIZE = 640
STRIDE = 32

accepted_types = [
    'image/bmp', 'image/jpg', 'image/jpeg', 'image/png', 'image/tif', 
    'image/tiff', 'image/dng', 'image/webp', 'image/mpo'
]

# 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-sagemaker

CONF_THRES = 0.65


def model_fn(onnx_model_dir: str, context=None):
    """
    Lee un archivo onnx dado

    """
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_dir, providers=providers)
    
    model = {
        "session": session,
        "outname": [i.name for i in session.get_outputs()],
        "inname": [i.name for i in session.get_inputs()],
    }

    return model


def input_fn(request_body, request_content_type, context):
    
    if request_content_type in accepted_types:        
        img = np.frombuffer(request_body, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        im, ratio, dwdh = prepare_input(img)
        # print(img.shape)

        input_object = {
            "original_img": img,
            "preprocessed_img": im,
            "ratio": ratio,
            "dwdh": dwdh,
        }
        
        return input_object

    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise ValueError("Content-Type not supported")


def predict_fn(input_object, model, context):
    """
    Runs inference on preprocessed input image

    Returns a list of predictions info.
    e.g. [[x1, y1, x2, y2, score, class] ...]
    """
    
    # Unpack variables
    original_img = input_object["original_img"]
    input_img = input_object["preprocessed_img"]
    ratio = input_object["ratio"]
    dwdh = input_object["dwdh"]

    # Unpack model dictionary
    session = model["session"]
    outname = model["outname"]
    inname = model["inname"]

    inp = {inname[0]: input_img}
    # print(f"input_img shape: {input_img.shape}")
    outputs = session.run(outname, inp)[0]
    outputs = outputs[:, 1:] # 0 index elements its batch id, not needeed

    translated_boxes = postprocess_boxes(outputs[:,:4], ratio, dwdh)

    scores = outputs[:,5]
    classes = outputs[:,4].astype(int).tolist()
    # labels = [names[int(cl)] + ": " + str(round(float(score), 3)) for cl, score in zip(classes, scores)]

    result = {
        "original_img": original_img,
        "bboxes": translated_boxes,
        "scores": scores,
        "classes": classes,
    }

    return result


def output_fn(prediction, response_content_type, context):

    original_img = prediction["original_img"]
    translated_boxes = prediction["bboxes"]
    scores = prediction["scores"]
    classes = prediction["classes"]

    result_dict = boxes_to_clusters(translated_boxes, classes)

    # Draw bounding boxes on image copy
    annotated_img = original_img.copy()
    # print(translated_boxes)
    for i in range(len(translated_boxes)):
        label = names[classes[i]] + " " + str(int(round(scores[i], 2)*100)) + "%"
        draw_annot(annotated_img, translated_boxes[i].astype(int), label, colors[classes[i]])
    # cv2.imwrite("result.png", annotated_img)

    # Convert np array to list so it can be serialized
    annotated_img = annotated_img.tolist()

    result = {
        "products_info": result_dict,
        "annotated_img": annotated_img,
    }

    result = json.dumps(result)
    return result.encode()
