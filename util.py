import cv2
import numpy as np
import pandas as pd


colors = [
    (52, 64, 235), 
    (181, 81, 18), 
    (250, 175, 25), 
    (23, 21, 18), 
    (76, 217, 48),
    (255, 140, 8), 
    (20, 20, 20), 
    (255, 51, 51), 
    (227, 211, 68), 
    (137, 227, 68), 
    (251, 255, 18), 
    (250, 7, 16),
]

names = ['7 Mares', 'ML Chiltepin', 'ML Habanero', 'ML HabaneroT', 'ML HabaneroV', 'ML PicanteN', 
         'Pekin', 'S Amor', 'S AmorC', 'S AmorL', 'Son Limon', 'Son Roja']

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def postprocess_boxes(boxes: np.array, ratio: float, dwdh: tuple):
    """Translate bounding boxes from model input shape to original image shape

    Args:
        boxes (np.array): model bounding boxes output
        r (float): scale ratio
        dwdh (tuple): image padding

    Returns:
        torch.tensor: the new bounding boxes
    """
    boxes -= np.array(dwdh*2)
    boxes /= ratio
    return boxes


def prepare_input(img: np.array, img_size: tuple=(640,640)) -> tuple:
    im, ratio, dwdh = letterbox(img, img_size, auto=False)
    im = im[:, :, ::-1].transpose(2, 0, 1)
    im = np.expand_dims(im, 0)
    im = np.ascontiguousarray(im)

    im = im.astype(np.float32)
    im /= 255
        
    return im, ratio, dwdh


def draw_annot(img, box, text, color):
    cv2.rectangle(img, box[:2], box[2:],color,1)
    cv2.rectangle(img, (int(box[0]), int(box[1])-11), (int(box[2]), int(box[1])-2), color, -1)
    cv2.putText(img,text,(int(box[0]), int(box[1]) - 3),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255), thickness=1)


def boxes_to_clusters(boxes, classes):
    # Process bounding boxes, find centers and create result dictionary for products
    rects = [r for r in boxes]
    centros = np.array([(r[0] + (r[2]-r[0])/2, r[1] + (r[3]-r[1])/2, names[cat]) for r, cat in zip(rects, classes)])

    objects_df = pd.DataFrame(centros, columns=["centro_x", "centro_y", "producto"])
    objects_df["centro_x"] = objects_df["centro_x"].astype("float")
    objects_df["centro_y"] = objects_df["centro_y"].astype("float")
    objects_df = objects_df.sort_values(by=["centro_y", "centro_x"], ascending=[True, True])

    # Create products dict
    conteo, sep = np.histogram(objects_df["centro_y"])
    result_dict = {}
    n_filas = 0
    for i in range(len(conteo)):
        if conteo[i] != 0:
            n_filas += 1
            productos = {}
            for _, y,class_name in objects_df.values:
                if sep[i+1] >=  y >= sep[i]:
                    productos.setdefault(class_name, 0)
                    productos[class_name] += 1

            result_dict[f"fila {n_filas}"] = productos
    
    return result_dict



# def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = img.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return img, ratio, (dw, dh)