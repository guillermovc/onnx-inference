from collections import namedtuple
import json
from pprint import pprint

import cv2
import numpy as np

from inference import input_fn, output_fn, predict_fn, model_fn

Context = namedtuple("Context", "system_properties")

MODEL_DIR = "./best9feb_nms.onnx"
context = Context({"gpu_id" : 0})


def run(request_body, input_type):
    model = model_fn(MODEL_DIR, context)
    inp = input_fn(request_body, input_type, context)
    out = predict_fn(inp, model, context)
    out = output_fn(out, "application/json", context)

    return out
    

def main():
    with open("./prueba3.jpg", "rb") as f:
        body = f.read()
    result = run(body, "image/jpeg")
    # print(type(result))
    result = result.decode()
    result = json.loads(result)
    # print(type(result))
    
    products_info = result["products_info"]
    annotated_img = result["annotated_img"]
    annotated_img = np.array(annotated_img, dtype=np.uint8)
    # print(annotated_img.shape)
    # print(annotated_img.dtype)
    
    # annotated_img = cv2.resize(annotated_img, (640, 800))

    print("Detecciones de la imagen")
    pprint(products_info)
    cv2.imwrite("result_nuevo.png", annotated_img)
    print(f"Imagen guardada en result_nuevo.png")
    # cv2.imshow("Productos encontrados", annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result

if __name__ == "__main__":
    main()