import argparse
import typing
import PySimpleGUI as sg
import cv2 as cv
import numpy as np

import detectors as det


############ GUI Layout
image_column = [
    [  
        sg.Image(filename="", key="-IMAGE-") 
    ]
]

qr_codes_column = [
    [ 
        sg.Listbox(values=[], size=(40,20), key="-QRCODES-")
    ],
    [   
        sg.Button("Clear", size=(10,1)),
        sg.Button("Export", size=(10,1))
    ]
]

layout = [
    [
        sg.Column(image_column),
        sg.VSeparator(),
        sg.Column(qr_codes_column),

    ]
]
############


def write_qr_codes_to_file(file_name: str, qr_codes: typing.Set[str]):
    with open(file_name, "w") as file:
        for qr_code in qr_codes:
            file.write(qr_code + "\n")


def draw_text(
        image: np.ndarray, 
        text: str, 
        pos: typing.Tuple[int,int], 
        color: typing.Tuple[int,int,int],
        font: int = cv.FONT_HERSHEY_SIMPLEX,
        font_scale: int = 1,
        font_thickness: int = 1,
    ):
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(image, pos, (x + text_w, y + text_h + 1), color, -1)
    if text:
        cv.putText(image, text, (x, y + text_h + font_scale - 1), font, font_scale, (0, 0, 0), font_thickness)


def draw_bounding_boxes(image: np.ndarray, boxes: typing.List[det.BoundingBox]):
    color = (0, 255, 255)
    thickness = 3
    for box in boxes:
        cv.rectangle(image, box.upper_left_corner(), box.lower_right_corner(), color, thickness)
        qr_code = box.get_qr_code()
        if qr_code:
            draw_text(image, qr_code, box.upper_left_corner(), color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="path to weights (.pt) file for YOLOv8 model")
    parser.add_argument("--width", type=int, help="width of images, default = 480", default=480)
    parser.add_argument("--height", type=int, help="height of images, default = 600", default=600)
    parser.add_argument("-c", "--confidence", type=float, help="confidence threshold, default = 0.8", default=0.8)
    args = parser.parse_args()

    if (not args.weights):
        print("No weights file was provided")
        exit()

    window = sg.Window("QR Code Detector", layout)
    object_detector = det.BoxObjectDetector(args.weights, args.width, args.height, args.confidence)
    qr_code_finder = det.QRCodeFinder()

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED:
            break
        elif event == "Clear":
            qr_code_finder.clear_qr_codes()
            continue
        elif event == "Export":
            name_file = sg.PopupGetFile('Enter Filename', save_as=True)
            if name_file:
                write_qr_codes_to_file(name_file, qr_code_finder.get_qr_codes()) 
            continue

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        bounding_boxes = object_detector.find_objects(frame)
        qr_code_finder.find_qr_codes(frame, bounding_boxes) # will add qr codes to bounding boxes
        qr_codes = qr_code_finder.get_qr_codes()
        draw_bounding_boxes(frame, bounding_boxes)

        imgbytes = cv.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
        window["-QRCODES-"].update(qr_codes)

    window.close()
