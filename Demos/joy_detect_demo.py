#!/usr/bin/env python3

from picamera import PiCamera
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.annotator import Annotator
from aiy.leds import Leds, Color
from aiy.board import Board

import contextlib
import time

scale_x = 320 / 1640
scale_y = 240 / 1232

JOY_COLOR = (255, 70, 0)
SAD_COLOR = (0, 0, 64)
ALPHA = 64

# Records
max_joy = 0.0
max_faces = 0

def avg_joy_score(faces):
    if faces:
        return sum(face.joy_score for face in faces) / len(faces)
    return 0.0

def transform(bounding_box):
    x, y, width, height = bounding_box
    return (scale_x * x, scale_y * y, scale_x * (x + width),
            scale_y * (y + height))

def reset_records():
    global max_joy, max_faces
    print('Records reset ..')
    max_joy = 0.0
    max_faces = 0

with contextlib.ExitStack() as stack:
    leds   = stack.enter_context(Leds())
    board = stack.enter_context(Board())
    camera = stack.enter_context(PiCamera(sensor_mode=4, resolution=(820, 616)))

    annotator = Annotator(camera, dimensions=(320, 240), bg_color=(0,0,0,0))
    camera.start_preview()

    print ("Loading model - hold on ..")

    board.button.when_pressed = reset_records

    # Do inference on VisionBonnet
    with CameraInference(face_detection.model()) as inference:
        try:   
            print("Ready ..")
            for result in inference.run():
                leds.update(Leds.rgb_on(Color.RED))
                annotator.clear()
                faces = face_detection.get_faces(result)

                if len(faces) >= 1:
                    leds.update(Leds.rgb_on(Color.GREEN))
                    for face in faces:
                        bbox=transform(face.bounding_box)
                        color=Color.blend(JOY_COLOR, SAD_COLOR, face.joy_score)
                        annotator.bounding_box(bbox, fill=color+(ALPHA,), outline=color)
                        score_string = f"Joy: {face.joy_score*100:.1f}%"
                        annotator.text((bbox[0], bbox[1]), score_string, color)

                average_joy = avg_joy_score(faces)
                max_joy = max(max_joy, average_joy)
                max_faces = max(len(faces), max_faces)
                status=f"Average joy  : {average_joy*100:7.3f}% -- JOY RECORD  : {max_joy*100:7.3f}% !!"
                annotator.text((2,220), status, (255, 255, 0))
                status=f"Current faces: {len(faces):7d}  -- FACES RECORD: {max_faces:7d}  !!"
                annotator.text((2,210), status, (255, 255, 0))
                annotator.update()

        except KeyboardInterrupt:
            print("Interrupted ..")

    leds.update(Leds.rgb_off())
    camera.stop_preview()

    # Servo back to the middle upon ending
    print("Done")
