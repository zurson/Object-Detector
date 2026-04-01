import time

from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Settings
counted_items = set()
current_ids = []
current_classes = []

confidence = 0.5
counter_area_in_pixels = 30
fps_each_frame = 5
curr_frame_counter = 0
fps = 0
window_name = 'iDetector Pro'

frame_width = 1280
frame_height = 720
frame_width_padding = frame_width // 3
frame_height_padding = frame_height // 7
text_pos_x = frame_width_padding // 28
text_pos_y = frame_height_padding // 24
fps_x = frame_width - 110

counter_line_points = [(500, 0), (500, 720)]
font = ImageFont.truetype("arial.ttf", 30)


def get_items_amounts():
    pens = 0
    crayons = 0
    markers = 0
    defects = 0

    for val in counted_items:
        _, name = val

        match name:
            case 'pen':
                pens += 1
            case 'crayon':
                crayons += 1
            case 'marker':
                markers += 1
            case 'defect':
                defects += 1

    return pens, crayons, markers, defects


def update_stats(draw):
    pens, crayons, markers, defects = get_items_amounts()
    draw.text((text_pos_x, text_pos_y), 'Pens: ' + str(pens), font=font, fill=(0, 255, 0))
    draw.text((text_pos_x, text_pos_y + 30), 'Crayons: ' + str(crayons), font=font, fill=(0, 255, 0))
    draw.text((text_pos_x, text_pos_y + 60), 'Markers: ' + str(markers), font=font, fill=(0, 255, 0))
    draw.text((text_pos_x, text_pos_y + 90), 'Defects: ' + str(defects), font=font, fill=(0, 255, 0))

    draw.text((fps_x, text_pos_y), 'FPS: ' + str(round(fps)), font=font, fill=(0, 255, 0))


def count_fps(start_time):
    elapsed_time = time.time() - start_time

    global curr_frame_counter
    global fps

    if curr_frame_counter == fps_each_frame:
        fps = 1 / elapsed_time
        curr_frame_counter = 0

    curr_frame_counter += 1
    start_time = time.time()

    return start_time


def setup_counter(model):
    counter = object_counter.ObjectCounter()

    counter.set_args(
        view_img=False,
        reg_pts=counter_line_points,
        classes_names=model.names,
        draw_tracks=True,
        view_in_counts=False,
        view_out_counts=False
    )

    return counter


def open_video_stream(src):
    cap = cv2.VideoCapture(src)

    if cap.isOpened() is False:
        print("Stream is not valid!")
        exit(1)

    return cap


def setup_model(dataset, device):
    model = YOLO(dataset).to(device)
    return model


def get_frame(cap):
    _, frame = cap.read()
    return frame


def insert_stats_to_frame(frame):
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    draw = ImageDraw.Draw(frame)
    update_stats(draw)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)

    return frame


def count_items(model, results, counter, frame):
    global current_ids
    global current_classes

    current_classes = results[0].boxes.cls
    current_ids = results[0].boxes.id
    coords_list = results[0].boxes

    counter_line_x = counter_line_points[0][0]

    for i, coord in enumerate(coords_list):
        x, _, _, _ = coord.xywh[0, :]
        x = float(x)

        if counter_line_x - counter_area_in_pixels < x < counter_line_x:

            if current_ids is None or current_classes is None:
                break

            item_id = int(current_ids[i])
            item_class_name = model.names[int(current_classes[i])]

            new_item = (item_id, item_class_name)
            counted_items.add(new_item)

    counter.start_counting(frame, results)


def run_program():
    model = setup_model('weights.pt', 'cuda')
    cap = open_video_stream("http://192.0.0.4:8080/video")
    # cap = open_video_stream(0)
    # cap = open_video_stream("http://192.168.158.164:8080/video")
    counter = setup_counter(model)
    start_time = time.time()
    global fps

    while cap.isOpened():
        frame = get_frame(cap)
        start_time = count_fps(start_time)

        results = model.track(frame, persist=True, show=False, conf=confidence, verbose=False)
        count_items(model, results, counter, frame)

        frame = insert_stats_to_frame(frame)

        cv2.imshow(window_name, frame)
        # cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


run_program()
