import subprocess
import sys
import tempfile
import streamlit as st
import numpy as np
from PIL import Image, ExifTags
from ultralytics import YOLO
import easyocr
import pandas as pd
import uuid
import os
import csv
from util import get_car, read_lp, write_csv
from scipy.interpolate import interp1d
import cv2
from visualize import visualize  # Import the visualize function
import git

# Initialize and update git submodules
subprocess.run([sys.executable, "-m", "pip", "install", "gitpython"])
repo = git.Repo('.')
repo.submodule_update(init=True, recursive=False)

from sort.sort import Sort


# App configuration
st.set_page_config(
    page_title="Automatic Hungarian  License Plate Recognition",
    page_icon="./Assets/license-plate.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'mailto:ashumarks@gmail.com',
        'Report a bug': "https://github.com/Rasstaw/license_plate_recognition",
        'About': "https://github.com/Rasstaw/license_plate_recognition"
                 "vehicle License plate detection and recognition streamlit web application"
    }
)

############################################################################
# setting file Paths
############################################################################
demo = "./Assets/Images/demo.jpg"

# Check if the directory exists, if not, create it
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

folder_path = "./output/licenses_plates_imgs_detected/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# instance of easyocr and vehicle classes
lp_reader = easyocr.Reader(['en'], gpu=True)
vehicles = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}

LICENSE_PLATE_DETECTION_MODEL_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

#########################################################################
# Correction map for common OCR mis_recognitions
##########################################################################
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'S': '5', 'B': '8', 'G': '9'}

dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '5': 'S', '8': 'B', '9': 'G'}

correction_map = {
    'L': '4', 'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '9', 'J': '3'
}


def run_visualize_script(video_path, interpolated_csv_path, output_video_path):
    try:
        visualize(video_path, interpolated_csv_path, output_video_path)
        st.success(f"building visuals...✅ Output saved to {output_video_path}.")
    except Exception as e:
        st.error(f"Error running visualize.py: {e}")


def display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        stframe.image(frame, channels="BGR")
    cap.release()


# Function to load YOLO model (can be reused for both models)
@st.cache_resource(ttl=30 * 24 * 3600)  # Cache for 1month it's never going to change so ...
def load_yolo_model(model_dir):
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' not found.")
    return YOLO(model_dir)


###############################################################################
# Processing the video
###############################################################################
def process_video(input_video_path, output_csv_path):
    results = {}
    mot_tracker = Sort()

    # Load COCO vehicle detection license plate detector model only once and use the cache on the subsequent requests
    coco_model = load_yolo_model(COCO_MODEL_DIR)
    lp_detector = load_yolo_model(LICENSE_PLATE_DETECTION_MODEL_DIR)

    # read video
    cap = cv2.VideoCapture(input_video_path)
    # read frames
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, confidence])

            # track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))
            # detect license plates extract their information and assign them to a car
            license_plates = lp_detector(frame)[0]

            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    lp_crop_gray_de_noised = cv2.fastNlMeansDenoising(license_plate_crop_gray, None, 30, 7, 21)

                    lp_crop_gray_de_noised_hist_equalised_thresh = cv2.adaptiveThreshold(
                        lp_crop_gray_de_noised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                        11, 2
                    )

                    license_plate_text, license_plate_text_score = read_lp(lp_crop_gray_de_noised_hist_equalised_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

    cap.release()
    # write results
    write_csv(results, output_csv_path)
    return output_csv_path, results


def interpolate_bounding_boxes(input_csv_path, output_csv_path):
    # Load the CSV file
    with open(input_csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        data = list(csv_reader)

    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0,
                                           kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {'frame_nmr': str(frame_number), 'car_id': str(car_id),
                   'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
                   'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i]))}

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if
                                int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][
                    0]
                row['license_plate_bbox_score'] = original_row[
                    'license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row[
                    'license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data


def correct_recognized_text(text):
    corrected_text = ''
    for char in text:
        if char in correction_map:
            corrected_text += correction_map[char]
        else:
            corrected_text += char
    return corrected_text


def license_complies(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) == 6 and text[:3].isalpha() and text[3:].isdigit():
        return True
    elif len(text) == 7 and text[:3].isalpha() and text[3] == '-' and text[4:].isdigit():
        return True
    elif len(text) == 8 and text[:2].isalpha() and text[3:5].isalpha() and text[6:].isdigit():
        return True
    elif len(text) == 9 and text[:5].isalpha() and text[6:].isdigit():
        return True
    return False


def format_lp(text):
    """
        Format the license plate text by converting characters using the mapping dictionaries.

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
    lp_ = ''

    if len(text) == 6:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char,
                   3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int}
        for j in range(len(text)):
            if j in mapping and text[j] in mapping[j].keys():
                lp_ += mapping[j][text[j]]
            else:
                lp_ += text[j]
    elif len(text) == 7:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char,
                   4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int}
        for j in range(len(text)):
            if j in mapping and text[j] in mapping[j].keys():
                lp_ += mapping[j][text[j]]
            else:
                lp_ += text[j]

    elif len(text) == 8:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 3: dict_int_to_char,
                   4: dict_int_to_char, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int}
        for j in range(len(text)):
            if j in mapping and text[j] in mapping[j].keys():
                lp_ += mapping[j][text[j]]
            else:
                lp_ += text[j]

    elif len(text) == 9:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 3: dict_int_to_char,
                   4: dict_int_to_char, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int}
        for j in range(len(text)):
            if j in mapping and text[j] in mapping[j].keys():
                lp_ += mapping[j][text[j]]
            else:
                lp_ += text[j]
    return lp_


def extract_lp(lp_crop, img):
    confidences = 0

    # Convert to gray
    lp_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
    # Apply de_noising
    lp_crop_gray_de_noised = cv2.fastNlMeansDenoising(lp_crop_gray, None, 30, 7, 21)
    # Apply adaptive thresholding
    lp_crop_gray_de_noised_hist_equalised_thresh = cv2.adaptiveThreshold(
        lp_crop_gray_de_noised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    st.text("")
    st.sidebar.image(lp_crop_gray_de_noised_hist_equalised_thresh,
                     caption='gray license plate cropped, de_noised and adaptive threshold applied',
                     use_column_width=True)

    recognitions = lp_reader.readtext(lp_crop_gray_de_noised)
    if not recognitions:
        return None, None

    rectangle_size = (lp_crop_gray_de_noised_hist_equalised_thresh.shape[0] *
                      lp_crop_gray_de_noised_hist_equalised_thresh.shape[1])
    plate = []

    for recognition in recognitions:
        length = np.sum(np.subtract(recognition[0][1], recognition[0][0]))
        height = np.sum(np.subtract(recognition[0][2], recognition[0][1]))
        b_box, text, confidence = recognition
        text = text.upper().replace(' ', '').replace('.', '-').replace("'", ' ').replace(":", '-')

        if length * height / rectangle_size > 0.17:
            confidences += confidence
            plate.append(text)

            if text is not None and confidence is not None:
                if license_complies(text):
                    return format_lp(text), confidence
                return text, confidence

    if plate:
        return " ".join(plate), confidences / len(plate)
    else:
        return None, 0


def model_prediction(car_img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    car_img = cv2.cvtColor(car_img, cv2.COLOR_RGB2BGR)

    # Load COCO vehicle detection model only once
    with st.spinner("Detecting vehicles..."):
        coco_model = load_yolo_model(COCO_MODEL_DIR)
        object_detections = coco_model(car_img)[0]
        st.success("Detecting vehicles...✅")

    # Load license plate detector model only once
    with st.spinner("Detecting license plates..."):
        lp_detector = load_yolo_model(LICENSE_PLATE_DETECTION_MODEL_DIR)
        license_detections = lp_detector(car_img)[0]
        st.success("Detecting license plates...✅")

    # Dictionary for class names
    class_names = {
        2: 'Car',
        3: 'Motorbike',
        5: 'Bus',
        7: 'Truck'
    }

    # if Vehicles are detected get their bounding boxes, classID and draw a rectangle around it
    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_confidence, class_id = detection

            if int(class_id) in vehicles:
                cv2.rectangle(car_img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 5)

                # Get the class name from the dictionary
                class_name = class_names.get(int(class_id), 'Unknown')

                # Prepare the label with class ID and car confidence
                label = f'{class_name}  confidence : {car_confidence:.2f}'

                # Get the label size
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)

                # Set the top-left corner of the label
                top_left = (int(xcar1), int(ycar1) - 10)

                # Draw a filled rectangle behind the text for better visibility
                cv2.rectangle(car_img, (top_left[0], top_left[1] - label_height - baseline),
                              (top_left[0] + label_width, top_left[1] + baseline),
                              (0, 0, 255), cv2.FILLED)

                # Put the label text on the image
                cv2.putText(car_img, label, (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                            3)

    else:
        st.error("No vehicle detected please check if your input is in upright position")
        st.image(car_img)
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_confidence = 0

    # if License plates are detected get their bounding boxes, crop them,and save them with a unique name
    with st.spinner("Recognising License Plate text..."):
        if len(license_detections.boxes.cls.tolist()) != 0:
            lp_crops_total = []
            recognized_texts = []

            for lp in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = lp

                # Draw rectangle around the detected license plate
                cv2.rectangle(car_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)

                # Prepare the label with "License Plate" and confidence
                label = f'License Plate: confidence {confidence:.2f}'

                # Get the label size
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)

                # Set the top-left corner of the label
                top_left = (int(x1), int(y1) - 10)

                # Draw a filled rectangle behind the text for better visibility
                cv2.rectangle(car_img, (top_left[0], top_left[1] - label_height - baseline),
                              (top_left[0] + label_width, top_left[1] + baseline),
                              (0, 255, 0), cv2.FILLED)

                # Put the label text on the image
                cv2.putText(car_img, label, (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                            3)

                lp_crop = car_img[int(y1):int(y2), int(x1): int(x2), :]
                img_name = '{}.jpg'.format(uuid.uuid1())

                # saving the cropped License plates
                cv2.imwrite(os.path.join(folder_path, img_name), lp_crop)

                # read license plate text
                lp_text, lp_confidence = extract_lp(lp_crop, car_img)
                licenses_texts.append(lp_text)

                if not lp_text:
                    st.image(lp_crop)
                    st.error("No license plate text recognized.")

                if lp_text is not None and lp_confidence is not None:
                    lp_crops_total.append(lp_crop)
                    results[license_numbers] = {
                        'car': {'b_box': [xcar1, ycar1, xcar2, ycar2], 'car_confidence': car_confidence},
                        'lp': {'b_box': [x1, y1, x2, y2],
                               'text': lp_text,
                               'b_box_confidence': confidence,
                               'confidence': lp_confidence}
                    }

                    license_numbers += 1

                if lp_text:
                    recognized_texts.append(f"{len(recognized_texts) + 1}. {lp_text}")

            # Display all recognized license plate texts in a single success message
            if recognized_texts:
                recognized_texts_str = ":\n" + "\n".join(recognized_texts)
                st.success(f"Recognized License Plate text  {recognized_texts_str} ")

            return [car_img, licenses_texts, lp_crops_total, results]

        else:
            img_wth_box = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
            st.error("No license plates Detected.")
            return [img_wth_box, [], [], results]


def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, None)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image


def write_to_csv(results, detections_csv_file):
    with open(detections_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['Image Name', 'Vehicle b_box', 'Car confidence',
                      'License Plate b_box', 'License Plate Text',
                      'b_box confidence', 'Text confidence']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_name, image_results in results.items():
            for license_number, details in image_results.items():
                writer.writerow({
                    'Image Name': image_name,
                    'Vehicle b_box': [
                        f"{details['car']['b_box'][0]:.2f}" if 'car' in details else "N/A",
                        f"{details['car']['b_box'][1]:.2f}" if 'car' in details else "N/A",
                        f"{details['car']['b_box'][2]:.2f}" if 'car' in details else "N/A",
                        f"{details['car']['b_box'][3]:.2f}" if 'car' in details else "N/A"
                    ],
                    'Car confidence': f"{details['car']['car_confidence']:.2f}" if 'car' in details else "N/A",
                    'License Plate b_box': [
                        f"{details['lp']['b_box'][0]:.2f}" if 'lp' in details else "N/A",
                        f"{details['lp']['b_box'][1]:.2f}" if 'lp' in details else "N/A",
                        f"{details['lp']['b_box'][2]:.2f}" if 'lp' in details else "N/A",
                        f"{details['lp']['b_box'][3]:.2f}" if 'lp' in details else "N/A"
                    ],
                    'License Plate Text': details['lp']['text'] if 'lp' in details else "N/A",
                    'b_box confidence': f"{details['lp']['b_box_confidence']:.2f}" if 'lp' in details else "N/A",
                    'Text confidence': f"{details['lp']['confidence']:.2f}" if 'lp' in details else "N/A"
                })


def main():
    # setting the title of the app
    with st.container():
        # Header section
        st.markdown(
            """
            <h1 style='text-align: center; font-size: 28px;'>
                Automatic Hungarian License Plate Recognition
            </h1>
            """, unsafe_allow_html=True)

    # Create a selection box to choose recognition mode
    app_mode = st.selectbox(
        "Choose Input Type", ["Run on Single Image", "Run on Multiple Images", "Run on Video"]

    )
    st.write("you selected to : ", app_mode)
    st.header("")

    if app_mode == "Run on Single Image":
        uploaded_file = st.file_uploader("upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.text("Your uploaded Image is displayed on the sidebar:")
            st.sidebar.header("Uploaded Image")
            st.sidebar.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
            st.sidebar.write("--------")

            if st.button("Apply Recognition"):
                all_results = {}
                file_name = uploaded_file.name
                image = Image.open(uploaded_file)
                image = correct_image_orientation(image)
                image = np.array(image)

                with st.spinner(f"Processing image: {file_name}"):
                    predictions = model_prediction(image)
                    if len(predictions) == 4:
                        img_wth_box, licenses_texts, lp_crops_total, results = predictions
                        st.sidebar.image(img_wth_box, caption=f"detected vehicles and License plates on image:"
                                                              f" {file_name}", use_column_width=True)

                        # Collect results for CSV writing
                        if licenses_texts:
                            all_results[file_name] = results

                    else:
                        st.image(predictions[0], caption='No license plates detected', use_column_width=True)

                st.markdown("---")
                st.success(f"Processing image: {file_name}✅")

                # Write all results to a CSV file
                write_to_csv(all_results, "./output/single_image_detection_results.csv")

                st.text("Recognition Records")
                df = pd.read_csv(f"./output/single_image_detection_results.csv")
                st.dataframe(df)

    elif app_mode == "Run on Multiple Images":
        uploaded_files = st.file_uploader("Upload image files...", type=["jpg", "jpeg", "png"],
                                          accept_multiple_files=True)
        if uploaded_files:
            st.text("Your uploaded Images are displayed on the sidebar:")
            st.sidebar.header("Uploaded files")

            for uploaded_file in uploaded_files:
                st.sidebar.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
                st.sidebar.write("--------")

            if st.button("Apply Recognitions"):
                all_results = {}

                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    image = Image.open(uploaded_file)
                    image = correct_image_orientation(image)
                    image = np.array(image)

                    with st.spinner(f"Processing image: {file_name}"):
                        predictions = model_prediction(image)
                        if len(predictions) == 4:
                            img_wth_box, licenses_texts, lp_crops_total, results = predictions
                            st.sidebar.image(img_wth_box, caption=f"detected vehicles and License plates on image:"
                                                                  f" {file_name}", use_column_width=True)

                            if licenses_texts:
                                all_results[file_name] = results

                        else:
                            st.image(predictions[0], caption='No license plates detected', use_column_width=True)
                    st.markdown("---")

                # Write all results to a CSV file
                write_to_csv(all_results, "./output/multi_image_detection_results.csv")

                st.text("Recognition Records")
                df = pd.read_csv(f"./output/multi_image_detection_results.csv")
                st.dataframe(df)

    elif app_mode == "Run on Video":
        uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_video is not None:
            t_file = tempfile.NamedTemporaryFile(delete=False)
            t_file.write(uploaded_video.read())
            input_video_path = t_file.name
            st.sidebar.video(input_video_path)

            output_csv_path = './output/Detection_Results_from_video.csv'
            interpolated_csv = output_csv_path.replace('.csv', '_interpolated.csv')
            output_video_path = './output/output_visualized.mp4'

            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    output_csv_path, results = process_video(input_video_path, output_csv_path)
                st.success(f"Processing video...✅ Results saved to {output_csv_path}")

            if st.button("Interpolate Missing Data"):
                with st.spinner("Interpolating data..."):
                    interpolated_data = interpolate_bounding_boxes(output_csv_path, interpolated_csv)

                    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score',
                              'license_number', 'license_number_score']
                    with open(interpolated_csv, 'w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=header)
                        writer.writeheader()
                        writer.writerows(interpolated_data)
                st.success(f"Interpolating data...✅ Results saved to {interpolated_csv}")

            if st.button("Visualize Results"):
                with st.spinner("building visuals..."):
                    run_visualize_script(input_video_path, interpolated_csv, output_video_path)

            if st.button("View Video"):
                display_video('output_video.mp4')


if __name__ == "__main__":
    main()
