import streamlit as st
import numpy as np
from PIL import Image, ExifTags
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import uuid
import os
import csv

st.set_page_config(
    page_title="Automatic Hungarian License Plate Recognition",
    page_icon="./Assets/license-plate.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'mailto:ashumarks@gmail.com',
        'Report a bug': "https://github.com/Rasstaw/AHLPR",
        'About': "https://github.com/Rasstaw/AHLPR This is a header. "
                 "vehicle Detection, License plate detection and recognition streamlit web application"
    }
)

############################################################################
# Recognition from Image
############################################################################

# file Paths
demo = "./Assets/Images/demo.jpg"
folder_path = "./licenses_plates_imgs_detected/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

# instance of easyocr and vehicle classes
reader = easyocr.Reader(['en'], gpu=False)
vehicles = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}


dict_char_to_int = {
    'O': '0', 'I': '1', 'J': '3', 'G': '6', 'S': '5', 'L': '4',
    'Z': '2', 'B': '8'
}
dict_int_to_char = {
    '0': 'O', '1': 'I', '3': 'J', '6': 'G', '5': 'S', '4': 'L'
}

# Correction map for common OCR misrecognitions
correction_map = {
    'L': '4', 'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'
}


# Function to load YOLO model (can be reused for both models)
@st.cache_resource(ttl=30*24*3600)  # Cache for 1month it's neve going to change so ...
def load_yolo_model(model_dir):
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' not found.")
    return YOLO(model_dir)


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
    if len(text) == 7 and text[:3].isalpha() and text[3:].isdigit():
        return True
    elif len(text) == 8 and text[:3].isalpha() and text[3] == '-' and text[4:].isdigit():
        return True
    return False


def format_license_plate(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char,
               4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int}

    for j in range(len(text)):
        if j in mapping and text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def extract_license_plate(license_plate_crop, img):
    scores = 0

    # Convert to gray
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    # Apply de_noising
    license_plate_crop_gray_de_noised = cv2.fastNlMeansDenoising(license_plate_crop_gray, None, 30, 7, 21)

    # Apply adaptive thresholding
    license_plate_crop_gray_de_noised_hist_equalised_thresh = cv2.adaptiveThreshold(
        license_plate_crop_gray_de_noised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    st.text("")
    # st.text("")
    st.sidebar.image(license_plate_crop_gray_de_noised,
                     caption='gray license plate cropped, de_noised and adaptive threshold applied',
                     use_column_width=True)

    recognitions = reader.readtext(license_plate_crop_gray_de_noised)
    if not recognitions:
        return None, None

    rectangle_size = (license_plate_crop_gray_de_noised.shape[0] *
                      license_plate_crop_gray_de_noised.shape[1])
    plate = []

    for recognition in recognitions:
        length = np.sum(np.subtract(recognition[0][1], recognition[0][0]))
        height = np.sum(np.subtract(recognition[0][2], recognition[0][1]))
        bbox, text, confidence = recognition
        text = text.upper().replace(' ', '').replace('.', '-')

        if length * height / rectangle_size > 0.17:
            scores += confidence
            plate.append(text)

            if text is not None and confidence is not None:
                if license_complies(text):
                    return format_license_plate(text), confidence
                return text, confidence

    if plate:
        return " ".join(plate), scores / len(plate)
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
        license_plate_detector = load_yolo_model(LICENSE_MODEL_DETECTION_DIR)
        license_detections = license_plate_detector(car_img)[0]
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
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles:
                cv2.rectangle(car_img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 5)

                # Get the class name from the dictionary
                class_name = class_names.get(int(class_id), 'Unknown')

                # Prepare the label with class ID and car score
                label = f'{class_name}  Score : {car_score:.2f}'

                # Get the label size
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)

                # Set the top-left corner of the label
                top_left = (int(xcar1), int(ycar1) - 10)

                # Draw a filled rectangle behind the text for better visibility
                cv2.rectangle(car_img, (top_left[0], top_left[1] - label_height - baseline),
                              (top_left[0] + label_width, top_left[1] + baseline),
                              (0, 0, 255), cv2.FILLED)

                # Put the label text on the image
                cv2.putText(car_img, label, (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    else:
        st.error("No vehicle detected please check if your input is in upright position")
        st.image(car_img)
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_score = 0

    # if License plates are detected get their bounding boxes, crop them,and save them with a unique name
    with st.spinner("Recognising License Plate text..."):
        if len(license_detections.boxes.cls.tolist()) != 0:
            license_plate_crops_total = []
            recognized_texts = []

            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Draw rectangle around the detected license plate
                cv2.rectangle(car_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)

                # Prepare the label with "License Plate" and score
                label = f'License Plate: Score {score:.2f}'

                # Get the label size
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)

                # Set the top-left corner of the label
                top_left = (int(x1), int(y1) - 10)

                # Draw a filled rectangle behind the text for better visibility
                cv2.rectangle(car_img, (top_left[0], top_left[1] - label_height - baseline),
                              (top_left[0] + label_width, top_left[1] + baseline),
                              (0, 255, 0), cv2.FILLED)

                # Put the label text on the image
                cv2.putText(car_img, label, (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

                license_plate_crop = car_img[int(y1):int(y2), int(x1): int(x2), :]
                img_name = '{}.jpg'.format(uuid.uuid1())

                # saving the cropped License plates
                cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

                # read license plate text
                license_plate_text, license_plate_text_score = extract_license_plate(license_plate_crop, car_img)
                licenses_texts.append(license_plate_text)

                if not license_plate_text:
                    st.image(license_plate_crop)
                    st.error("No license plate text recognized.")

                if license_plate_text is not None and license_plate_text_score is not None:
                    license_plate_crops_total.append(license_plate_crop)
                    results[license_numbers] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                          'text': license_plate_text,
                                          'bbox_score': score,
                                          'text_score': license_plate_text_score}
                    }

                    license_numbers += 1

                if license_plate_text:
                    recognized_texts.append(f"{len(recognized_texts) + 1}. {license_plate_text}")

            # write_to_csv(results, f"./csv_detections/detection_results.csv")

            # img_wth_box = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)

            # Display all recognized license plate texts in a single success message
            if recognized_texts:
                recognized_texts_str = ":\n" + "\n".join(recognized_texts)
                st.success(f"Recognized License Plate text  {recognized_texts_str} ")

            return [car_img, licenses_texts, license_plate_crops_total, results]

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
        fieldnames = ['Image Name', 'Vehicle BBox', 'Car Score',
                      'License Plate BBox', 'License Plate Text',
                      'BBox Score', 'Text Score']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_name, image_results in results.items():
            for license_number, details in image_results.items():
                writer.writerow({
                    'Image Name': image_name,
                    'Vehicle BBox': [
                        f"{details['car']['bbox'][0]:.2f}" if 'car' in details else "N/A",
                        f"{details['car']['bbox'][1]:.2f}" if 'car' in details else "N/A",
                        f"{details['car']['bbox'][2]:.2f}" if 'car' in details else "N/A",
                        f"{details['car']['bbox'][3]:.2f}" if 'car' in details else "N/A"
                    ],
                    'Car Score': f"{details['car']['car_score']:.2f}" if 'car' in details else "N/A",
                    'License Plate BBox': [
                        f"{details['license_plate']['bbox'][0]:.2f}" if 'license_plate' in details else "N/A",
                        f"{details['license_plate']['bbox'][1]:.2f}" if 'license_plate' in details else "N/A",
                        f"{details['license_plate']['bbox'][2]:.2f}" if 'license_plate' in details else "N/A",
                        f"{details['license_plate']['bbox'][3]:.2f}" if 'license_plate' in details else "N/A"
                    ],
                    'License Plate Text': details['license_plate']['text'] if 'license_plate' in details else "N/A",
                    'BBox Score': f"{details['license_plate']['bbox_score']:.2f}" if 'license_plate' in details else "N/A",
                    'Text Score': f"{details['license_plate']['text_score']:.2f}" if 'license_plate' in details else "N/A"
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
                        img_wth_box, licenses_texts, license_plate_crops_total, results = predictions
                        st.sidebar.image(img_wth_box, caption=f"detected vehicles and License plates on image:"
                                                              f" {file_name}", use_column_width=True)

                        # Collect results for CSV writing
                        if licenses_texts:
                            all_results[file_name] = results

                    else:
                        st.image(predictions[0], caption='No license plates detected', use_column_width=True)

                st.markdown("---")
                # Write all results to a CSV file
                write_to_csv(all_results, "./csv_detections/single_image_detection_results.csv")

                st.text("Recognition Records")
                df = pd.read_csv(f"./csv_detections/single_image_detection_results.csv")
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
                            img_wth_box, licenses_texts, license_plate_crops_total, results = predictions
                            st.sidebar.image(img_wth_box, caption=f"detected vehicles and License plates on image:"
                                                                  f" {file_name}", use_column_width=True)

                            if licenses_texts:
                                all_results[file_name] = results

                        else:
                            st.image(predictions[0], caption='No license plates detected', use_column_width=True)
                    st.markdown("---")

                # Write all results to a CSV file
                write_to_csv(all_results, "./csv_detections/multi_image_detection_results.csv")

                st.text("Recognition Records")
                df = pd.read_csv(f"./csv_detections/multi_image_detection_results.csv")
                st.dataframe(df)

    # elif app_mode == "Run on Picture":
        # img = st.sidebar.camera_input("Take a picture")
        # handle_image_input(img)

    elif app_mode == "Run on Video":
        st.error("Video recognition mode is not yet implemented.")


if __name__ == "__main__":
    main()
