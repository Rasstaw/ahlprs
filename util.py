import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'S': '5', 'B': '8', 'G': '9'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '5': 'S', '8': 'B', '9': 'G'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


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


def format_license(text):
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


def read_lp(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '').replace('.', '-').replace("'", ' ').replace(":", '-')
        print(f"OCR Detection - Processed Text: {text}")

        if license_complies(text):
            formatted_text = format_license(text)
            print(f"Formatted License Plate: {formatted_text}")
            return formatted_text, score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1