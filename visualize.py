import ast
import numpy as np
import pandas as pd
import logging
import cv2


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def is_within_frame(coords, frame_width, frame_height):
    x1, y1, x2, y2 = coords
    return 0 <= x1 < frame_width and 0 <= y1 < frame_height and 0 <= x2 < frame_width and 0 <= y2 < frame_height


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def adjust_bbox_to_frame(x1, y1, x2, y2, frame_width, frame_height):
    x1 = clamp(x1, 0, frame_width - 1)
    y1 = clamp(y1, 0, frame_height - 1)
    x2 = clamp(x2, 0, frame_width - 1)
    y2 = clamp(y2, 0, frame_height - 1)
    return x1, y1, x2, y2


def visualize(video_path, results_path, output_path):
    # results = pd.read_csv('./output/Detection_Results_from_video_interpolated.csv')
    results = pd.read_csv(results_path)


    # load video
    # video_path = './short_but_clean.mp4'
    # frame = './output/output_video.mp4'
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {'license_crop': None,
                                 'license_plate_number': results[(results['car_id'] == car_id) &
                                                                 (results['license_number_score'] == max_)][
                                     'license_number'].iloc[0]}
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                 (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                  (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[
                                              0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ',
                                                                                                                   ','))

        if is_within_frame((x1, y1, x2, y2), width, height):
            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
            license_plate[car_id]['license_crop'] = license_crop
        else:
            logging.error(f'Invalid license plate bbox for car_id {car_id}: {x1, y1, x2, y2}')

    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                    df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ',
                                                                                                                     ','))

                if not is_within_frame((car_x1, car_y1, car_x2, car_y2), width, height):
                    logging.error(
                        f'Invalid car bbox for car_id {df_.iloc[row_indx]["car_id"]}: {car_x1, car_y1, car_x2, car_y2}')
                    car_x1, car_y1, car_x2, car_y2 = adjust_bbox_to_frame(car_x1, car_y1, car_x2, car_y2, width, height)

                logging.debug(f'Drawing car bbox: {car_x1, car_y1, car_x2, car_y2}')
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 10,
                            line_length_x=200, line_length_y=200)

                # draw license plate
                x1, y1, x2, y2 = ast.literal_eval(
                    df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ')
                    .replace('  ', ' ').replace(' ', ','))

                if not is_within_frame((x1, y1, x2, y2), width, height):
                    logging.error(f'Invalid license plate bbox for car_id {df_.iloc[row_indx]["car_id"]}: {x1, y1, x2, y2}')
                    x1, y1, x2, y2 = adjust_bbox_to_frame(x1, y1, x2, y2, width, height)

                logging.debug(f'Drawing license plate bbox: {x1, y1, x2, y2}')
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 8)

                try:
                    # Draw white rectangle above the detected license plate
                    license_plate_height = int(y2) - int(y1)
                    license_plate_width = int(x2) - int(x1)
                    white_rect_y1 = int(y1) - license_plate_height
                    white_rect_y2 = int(y1)

                    if white_rect_y1 < 0:
                        white_rect_y1 = 0  # Ensure the rectangle is within frame bounds

                    cv2.rectangle(frame, (int(x1), white_rect_y1), (int(x2), white_rect_y2), (255, 255, 255), -1)

                    license_plate_number = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        5)

                    text_x = int(x1) + (license_plate_width - text_width) // 2
                    text_y = white_rect_y1 + (license_plate_height + text_height) // 2

                    logging.debug(
                        f'Placing license plate text: {license_plate_number} at coordinates: ({text_x}, {text_y})')
                    cv2.putText(frame, license_plate_number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)

                except Exception as e:
                    logging.error(f'Error processing frame {frame_nmr} for car_id {df_.iloc[row_indx]["car_id"]}: {e}')

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

    out.release()
    cap.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize license plate recognition results on video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--results", type=str, required=True, help="Path to the CSV file with detection results.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output video file.")

    args = parser.parse_args()

    visualize(args.video, args.results, args.output)
