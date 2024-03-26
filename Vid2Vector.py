import os

import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
OUTPUT_Str = 'output_'
SCALE_REDUCTION = 4


def main():
    # make directories if they don't exist
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Process all video files in the input directory
    for file in os.listdir(INPUT_DIR):
        if file.endswith('.mp4'):
            input_video = os.path.join(INPUT_DIR, file)
            output_video = os.path.join(OUTPUT_DIR, OUTPUT_Str + file)

            # Vectorize the video
            vectorize(input_video, output_video)


def vectorize(input_video, output_video):
    # Check if the video file exists
    if not os.path.isfile(input_video):
        print(f"Video file {input_video} does not exist.")
        return

    print(f"Processing video file: {input_video}")

    # Open the video stream
    cap = cv2.VideoCapture(input_video)

    # Get the video's frame size and frames per second (fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'Frame count: {framecount}')

    # Define the new resolution
    new_width = frame_width // SCALE_REDUCTION
    new_height = frame_height // SCALE_REDUCTION

    # Create a video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

    # Initialize last_point to None
    last_point = None

    i = 0
    # Process each frame

    while cap.isOpened():
        progressbar(i, framecount)
        i += 1

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Resize the frame
            frame = cv2.resize(frame, (new_width, new_height))

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

            # Perform thresholding
            thresholded_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, 11, 2)

            # Apply edge detection
            edged_frame = cv2.Canny(thresholded_frame, 50, 100)

            # Find contours in the frame
            contours, _ = cv2.findContours(edged_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a blank frame
            blank_frame = np.zeros_like(frame)

            # Check if contours list is not empty
            if contours:
                # Calculate the centroid of each contour and flatten it to make it 1-D
                centroids = [np.ravel(np.mean(contour, axis=0)) for contour in contours]

                # Check if centroids list is not empty
                if centroids:
                    # If last_point is not None, sort the centroids based on their Euclidean distance to last_point
                    if last_point is not None:
                        centroids.sort(key=lambda centroid: np.linalg.norm(centroid - last_point))

                    # Perform hierarchical clustering on the centroids
                    try:
                        Z = linkage(centroids, 'ward')

                        # Get the order in which the centroids were grouped
                        order = dendrogram(Z, no_plot=True)['leaves']

                        # Sort the contours based on this order
                        contours = [contours[i] for i in order]
                    except:
                        print("Error in hierarchical clustering")

                # Concatenate all contours into one
                all_contours = np.concatenate(contours)

                # Create a mask of the contours with a thickness of 1
                mask = np.zeros_like(frame)

                # Blur the mask
                blurred_mask = cv2.GaussianBlur(mask, (99, 99), 30)

                # Draw the blurred mask on the blank frame
                blank_frame = cv2.addWeighted(blank_frame, 1, blurred_mask, 0.5, 0)

                for contour in contours:
                    # Draw the contour on the mask
                    cv2.drawContours(mask, [contour], -1, (0, 255, 0), 1)
                    cv2.drawContours(blank_frame, [contour], -1, (0, 255, 0), 1)

                # Update last_point to the last point of the last contour
                last_point = np.ravel(contours[-1][-1])

            # Write the blank frame with contours to the video file
            out.write(blank_frame)
        else:
            print(f'Video processing completed. Output saved to {output_video}')
            print("\n" + ("-" * 50) + "\n")
            break

    # Release the video capture and writer
    cap.release()
    out.release()


def progressbar(i, upper_range):
    percentComplete = int(i / (upper_range) * 100)

    # Some functions in this project take some time to run due to loops.
    # This gives visual indication of progress
    progress_string = f'\r{("#" * percentComplete)}{("_" * ((100) - percentComplete))} {percentComplete} / {100} [ Printing Frame: {i} ]'
    if i == upper_range:
        print(progress_string)
    else:
        print(progress_string, end='')


if __name__ == '__main__':
    main()
