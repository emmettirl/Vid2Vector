import os

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np

from moviepy.editor import VideoFileClip, AudioFileClip
from utils import progressbar


INPUT_DIR = 'audio2Vector_input'
OUTPUT_DIR = 'audio2Vector_output'
TMP_OUTPUT_STR = 'temp_output_'
AV_OUT_STR =  "audio2Vector_output_"
XLIM_DIVISOR = 3

def main():

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.listdir(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} is empty. Add .wav files to the directory and run the script again.")
        return

    for file in os.listdir(INPUT_DIR):
        if file.endswith('.wav') or file.endswith('.mp3'):
            input_audio = os.path.join(INPUT_DIR, file)
            output_audio = os.path.join(OUTPUT_DIR, TMP_OUTPUT_STR + os.path.splitext(file)[0] + '.mp4')

            num_frames = get_number_of_frames(input_audio)
            print(f'The number of frames in the audio track is {num_frames}.')

            print("processing audio file: ", input_audio)
            toVid(input_audio, output_audio, num_frames)
            print(f'Audio file created: {output_audio}\n\n {"-" * 50} \n\n')

    #delete the temp files
    for file in os.listdir(OUTPUT_DIR):
        if file.startswith(TMP_OUTPUT_STR):
            os.remove(os.path.join(OUTPUT_DIR, file))


def toVid(input_audio, output_video, num_frames):
    # Load the audio file
    y, sr = librosa.load(input_audio)

    # Calculate the Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)

    # Get the magnitude of the STFT
    D_magnitude = np.abs(D)

    # Define the video's frames per second (fps)
    fps = sr/512

    # Define the dimensions of the video writer
    dimensions = (D_magnitude.shape[0], D_magnitude.shape[0])

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_video, fourcc, fps, dimensions)

    img = None

    for frame_number in range(int(num_frames)):
        if frame_number % 5 == 0:
            progressbar(frame_number, num_frames)

            # Get the specified audio frame (column in the STFT)
            audio_frame = D_magnitude[:, frame_number]

            # Normalize the audio frame to fit in the range of 0-255
            audio_frame_normalized = ((audio_frame - np.min(audio_frame)) / (
                    np.max(audio_frame) - np.min(audio_frame))) * 255

            makePlot(audio_frame_normalized)

            tempImg = makePlot(audio_frame_normalized)

            img = cv2.resize(tempImg, dimensions, interpolation=cv2.INTER_AREA)

        # Write the image to the video file
        out.write(img)

    # Release the video writer
    out.release()

    add_audio_to_video(output_video, input_audio, f"{OUTPUT_DIR}/av_{os.path.basename(output_video)}")

    print(f'Video file created: {OUTPUT_DIR}/av_{os.path.basename(output_video)}')


def get_number_of_frames(input_audio):
    # Load the audio file
    y, sr = librosa.load(input_audio)

    # Calculate the Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)

    # Get the number of frames (columns in the STFT)
    num_frames = D.shape[1]

    return num_frames

def add_audio_to_video(video_file, audio_file, output_file):

    # Load the video file
    video = VideoFileClip(video_file)

    # Load the audio file
    audio = AudioFileClip(audio_file)

    # Add the audio to the video
    video_with_audio = video.set_audio(audio)

    # Write the result to a file
    video_with_audio.write_videofile(output_file, codec='libx264')

def makePlot(audio_frame_normalized):
    # Create a figure with a transparent background
    plt.figure(facecolor='none')

    # Plot the audio frame as a bar chart and return the BarContainer object
    plt.plot(np.arange(len(audio_frame_normalized)) * 0.25, audio_frame_normalized * 10, color='#00FF00')       # ++
    plt.plot(np.arange(len(audio_frame_normalized)) * 0.25, -audio_frame_normalized * 10, color='#00FF00')      # +-

    plt.plot(np.arange(len(audio_frame_normalized)) * -0.25, audio_frame_normalized * 10, color='#00FF00')      # -+
    plt.plot(np.arange(len(audio_frame_normalized)) * -0.25, -audio_frame_normalized * 10, color='#00FF00')     # --

    # Set the limits of the x-axis to display only the first half
    xlim = len(audio_frame_normalized) * 0.25 / XLIM_DIVISOR
    plt.xlim(-xlim, xlim)

    plt.axis('off')


    # Draw the canvas and get the RGB data
    plt.gcf().canvas.draw()
    img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

    return img


if __name__ == '__main__':
    main()