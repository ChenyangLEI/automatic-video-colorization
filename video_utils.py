import os
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imsave
from os.path import isfile, join
import cv2
from moviepy.editor import *

parser = argparse.ArgumentParser()

parser.add_argument("--video_dir", default="demo.mkv", help="choose the dir of video")
parser.add_argument("--out_frames_dir", default="demo.mkv", help="choose the dir of video")

parser.add_argument("--colorized_frames_dir", default="demo.mkv", help="choose the dir of video")
parser.add_argument("--colorized_video_dir", default="demo.mkv", help="choose the dir of video")

parser.add_argument("--video2frames", action="store_true", help="convert video to frames")
parser.add_argument("--frames2video", action="store_true", help="convert frames to video")
parser.add_argument("--add_sound", action="store_true", help="convert video to frames")

parser.add_argument("--fps", default=30, help="convert frames to video")


ARGS = parser.parse_args()


def video2frames(video_dir, out_frames_dir="None"):
    os.makedirs(out_frames_dir, exist_ok=True)
    video = VideoFileClip(video_dir)
    audio = video.audio
    audio.write_audiofile(out_frames_dir + ".mp3")

    vidcap = cv2.VideoCapture(video_dir)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break    
        if image is None:
            print("Fps is {}".format(fps))
            return 0    
        if count % 100 == 0:
            print("Video to frames: {}/frame{:04d}.png    Image shape:" .format(out_frames_dir, count),    image.shape)
        cv2.imwrite("{}/frame{:04d}.png".format(out_frames_dir, count), image)     # save frame as JPEG file
        count += 1
    vidcap.release()
    audio.release()
    print("Fps is {}".format(fps))
    return int(fps)


def frames2video(frames_dir= './images/testing/',  colorized_video_dir = 'output.mp4', fps=24):
    frame_array = []    
    files = sorted(glob(frames_dir+"/*"))
    print(frames_dir, len(files))
    for i in range(len(files)):
        filename= files[i]
        #reading each files
        if i % 100 == 0:
            print("frames2video:", filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)        #inserting the frames into an image array
    out = cv2.VideoWriter(colorized_video_dir, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    
def add_mp3(source_video, target_video):
    fg_video = VideoFileClip(source_video)
    video = VideoFileClip(target_video)
    audio = fg_video.audio
    videoclip2 = video.set_audio(audio)
    name1 = target_video.split('.', 1)[0] + "_withsound.mp4"
    videoclip2.write_videofile(name1)


if ARGS.video2frames:
    fps = video2frames(ARGS.video_dir, ARGS.out_frames_dir)
    # print("fps is {}".format(fps))

if ARGS.frames2video:
    frames2video(ARGS.colorized_frames_dir, ARGS.colorized_video_dir, ARGS.fps)
    print(ARGS.colorized_video_dir)

if ARGS.add_sound:
    add_mp3(ARGS.video_dir, ARGS.colorized_video_dir)