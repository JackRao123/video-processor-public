import requests
import subprocess
import json
import os
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from moviepy.editor import *
from pytube import YouTube
import random
import scrapetube
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip  # herere!

import time
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def get_video_length(filename):
    clip = VideoFileClip(filename)
    duration = clip.duration
    clip.close()
    return duration

# deletes any videos that are shorter than min_time, or longer than max_time, in folder directory.
def delete_videos(min_time, max_time, directory):  
    for filename in os.scandir(directory):
        filename = filename
        file_path = filename.path

        duration = get_video_length(file_path)

        if duration < min_time:
            os.remove(file_path)
        else:
            if duration > max_time:
                os.remove(file_path)

#Kind of useless
def rename_videos(directory):
    count = 1  # !!!
    os.mkdir(directory + "NEW")
    for filename in os.scandir(directory):
        filename = filename
        file_path = filename.path
        print(f"filename = {file_path}")
        new_name = (
            directory + rf"NEW\VIDEO{count}.mp4"
        )  # the "NEW"  makes it so that it puts it in a new folder, because otherwise as names change it will rename files multiple times and fuck up the numbering.this also means we have to change 'directory' every time we run this, but we'll only have to run it once.
        # os.mkdir makes a new directory, places everything in ther.
        os.rename(file_path, new_name)  # renames it, os.rename(oldname,newname)

        count = count + 1

#Removes audio from a mp4. Takes full file dir. Saves in exact same place.
def remove_audio(file_dir):
    with VideoFileClip(file_dir) as video_clip:
        video_clip = video_clip.without_audio()

        file_dir_reverse = file_dir[::-1]
        file_dir_reverse = file_dir_reverse[file_dir_reverse.index('.')+1:]
        file_dir_without_extension = file_dir_reverse[::-1]
        output_path = f"{file_dir_without_extension}TEMP.mp4"

        video_clip.write_videofile(output_path)
        video_clip.close()

    os.remove(file_dir)
    os.rename(output_path,file_dir)




def trimVid(input_name, start, end, output_name, remove_audio=False):
    # ffmpeg_extract_subclip(input_name,start,end,targetname = output_name)
    # ffmpeg  is sooooo much faster but makes random frames stutter
    clip = VideoFileClip(input_name).subclip(start, end)
    if remove_audio == True:
        clip = clip.without_audio()

    clip.write_videofile(
        output_name,
        verbose=True,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        preset="medium",
        ffmpeg_params=[
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-pix_fmt",
            "yuv420p",
        ],
    )
    # ffmpeg_extract_subclip(input_name, start, end, targetname=output_name)

 

def separateVideo(min_length, max_length, input_name, new_directory, remove_audio=False):  # separates a long video into multiple small ones
    # new directory is necessary, it saves new ones in there so thta it doenst process clips multiple times.
    print(input_name)
    vid_length = get_video_length(input_name)

    lengths_array = []  # array of all the lengths of all the subclips.
    lengths_array = np.array(lengths_array)

    total_length = 0

    current_length = random.uniform(min_length, max_length)
    while (
        total_length + current_length < vid_length
    ):  # separates big clip into small ones
        total_length = total_length + current_length
        lengths_array = np.append(lengths_array, current_length)
        current_length = random.uniform(min_length, max_length)

    remainder = vid_length - total_length

    print(f"len = {len(lengths_array)}")
    if len(lengths_array) == 0:
        print(vid_length)
        lengths_array = np.array([0])
        print(lengths_array)

    lengths_array = lengths_array + remainder / len(
        lengths_array
    )  # leftover video, adds it back on.

    print(f"lengths array = {lengths_array}")
    # now, we make the clips

    # makeing sure for "new directory" that we make the new directory first.
    if os.path.exists(new_directory) == False:
        os.mkdir(new_directory)

    current_time = 0
    for i in range(len(lengths_array)):
        # we need to do some string processing, and then save it in a new folder.
        indexes_of_backslash = [x for x, char in enumerate(input_name) if char == "\\"]
        desired_index_of_backslash = indexes_of_backslash[-1]
        actual_video_name = input_name[desired_index_of_backslash + 1 :]
        new_file_name = rf"{new_directory}\{actual_video_name[0:len(actual_video_name)- 4]}SUBCLIP{i+1}.mp4"
        print(f"newfilename = {new_file_name}")
        # rf'{input_name[0:len(input_name)- 4]}SUBCLIP{i}.mp4'

        trimVid(
            input_name,
            current_time,
            current_time + lengths_array[i - 1],
            new_file_name,
            remove_audio=True,
        )
        current_time = current_time + lengths_array[i - 1]

    # clip = VideoFileClip(input_name).subclip


if __name__ == "__main__":
    channels = [
 
    
     'petlover351'
    ]



    for name in channels:
        folder_in =  rf'C:\Users\raoj6\Videos\Youtube Downloads\Good Unprocessed\{name}'
        folder_out = rf'C:\Users\raoj6\Videos\Youtube Downloads\Good Processed\{name}'

        for file in os.listdir(folder_in):
            separateVideo(7,10,os.path.join(folder_in, file), folder_out)








#     folder_in = r'C:\Users\raoj6\Videos\Youtube Downloads\Good Unprocessed\jtbcatsplus'
#     folder_out = r'C:\Users\raoj6\Videos\Youtube Downloads\Good Processed\jtbcatsplus'
#     for file in os.listdir(folder_in):
#         separateVideo(7,10,os.path.join(folder_in, file), folder_out)
   
# r'https://www.youtube.com/@bellaandquantas/shorts',
#     r'https://www.youtube.com/@Akay_1999/shorts',
#     r'https://www.youtube.com/@vickynga/shorts',
#     r'https://www.youtube.com/@Kd_unite/shorts',
#     r'https://www.youtube.com/@kittyear/shorts',
#     r'https://www.youtube.com/@withmycat2111/shorts',
#     r'https://www.youtube.com/@Cat-James/shorts'
#     r'https://www.youtube.com/@petlover351/shorts', 
#     r'https://www.youtube.com/@MaltipooVip/shorts',
#     r'https://www.youtube.com/@Corgi_Maks/shorts']