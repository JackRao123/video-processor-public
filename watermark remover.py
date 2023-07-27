from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.VideoFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip
 
from PIL import ImageFont, ImageDraw, Image
from collections import deque
import cv2
import time
import os
import numpy as np
import random
import textwrap
import statistics
import traceback
import openai
from pydub import AudioSegment
import librosa
import soundfile as sf
#shitass global variables :/
blank_text_pngs_created = False
blank_text_png_index = 0


def check_overlap(rect1, rect2):
    """Check if two rectangles overlap"""
    
    # Get the coordinates of the left-most and right-most points for each rectangle
    x1 = min(rect1[0][0], rect1[1][0], rect1[2][0], rect1[3][0])
    x2 = max(rect1[0][0], rect1[1][0], rect1[2][0], rect1[3][0])
    x3 = min(rect2[0][0], rect2[1][0], rect2[2][0], rect2[3][0])
    x4 = max(rect2[0][0], rect2[1][0], rect2[2][0], rect2[3][0])
    
    # Get the coordinates of the top-most and bottom-most points for each rectangle
    y1 = min(rect1[0][1], rect1[1][1], rect1[2][1], rect1[3][1])
    y2 = max(rect1[0][1], rect1[1][1], rect1[2][1], rect1[3][1])
    y3 = min(rect2[0][1], rect2[1][1], rect2[2][1], rect2[3][1])
    y4 = max(rect2[0][1], rect2[1][1], rect2[2][1], rect2[3][1])
    
    # Check if the rectangles overlap
    if x1 <= x4 and x2 >= x3 and y1 <= y4 and y2 >= y3:
        return True
    else:
        return False


def check_overlap_rect_list(rectlist1, rectlist2):
    for rect1 in rectlist1:
        for rect2 in rectlist2:
            if(check_overlap(rect1, rect2)== True):
                return True
            
    return False


    
def ensure_rectangle_not_overlapping(rectangle, all_rectangles):
    #this function still doesn't ensure that the text doesn't overlap, only the rectangles
    #because the rectangle might not fully bound the text.
    if(len(all_rectangles)==0):
        return rectangle
    any_overlap = check_overlap_rect_list([rectangle], all_rectangles)
    
    if(any_overlap==False):
        return rectangle#dont need to do anything
    else:
        tries =0
        max_tries = 100

        going_downward = ((rectangle[0][1] + rectangle[3][1])/2 < 1920/2  )

        while(check_overlap_rect_list([rectangle], all_rectangles)== True and tries<max_tries):
            multipler =1
            if(going_downward==False):
                multiplier = -1


            for i in range(0,3):
                rectangle[i][1] = rectangle[i][1] + (multipler)*20

            if(rectangle[0][1] < 50):
                going_downward = True
            else:
                if(rectangle[3][1] > 1920 - 300):
                    #might have to change this if not 1080p i dont remember if we ste everything as 1080p
                    going_downward= False
            

            tries = tries + 1

    return rectangle
   

    

def create_blank_text_png( x,y,font, text, fill_color, outline_color, outline_width, index):
    blank_text_png = Image.new("RGBA", (1080, 1920), (0,0,0,0))
    blank_text_draw = ImageDraw.Draw(blank_text_png)

    x=round(x)
    y=round(y)
     
    for x1 in range(x-outline_width, x+outline_width, 1):
        for y1 in range(y-outline_width, y+outline_width, 1):
            blank_text_draw.text((x1, y1), text, font=font, fill=outline_color)  # Draw the text

     
    blank_text_draw.text((x, y), text, font=font, fill=fill_color)  # Draw the text
    blank_text_png.save(rf"C:\Users\raoj6\Desktop\Python projects\TikTok\Prefabs\blank_text_pngs\temp_blank_image_with_text{index}.png", "PNG"  )




def bold_font(image, x,y,font, text, fill_color, outline_color, outline_width):
    blank_text_png = Image.open("temp_blank_image_with_text.png")

    image.paste(blank_text_png, (0,0), blank_text_png )
    

    return image


 
 
def paraphrase(prompt):
    openai.api_key = "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"
    response = openai.Completion.create(
        engine='text-curie-001',
        prompt=prompt,
        temperature=0.54,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.1
    )

    response = (response['choices'][0]['text']).replace('\n', '')
    return response

 
class rect_class:
    def __init__(self, corner_rectangle_bounds):  # corner_rectangle bounds is the  [ (), (), (), ()] [topleft, topright, bottomright, bottomleft]
        self.topleft = corner_rectangle_bounds[0]
        self.topright = corner_rectangle_bounds[1]
        self.bottomright = corner_rectangle_bounds[2]
        self.bottomleft = corner_rectangle_bounds[3]

        self.area = abs((self.topright[0] - self.topleft[0]) * (self.bottomleft[1] - self.topleft[1]))

        self.center = ((self.topright[0] + self.topleft[0]) / 2, (self.bottomleft[1] + self.topleft[1]) / 2)

    def distance(self, other):
        x_diff = self.x - other.x
        y_diff = self.y - other.y
        return np.sqrt(x_diff ** 2 + y_diff ** 2)

    def angle_of_inclination(self, other):  # angle of inclination from lower to higher, in degr
        x_diff = abs(self.center[0] - other.center[0])
        y_diff = abs(self.center[1] - other.center[1])
        return abs(np.degrees(abs(np.arctan(y_diff / x_diff))))


def color_match(image, accuracy, cords, target_color):
    try:
        pixel = image.getpixel((cords))
    except:

        pass
    match = abs(pixel[0] - target_color[0]) < accuracy and abs(pixel[0] - target_color[0]) < accuracy and abs(pixel[0] - target_color[0]) < accuracy

    return match


def find_bounds_of_cluster(cluster):
    x_arr = []
    y_arr = []

    for pixel in cluster:
        x_arr.append(pixel[0])
        y_arr.append(pixel[1])

    min_x = min(x_arr)
    max_x = max(x_arr)
    min_y = min(y_arr)  # (0,0) is top left corner, (width, height) is bottom right corner which is why "top" is "min"
    max_y = max(y_arr)

    left_bound = cluster[x_arr.index(min_x)]
    right_bound = cluster[x_arr.index(max_x)]
    top_bound = cluster[y_arr.index(min_y)]
    bottom_bound = cluster[y_arr.index(max_y)]

    return (left_bound, top_bound, right_bound, bottom_bound)


def find_smallest_distance(cluster1, cluster2, horizontal=False, vertical=False):  # finds the smallest distance between the bounds of two clusters (tl,tr,bl,br), (tl1,tr1,bl1,br1),
    # horizontal and vertical are bool that represent if we are only considering horizontal/vertical distances respectively.
    # e.g. if horizontal = True, then we only find smallest distance horizontally.
    distances = []
    horizontal_distances = []  # only considers horizontal
    vertical_distances = []  # only considers vertical

    for cord1 in cluster1:
        for cord2 in cluster2:
            horizontal_distances.append(abs(cord1[0] - cord2[0]))
            vertical_distances.append(abs(cord1[1] - cord2[1]))
            distances.append(np.linalg.norm(np.array(cord1) - np.array(cord2)))

    if horizontal == True:
        min_horizontal_dist = min(horizontal_distances)
        return min_horizontal_dist

    if vertical == True:
        min_vertical_dist = min(vertical_distances)
        return min_vertical_dist

    min_dist = min(distances)
    # print(f"Cluster1 = {cluster1}, Cluster2 = {cluster2}, min_dist = {min_dist}")

    return min_dist


def combine_two_clusters(cluster1, cluster2):  # this returns a single cluster that encompasses both clusters
    # cluster1 = [(x,y),(x,y),(x,y),(x,y)]
    # cluster2 = [(x,y),(x,y),(x,y),(x,y)]
    # (left_bound, top_bound, right_bound, bottom_bound)

    left_bound = cluster1[0]
    top_bound = cluster1[1]
    right_bound = cluster1[2]
    bottom_bound = cluster1[3]

    if cluster1[0][0] > cluster2[0][0]:
        left_bound = cluster2[0]

    if cluster1[1][1] > cluster2[1][1]:
        top_bound = cluster2[1]

    if cluster1[2][0] < cluster2[2][0]:
        right_bound = cluster2[2]

    if cluster1[3][1] < cluster2[3][1]:
        bottom_bound = cluster2[3]

    return (left_bound, top_bound, right_bound, bottom_bound)


def pil_to_cv2(pil_image):
    image = np.array(pil_image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a Pillow image from the converted image
    return Image.fromarray(image)


def merge_all_clusters(clusters, max_distance_between_clusters, maximum_angle):  # angle of inclination/depression
    # the 'clusters' are actally clusters
    cluster_bounds = []  # cluster bounds represents the leftmost, topmost, rightmost, bottommost POINT. won't look like a rectangle.
    for cluster in clusters:
        cluster_bound = find_bounds_of_cluster(cluster)
        cluster_bound = [bound for bound in cluster_bound]  # converting from ( (), (), (), ()) to [ (), (), (), ()]
        cluster_bounds.append([cluster_bound, False, False])

    # cluster_bounds[]    [[cluster_bound, done_connecting, ever_connected], [] ... ]

    all_clusters_connected = False
    while all_clusters_connected == False:
        # all_clusters_connected = True

        cluster1 = []
        # choosing a cluster1
        n = 0
        while n < len(cluster_bounds):  # while the cluster we choose is "done connecting", we choose another one
            if cluster_bounds[n][1] == True:
                n = n + 1
            else:
                break

        if n == len(cluster_bounds):
            # this means that every single cluster-Bound is "done connectign", so we just break.
            break
        else:
            cluster1 = cluster_bounds[n][0]

        cluster2 = []
        # choosing a cluster2
        m = 0
        smallest_dist_between_c1_c2 = 999999  # smallest distance between cluster1, cluster 2
        index_of_smallest_dist = 999
        while m < len(cluster_bounds):
            if cluster_bounds[m][1] == False and cluster_bounds[m][0] is not cluster1:  # while the cluster we choose is "done connecting", we choose another one # also make sure its not just cluster1
                cluster2 = cluster_bounds[m][0]

                vert_dist = find_smallest_distance(cluster1, cluster2, vertical=True)
                horizontal_dist = find_smallest_distance(cluster1, cluster2, horizontal=True)

                if vert_dist < max_distance_between_clusters * np.sin(np.radians(maximum_angle)) and horizontal_dist < max_distance_between_clusters * np.cos(np.radians(maximum_angle)):
                    rect1 = rect_class(cluster1)
                    rect2 = rect_class(cluster2)
                    angle = rect1.angle_of_inclination(rect2)
                    # if multiple lines of rectangles are merged then the centre of the big rectangle will be shifted up/down and it will not be able to
                    # merge with any other rectangles on the side, because angle will be too large.
                    # thus we find value of angle with 2 rectangles, which consists of this big rectangle split in 2
                    # recta is on top rectb on bottom.
                    bigger_rect = 0  # initialise variable
                    smaller_rect = 0
                    if rect1.area > rect2.area:

                        bigger_rect = cluster1
                        smaller_rect = cluster2
                    else:
                        bigger_rect = cluster2
                        smaller_rect = cluster1

                    # temporarily convert [(), (), (), ()] into [[], [], [], []]
                    bigger_rect = [[cord[0], cord[1]] for cord in bigger_rect]
                    smaller_rect = [[cord[0], cord[1]] for cord in smaller_rect]

                    bigger_rect_a = bigger_rect.copy()
                    bigger_rect_b = bigger_rect.copy()
                    bigger_rect_a[2][1] = (bigger_rect[1][1] + bigger_rect[2][1]) / 2  # set the new bottom right of bigger rect a to halfway of the old bigger_rect
                    bigger_rect_a[3][1] = (bigger_rect[1][1] + bigger_rect[2][1]) / 2
                    bigger_rect_b[0][1] = (bigger_rect[1][1] + bigger_rect[2][1]) / 2  # set the new top left to the middle of old one
                    bigger_rect_b[1][1] = (bigger_rect[1][1] + bigger_rect[2][1]) / 2

                    # copnvert it back so we can use with rect_class()
                    bigger_rect_a = [(cord[0], cord[1]) for cord in bigger_rect_a]
                    bigger_rect_b = [(cord[0], cord[1]) for cord in bigger_rect_b]
                    smaller_rect = [(cord[0], cord[1]) for cord in smaller_rect]

                    bigger_rect_a = rect_class(bigger_rect_a)
                    bigger_rect_b = rect_class(bigger_rect_b)
                    smaller_rect = rect_class(smaller_rect)

                    angle_a = bigger_rect_a.angle_of_inclination(smaller_rect)
                    angle_b = bigger_rect_b.angle_of_inclination(smaller_rect)

                    if np.isnan(angle) == False and (angle < maximum_angle or angle_a < maximum_angle or angle_b < maximum_angle):
                        smallest_dist_between_c1_c2 = find_smallest_distance(cluster1, cluster2)
                        index_of_smallest_dist = m
            m = m + 1

        if smallest_dist_between_c1_c2 == 999999:  # if it hasnt been changed
            # this means we can't find a cluster2, meaning this cluster1 is too far/angle too high to be connected
            cluster_bounds[n][1] = True


        else:

            cluster2 = cluster_bounds[index_of_smallest_dist][0]

            combined_cluster = combine_two_clusters(cluster1, cluster2)
            # get rid of cluster1 and cluster2 from the original array.

            new_cluster_bounds = []
            for cluster in cluster_bounds:
                if cluster[0] is not cluster1 and cluster[0] is not cluster2:
                    new_cluster_bounds.append(cluster)

            combined_cluster_full = [combined_cluster, False, True]
            new_cluster_bounds.append(combined_cluster_full)

            cluster_bounds = new_cluster_bounds

        # now we have chosen the two clusters,  cluster1 and clustr2 we can perform operation on them.

    return_arr = []
    for cluster in cluster_bounds:
        if cluster[2] == True:
            return_arr.append(cluster[0])

    # now we have to merge them vertically, because this will put put them all in lines
    combined_rectangles = combine_rectangles_vertically(return_arr, 60)

    return combined_rectangles

    # each element in this return_arr is a 4-element list containing (x,y) values of topmost, rightmost, leftmost, bottommost points of the cluster.

    return return_arr


def combine_rectangles_vertically(rectangles, max_dist):
    if len(rectangles) == 1:
        return rectangles

    cluster_bounds = []
    for rectangle in rectangles:
        cluster_bounds.append([rectangle, False, False])

    all_clusters_connected = False
    while all_clusters_connected == False:
        # all_clusters_connected = True

        cluster1 = []
        # choosing a cluster1
        n = 0
        while n < len(cluster_bounds):  # while the cluster we choose is "done connecting", we choose another one
            if cluster_bounds[n][1] == True:
                n = n + 1
            else:
                break

        if n == len(cluster_bounds):
            # this means that every single cluster-Bound is "done connectign", so we just break.
            break
        else:
            cluster1 = cluster_bounds[n][0]

        cluster2 = []
        # choosing a cluster2
        m = 0
        smallest_dist_between_c1_c2 = 999999  # smallest distance between cluster1, cluster 2
        index_of_smallest_dist = 999
        while m < len(cluster_bounds):
            if cluster_bounds[m][1] == False and cluster_bounds[m][0] is not cluster1:  # while the cluster we choose is "done connecting", we choose another one # also make sure its not just cluster1
                cluster2 = cluster_bounds[m][0]

                vert_dist = find_smallest_distance(cluster1, cluster2, vertical=True)

                if vert_dist < max_dist and vert_dist < smallest_dist_between_c1_c2:
                    smallest_dist_between_c1_c2 = vert_dist
                    index_of_smallest_dist = m

            m = m + 1

        if smallest_dist_between_c1_c2 == 999999:  # if it hasnt been changed
            # this means we can't find a cluster2, meaning this cluster1 is too far/angle too high to be connected
            cluster_bounds[n][1] = True


        else:

            cluster2 = cluster_bounds[index_of_smallest_dist][0]
            combined_cluster = combine_two_clusters(cluster1, cluster2)
            # get rid of cluster1 and cluster2 from the original array.

            new_cluster_bounds = []
            for cluster in cluster_bounds:
                if cluster[0] is not cluster1 and cluster[0] is not cluster2:
                    new_cluster_bounds.append(cluster)

            combined_cluster_full = [combined_cluster, False, True]
            new_cluster_bounds.append(combined_cluster_full)

            cluster_bounds = new_cluster_bounds

        # now we have chosen the two clusters,  cluster1 and clustr2 we can perform operation on them.
    return_arr = []
    for cluster in cluster_bounds:
        # we can't have the "if they were never merged, don't return', because then that would get rid of any rectangles that are just by themselves, e.g. single lines of text.
        return_arr.append(cluster[0])
    return return_arr

 




def add_text_to_video(vid_dir, font, text_arr, text_color, position, line_thickness, save_dir, fps, dimensions, rectangles):
    # how to call module
    # vid_dir is filename of the video we want to process. use r'' for viddir
    # font is a cv2 font
    # text_arr is selected script. it is not processed 1d array, each element is for another text box.
    # text color is (r,g,b) tuple
    # text_size is a int
    # position is (x,y) note that top left is (0,0) bottom right is whatever vid resolution is, and the top left corner of rect is placed at x,y
    # line thickness is shoudl be 2 or 3
    # save_dir is where it should be saved, including filename
    # fps is desired fps
    # dimensions is dimensions of video desired
    # rectangles is the rectangles of text to be whitedout

    global blank_text_pngs_created
    global blank_text_png_index

    video = cv2.VideoCapture(vid_dir)
    frames = []
    ret = True

    # textsize = cv2.getTextSize(text_arr[0][0], font, text_size, line_thickness)
    # textheight = textsize[0][1] + textsize[1]

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    black = cv2.imread(r'C:\Users\raoj6\Desktop\Python projects\TikTok\Prefabs\blackbackground.jpg')
    black = cv2.resize(black, dimensions)  # here

    rects_and_texts = []  # [rect, text_arr, color]
    i = 0
    while i < (max([len(rectangles), len(text_arr)])):

        rect_and_text = [[], [""], False]
        if i < len(rectangles):  # then we do a colored rectangle
            rectangle = rectangles[i]
 

            if rectangle[0][0] > 100:
                ran = random.randint(60, 100)
                rectangle[0][0] = ran
                rectangle[3][0] = ran

            if rectangle[1][0] < 980:
                ran = random.randint(980, 1020)
                rectangle[1][0] = ran
                rectangle[2][0] = ran
 

            rectangle = ensure_rectangle_not_overlapping(rectangle,[rect_and_text[0] for rect_and_text in rects_and_texts] )
            rect_and_text[0] = rectangle
            rect_and_text[2] = True




        else:  # we just create a rectangle, make sure its not colored
            
            
            ran = random.randint(60, 100)
            rect = [[ran, 0], [1080 - ran, 0], [1080 - ran, 300], [ran, 300]]
            bruh = random.randint(0,1)
            if(bruh == 0):
                ran = random.randint(50,650)
                for i in range(0,3):
                    rect[i][1] = rect[i][1] +ran
            else:
                ran = random.randint( 1920/2+   (1920/2 - 650), 1920-300)
                for i in range(0,3):
                    rect[i][1] = rect[i][1] +ran

            # theres an error where the text will overlap, its because when we make these rectangles it doesn't consider if theres already a rectangle at this position
            # to fix we could have an index, that we increment, and add a number like 400*index to the y value to move it away, so they dont overlap but this may cause more issues.

            # attempted fix is below

            rect = ensure_rectangle_not_overlapping(rect, [rect_and_text[0] for rect_and_text in rects_and_texts])

            rect_and_text[0] = rect
            rect_and_text[2] = False


        rects_and_texts.append(rect_and_text)
        i = i + 1

    if len(rects_and_texts) == 0:
        print("NO LENGTH")
        print(f"TEXT ARR = {text_arr}")

    processed_scripts = []
    i = 0
    while i < len(text_arr):
        try:
            processed_scripts.append(wrap_text(text_arr[i], font, rects_and_texts[i][0][1][0] - rects_and_texts[i][0][0][0]))  ######
        except:
            processed_scripts.append(wrap_text(text_arr[i], font))
        i = i + 1
    text_arr = processed_scripts.copy()
    text_arr.reverse()  # do this so that the scripts we want to appear at the 'top' of the screen appear at the lower 'pixels' (because the lower on screen is higher pixel value)

    i = 0
    while i < (max([len(rectangles), len(text_arr)])):
        if i < len(text_arr):
            rects_and_texts[i][1] = text_arr[i]
            line_height = int(font.getsize('hg')[1])

            height = line_height * len(text_arr[i])

            if (rects_and_texts[i][0][3][1] - rects_and_texts[i][0][0][1]) < height:
                rects_and_texts[i][0][2][1] = rects_and_texts[i][0][0][1] + height
                rects_and_texts[i][0][3][1] = rects_and_texts[i][0][0][1] + height
        else:
            rects_and_texts[i][1] = []
        i = i + 1

    ret, frame = video.read()
    ret, frame = video.read()#twice just to be safe
    max_frame_margin = 0.05 #a percentage representing the most that can be taken off from left/right/top/bottom
 
    height, width, channels = frame.shape
    
    x1 = round(random.uniform(0,max_frame_margin*width))
    y1 = round(random.uniform(0,max_frame_margin*height))
    x2 = round(random.uniform(width*(1-max_frame_margin),width))
    y2 = round(random.uniform(height*(1-max_frame_margin),height))

   



    


    ret, frame = video.read()
    while ret == True:
        
        
        
        frame = frame[y1:y2, x1:x2]
        frame = cv2.resize(frame, (width, height))
     

        try:
            h1, w1, _ = black.shape
            h2, w2, _ = frame.shape

            frame = cv2.resize(frame, (dimensions[0], round(dimensions[0] * h2 / w2)))  # and here

            result = black.copy()  # get the background
            h1, w1, _ = black.shape
            h2, w2, _ = frame.shape
            x = 0
            y = int(h1 / 2 - h2 / 2)
            # Overlay the frame onto background
            result[y:y + h2, x:x + w2] = frame

            frame = result

        except Exception as e:
            pass

        if frame is not None:

            pil_frame = cv2_to_pil(frame)

            new_image_draw = ImageDraw.Draw(pil_frame)

            for rect_and_text in rects_and_texts:
                current_rect = rect_and_text[0]
                current_script = rect_and_text[1]
                draw_rect = rect_and_text[2]  # if we draw the rect or not. if we don't, we just use it to position the text.

                line_num = 0

                if draw_rect == True:
                    tuple_rect = [(corner[0], corner[1]) for corner in current_rect]

                    new_image_draw.polygon(tuple_rect, fill="white")  # outline = "blue"

                # print(f"rect = {current_rect}")

            for k in range(len(rects_and_texts)):
                rect_and_text = rects_and_texts[k]

                current_rect = rect_and_text[0]
                current_script = rect_and_text[1]
                draw_rect = rect_and_text[2]  # if we draw the rect or not. if we don't, we just use it to position the text.

                line_num = 0

                for line in current_script:
                    cords = (current_rect[0][0], current_rect[0][1] + line_num * int(font.getsize('hg')[1]))  # + (rect_count)* 400

                    fill_color = (255,255,255,255)
                    outline_color = (0,0,0,255)#( random.randint(0,255),random.randint(0,255),random.randint(0,255),255)
                    outline_width = 6 #THIS LINE OF CODE IS RESPONSIBLE FOR BOLD TEXT
                    
                    if blank_text_pngs_created == False:
                        create_blank_text_png(cords[0], cords[1], font , line,   fill_color, outline_color, outline_width, blank_text_png_index )
                        
                     
                    blank_text_png_dir = rf'C:\Users\raoj6\Desktop\Python projects\TikTok\Prefabs\blank_text_pngs\temp_blank_image_with_text{blank_text_png_index}.png'
                    blank_text_png = Image.open(blank_text_png_dir)

                    
                    pil_frame.paste(blank_text_png, (0,0), blank_text_png)
                    
                    
                    
                    blank_text_png_index = blank_text_png_index + 1

                    



                    line_num = line_num + 1
                # print(f"Script = {current_script}")

            blank_text_pngs_created = True #
            blank_text_png_index = 0



            try:
                cv2_frame = pil_to_cv2(pil_frame)
                # print("Frame Completed")
                frames.append(cv2_frame)
            except:
                print(f"Error in line cv2_frame = pil_to_cv2(pil_frame)")
                pass
        ret, frame = video.read()
    #we are done with the blank_text_pngs for this video
    blank_text_pngs_created = False  #making it false for the next video.
    folder_path = r'C:\Users\raoj6\Desktop\Python projects\TikTok\Prefabs\blank_text_pngs'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    #clearing the folder for the next video.
    


     # now that we have processed all the frames, we turn it into a video
    output_vid = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'MP4V'), fps, dimensions)
    # the last parameter, has to be the original resolution of the video otherwise it will be corrupt.

    #probabilities: case, probability. Case is taken as output essentially and used for the match-case
    probabilities = {
        0: 0.25,
        1: 0.75,
        2: 0.0
    }
    random_int = random.uniform(0,1)
    total_probability = 0 
    case= 0 #default
    for key, value in probabilities.items():
        total_probability = total_probability + value
        if total_probability  > random_int:
            case = key
            break


    



    match case:
        case 0:
            #encode the video normally.
            for i in range(len(frames) - 1):
                output_vid.write(frames[i])
            output_vid.release()    
        
        case 1:
            #encode the video in reverse
            for i in range(len(frames)-1, 0, -1):
                output_vid.write(frames[i])
            output_vid.release()  

        case 2:
            #encode the video randomly
            while(len(frames)>0):
                rand_num=random.randint(0, len(frames)-1)
                rand_frame = frames[rand_num]
                output_vid.write(rand_frame)
                frames.pop(rand_num)
            output_vid.release()
            


    display_video = False  # if you want the video to be displayed

    if display_video == True:
        for i in range(len(frames) - 1):
            cv2.imshow('video', frames[i])
            print(f"fps = {fps}")
            if cv2.waitKey(round(1000 / fps)) & 0xFF == ord('q'):  # press q to quit.
                video.release()
                cv2.destroyAllWindows()
                break

    video.release()
    cv2.destroyAllWindows()


def wrap_text(text, font, width=1080 - 60 - 60):
    # Load the font

    # Get the width of a single space character
    space_width, _ = font.getsize(" ")

    # Split the text into words

    words = text.split(" ")

    # Initialize the list of lines
    lines = []

    # Initialize the current line
    line = ""

    # Iterate over the words
    for word in words:
        # Get the size of the word
        word_width, _ = font.getsize(word)

        # Check if the word fits on the current line
        if font.getsize(line)[0] + word_width + space_width < width:
            # If it fits, add the word to the line
            line += word + " "
        else:
            # If it doesn't fit, add the line to the list of lines
            if line[0] == " ":
                line = line[1:]
            lines.append(line)

            # Reset the current line and add the word
            line = word + " "

    # Add the final line to the list of lines
    if line[0] == " ":
        line = line[1:]
    lines.append(line)

    return lines

 

def rand_tf():
    return [True, False][random.randint(0,1)]
 


class AudioEditor:
    def __init__(self):
        self.temp_num = random.randint(1000000000,9999999999) #this is a number used as identifier for the temp file.
        pass

    def change_speed(self, mp_audio_file_clip, new_speed):
        temp_filename = f"{self.temp_num}change_speed.mp3"
        
        mp_audio_file_clip.write_audiofile(temp_filename)
        slow_mp3 = AudioSegment.from_file(temp_filename)
        fast_mp3 = slow_mp3.speedup(playback_speed = new_speed)
        
        fast_mp3.export(temp_filename, format="mp3")
        mp_audio_file_clip = AudioFileClip(temp_filename)

         
        return mp_audio_file_clip

    def fade_in_out(self, input_path, output_path, duration, fade_in= False, fade_out = False ):
        audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)
        audio_data_length = audio_data.shape[1]
        audio_duration_s = audio_data_length/sample_rate
        if(duration>audio_duration_s/2):
            duration = 0.5*audio_duration_s*0.9

        length_of_modification_array = int( duration*sample_rate)
        start_multiplier = 0.1
        end_multiplier = 1
        modification_array = np.linspace(start_multiplier,end_multiplier,int(length_of_modification_array))
        modification_array_reversed = modification_array[::-1]

        if(fade_in):
            audio_data[0][:length_of_modification_array] *= modification_array
            audio_data[0][audio_data_length-length_of_modification_array :]  *= modification_array
        if(fade_out):
            audio_data[1][:length_of_modification_array] *= modification_array
            audio_data[1][audio_data_length-length_of_modification_array :]  *= modification_array

        sf.write(output_path, audio_data.T, sample_rate, format="WAV" )

    def add_noise(self, input_path, output_path, noise_amount):
        audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)
        noise = np.random.normal(0, noise_amount, audio_data.shape)
        noisy_audio_data = audio_data + noise
        sf.write(output_path, noisy_audio_data.T, sample_rate, format='wav')

    def alter_mp3(self, audio_file_clip, video_duration, noise_amount=0, fade_duration=0, fade_in=False, fade_out=False, new_speed=1.01):
        #Note - new_speed >1. It will break if its <=1.
        fade_duration = fade_duration/new_speed
        mp_audio_clip = audio_file_clip
        mp_audio_clip=self.change_speed(mp_audio_clip, new_speed)

        
        audio_duration = mp_audio_clip.duration
        if(audio_duration < video_duration):
            start_time = 0 
        else:
            max_start_time  = audio_duration - video_duration
            start_time = random.uniform(0,max_start_time)
            mp_audio_clip = mp_audio_clip.subclip(start_time, start_time + video_duration)

 
 
         
      
         
        self.current_file = f"{self.temp_num}add_noise.wav"
        mp_audio_clip.write_audiofile(self.current_file)

        if(noise_amount>0):
            self.add_noise(self.current_file, f"{self.temp_num}noise_added.wav", noise_amount)
            self.current_file = f"{self.temp_num}noise_added.wav"
        if(fade_in or fade_out):
            self.fade_in_out(self.current_file, f"{self.temp_num}faded.wav", fade_duration, fade_in, fade_out)
            self.current_file = f"{self.temp_num}faded.wav"
        mp_audio_clip = AudioFileClip(self.current_file)


        mp_audio_clip.write_audiofile("temp_audiofilemp3.mp3", codec = 'libmp3lame')
        

        mp_audio_clip = AudioFileClip("temp_audiofilemp3.mp3")
        
        
       
        temp_names = [f"{self.temp_num}add_noise.wav", f"{self.temp_num}noise_added.wav", f"{self.temp_num}faded.wav", f"{self.temp_num}change_speed.mp3"]
        temp_names.append("temp_audiofilemp3.mp3")
        return mp_audio_clip, temp_names

 


  
def add_music(video_dir, music_dir, output_dir):  # threadnum
    # when we add text, we use opencv, but when we use add_music, we use moviepy. this we need to save a temp file for add_music to read. Also, if doing multithreading we will need different temp names.

    video = VideoFileClip(video_dir)
    video = video.without_audio()
    audio = AudioFileClip(music_dir)

 
    #altering mp3 
    audio_editor = AudioEditor()
    audio, temp_names = audio_editor.alter_mp3(audio, video.duration, random.uniform(0,0.005), random.uniform(0.2, (video.duration)/5), rand_tf(), rand_tf(), random.uniform(1.01, 1.15))
    

    


    #altering mp3


    video_length = video.duration  # we need to know the length of video cuz normally if we add video and audio the audio at the end will stick out so we gotta trim it

    final_audio = CompositeAudioClip([audio])  # its meant to be CompositeAudioClip([audio1, audio2, audio3...]), but i think if you want to do this then you need
    video.audio = final_audio
    video = video.subclip(0, video_length + 0.1)

    output_dir = output_dir.replace('\n', "").replace('\r', "")  # gets rid of \r and \n characters so that the next line doesn't result in an error.
    # because apparently you cant put \n\r in a file name?

    if len(output_dir) > 250:
        output_dir = output_dir[:249]  # also max file path length is 260 in windows

    video.write_videofile(output_dir, threads=128, verbose=False, logger=None, audio_codec = 'aac')
    #as of 31/03/2023, mp4s on mobile only work if i use aac codec.?

    #here, we close the audio so that we can delete all our temp files created in alter_mp3()
    audio.close()
    for name in temp_names:
        try:
            os.remove(name)
        except:
            pass


def find_clusters(image_dir, target_color, accuracy, minimum_size):
    # Open image
    image = Image.open(image_dir)
    image = np.array(image)

    # Define the range of white color
    lower_white = np.array([255 - accuracy, 255 - accuracy, 255 - accuracy])
    upper_white = np.array([255, 255, 255])

    # Threshold the image to obtain only white pixels
    mask = cv2.inRange(image, lower_white, upper_white)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    clusters = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minimum_size:
            # Get the coordinates of all points in the contour
            points = cnt.reshape(cnt.shape[0], 2)
            cluster = [(x, y) for x, y in points]
            clusters.append(cluster)

    return clusters


 

def generate_rectangles(clusters, padding, required_scaling):
    rectangles = []
    for cluster in clusters:
        x_arr = [pixel[0] for pixel in cluster]
        y_arr = [pixel[1] for pixel in cluster]
        rectangle = [[min(x_arr) - padding, min(y_arr) - padding], [max(x_arr) + padding, min(y_arr) - padding], [max(x_arr) + padding, max(y_arr) + padding], [min(x_arr) - padding, max(y_arr) + padding]]  # (tl, tr, br, bl)

        rectangle = [[cord[0] * required_scaling[0], cord[1] * required_scaling[1]] for cord in rectangle]

        rectangle_height = max(y_arr) - min(y_arr)

        # got rid of some of the parts, so that it doesn't make the rectagnles smallre, only bigger
        # cuz previosly it made the rectangles smaller and thus it wouldnt cover up the text properly

        # if rectangle[0][1] < 50:
        #     ran = random.randint(50, 80)
        #     rectangle[0][1] = ran
        #     rectangle[1][1] = ran
        #     rectangle[2][1] = ran + rectangle_height
        #     rectangle[3][1] = ran + rectangle_height

        # if rectangle[0][0] < 100:
        #     ran = random.randint(100, 120)
        #     rectangle[0][0] = ran
        #     rectangle[3][0] = ran

        if rectangle[0][0] > 300:
            ran = random.randint(60, 300)
            rectangle[0][0] = ran
            rectangle[3][0] = ran

        if rectangle[1][0] < 780:
            ran = random.randint(780, 1020)
            rectangle[1][0] = ran
            rectangle[2][0] = ran

        # if rectangle[1][0] > 1020:
        #     ran = random.randint(980, 1020)
        #     rectangle[1][0] = ran
        #     rectangle[2][0] = ran

        rectangles.append(rectangle)

    biggest_area = 0
    index_of_largest = 0
    i = 0
    while i < (len(rectangles) - 1):
        rect_area = (rectangle[1][0] - rectangle[0][0]) * (rectangle[3][1] - rectangle[0][1])
        if rect_area > biggest_area:
            index_of_largest = i
        i = i + 1

    if i != 0:
        largest_rect = rectangles[i]  # this is the one we will be putting the text in.
        rectangles.insert(0, rectangles.pop(index_of_largest))  # puts the biggest rectangle at the front

    return rectangles


def process_video(vid_dir, music_files_list, save_folder, channel_name, enable_filtering, video_name=None):  # we can set a video name if we want.
    padding = 35  # number of pixels to pad on each side of rectangles
    fontsize = random.randint(37,47)

    font_dir = r'C:\Users\raoj6\Videos\fonts'
    fonts = os.listdir(font_dir)
    chosen_font_name = random.choice(fonts)

    font_dir = rf'C:\Users\raoj6\Videos\fonts\{chosen_font_name}'

    chosenfont = ImageFont.truetype(font_dir, fontsize)

    dimensions = (1080, 1920)
    actual_dimensions = (cv2.VideoCapture(vid_dir).get(cv2.CAP_PROP_FRAME_WIDTH), cv2.VideoCapture(vid_dir).get(cv2.CAP_PROP_FRAME_HEIGHT))

    required_scaling = (dimensions[0] / actual_dimensions[0], dimensions[1] / actual_dimensions[1])

    framerate = round(cv2.VideoCapture(vid_dir).get(cv2.CAP_PROP_FPS))

    # +====================

    video = cv2.VideoCapture(vid_dir)

    for i in range(30):
        ret, frame = video.read()
        if ret:
            cv2.imwrite("temp.png", frame)
            break

    base_frame = frame
    cv2.imwrite("temp_base_frame.png", base_frame)  # base frame used to determine location of whiteout boxes. Saved to a file to be reopened by pillow.

    start_time = time.time()
    clusters = find_clusters("temp_base_frame.png", (255, 255, 255, 255), 10, 10)  # (image_dir, target_color, accuracy, minimum_size)
  
    clusters = merge_all_clusters(clusters, 80, 14)  # clusters, target_color, accuracy, max_distance_between_clusters)

    rectangles = generate_rectangles(clusters, padding, required_scaling)

  
    file_actual_name = "yourfilename"
   
    selected_scripts = ["placeholder", "replace this"]
    i = 0

    # selected_scripts[i]

    positioning = (random.randint(40, 70), (random.randint(85, 200)))  # [(random.randint(40,70), (random.randint(85,200))), (random.randint(40,70), (random.randint(1400,1600)))][random.randint(0,1)]

    add_text_to_video(vid_dir,
                      chosenfont,
                      selected_scripts,
                      (255, 255, 255),

                      positioning,
                      2,
                      r'temp.mp4',
                      framerate,
                      dimensions,
                      rectangles)

    start_index = vid_dir.rfind("\\")
    end_index = vid_dir.rfind(".")

    num_of_captions = random.randint(5, 10)
    
    index = 0
    if video_name == None:
        output_filename = rf'{save_folder}\{file_actual_name}.mp4'
        while os.path.exists(output_filename):
            output_filename = rf'{save_folder}\{file_actual_name}{index}.mp4'
            index = index + 1


    else:   
        output_filename = rf'{save_folder}\{video_name}.mp4'
        while os.path.exists(output_filename):
            output_filename = rf'{save_folder}\{video_name}{index}.mp4'
            index = index + 1

    chosen_sound = music_files_list[random.randint(0, len(music_files_list) - 1)]

    add_music(r'temp.mp4', chosen_sound, output_filename)

 


 
 

if __name__ == "__main__":
    folder_to_process = rf'i'

    sounds = []
    for filename in os.listdir(rf'..\Videos\Sounds'):
        sounds.append(rf'..\Videos\Sounds\{filename}')

    fails = []
    channel_name = folder_to_process[folder_to_process.rfind("\\") + 1:]
    output_folder = rf'..\Videos\AUTOMATED\{channel_name}'
    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)

    n = 0
    for i in range(5):
        for filename in os.listdir(folder_to_process):

            while f"VIDEO{n}" in os.listdir(output_folder):
                n = n + 1

            input_vid = rf'{folder_to_process}\{filename}'

            try:
                process_video(input_vid, sounds, output_folder, channel_name, False, video_name=None)  # f"VIDEO{n}"
                print(f"{filename} processed successfully.")
            except Exception as e:
                print(traceback.print_exc())
                print(f"{filename} processed unsuccessfully.")

            n = n + 1

    # logging
    for fail in fails:
        print(fail) 