import glob, os, errno
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

ORIGINAL_PATH = "./data/PetImages"
NEW_PATH = "./data/PetImagesStandard"

def convert_files(from_folder, to_folder):
    for filename in os.listdir(from_folder):
        if filename.endswith('.jpg'):
            img = cv2.imread(from_folder + "/" + filename)
            if not img is None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                final = cv2.resize(gray, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(to_folder + "/" + filename, final)

try: 
    os.makedirs(NEW_PATH)
    os.makedirs(NEW_PATH + "/Cat")
    os.makedirs(NEW_PATH + "/Dog")
except OSError as e:
    if e.errno == errno.EEXIST:
        raise

print("Converting cats...")
convert_files(ORIGINAL_PATH + "/Cat", NEW_PATH + "/Cat") 
print("Converting dogs...")
convert_files(ORIGINAL_PATH + "/Dog", NEW_PATH + "/Dog") 
    

 


