from zipfile import ZipFile
import os
import io
import errno
from skimage.color import rgb2gray
from skimage.transform import resize
from matplotlib import pyplot as plt
from urllib import request

ZIP_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
ZIP_PATH = "./dogs-vs-cats.zip"
DATA_PATH = "./data"


FINAL_IMG_SIZE = (80, 80)
cmap_type = plt.cm.gray

def download_dataset(url=ZIP_URL, zip_dest=ZIP_PATH):
    if not os.path.exists(zip_dest):
        print(f"Requesting file from {url}")
        response = request.urlopen(url)
        print(f"Reading response.")
        zip_content = response.read()
        print(f"Writing response to {zip_dest}")
        with open(zip_dest, 'wb') as zip_file:
            zip_file.write(zip_content)
    else:
        print('Zip file exists. Unzipping and preprocessing.')
    return

def extract_data(zip_path=ZIP_PATH, dest_path=DATA_PATH):
    # Check if the dogs-vs-cats.zip file is in the current directory
    if not os.path.exists(zip_path):
        raise FileNotFoundError( errno.ENOENT, os.strerror(errno.ENOENT), zip_path)
    else:
        print(f"Found file {zip_path}. Unzipping contents to {dest_path}")
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
    
        unzip_file = open(zip_path, 'rb')
        unzipper = ZipFile(unzip_file)
        for file in unzipper.namelist():
            if 'jpg' in file:
                file_path = os.path.join(dest_path, file[10:])
                parent_dir = os.path.join(file_path.split('/')[0], file_path.split('/')[1], file_path.split('/')[2])
                if not os.path.exists(parent_dir):
                    os.mkdir(parent_dir)
                print(f"Extracting {file} to {file_path}")
                with open(file_path, 'wb') as img_file:
                    img_file.write(unzipper.read(file))
    os.remove(ZIP_PATH)
    return

def preproc(directory):
    img_files = os.listdir(directory)
    for img_file in img_files:
        if os.path.isfile(os.path.join(directory, img_file)):
            if img_file.endswith('.jpg'):
                try:

                    img = plt.imread(os.path.join(directory, img_file))
                    print(f"Processing file: {os.path.join(directory, img_file)}")
                    gray_img = rgb2gray(img)
                    resized_img = resize(gray_img, FINAL_IMG_SIZE)
                    plt.imsave(os.path.join(directory, img_file), resized_img, cmap=cmap_type)
                except IOError:
                    print(f"{img_file} is corrupt. Deleting file.")
                    os.remove(os.path.join(directory, img_file))
                    pass
    return



if __name__ == "__main__":
    download_dataset(ZIP_URL, ZIP_PATH)
    extract_data(ZIP_PATH, DATA_PATH)
    for directory in os.listdir(DATA_PATH):
        if os.path.isdir(os.path.join(DATA_PATH, directory)):
            preproc(os.path.join(DATA_PATH, directory))
        else:
            print(f"{directory} is not a directory.")
            pass
    print("Preprocessing complete.")

