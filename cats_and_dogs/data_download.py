
from zipfile import ZipFile
import os
import io
import errno
from urllib import request

ZIP_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
ZIP_PATH = "./dogs-vs-cats.zip"
DATA_PATH = "./data"

def download_dataset(url=ZIP_URL, zip_dest=ZIP_PATH, data_path=DATA_PATH):
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

def check_and_download(zip_url=ZIP_URL, zip_path=ZIP_PATH, data_path=DATA_PATH):
    if os.path.exists(data_path):
        # Checking if file size of the data directory is greater than the lowest possible value (64bytes) and if the directory or its subdirectories already contain the images
        if os.path.getsize(data_path) > 64 & any(fname.endswith('.jpg') for fname in os.listdir(data_path)):
            print("Images already exist in the directory. Proceeding to preprocessing and modelling stages.")
            return
    else:
        download_dataset(ZIP_URL, ZIP_PATH)
        extract_data(ZIP_PATH, DATA_PATH)
    return
