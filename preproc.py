from zipfile import ZipFile
import os
import errno


ZIP_PATH = "./dogs-vs-cats.zip"
OP_PATH = "./data"
def extract_data(zip_path=ZIP_PATH, dest_path=OP_PATH):
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
            print(f"Extracting {file} to {dest_path}.")
            unzipper.extract(file, dest_path)
            if "zip" in file:
                if not os.path.exists(os.path.join(dest_path, file[:-4])):
                    os.mkdir(os.path.join(dest_path, file[:-4]))
                sub_dir_path = os.path.join(dest_path, file)
                sub_dir = open(sub_dir_path, 'rb')
                unzip_sub_dir = ZipFile(sub_dir_path)
                for sub_dir_file in unzip_sub_dir.namelist():
                    print(f"Extracting subdirectory {sub_dir_file} to {os.path.join(dest_path, file[:-4])}")
                    unzip_sub_dir.extract(sub_dir_file, dest_path)
    return

if __name__ == "__main__":
    extract_data(ZIP_PATH, OP_PATH)
