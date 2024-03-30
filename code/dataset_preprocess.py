import os
import shutil
import zipfile

from PIL import Image

DATA_DIR = "../Dataset"
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']
file_lev = []


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def delete_file(img_path):
    if os.path.isdir(img_path):
        os.rmdir(img_path)
    elif os.path.isfile(img_path):
        os.remove(img_path)
        print(f"Invalid image file, delete {img_path}")
    else:
        print(f"Invalid path, please check {img_path}")
        exit(-1)


def is_jpg(img_path):
    try:
        i = Image.open(img_path)
        return i.format == 'JPEG'
    except IOError:
        return False


def filter_file(img_path):
    if not (is_image_file(img_path) and is_jpg(img_path)):
        delete_file(img_path)


def is_dir_empty(dir_name):
    return len(os.listdir(dir_name)) == 0


def go_to_file_lev(surface_lev):
    if is_dir_empty(surface_lev):
        file_lev.append(surface_lev)
        return
    if os.path.isdir(surface_lev):
        for files in os.listdir(surface_lev):
            if os.path.isdir(os.path.join(surface_lev, files)):
                go_to_file_lev(os.path.join(surface_lev, files))
            else:
                file_lev.append(surface_lev)
                return


def delete_empty_file(my_dir):
    while 1:
        count = 0
        file_lev.clear()
        go_to_file_lev(my_dir)
        for dirs in file_lev:
            if is_dir_empty(dirs):
                count = 1
                delete_file(dirs)
        if count == 0:
            print("Clear all the empty dirs!")
            break


def filter_dataset(data_path):
    """Remove all the files that are not in JPEG format"""
    for sub_dir in os.listdir(data_path):
        file_path = os.path.join(data_path, sub_dir)
        if os.path.isfile(file_path):
            filter_file(file_path)
    go_to_file_lev(data_path)
    for dirs in file_lev:
        if is_dir_empty(dirs):
            delete_file(dirs)
        else:
            for img_name in os.listdir(dirs):
                filter_file(os.path.join(dirs, img_name))
    file_lev.clear()


def split_dataset(data_path, eval_split=0.1):
    """Split the dataset into train and evaluate"""
    go_to_file_lev(data_path)
    os.makedirs(os.path.join(data_path, "train"))
    os.makedirs(os.path.join(data_path, "eval"))
    for dirs in file_lev:
        to_be_cls = os.listdir(dirs)
        train_size = int(len(to_be_cls) * (1 - eval_split))
        os.makedirs(os.path.join(data_path, "train", os.path.basename(dirs)))
        os.makedirs(os.path.join(data_path, "eval", os.path.basename(dirs)))
        for i, file_name in enumerate(os.listdir(dirs)):
            source_file = os.path.join(dirs, file_name)
            if i <= train_size:
                target_file = os.path.join(data_path, "train", os.path.basename(dirs), file_name)
            else:
                target_file = os.path.join(data_path, "eval", os.path.basename(dirs), file_name)
            shutil.move(source_file, target_file)
        delete_file(dirs)
    delete_empty_file(data_path)


def extract_dataset(zip_path, save_path):
    """Extract target file"""
    if not os.path.exists(zip_path):
        raise ValueError(f"{zip_path} is not a valid path!")
    try:
        print(f"Begin to extract!")
        zip_path = zipfile.ZipFile(zip_path)
        for names in zip_path.namelist():
            zip_path.extract(names, save_path)
        zip_path.close()
        print(f"Successfully extract at {os.path.join(save_path, DATA_DIR)}")
    except ValueError:
        raise ValueError(f"{save_path} is not a valid path!")


if __name__ == '__main__':
    save_dir = os.path.abspath(DATA_DIR)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    zip_file = "../kaggleCatsAndDogs_3367a.zip"
    extract_dataset(zip_file, save_dir)
    print("filter invalid images!")
    dataset_path = os.path.join(save_dir)
    filter_dataset(dataset_path)
    print("filter invalid images done, then split dataset to train and eval")
    split_dataset(dataset_path, eval_split=0.1)
    print(f"final dataset at {dataset_path}")
