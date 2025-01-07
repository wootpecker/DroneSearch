import os
import moviepy.video.io.ImageSequenceClip
import ds2_augment_dataset
import utils
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

SIMULATIONS = ["01_Winter", "02_Spring", "03_Summer", "04_Autumn"]
FOLDER_IMAGES = "data/images/"
FOLDER_DATA = "data/"
FOLDER_VIDEOS = "data/videos/"
FPS = 30

def main():
    for simulation in SIMULATIONS:
        test_correctness(simulation)
        #test_create_video()
    dataset_GDM, dataset_GSL = utils.load_dataset("train", augmented=True)

def test_correctness(dataset=SIMULATIONS[0]):
    dataset_GDM, dataset_GSL = utils.load_dataset(dataset, augmented=True)
    dataset_GSL = dataset_GSL.numpy()
    count = 0
    for x in range(dataset_GSL.shape[0]):
        if(len(dataset_GSL.shape)<3):
            if (dataset_GSL[x] != torch.tensor([x // 64, x % 64]).numpy()).any():
                count += 1
                print(f"[ERROR] dataset_GSL[{x}]: {dataset_GSL[x]};     x0:{x // 64}, x1:{x % 64}")
        else:
            for y in range(dataset_GSL.shape[1]):
                if (dataset_GSL[x][y] != torch.tensor([x // 64, x % 64]).numpy()).any():
                    count += 1
                    print(f"[ERROR] dataset_GSL[{x}]: {dataset_GSL[x]};     x0:{x // 64}, x1:{x % 64}")
    print(f"[RESULT] Mistakes in Source Location: {count}")

def test_create_video(dataset=SIMULATIONS[0]):
    dataset_GDM, dataset_GSL = utils.load_dataset(dataset, augmented=True)
    for x in range(dataset_GDM.shape[0]):
        utils.save_image(dataset_GDM[x][0], index=x, title=f"{dataset}")
    create_video_from_images(f"{dataset}", image_folder=FOLDER_IMAGES, fps=FPS)

def save_image(image, index, title=""):
    """
    Saves a single image.
    
    Args:
        image (Tensor): The image to save.
        title (str): The title of the plot.
    """
    target_dir_path = Path(FOLDER_IMAGES)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    title = f"{title}_{index:04d}"
    file_path = target_dir_path / f"{title}.png"
    plt.imsave(file_path, image, cmap='viridis')
    print(f"[INFO] Image saved at: {file_path}")

def create_video_from_images(video_name, image_folder=FOLDER_IMAGES, fps=30):
    """
    Creates a video from a series of images.
    
    Args:
        image_folder (str): The folder containing the images.
        video_name (str): The name of the output video file.
        fps (int): Frames per second for the video.
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure the images are in the correct order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    target_dir_path = Path(FOLDER_VIDEOS)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    video_name=os.path.join(target_dir_path,f"{video_name}.mp4")
    #video_name = os.path.join(image_folder, video_name)
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print(f"[INFO] Video saved at: {video_name}")


def test_create_different_video():
    dataset_GDM,dataset_GSL=utils.load_dataset(SIMULATIONS[0], augmented=True)
    #image_files = [os.path.join(image_folder,img) for img in os.listdir(image_folder) if img.endswith(".png")]
    image_files = []
    for x in dataset_GDM:
        img = utils.save_image(x[0], title="GDM")
        image_files.append(img)
        if len(image_files) > 10:
            break
    print(f"image_files: {image_files}")
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=FPS)
    clip.write_videofile(f'{SIMULATIONS[0]}.mp4')




if __name__ == "__main__":
    main()



