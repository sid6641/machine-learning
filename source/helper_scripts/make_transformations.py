import os
import cv2

files = "../dataset/perfect"


image_height = 286
image_width = 384

images_dump_folder_left = "./images/go_left"
images_dump_folder_right = "./images/go_right"
images_dump_folder_up = "./images/go_up"
images_dump_folder_down = "./images/go_down"
images_dump_folder_back = "./images/go_back"
images_dump_folder_center = "./images/go_center"

# strength value denotes the ratio of length chosen to total length 
# [0, 1] -> 0 - no crop, image remains the same
# [0, 1] -> 1 - full crop, nothing left in the cropped image after that
strength_value = 0.1
write_in_go_back = False
dump_center_image = True


def do_left(img):
    crop_width = int(strength_value*image_width)
    half_crop_height = int(0.5*strength_value*image_height)
    new_img = img[half_crop_height:image_height-half_crop_height, crop_width:]
    new_img = cv2.resize(new_img, (image_width, image_height))
    return new_img

def do_right(img):
    crop_width = int(strength_value*image_width)
    half_crop_height = int(0.5*strength_value*image_height)
    new_img = img[half_crop_height:image_height-half_crop_height, :-crop_width]
    new_img = cv2.resize(new_img, (image_width, image_height))
    return new_img

def do_up(img):
    half_crop_width = int(0.5*strength_value*image_width)
    crop_height = int(strength_value*image_height)
    new_img = img[crop_height:, half_crop_width:image_width-half_crop_width]
    new_img = cv2.resize(new_img, (image_width, image_height))
    return new_img


def do_down(img):
    half_crop_width = int(0.5*strength_value*image_width)
    crop_height = int(strength_value*image_height)
    new_img = img[:-crop_height, half_crop_width:image_width-half_crop_width]
    new_img = cv2.resize(new_img, (image_width, image_height))
    return new_img

def do_center(img):
    half_crop_width = int(0.5*strength_value*image_width)
    half_crop_height = int(0.5*strength_value*image_height)
    new_img = img[half_crop_height:image_height-half_crop_height, half_crop_width:image_width-half_crop_width]
    new_img = cv2.resize(new_img, (image_width, image_height))
    return new_img


def main():
    for file in os.listdir(files):
        if file.startswith('.'):
            continue
        img = cv2.imread(os.path.join(files, file))
        new_img_left = do_left(img)
        new_img_right = do_right(img)
        new_img_up = do_up(img)
        new_img_down = do_down(img)
    
        newname_left = file[:-4] + "-str-" + str(int(strength_value*100)) + "-left.jpg"
        newname_right = file[:-4] + "-str-" + str(int(strength_value*100)) + "-right.jpg"
        newname_up = file[:-4] + "-str-" + str(int(strength_value*100)) + "-up.jpg"
        newname_down = file[:-4] + "-str-" + str(int(strength_value*100)) + "-down.jpg"

        # cv2.imwrite(os.path.join(images_dump_folder_left, newname_left), new_img_left)
        # cv2.imwrite(os.path.join(images_dump_folder_right, newname_right), new_img_right)
        # cv2.imwrite(os.path.join(images_dump_folder_up, newname_up), new_img_up)
        # cv2.imwrite(os.path.join(images_dump_folder_down, newname_down), new_img_down)

        if dump_center_image:
            new_img_center = do_center(img)
            newname_center = file[:-4] + "-str-" + str(int(strength_value*100)) + "-center.jpg"
            cv2.imwrite(os.path.join(images_dump_folder_center, newname_center), new_img_center)

        if write_in_go_back:
            cv2.imwrite(os.path.join(images_dump_folder_back, newname_left), new_img_left)
            cv2.imwrite(os.path.join(images_dump_folder_back, newname_right), new_img_right)
            cv2.imwrite(os.path.join(images_dump_folder_back, newname_up), new_img_up)
            cv2.imwrite(os.path.join(images_dump_folder_back, newname_down), new_img_down)

main()










