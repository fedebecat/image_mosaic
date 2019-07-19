import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


# Get color histogram
def get_hist(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


sample_dir = './samples/'  # Sample directory
im_path = './gioconda.png'  # Image path
img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
out_img = np.ones((img.shape[0], img.shape[1], 4))
samples = []
hists = []
rows = 50  # Number of samples per row
do_shift = True  # change to shift the samples. Useful when samples are circles to reduce white spaces.

# load samples
for image_path in glob(sample_dir + '*.png'):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    samples.append(image)
    hists.append(get_hist(image))

circle_size = img.shape[0]//rows
offset = 0
step = circle_size//2 - offset
for n, r in tqdm(enumerate(range(step, img.shape[0] - step, circle_size))):
    if n % 2 == 1 and do_shift:
        offset = circle_size//2
    else:
        offset = 0
    for c in range(step + offset, img.shape[1] - step, circle_size):
        crop = img[r-step:r+step, c-step:c+step, :]
        cur_hist = get_hist(crop)
        distances = [cv2.compareHist(cur_hist, x, cv2.HISTCMP_INTERSECT) for x in hists]
        cur_sample = samples[np.argmax(distances)]
        out_img[r - step:r + step, c - step:c + step] = cv2.resize(cur_sample, crop.shape[:2])

# Save output
suffix = ''
if do_shift:
    suffix += '_shifted'
cv2.imwrite('out_montage{}.png'.format(suffix), out_img)
