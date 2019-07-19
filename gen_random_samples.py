import cv2
import numpy as np

num_samples = 500
sample_size = (350, 350, 4)
img_circle = np.zeros((sample_size[0], sample_size[1]))
cv2.circle(img_circle, (sample_size[0]//2, sample_size[0]//2), sample_size[0]//2,
           (255, 255, 255), thickness=-1)

for s in range(num_samples):
    sample = np.zeros(sample_size, dtype=np.int32)
    for c in range(3):
        sample[:, :, c] = np.random.randint(255) # BGR
    print(sample.shape)
    sample[:, :, 3] = (img_circle>0)*255 # alpha
    cv2.waitKey(100)
    cv2.imwrite('./samples/{}.png'.format(s), sample)
