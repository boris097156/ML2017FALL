import sys
import os
import math
import numpy as np
from skimage import io

IMG_DIR = sys.argv[1] + '/'
IMG_AMOUNT = 415
IMG_SIDE = 600
CHANNEL = 3
TOP_NUM = 4
REC_NUM = 4

def read_images():
	imgs = []
	for f in os.listdir(IMG_DIR):
		print(f)
		if '.jpg' in f:
			img_name = os.path.join(IMG_DIR, f)
			img = io.imread(img_name)
			imgs.append(img)
	return np.asarray(imgs, dtype=np.float64)

def save_img(img, name):
	img = np.asarray(img, dtype=np.uint8).reshape((IMG_SIDE, IMG_SIDE, CHANNEL))
	io.imsave((str(name)+'.jpg'), img, quality=100)

def mean_face(imgs):
	imgs_mean = imgs.mean(axis=0, keepdims=True)
	save_img(imgs_mean, 'mean')
	return imgs_mean

def top_eigen(imgs, imgs_mean):
	imgs_ctr = np.array((imgs - imgs_mean)).reshape(IMG_AMOUNT, IMG_SIDE*IMG_SIDE*CHANNEL)
	U, s, V = np.linalg.svd(imgs_ctr, full_matrices=False)
	#eigenValues = np.power(s, 2)
	eigValue_sum = np.sum(s)
	top = np.copy(V[:TOP_NUM])
	for i,t in enumerate(top):
		t -= np.min(t)
		t /= np.max(t)
		t = (t*255).astype(np.uint8)
		print("%.2f %%" % (s[i]*(100.0)/float(eigValue_sum)))
		save_img(t, 'eigen%d' % i)
	return V
	'''
	percentage
	4.14 %
	2.95 %
	2.39 %
	2.21 %
	'''

def reconstruct(imgs):
	target_img = sys.argv[2]
	target_img = int(target_img[:-4])
	#print(target_img)
	imgs_mean = imgs.mean(axis=0, keepdims=True)
	imgs_ctr = np.array((imgs - imgs_mean)).reshape(IMG_AMOUNT, IMG_SIDE*IMG_SIDE*CHANNEL)
	U, s, V = np.linalg.svd(imgs_ctr, full_matrices=False)
	#img_list = [0, 10, 20, 30]
	imgs_ctr = (imgs-imgs_mean).reshape(IMG_AMOUNT, IMG_SIDE*IMG_SIDE*CHANNEL)
	for i in range(1):
		no = target_img
		p = np.dot(imgs_ctr[no], V.T)
		new = np.dot(p[:TOP_NUM], V[:TOP_NUM])
		new = new.reshape(1, IMG_SIDE, IMG_SIDE, CHANNEL)
		new += imgs_mean
		new -= np.min(new)
		new /= np.max(new)
		new = (new*255)
		save_img(new, 'reconstruction')

def main():
	imgs = read_images()
	#mean = mean_face(imgs)
	#eigenVector = top_eigen(imgs, mean)
	reconstruct(imgs)

if __name__ == '__main__':
	main()