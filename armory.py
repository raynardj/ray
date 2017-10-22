import numpy as np


def preproc(img,rgb_mean=[123.68, 116.779, 103.939]):
	"""
	preprocessing
	Turn rgb into bgr on picture
	img:input image, numpy array
	rgb_mean:a list of r,g,b value, default [123.68, 116.779, 103.939]
	"""
	rn_mean = np.array(rgb_mean, dtype=np.float32)
	return (img - rn_mean)[:, :, :, ::-1]

def deproc(img,shp=None,rgb_mean=[123.68, 116.779, 103.939]):
	"""
	deprocessing
	Turn bgr into rgb on picture
	img:input image, numpy array
	shp:output shape, default None and using the input shape
	rgb_mean:a list of r,g,b value, default [123.68, 116.779, 103.939]
	"""
	rn_mean = np.array(rgb_mean, dtype=np.float32)
	if not shp:
		shp=img.shape
	return np.clip(img.reshape(shp)[:, :, :, ::-1] + rn_mean, 0, 255)
