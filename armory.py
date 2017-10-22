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

def check_img_folder(img_dir):
	"""
    Check if we can open all the images under a folder,
    if not rename the file to *.damage,
    input the image folder directory
    check_img_folder("/data/isface/pics/faces/")
    """
	if list(img_dir)[-1]=="/":
		endsym="*"
	else:
		endsym="/*"
	img_path=glob(img_dir+endsym)
	print("="*50)
	print("Samples of folders")
	for j in img_path[:20]:
		print("-"*45)
		print(j)
	#print("-"*45)
	right_count=0;fault_count=0
	for i in img_path:
		# from PIL import Image
		try:
			tmpimg=Image.open(i)
			tmpimg.close()
			right_count+=1
		except:
			os.system("mv %s %s"%(i,i+".damaged"))
			fault_count+=1
	print("="*50)
	print("%s images ok, %s images not ok, %s images in total under folder %s"%(right_count,
                                                                                fault_count,
                                                                                right_count+fault_count,
                                                                               img_dir))
	print("="*50)