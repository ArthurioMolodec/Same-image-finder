import numpy as np
import cv2
import threading
from matplotlib import pyplot as plt
import os
from PIL import Image
import time
sift = cv2.xfeatures2d.SIFT_create()


def resize_image(input_image_path,output_image_path,scale,output):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    if(output):
        print('The original image size is {wide} wide x {height} high'.format(wide=width, height=height))
 
    resized_image = original_image.resize((int(width//scale[0]),int(height//scale[1])))
    width, height = resized_image.size
    if(output):
        print('The resized image size is {wide} wide x {height} high'.format(wide=width, height=height))
    # resized_image.show()
    resized_image.save(output_image_path)
 
def compression(array_files,name_postfix="changed",folder="changed",scaler={-1:[0,0]},output=False,scaler_array_pos=0):
	for i in array_files:
		original_image = Image.open(i)
		width, height = original_image.size
		#if(width<=50 or height<=50):
			#return False
		minimum = min([width, height])
		for max,scalers in scaler.items():
			if(max>=minimum or max==-1):
				scaler_c = scalers[scaler_array_pos]
				break
		resize_image(i,folder+"/"+".".join(i.split(".")[:-1])+name_postfix+"."+i.split(".")[::-1][0],scale=(scaler_c,scaler_c),output=output)
		return True

######################################################

#FIRST VARIANT

def img_recon1(img1,img_current,dist_multiplier,output):
	img2 = cv2.imread(img_current,0) # trainImage

	# Initiate SIFT detector
	#sift = cv2.SIFT()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params,search_params)

	matches = flann.knnMatch(des1,des2,k=2)
	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in range(len(matches))]
	# ratio test as per Lowe's paper
	matches_count = 0
	matches_len = len(matches)
	matches_count_no_distance_check = 0
	for i,(m,n) in enumerate(matches):
		matches_count_no_distance_check+=1
		if m.distance < dist_multiplier*n.distance:
			matchesMask[i]=[1,0]
			matches_count+=1
	draw_params = dict(matchColor = (0,255,0),
					   singlePointColor = (255,0,0),
					   matchesMask = matchesMask,
					   flags = 0)
	return ({"matches_count":matches_count,"matches_count_no_distance_check":matches_count_no_distance_check,"matches_len":matches_len})
	# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
	# plt.imshow(img3,),plt.show()
	

######################################################

#SECOND VARIANT

def img_recon2(img1,img_current,dist_multiplier,output):
	img2 = cv2.imread(img_current,0) # trainImage
	
	
	# Initiate SIFT detector
	# sift = cv2.SIFT()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < dist_multiplier*n.distance:
			good.append([m])
	return ({"matches_count":len(good)})

	# print(len(good))
	# cv2.drawMatchesKnn expects list of lists as matches.
	# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

	# plt.imshow(img3),plt.show()
def tryparse(input):
	try:
		return bool(float(input)) if float(input)!=0 else True
	except:
		return False
ALGOS = {1:{"SCALER_RANGE":[{"MIN":2,"MAX":3,"STEP":0.5},{"MIN":1/3,"MAX":1/2,"STEP":0.05}],"DIST_MULTIPLIER_RANGE":{"MIN":0.7,"MAX":0.8,"STEP":0.05}},2:{"SCALER_RANGE":[{"MIN":2,"MAX":3,"STEP":0.5},{"MIN":1/3,"MAX":1/2,"STEP":0.05}],"DIST_MULTIPLIER_RANGE":0.75}}
def GetSameImageAlgo(ALGO_ID,imgs_orig="what",imgs_search_in="there"):
	results = []
	if(ALGO_ID==1):
		ALGO_DATA = ALGOS[ALGO_ID]
		SCALER_RANGE = ALGO_DATA['SCALER_RANGE']
		DIST_MULTIPLIER_RANGE = ALGO_DATA['DIST_MULTIPLIER_RANGE']
		SCALER_RANGE_WHAT = np.arange(SCALER_RANGE[0]['MIN'],SCALER_RANGE[0]['MAX']+SCALER_RANGE[0]['STEP'],SCALER_RANGE[0]['STEP'])
		DIST_MULTIPLIER_RANGE = np.arange(DIST_MULTIPLIER_RANGE['MIN'],round(DIST_MULTIPLIER_RANGE['MAX']+DIST_MULTIPLIER_RANGE['STEP'],4),DIST_MULTIPLIER_RANGE['STEP'])
		SCALER_RANGE_THERE = np.arange(SCALER_RANGE[1]['MIN'],SCALER_RANGE[1]['MAX']+SCALER_RANGE[1]['STEP'],SCALER_RANGE[1]['STEP'])

		for SCALER_WHAT in SCALER_RANGE_WHAT:
			for SCALER_THERE in SCALER_RANGE_THERE:
				for DIST_MULTIPLIER in DIST_MULTIPLIER_RANGE:
					try:
						results.append(GetSame(imgs_orig,imgs_search_in,dist_multiplier=float(DIST_MULTIPLIER),scaler=[float(SCALER_WHAT),float(SCALER_THERE)]))
						print("Added")
					except Exception as e:
						print("Error due {} {} {}".format(DIST_MULTIPLIER,SCALER_WHAT,SCALER_THERE))
						print(e)
		not_sorted=True
		while(not_sorted):
			not_sorted=False
			for m in range(len(results)-1):
				if(results[m+1]['RESULT'][0]['result']>results[m]['RESULT'][0]['result']):
					results[m],results[m+1] = results[m+1],results[m]
					not_sorted=True
					continue
	if(ALGO_ID==2):
		ALGO_DATA = ALGOS[ALGO_ID]
		SCALER_RANGE = ALGO_DATA['SCALER_RANGE']
		DIST_MULTIPLIER_RANGE = ALGO_DATA['DIST_MULTIPLIER_RANGE']
		SCALER_RANGE_WHAT = np.arange(SCALER_RANGE[0]['MIN'],SCALER_RANGE[0]['MAX']+SCALER_RANGE[0]['STEP'],SCALER_RANGE[0]['STEP'])
		DIST_MULTIPLIER = DIST_MULTIPLIER_RANGE
		SCALER_RANGE_THERE = np.arange(SCALER_RANGE[1]['MIN'],SCALER_RANGE[1]['MAX']+SCALER_RANGE[1]['STEP'],SCALER_RANGE[1]['STEP'])

		for SCALER_WHAT in SCALER_RANGE_WHAT:
			for SCALER_THERE in SCALER_RANGE_THERE:
				print("Error due {} {} {}".format(DIST_MULTIPLIER,SCALER_WHAT,SCALER_THERE))

				results.append(GetSame(imgs_orig,imgs_search_in,dist_multiplier=float(DIST_MULTIPLIER),scaler=[float(SCALER_WHAT),float(SCALER_THERE)]))
		not_sorted=True
		while(not_sorted):
			not_sorted=False
			for m in range(len(results)-1):
				if(results[m+1]['RESULT'][0]['result']>results[m]['RESULT'][0]['result']):
					results[m],results[m+1] = results[m+1],results[m]
					not_sorted=True
					continue
	return results
def GetSame(imgs_orig="what",imgs_search_in="there",dist_multiplier=0.75,scaler=[3,1/3],output=False):
	fin_res=[]
	fin_res_bad=[]
	start_time = time.time()
	if type(imgs_orig) == str:
		imgs_orig = list(map(lambda a : imgs_orig+"/"+a, os.listdir(path=imgs_orig)))
	if type(imgs_search_in) == str:
		imgs_search_in = list(map(lambda a : imgs_search_in+"/"+a, os.listdir(path=imgs_search_in)))
	
	
	results = []
	a = dist_multiplier
	if(tryparse(a)):
		a = float(a)
	else:
		a = 0.75
		
	# b = scaler[0]
	# if(tryparse(b)):
		# b = float(b)
	# else:
		# b = 3
	
	# c = scaler[1]
	# if(tryparse(c)):
		# c = float(c)
	# else:
		# c = 1/3
	dist_multiplier = a
	# scaler[0] = b
	# scaler[1] = c

	for i in imgs_orig:
		i_orig = i
		# i = "what\\"+i
		if(not compression([i],scaler=scaler,scaler_array_pos=0,output=output)):
			fin_res_bad.append({"what":i,"there":"-","result":"-1","comment":"dropped (too small)"}) 
			continue
		i = "changed"+"/"+".".join(i.split(".")[:-1])+"changed"+"."+i.split(".")[::-1][0]
		img1 = cv2.imread(i,0)
		for img_current in imgs_search_in:
			if(not compression([img_current],scaler=scaler,name_postfix="changed1",scaler_array_pos=1,output=output)):
				fin_res_bad.append({"what":"-","there":img_current,"result":"-1","comment":"dropped (too small)"}) 
				continue
			img_current = "changed"+"/"+".".join(img_current.split(".")[:-1])+"changed1"+"."+img_current.split(".")[::-1][0]
			# print(img_current)
			# img_current = "there\\"+img_current
			if(output):
				print(dist_multiplier)
			data = img_recon1(img1,img_current,dist_multiplier,output)
			if(output):
				print(i,"->",img_current,data['matches_count'])
			results.append({"what":i,"there":img_current,"result":data['matches_count']})
			if(output):
				print('---')
	not_sorted=True
	while(not_sorted):
		not_sorted=False
		for m in range(len(results)-1):
			if(results[m+1]['result']>results[m]['result']):
				results[m],results[m+1] = results[m+1],results[m]
				not_sorted=True
				continue
	
	for i in results:
		add=True
		for res in fin_res:
			if(i['what'] == res['what'] or i['there'] == res['there']):
				add=False
		if(add and i['result']>0):
			fin_res.append({"what":i['what'],"there":i['there'],"result":i['result']}) 
	fin_res = fin_res+fin_res_bad
	if(output):
		for i in results:
			print(i)
	if(output):
		print("")
		print("-"*55)
		print("dist_multiplier = ",dist_multiplier)
		print("scaler = ",scaler)
		for k in fin_res:
			print(k)
		print("-"*55)
	return {"dist_multiplier":dist_multiplier,"scaler":scaler,"RESULT":fin_res}
