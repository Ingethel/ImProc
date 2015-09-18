from scipy.misc import imread, imsave, imresize
import threading
import numpy
import math
import os

# class used to stitch images
# contains additional methods for image filtering
class Panorama:

	# initialise variables,
	# used for calculating merging point of each thread
	_min = [None]*2
	_bestThesi = [None]*2
	_bestShift = [None]*2

	#variables for Sobel operation
	_SobelGx = numpy.array([[-1, 0, 1],
						 [-2, 0, 2],
						 [-1, 0, 1]])
	_SobelGy = numpy.array([[1, 2, 1],
						 [0, 0, 0],
						 [-1,-2,-1]])

	#variables for Gaussian blur
	_blur = (1/float(16))*numpy.array([[1,2,1],
							[2,4,2],
							[1,2,1]])

	#variable for edge detection
	_edge = numpy.array([[-1,-1,-1],
					  [-1, 8,-1],
					  [-1,-1,-1]])

	# initialise object of class
	def __init__(self): pass

	#method to handle invalid values in the image
	def handleOverflows(self,image):
		# if a value in the array is larger than 255, reassign it as 255
		image[image>255]=255
		# if a value in the array is smaller than 0, reassign it as 0
		image[image<0]=0
		return image

	#method to apply blur
	def blur(self, image):
		# get image dimensions
		(h,w,dim) = image.shape
		# create new image and copy values
		blurred = image[:]
		for x in range(1,w-1):
			for y in range(1,h-1):
				for d in range(dim):
					# do convolution with respected kernel
					blurred[y,x,d] = self._blur[0,0]* image[y-1,x-1,d] + self._blur[0,1]* image[y-1,x,d] + self._blur[0,2]* image[y-1,x+1,d] + self._blur[1,0]* image[y,x-1,d] + self._blur[1,1]* image[y,x,d] + self._blur[1,2]* image[y,x+1,d] + self._blur[2,0]* image[y+1,x-1,d] + self._blur[2,1]* image[y+1,x,d]+ self._blur[2,2]* image[y+1,x+1,d]
		# check for invalid values
		self.handleOverflows(blurred)
		return blurred

	#method to apply edge detection
	def edge(self, image):
		# get image dimensions
		(h,w) = image.shape
		# crate new image
		edged = numpy.zeros((h,w))
		for x in range(1,w-1):
			for y in range(1,h-1):
				# do convolution with respected kernel
				edged[y,x] = self._edge[0,0]* image[y-1,x-1] + self._edge[0,1]* image[y-1,x] + self._edge[0,2]* image[y-1,x+1] + self._edge[1,0]* image[y,x-1] + self._edge[1,1]* image[y,x] + self._edge[1,2]* image[y,x+1] + self._edge[2,0]* image[y+1,x-1] + self._edge[2,1]* image[y+1,x]+ self._edge[2,2]* image[y+1,x+1]
		# check for invalid values
		self.handleOverflows(edged)
		return edged

	#method to apply edge detection - Sobel filter
	def sobel(self, image):
		# get image dimensions
		(h,w) = image.shape
		# crate new image
		G = numpy.zeros((h,w))
		for x in range(1,w-1):
			for y in range(1,h-1):
				# do convolution with respected kernel to calculate partial derivatives
				Gx = self._SobelGx[0,0]* image[y-1,x-1] + self._SobelGx[0,1]* image[y-1,x] + self._SobelGx[0,2]* image[y-1,x+1] + self._SobelGx[1,0]* image[y,x-1] + self._SobelGx[1,1]* image[y,x] + self._SobelGx[1,2]* image[y,x+1] + self._SobelGx[2,0]* image[y+1,x-1] + self._SobelGx[2,1]* image[y+1,x]+ self._SobelGx[2,2]* image[y+1,x+1]
				Gy = self._SobelGy[0,0]* image[y-1,x-1] + self._SobelGy[0,1]* image[y-1,x] + self._SobelGy[0,2]* image[y-1,x+1] + self._SobelGy[1,0]* image[y,x-1] + self._SobelGy[1,1]* image[y,x] + self._SobelGy[1,2]* image[y,x+1] + self._SobelGy[2,0]* image[y+1,x-1] + self._SobelGy[2,1]* image[y+1,x]+ self._SobelGy[2,2]* image[y+1,x+1]
				# calculate magnitude from partial derivatives
				G[y,x] = math.sqrt(Gx**2 + Gy**2)
		# check for invalid values
		self.handleOverflows(G)
		return G

	#method to perform median filtering
	def medianFilter(self, image):
		# get image dimensions
		(h,w,dim) = image.shape
		# create new image and copy values
		median = image[:]
		# create list to hold candidates for pixel value
		candidates = numpy.zeros(9)
		for d in range(dim):
			for x in range(1,w-1):
				for y in range(1,h-1):
					# take current and the 8 neighbouring pixel values
					candidates[0] = image[y-1,x-1,d]
					candidates[1] = image[y,x-1,d]
					candidates[2] = image[y+1,x-1,d]
					candidates[3] = image[y-1,x,d]
					candidates[4] = image[y,x,d]
					candidates[5] = image[y+1,x,d]
					candidates[6] = image[y-1,x+1,d]
					candidates[7] = image[y,x+1,d]
					candidates[8] = image[y+1,x+1,d]
					# sort them
					candidates.sort()
					# pick the middle ranked
					median[y,x,d]=candidates[4]
		return median

	# method to convert RGB image to grey scaled
	def grayscale(self, image):

		# get image dimensions
		(h,w,dims) = image.shape
		# create new image
		grayscaleImage = numpy.zeros((h,w))

		# convert based on given RGB weights
		red = image[:,:,0]
		green = image[:,:,1]
		blue = image[:,:,2]
		grayscaleImage = 0.2125*red + 0.7154*green + 0.0721*blue

		return grayscaleImage

	# method to calculate updated coordinates for cylindrical projection
	def convertPoint(self,x,y,w,h):
		# get current coordinates based on (0,0) being on center
		pc = [y-h/2,x-w/2]
		# focal length defined as width of image
		f = w
		# radius of cylinder
		r = w

		# calculate z coordinate of projected plane
		z0 = f - math.sqrt(r**2-(w/2)**2)
		a = ((pc[1]**2)/float(f**2))+1
		zc = (2*z0+math.sqrt(4*z0**2-4*a*(z0**2-r**2)))/float(2*a)
		# calculate new x and y coordinates based on (0,0) being at top left of image
		x = int((pc[1]*zc/float(f))+(w/2))
		y = int((pc[0]*zc/float(f))+(h/2))
		# save them as a point
		pf = [y, x]

		return pf

	# method to perform cylindrical transform
	def cylindricalTransform(self,image):
		# get image dimensions
		(h,w,dim) = image.shape
		# create new image
		newImage = numpy.zeros((h,w,dim))

		for y in range (h):
			for x in range(w):
				# get updated coordinates for current pixel
				currentPos = self.convertPoint(x,y,w,h)
				# if in bounds of image dimensions
				if 0<=currentPos[0]<h and 0<=currentPos[1]<w:
					newImage[y,x] = image[currentPos[0], currentPos[1]]

		return newImage

	# method to find a merging point between two images with shifting property
	def findMin(self, image1, image2, numberOfThread):

		# get image dimensions
		(h1,w1) = image1.shape
		(h2,w2) = image2.shape
		# minimum height of two images, used for out of range handling
		height = min(h1,h2)
		# set shift range
		shift = [-3,-2, -1, 0, 1, 2, 3]
		# current minimum difference
		minimum = 100000000
		# current merging position
		bestThesi = 0

		# find the two lines of pixels
		# with the minimum difference between the two images
		for i in shift:
			for x in range(w2):
				sum = 0
				for y in range(height):
					if y+shift[i]>=0 and y+shift[i]<h2:
						sum += abs(image1[y, w1-1]-image2[y+shift[i], x])
				if sum < minimum:
					minimum = sum
					bestThesi = x
					bestShift = shift[i]

		# save results for further use
		self._min[numberOfThread] = minimum
		self._bestThesi[numberOfThread] = bestThesi
		self._bestShift[numberOfThread] = bestShift

		return None

	# method to merge two images with shifting property
	def merge(self, image1_gray, image2_gray, thesi, shift, image1, image2):

		# get image dimensions
		(h1,w1,dim) = image1.shape
		(h2,w2,dim) = image2.shape
		# initialise dimensions of merged image
		height = min(h1,h2)
		width = w1+w2-thesi
		# create new images
		mergedImage_gray = numpy.zeros((height,width))
		mergedImage = numpy.zeros((height,width,dim))
		# calculate overlap area
		startOfOverlap = w1-thesi
		# handle length of overlap area and reassign if needed
		while w1/startOfOverlap >= 2:
			startOfOverlap = startOfOverlap+round((w1-startOfOverlap)/2)
		# initialise weights for blending process
		weight1 = 2.0
		weight2 = 0.0
		steps = w1-startOfOverlap
		if steps != 0:
			stepSize = 2.0/steps

		# merging
		for x in range(width):

			# only first image
			if x < startOfOverlap:
				for y in range(height):
					mergedImage_gray[y,x] = image1_gray[y,x]
					mergedImage[y,x] = image1[y,x]

			# overlap area
			# gradually change between the two images
			elif x < w1:
				weight1 -= stepSize
				weight2 += stepSize
				for y in range(height):
					if y+shift>=0 and y+shift<h2:
						mergedImage_gray[y,x] = (image1_gray[y,x]*weight1 + image2_gray[y+shift,thesi+x-w1]*weight2)/2
						mergedImage[y,x] = (image1[y,x]*weight1 + image2[y+shift,thesi+x-w1]*weight2)/2
					else:
						mergedImage_gray[y,x] = image1_gray[y,x]
						mergedImage[y,x] = image1[y,x]

			# only second image
			else:
				for y in range(height):
					if y+shift>=0 and y+shift<h2:
						mergedImage_gray[y,x] = image2_gray[y+shift,thesi+x-w1]
						mergedImage[y,x] = image2[y+shift,thesi+x-w1]
					else:
						mergedImage_gray[y,x] = 0
						mergedImage[y,x] = 0

		# crop shifted rows
		if shift >= 0:
			newImage_gray = mergedImage_gray[0:height-shift,::]
			newImage = mergedImage[0:height-shift,::]
		if shift < 0:
			newImage_gray = mergedImage_gray[-shift:height,::]
			newImage = mergedImage[-shift:height,::]

		image = [newImage_gray, newImage]
		return image

	# method that takes a list of images and creates one panoramic
	def stitch(self, list):

		# id numbers for parallel threads
		threadID = [0,1]

		# creation of a second list as a grey-scaled version of the original
		# in order to compute faster a merging point between two candidate images
		grayListOfImages = []
		for a in list:
			grayListOfImages.append(self.grayscale(a))

		# merge all the images in the list
		# stop when only one image is contained in the list
		# that image is the merged image of all previous
		while len(list) != 1 :
			image1 = list.pop(len(list)-1)
			image1_gray = grayListOfImages.pop( len(grayListOfImages)-1 )
			image2 = list.pop(len(list)-1)
			image2_gray = grayListOfImages.pop( len(grayListOfImages)-1 )

			# introduction of parallel threads
			# used to identify left and right image
			# through means of comparing the merging points
			process1 = threading.Thread(target=self.findMin, args=(image1_gray, image2_gray, threadID[0]))
			process2 = threading.Thread(target=self.findMin, args=(image2_gray, image1_gray, threadID[1]))
			# start threads
			process1.start()
			process2.start()
			# wait for them to end before continuing
			process1.join()
			process2.join()

			# merge the images based on the optimal merging point
			if self._min[0] < self._min[1]:
				image = self.merge(image1_gray, image2_gray, self._bestThesi[0], self._bestShift[0], image1, image2)
			else:
				image = self.merge(image2_gray, image1_gray, self._bestThesi[1], self._bestShift[1], image2, image1)

			# insert the merged image back in the list
			grayListOfImages.append( image[0] )
			list.append( image[1] )

		# when out of the loop the only image contained
		# in the list will be the final panoramic image
		finalImage = list.pop()
		return finalImage

# check user input validity for kernel operations
def checkInput(input):
	if input.lower() != 'y' and input.lower() != 'n':
		print 'please answer with "Y" for yes and "N" for no'
		return 'k'
	else:
		return input

# check user input validity for data set name
def checkName(input, files):
	flag = 0
	for filename in files:
		if filename == input:
			flag = 1
			break
	if flag == 0:
		print 'enter an available file name'
		return 'k'
	else:
		return input

#main method
# creates list of images and calls methods to merge them
def main():

	# create object of Panorama class
	pano = Panorama()
	# list to hold the images
	listOfImages = []
	# root path for image files
	path = './images/dataSets/'
	# list to hold the file names for available image sets
	listOfFolders = [filename for filename in os.listdir(path)]
	# variable to print available folders
	folder = ''
	for i in range(len(listOfFolders)):
		folder = folder + str(listOfFolders[i]) + '\n'
	# ask for user input
	name = 'k'
	while name == 'k':
		name = checkName(raw_input('Choose file name for image set. Available: \n'+folder), listOfFolders)

	# load images
	path = path + str(name)+'/'
	images = [filename for filename in os.listdir(path) if (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp') )]
	images = [path + filename for filename in images]
	for image in images:
		listOfImages.append(imresize(imread(image),(192,256)))

	# perform the stitching algorithm
	finalImage = pano.stitch(listOfImages)
	# save the result
	imsave('./images/results/'+name+'.png', finalImage)

	# ask permission for extra operations
	A = 'k'
	while A == 'k':
		A = checkInput(raw_input('Apply Image Transforms?\n(Y/N)'))

	if A == 'y':

		# parameters for kernel operations
		C=G=M=S=E='k'
		# Cylindrical transform
		while C == 'k':
			C = checkInput(raw_input('Apply Cylindrical Transform?\n(Y/N)'))
		# Gaussian filter
		while G == 'k':
			G = checkInput(raw_input('Apply Gaussian Blur?\n(Y/N)'))
		# Median filter
		while M == 'k':
			M = checkInput(raw_input('Apply Median Filter?\n(Y/N)'))
		# Sobel filter
		while S == 'k':
			S = checkInput(raw_input('Apply Sobel Operator?\n(Y/N)'))
		# Edge filter
		while E == 'k':
			E = checkInput(raw_input('Apply Edge Detector?\n(Y/N)'))

		# perform accepted kernel operations
		if C.lower() == 'y':
			cylinder = pano.cylindricalTransform(finalImage)
			imsave('./images/results/Cylindrical_'+name+'.png', cylinder)
		if G.lower() == 'y':
			blur = pano.blur(finalImage)
			imsave('./images/results/Blurred_'+name+'.png', blur)
		if M.lower() == 'y':
			median = pano.medianFilter(finalImage)
			imsave('./images/results/Median_'+name+'.png', median)
		if S.lower() == 'y':
			sobel = pano.sobel(pano.grayscale(finalImage))
			imsave('./images/results/Sobel_'+name+'.png', sobel)
		if E.lower() == 'y':
			edge = pano.edge(pano.grayscale(finalImage))
			imsave('./images/results/Edged_'+name+'.png', edge)

# define runnable function
if  __name__ =='__main__': main()