# import the necessary packages
import cv2
from PIL import ImageChops, Image
import math
import operator
from functools import reduce
def rmsdiff(im1, im2):
    h = ImageChops.difference(im1, im2).histogram()
    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))

im1 = Image.open("/Users/pranoy/Desktop/xmile_tech/static/patient_data/dean/1.jpg")
im2 = Image.open("/Users/pranoy/Desktop/xmile_tech/static/patient_data/dean/4.jpg")

print(rmsdiff(im1,im2))