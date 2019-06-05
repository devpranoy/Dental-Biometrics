# importing packages
from flask import Flask ,render_template, redirect, url_for, session, request, logging
#import requests
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
#import json
import os
import torchvision.models as models
import torch.nn as nn
#Specify model architecture 
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
import glob
import cv2
from PIL import ImageChops, Image
import math
import operator
from functools import reduce


model_transfer = models.resnet50(pretrained=True)
for param in model_transfer.parameters():
    param.requires_grad = False
model_transfer.fc = nn.Linear(2048, 4, bias=True)
fc_parameters = model_transfer.fc.parameters()


class_names = ["Canines","Incisors","Premolars","Molars"]
standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

model_transfer.load_state_dict(torch.load('/Users/pranoy/Desktop/xmile_tech/model_transfer.pt', map_location='cpu'))

def rmsdiff(im1, im2):
    h = ImageChops.difference(im1, im2).histogram()
    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))



def load_input_image(img_path):    
    image = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.Grayscale(num_output_channels=3),
                                     transforms.ToTensor(), 
                                     standard_normalization])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image

def predict_teeth_transfer(model, class_names, img_path):
    # load the image and return the predicted teeth 
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]


file_path = ""

app = Flask(__name__) #app initialisation

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)

@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@nocache
@app.route('/detection', methods=['GET','POST']) #landing page intent
def detection():
    return render_template("SCAN.html")

@nocache
@app.route('/detection_scan', methods=['GET','POST']) #landing page intent
def detection_scan():
    for filename in glob.iglob('/Users/pranoy/Desktop/xmile_tech/static/for_scan/*', recursive=True):
        os.system("rm -rf %s"%filename)
    return render_template("SCAN_img.html")


@app.route('/', methods=['GET','POST']) #landing page intent
def home():
    return render_template("index.html")



@app.route('/login', methods=['GET','POST']) #landing page intent
def login():
    return render_template("login.html")


@app.route('/upload', methods=['GET','POST']) #landing page intent
def upload():
    if request.method=='POST':
        app.config['UPLOAD_FOLDER']="/Users/pranoy/Desktop/xmile_tech/uploads/"
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        # try:
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        path =  app.config['UPLOAD_FOLDER']+f.filename

        app.config['path'] = app.config['UPLOAD_FOLDER']+f.filename
        file_path = path
        img_path = path
        patient_name = request.form['patient_name']
        if not os.path.exists("/Users/pranoy/Desktop/xmile_tech/static/patient_data/"+patient_name):
            os.makedirs("/Users/pranoy/Desktop/xmile_tech/static/patient_data/"+patient_name)

        app.config['patient_name']= patient_name
        command_predict_yolo = "/Users/pranoy/xray/darknet_predictor/darknet detector test /Users/pranoy/xray/darknet_predictor/cfg/obj.data /Users/pranoy/xray/darknet_predictor/cfg/yolo-obj.cfg /Users/pranoy/xray/darknet_predictor/backup/yolo_dental.weights %s -thresh 0.22"%img_path
        #command_extract_img = "/Users/pranoy/ocr/darknet/darknet detector test /Users/pranoy/ocr/darknet/cfg/obj.data /Users/pranoy/ocr/darknet/cfg/yolo-obj.cfg /Users/pranoy/ocr/darknet/backup/yolo-obj.backup %s -thresh 0.1"%img_path
        #move_predictions = "mv /Users/pranoy/Desktop/YOCR_WEBSITE/predictions.jpg /Users/pranoy/Desktop/YOCR_WEBSITE/static/predictions.jpg"
        os.system(command_predict_yolo)
        os.system("mv predictions.jpg static/predict/")
        os.system("cp static/predict/predictions.jpg static/patient_data/"+patient_name)
        return redirect(url_for('detection'))
    return render_template("UPLOAD.html") #display the html template




@nocache
@app.route('/extract', methods=['GET','POST']) #landing page intent
def extract():
    print(app.config['path'])
    command_extract_img ="/Users/pranoy/xray/darknet/darknet detector test /Users/pranoy/xray/darknet_predictor/cfg/obj.data /Users/pranoy/xray/darknet_predictor/cfg/yolo-obj.cfg /Users/pranoy/xray/darknet_predictor/backup/yolo_dental.weights %s -thresh 0.22"%app.config['path']
    os.system(command_extract_img)
    os.system("rm -rf /Users/pranoy/Desktop/xmile_tech/predictions.jpg")
    files=[]
    for filename in glob.iglob('/Users/pranoy/Desktop/xmile_tech/*.jpg', recursive=True):
        os.system("mv %s /Users/pranoy/Desktop/xmile_tech/static/teeth/"%filename)
    for filename in glob.iglob('/Users/pranoy/Desktop/xmile_tech/static/teeth/*.jpg', recursive=True):
        #files.append(".."+filename[32:])
        files.append(filename)

    canines=[]
    molars=[]
    incisors=[]
    premolars=[]
    for file in files:
        prediction = predict_teeth_transfer(model_transfer,class_names,file)
        print(prediction)
        if prediction == "Premolars":
            premolars.append(file[32:])
        elif prediction =="Molars":
            molars.append(file[32:])
        elif prediction =="Canines":
            canines.append(file[32:])
        elif prediction =="Incisors":
            incisors.append(file[32:])
    
    return render_template("CLASSIFY.html",incisors=incisors,molars=molars,canines=canines,premolars=premolars)

@nocache
@app.route('/extract_img', methods=['GET','POST']) #landing page intent
def extract_img():
    final_score={}
    print(app.config['path'])
    command_extract_img ="/Users/pranoy/xray/darknet/darknet detector test /Users/pranoy/xray/darknet_predictor/cfg/obj.data /Users/pranoy/xray/darknet_predictor/cfg/yolo-obj.cfg /Users/pranoy/xray/darknet_predictor/backup/yolo_dental.weights %s -thresh 0.22"%app.config['path']
    os.system(command_extract_img)
    os.system("rm -rf /Users/pranoy/Desktop/xmile_tech/predictions.jpg")
    files=[]
    for filename in glob.iglob('/Users/pranoy/Desktop/xmile_tech/*.jpg', recursive=True):
        os.system("mv %s /Users/pranoy/Desktop/xmile_tech/static/for_scan"%filename)
    for filename in glob.iglob('/Users/pranoy/Desktop/xmile_tech/static/for_scan/*.jpg', recursive=True):
        #files.append(".."+filename[32:])
        files.append(filename)

    canines=[]
    molars=[]
    incisors=[]
    premolars=[]
    for file in files:
        prediction = predict_teeth_transfer(model_transfer,class_names,file)
        print(prediction)
        if prediction == "Premolars":
            premolars.append(file[32:])
        elif prediction =="Molars":
            molars.append(file[32:])
        elif prediction =="Canines":
            canines.append(file[32:])
        elif prediction =="Incisors":
            incisors.append(file[32:])
    for people in glob.iglob('/Users/pranoy/Desktop/xmile_tech/static/patient_data/*',recursive=True):
        score = []
        total_score=0
        for tooth in glob.iglob('/Users/pranoy/Desktop/xmile_tech/static/patient_data/%s/*.jpg'%people[52:], recursive=True):
            if tooth == "/Users/pranoy/Desktop/xmile_tech/static/patient_data/%s/predictions.jpg"%people[52:]:
                continue
            for file in files:    
                im1 = Image.open(file)
                im2 = Image.open(tooth)
                score.append(int(rmsdiff(im1,im2)))
        for i in score:
            total_score = i+total_score
        mean = total_score/len(score)
        final_score[str(people[53:]).capitalize()] = mean
        print("Score of Person",people[53:],"is ",mean)
    sorted_score = sorted(final_score.items(), key=lambda x: x[1])
    
    return render_template("prediction_scan.html",score=sorted_score,count=len(sorted_score))





@app.route('/save', methods=['GET','POST']) #landing page intent
def save():
    for filename in glob.iglob('/Users/pranoy/Desktop/xmile_tech/static/teeth/*.jpg', recursive=True):
        os.system("mv %s /Users/pranoy/Desktop/xmile_tech/static/patient_data/%s"%(filename,app.config['patient_name']))
    return redirect(url_for('upload'))


@app.route('/scan_img', methods=['GET','POST']) #landing page intent
def scan_img():
    if request.method=='POST':
        app.config['UPLOAD_FOLDER']="/Users/pranoy/Desktop/xmile_tech/uploads/"
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        # try:
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        path =  app.config['UPLOAD_FOLDER']+f.filename

        app.config['path'] = app.config['UPLOAD_FOLDER']+f.filename
        file_path = path
        img_path = path
        command_predict_yolo = "/Users/pranoy/xray/darknet_predictor/darknet detector test /Users/pranoy/xray/darknet_predictor/cfg/obj.data /Users/pranoy/xray/darknet_predictor/cfg/yolo-obj.cfg /Users/pranoy/xray/darknet_predictor/backup/yolo_dental.weights %s -thresh 0.22"%img_path
        #command_extract_img = "/Users/pranoy/ocr/darknet/darknet detector test /Users/pranoy/ocr/darknet/cfg/obj.data /Users/pranoy/ocr/darknet/cfg/yolo-obj.cfg /Users/pranoy/ocr/darknet/backup/yolo-obj.backup %s -thresh 0.1"%img_path
        #move_predictions = "mv /Users/pranoy/Desktop/YOCR_WEBSITE/predictions.jpg /Users/pranoy/Desktop/YOCR_WEBSITE/static/predictions.jpg"
        os.system(command_predict_yolo)
        os.system("mv predictions.jpg static/predict/")
        os.system("cp static/predict/predictions.jpg static/scan_data/")
        return redirect(url_for('detection_scan'))
    return render_template("UPLOAD_IMG.html")

if __name__=='__main__':
	app.run(debug=True,host="0.0.0.0",port=8000) 
    #use threaded=True instead of debug=True for production
    # use port =80 for using the http port



#sample code for form data recieve
# request.form['name']
# Sample Code for JSON send data to api

#url = 'URL_FOR_API'
#data = {'TimeIndex':time1 ,'Name':name,'PhoneNumber':phone}
#headers = {'content-type': 'application/json'}
#r=requests.post(url, data=json.dumps(data), headers=headers)
#data = r.json()
#print(data)


#Sample code for JSON recieve data from API

#url = 'URL_FOR_API'
#headers = {'content-type': 'application/json'}
#r=requests.get(url, headers=headers)
#data = r.json()
#count = data['Count']