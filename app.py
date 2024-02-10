import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def predict(filename):
   names=open(r"coco.names").read()
   names = names.strip().split("\n") 
   print(names)
   
   configuration_path = (r"yolov4.cfg")
   weights_path = (r"yolov4.weights")
   pro_min = 0.5 
   threshold = 0.2
   net = cv2.dnn.readNetFromDarknet(configuration_path,weights_path)
   layers = net.getLayerNames()
   output_layers=[layers[i - 1] for i in net.getUnconnectedOutLayers()]
   image = cv2.imread(filename)
   blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416,416), swapRB=True, crop=False)
   net.setInput(blob) # giving blob as input to our YOLO Network.
   t1=time.time()
   output = net.forward(output_layers)
   t2 = time.time()
   colours = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8') # randint(low, high=None, size=None, dtype='l')
   classes = []
   confidences = []
   boxes = []
   Height = image.shape[0]
   Width = image.shape[1]
    
   for out in output:
       for res in out:
            scores = res[5:]
            class_current = np.argmax(scores) # returning indices with max score and that would be our class as that will be 1 and rest will be 0
    
            # Getting the probability for current object by accessing the indices returned by argmax.
            confidence_current = scores[class_current]
    
            # Eliminating the weak predictions that is with minimum probability and this loop will only be encountered when an object will be there
            if confidence_current > 0.5:
                
                # Scaling bounding box coordinates to the initial image size
                # YOLO data format just keeps center of detected box and its width and height
                #that is why we are multiplying them elemwnt wise by width and height
                box = res[0:4] * np.array([Width, Height, Width, Height])  #In the first 4 indices only contains 
                #the output consisting of the coordinates.
                print(res[0:4])
                print(box)
    
                # From current box with YOLO format getting top left corner coordinates
                # that are x and y
                x, y, w, h = box.astype('int')
                x = int(x - (w / 2))
                y = int(y - (h / 2))
                
    
                # Adding results into the lists
                boxes.append([x, y, int(w), int(h)]) ## appending all the boxes.
                confidences.append(float(confidence_current)) ## appending all the confidences
                classes.append(class_current) ## appending all the classes
                
                
   results = cv2.dnn.NMSBoxes(boxes, confidences,0.2,0.4)
   g=[]
   for i in range(len(classes)):
        g.append(names[int(classes[i])])
   g=list(set(g))
   import csv
   with open('prediction.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in g:
            y = []
            y.append(i)
            writer.writerow(y)
            
   if len(results) > 0:

        for i in results.flatten():
    
            # Getting current bounding box coordinates
            x, y = boxes[i][0],boxes[i][1]
            width, height = boxes[i][2], boxes[i][3]
            
            colour_box_current = [int(j) for j in colours[classes[i]]]
    
            # Drawing bounding box on the original image
            cv2.rectangle(image, (x, y), (x + width, y + height),
                          colour_box_current,7 )
    
            # Preparing text with label and confidence 
            text_box_current = '{}: {:.4f}'.format(names[int(classes[i])], confidences[i])
    
            # Putting text with label and confidence
            cv2.putText(image, text_box_current, (x+2, y+2), cv2.FONT_HERSHEY_TRIPLEX, 1.5,(0,0,255))
            
   plt.rcParams['figure.figsize'] = (10,10)
   plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   plt.show()
   return g
    
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      y = predict(f.filename)
      return render_template('predict.html', result = y)
        
if __name__ == '__main__':
   app.run()