import numpy as np
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from matplotlib import pyplot as plt
import operator
from tkinter import *
import PIL
from PIL import Image, ImageDraw
import cv2
import time
import os
#(train_x, train_y), (test_x, test_y) = mnist.load_data()
# x, y = fetch_openml('mnist_784', version=1, return_X_y=True) # already flattened unlike mnist.load_data()
# train_x, test_x, train_y, test_y = train_test_split((x/255).astype('float32'), to_categorical(y), test_size=0.15, random_state=42)

#print(train_x.shape)

dot = lambda m1, m2: sum([i*j for (i, j) in zip(m1, m2)]) # row column matrix multiplication
sigmoid = lambda Z: 1 / (1+np.exp(-Z))
sigmoid_derivative = lambda Z: sigmoid(Z)*(1-sigmoid(Z))
softmax_derivative = lambda Z: softmax(Z)*(1-softmax(Z))
transpose = lambda matrix: np.array([[matrix[y][x] for y in range(len(matrix))] for x in range(len(matrix[0]))])
multiply = lambda m1, m2: np.array([[sum(map(operator.mul, row, col)) for col in transpose(m2)] for row in m1])
class DeepNN:
    def __init__(self, layer_dimensions = [784, 50, 10], learning_rate=0.1, epoches=100):
        self.layer_dims = layer_dimensions
        self.learning_rate, self.epoches = learning_rate, epoches
        self.params = {'W{}'.format(i): np.random.randn(*self.layer_dims[i-1:i+1][::-1]) for i in range(1, len(self.layer_dims))}
        self.weights_shapes = {k:v.shape for (k,v) in self.params.items()}
        self.params = {**self.params, **{"b{}".format(i):np.zeros((self.layer_dims[i])) for i in range(1, len(self.layer_dims))}}
    def forward(self, input):
        self.params["A0"] = input
        for i in range(1, len(self.layer_dims)):
            self.params["Z{}".format(i)] = (np.dot(self.params["W{}".format(i)], input if i == 1 else self.params["A{}".format(i-1)]))
            if i is not self.layer_dims:
                self.params["A{}".format(i)] = sigmoid(self.params["Z{}".format(i)]) # for now not generalizable. Will fix later.
        self.params["A2"] = softmax(self.params["Z2"]) # Fully connected layer uses softmax activation
        #print(self.params["A2"])
        return self.params["A2"]
    def backpropagation(self, actual, output):
        weights_update, biases_update = {}, {}
        error = 2*(output - actual) / output.shape[0] * softmax_derivative(self.params["Z2"])
        weights_update["W2"] = np.outer(error, self.params["A1"])
        # now for layer 1 with sigmoid activation
        error = np.dot(self.params["W2"].T, error) * sigmoid_derivative(self.params["Z1"])
        weights_update["W1"] = np.outer(error, self.params["A0"]) # A0 is input
        return (weights_update, biases_update)
    def update_network(self, weight_update, biases_update):
        for key, value in {**weight_update, **biases_update}.items():
            self.params[key] -= self.learning_rate * value 
    def predictionPrecision(self, test_x, test_y):
        predictions = []
        for x, y in zip(test_x, test_y):
            predictions.append(np.argmax(self.forward(x)) == np.argmax(y))
        return np.average(predictions)    
    def train_network(self, train_x, train_y, test_x, test_y):
        start_timer = time.time()
        for iteration in range(self.epoches):
            for x,y in zip(train_x, train_y):
                output = self.forward(x)
                weights_update, biases_update = self.backpropagation(y, output)
                self.update_network(weights_update, biases_update)
            precision = self.predictionPrecision(test_x, test_y)
            if precision > 0.9:
                print("Adequate. Saved.")
                self.saveWeights()
            print("Epoch: {0}, computed precision: {1:.2f}%".format(iteration+1, precision*100))
    def saveWeights(self):
        np.save('weights.npy', {k: v for k, v in self.params.items() if k.startswith('W')})
    def loadWeights(self):
        for key, value in np.load('weights.npy',allow_pickle=True).item().items():
            self.params[key] = value    

deepNN = DeepNN()
print(deepNN.weights_shapes)
#deepNN.train_network(train_x, train_y, test_x, test_y)
deepNN.loadWeights()
#print("Accuracy with loaded weights: {}%".format(100*deepNN.predictionPrecision(test_x, test_y)))

def save():
    os.remove(f'imageDigit.png')
    image1.save(f'imageDigit.png')
    image = cv2.imread('imageDigit.png', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in [contours[0]]:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.imshow("CNT", image)
        top,left = int(0.15*th.shape[0]),int(0.15*th.shape[1])
        roi = th[y-top:y+h+top,x-left:x+w+left] # copy make border faster?
        img = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
        # kernel = np.ones((3,3),np.uint8)
        # img = cv2.dilate(img,kernel,iterations = 1)
        #img = cv2.bitwise_not(img)
        img = img/255.0
        print("Prediction: {}".format(np.argmax(deepNN.forward((img).flatten()))))
        plt.imshow(np.array(img, dtype='float').reshape(28,28))
        plt.show()
def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y
def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=12)
    draw.line((lastx, lasty, x, y), fill='black', width=12)
    lastx, lasty = x, y
def clear():
    cv.delete("all")
    draw.rectangle([(0,0),image1.size], fill = 'white' )   
root = Tk()
lastx, lasty = None, None
image_number = 0
cv = Canvas(root, width=300, height=300, bg='white')
image1 = PIL.Image.new('RGB', (300, 300), 'white')
draw = ImageDraw.Draw(image1)
cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)
btn_save = Button(text="save", command=save)
btn_clear = Button(text="clear", command=clear)
btn_save.pack()
btn_clear.pack()
root.mainloop()



        
# print(sigmoid(2))
# 
# print(np.dot(np.random.randn(128, 784), np.zeros((1, 784)).T).shape)
# print(np.dot(np.random.randn(128, 784), train_x[0].flatten()).shape)
# print(np.zeros((1,20)))
# print(transpose(np.random.randn(128, 784)).shape)
# print(multiply(np.zeros((1, 784)), np.random.randn(128, 784)).shape)
# print(multiply(np.array([train_x[0].flatten()]), np.random.randn(128, 784)).shape)
