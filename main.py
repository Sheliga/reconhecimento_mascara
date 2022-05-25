#from email.mime import image
import cv2
import numpy as np 

from PIL import Image
from PIL import ImageDraw

import face_recognition


###################################
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import os

###################################
def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
###################################

def findFace(image_path):
    #image = face_recognition.load_image_file(image_path)
    image = image_path
    #vetor de coordenadas de cada rosto
    face_locations = face_recognition.face_locations(image)
    #print(face_locations)
    print(len(face_locations))


def compareFaces(known_image_path, unknown_image_path):
    image_known = face_recognition.load_image_file(known_image_path)
    known_face_enconding = face_recognition.face_encodings(image_known)[0]

    
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_face_enconding = face_recognition.face_encodings(unknown_image)[0]


    #comparando as imagens
    results = face_recognition.compare_faces([known_face_enconding], unknown_face_enconding)
    
    if (results[0]):
        print("Correspondem")
    else:
        print("Não correspondem")



def pullFaces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()
        #pil_image.save(f'{top}.jpg')

def identify_face(unknown_image_path, know_face_encodings, know_face_names):
    #carregar imagem de teste
    test_image = face_recognition.load_image_file(unknown_image_path)

    #encontrar faces na imagem de teste
    face_locations = face_recognition.face_locations(test_image)
    face_encondings = face_recognition.face_encodings(test_image, face_locations)   

    #converter pro formato PIL
    pil_image = Image.fromarray(test_image)

    #criar uma instancia da classe ImageDraw
    draw = ImageDraw.Draw(pil_image)

    # Loop pelos rostos encontrador na imagem teste 
    for(top, right, bottom, left), face_enconding in zip(face_locations, face_encondings):
        matches = face_recognition.compare_faces(know_face_encodings, face_enconding)

        name = "Pessoa Desconhecida"

        # se corresponder
        if True in matches:
            first_match_index = matches.index(True)
            name = know_face_names[first_match_index]

    # Desenhando a marcação
    draw.rectangle(((left, top), (right, bottom)), outline = (0, 0, 0))

    #desenhando o rotulo
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill = (0, 0, 0), outline = (0, 0, 0))
    draw.text((left + 6, bottom - text_height - 5), name, fill = (255, 255, 255, 255))

    del draw

    pil_image.show()

####
def carregar_imagem():
    imagem = cv2.imread("conhecido/img1.jpg")
    return imagem

def abrir_imagem():
    frame = carregar_imagem()
    #Imagem de saida
    cv2.imshow("Frame", frame)
    cv2.waitKey(0) 
 
    print("print")
    
def abrir_video(origem, faceNet, maskNet):
    pathVideo = origem
    
    captura = cv2.VideoCapture(pathVideo)

    while(1):
        ret, frame = captura.read()
        scale_percent = 50 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        detect_and_predict_mask(frame, faceNet, maskNet)
        findFace(frame)
        cv2.imshow("Video", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    captura.release()
    cv2.destroyAllWindows()

                                                                                       



########
#Inicializando setup
WEBCAM = 0
#origem = WEBCAM
origem = r"./videos/videoDoria.mp4"
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")
def main():
    abrir_video(origem, faceNet, maskNet)
    #abrir_webcam()
    







"""
# Carregar modelo de detecção
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

"""
if __name__ == '__main__':
    print("Main do projeto")
    main()
    cv2.destroyAllWindows()