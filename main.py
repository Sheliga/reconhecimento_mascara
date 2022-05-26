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
    return str(len(face_locations))


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
    print("unknow")
    print(unknown_image_path)
    #carregar imagem de teste
    unknown_image_path = cv2.cvtColor(unknown_image_path, cv2.COLOR_BGR2RGB)
    test_image = face_recognition.load_image_file(unknown_image_path)

    #encontrar faces na imagem de teste
    face_locations = face_recognition.face_locations(test_image)
    face_encondings = face_recognition.face_encodings(test_image, face_locations)   

    # Loop pelos rostos encontrador na imagem teste 
    for(top, right, bottom, left), face_enconding in zip(face_locations, face_encondings):
        matches = face_recognition.compare_faces(know_face_encodings, face_enconding)

        name = "Pessoa Desconhecida"

        # se corresponder
        if True in matches:
            first_match_index = matches.index(True)
            name = know_face_names[first_match_index]
            print(name)




####
def carregar_imagem(path):
    imagem = cv2.imread(path)
    return imagem

def abrir_imagem(path):
    frame = carregar_imagem(path)
    #Imagem de saida
    cv2.imshow("Frame", frame)
    findFace(frame)
    
    cv2.waitKey(0) 


def redimensionar_imagem(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def abrir_video(origem, faceNet, maskNet, scale_percent, image_path, know_face_encodings, know_face_names):
    pathVideo = origem
    
    captura = cv2.VideoCapture(pathVideo)

    while(1):
        ret, frame = captura.read()        
        frame = redimensionar_imagem(frame, scale_percent)

        #findFace(frame)
        
        ####
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        
        
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mascara" if mask > withoutMask else "Sem Mascara"
            color = (0, 255, 0) if label == "Mascara" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            name = findFace(frame)
            
            ####
            cv2.putText(frame, name, (startX, startY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            #identify_face(image_path, know_face_encodings, know_face_names)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        ####
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
scale_percent = 40 # percent of original size

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


#reconhecimento do individuo
known_face_path = r"D:\Projetos\Python\reconhecimento_mascara\src\conhecida\elon-musk.jpg"
image_path = r"D:\Projetos\Python\reconhecimento_mascara\src\teste\MuskMask.jpg"

unknown_image_path = r"src/desconhecida/salam-superJumbo.jpg"

#
individuo = face_recognition.load_image_file(known_face_path)
identify_face_encondig = face_recognition.face_encodings(individuo)[0]

#criando vetor de encondigs e nomes
know_face_encodings = [
    identify_face_encondig
]
know_face_names = [
    "Elon Musk"
]


def teste(image_path, know_face_encodings, know_face_names):
    frame = carregar_imagem(image_path)
    #Imagem de saida
    cv2.imshow("Frame", frame)
    findFace(frame)
    identify_face(frame, know_face_encodings, know_face_names)
    cv2.waitKey(0)
def main():
    #abrir_video(origem, faceNet, maskNet, scale_percent, image_path, know_face_encodings, know_face_names)
    #abrir_webcam()
    teste(image_path, know_face_encodings, know_face_names)
    
    #abrir_imagem(image_path)
    



if __name__ == '__main__':
    print("Main do projeto")
    main()
    cv2.destroyAllWindows()