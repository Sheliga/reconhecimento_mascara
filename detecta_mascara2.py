import cv2


def carregar_imagem(path):
    imagem = cv2.imread(path)
    return imagem


def main(imgPath):
    frame = carregar_imagem(imgPath)
    #Imagem de saida
    cv2.imshow("Frame", frame)
    cv2.waitKey(0) 
 
    print("print")


# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
if __name__ == '__main__':
    imgPath = "conhecido/img1.jpg"
    main(imgPath)