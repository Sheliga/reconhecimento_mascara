import cv2 as cv


def carregar_imagem():
    imagem = cv.imread("conhecido/img1.jpg")
    return imagem


def main():
    frame = carregar_imagem()
    #Imagem de saida
    cv.imshow("Frame", frame)
    cv.waitKey(0) 
 
    print("print")

if __name__ == '__main__':
    main()