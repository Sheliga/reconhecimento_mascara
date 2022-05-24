import cv2

def carregar_imagem():
    imagem = cv2.imread("conhecido/img1.jpg")
    return imagem


def main():
    frame = carregar_imagem()
    #Imagem de saida
    cv2.imshow("Frame", frame)
    cv2.waitKey(0) 
 
    print("print")

if __name__ == '__main__':
    main()