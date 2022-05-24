from PIL import Image
from PIL import ImageDraw

import face_recognition


def findFace(image_path):
    image = face_recognition.load_image_file(image_path)
    
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


image_path = './teste/MuskMask.jpg'


known_face_path = './conhecida/elon-musk.jpg'

unknown_image_path = './desconhecida/salam-superJumbo.jpg'




image = face_recognition.load_image_file(known_face_path)
identify_face_encondig = face_recognition.face_encodings(image)[0]

#criando vetor de encondigs e nomes
know_face_encodings = [
    identify_face_encondig
]

know_face_names = [
    "Elon Musk"
]

identify_face(image_path, know_face_encodings, know_face_names)