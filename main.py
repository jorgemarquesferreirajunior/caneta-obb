from utils import *
import time
import numpy as np
import cv2

dir_home = os.getcwd()
dir_dataset = os.path.join(dir_home, 'datasets')
dir_printscreens = os.path.join(dir_home, 'printscreens')
dir_resultados = os.path.join(dir_home, 'resultados')
path_modelo = os.path.join(dir_dataset, 'runs', 'obb', 'train', 'weights/best.pt')
detector_objetos = DetectorObjetos(path_modelo, GPU = True)

print('----------------------------------------------------------------------')
cont = 0
UtilitariosArquivo.esvaziar_pasta(dir_resultados)
while True:
    detectar = str(input("Detectar? [S/N]: ")).strip().upper()[0]
    if detectar != 'S': break

    print(f"Deteccao [{str(cont).zfill(2)}]")

    start = time.time()
    img = f"{str(cont).zfill(2)}.jpg"
    imagem = cv2.imread(os.path.join(dir_printscreens, img))
    confiancas, coordenadas, centros, coords_ab, inclinacoes = detector_objetos.prever_cv2(imagem)
    nome_img_result = f"pred-{str(len(os.listdir(dir_resultados)) + 1).zfill(2)}.jpg"
    path_img_result = os.path.join(dir_resultados, nome_img_result)
    deteccoes_unicas, deteccoes_sobrepostas = DetectorObjetos.ordenar_deteccoes(confiancas, centros, coordenadas, coords_ab, inclinacoes)
    CompiladorImagem.gerar_imagem_resultado_3(imagem, deteccoes_unicas, deteccoes_sobrepostas, path_img_result)
    mensagem = DetectorObjetos.gerar_msg2(deteccoes_unicas, deteccoes_sobrepostas)
    print(mensagem)

    cont+=1
    if cont > 49:
        cont = 0

    end = time.time()
    print(f"Tempo de Compilacao: {((end - start)*1000):.3f} milisegundos")
    print('----------------------------------------------------------------------')
   