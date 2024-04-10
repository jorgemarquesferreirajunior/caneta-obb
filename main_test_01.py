from utils import *
import time


dir_home = os.getcwd()
dir_dataset = os.path.join(dir_home, 'datasets')
dir_printscreens = os.path.join(dir_home, 'printscreens')
dir_resultados = os.path.join(dir_home, 'resultados')
path_modelo = os.path.join(dir_dataset, 'runs', 'obb', 'train', 'weights/best.pt')

detector_objetos = DetectorObjetos(path_modelo, GPU = True)
print('----------------------------------------------------------------------')

cont = 0
while True:
    detectar = str(input("Detectar? [S/N]: ")).strip().upper()[0]
    if detectar == 'S':
        cont+=1
        if cont > 49:
            cont = 0
        img = f"{str(cont).zfill(2)}.jpg"
        
        start = time.time()

        imagem = cv2.imread(os.path.join(dir_printscreens, img))
        # print(imagem.shape)

        confiancas, coordenadas, centros, coords_ab, inclinacoes = detector_objetos.prever_cv2(imagem)
        nome_img_result = f"imagem_predicao_{str(len(os.listdir(dir_resultados)) + 1).zfill(2)}.jpg"
        path_img_result = os.path.join(dir_resultados, nome_img_result)

        # CompiladorImagem.gerar_imagem_resultado(imagem, centros, coordenadas, inclinacoes, path_img_result)
        CompiladorImagem.gerar_imagem_resultado_2(imagem, centros, coordenadas, path_img_result)

        mensagem = DetectorObjetos.gerar_msg(confiancas, centros, inclinacoes)

        # print("")
        # print(coords_ab)
        # [print(f"{str(coo):<50}-->>", coo_ab) for coo, coo_ab in zip(coordenadas, coords_ab)]

        print(mensagem)
        end = time.time()
        print(f"Tempo de Compilacao: {((end - start)*1000):.5f} milisegundos")
        print('----------------------------------------------------------------------')
    else:
        break