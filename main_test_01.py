from utils import *
import time
import numpy as np
import cv2

CORES = [
    (255, 255, 255),  # Branco
    (0, 255, 255),    # Ciano
    (0, 255, 0),      # Verde claro
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Amarelo
    (0, 0, 255),      # Vermelho
    (0, 0, 125),
    (125, 0, 125),

]


def desenhar_retangulos(img, coords):
    for idx, deteccao in enumerate(coords):
        for ponto_atual in range(len(deteccao)):
            ponto_seguinte = (ponto_atual + 1) % len(deteccao)
            xi,yi = (deteccao[ponto_atual][0], deteccao[ponto_atual][1])
            xf,yf = (deteccao[ponto_seguinte][0], deteccao[ponto_seguinte][1])
            cv2.line(img, (xi, yi), (xf, yf),CORES[idx], 1)
            cv2.circle(img, (xi, yi), 3, CORES[ponto_atual], 2)


def retas(coords_ab, inclinacoes):
    count = 0
    sobrepostos = []
    idx_mais90 = [3,0,2,3,0,1,1,2]
    idx_menos90 = [0,1,3,0,1,2,2,3]
    retas_ab = []
    
    for i, coord in enumerate(coords_ab):
        retangulo = []
        for j in range(len(coord)):   
            if inclinacoes[i] > 90:
                linha = (coord[idx_mais90[j]][0], coord[idx_mais90[j]][1], coord[idx_mais90[j+4]][0], coord[idx_mais90[j+4]][1])

            else:
                linha = (coord[idx_menos90[j]][0], coord[idx_menos90[j]][1], coord[idx_menos90[j+4]][0], coord[idx_menos90[j+4]][1])
            retangulo.append(linha)
        retas_ab.append(retangulo)
            
    return retas_ab
            

def testa_sobreposicao(retas_ab):
    sobrepostos = []
    for r, retangulo in enumerate(retas_ab):
        for l in range(0, 1):
            # print(f"Retangulo[{r}]")
            for r2 in range(len(retas_ab)):
                for l2 in range(len(retangulo)):
                    if r != r2:
                        if retas_ab[r] not in sobrepostos:   
                            m1 = (retas_ab[r][l][3] - retas_ab[r][l][1]) / (retas_ab[r][l][2] - retas_ab[r][l][0] - 0.01)
                            b1 = round(retas_ab[r][l][1] - m1 * retas_ab[r][l][0])

                            m2 = (retas_ab[r2][l2][3] - retas_ab[r2][l2][1]) / (retas_ab[r2][l2][2] - retas_ab[r2][l2][0]- 0.01)
                            b2 = round(retas_ab[r2][l2][1] - m2 * retas_ab[r2][l2][0])
                            # print(f"retas_ab[{r}][{l}] = {retas_ab[r][l]} m1={m1:.2f} b1={b1:.2f} retas_ab[{r2}][{l2}] = {retas_ab[r2][l2]} m2={m2:.2f} b2={b2:.2f}")

                            
                            for x1 in range(retas_ab[r][l][0], 1+retas_ab[r][l][2]):
                                y1 = round(m1*x1 + b1)
                                for x2 in range(retas_ab[r2][l2][0], 1+retas_ab[r2][l2][2]):
                                    y2 = round(m2*x2 + b2)
                                    
                                    resps_x = [x2-1, x2+1]
                                    resps_y = [y2-1, y2+1]
                                    if x1 in resps_x and y1 in resps_y:
                                        # print(x1, y1, x2, y2)
                                        if r not in sobrepostos:
                                            sobrepostos.append(r)
                                        if r2 not in sobrepostos:
                                            sobrepostos.append(r2)
                                        break
        # print("")
    return sobrepostos

dir_home = os.getcwd()
dir_dataset = os.path.join(dir_home, 'datasets')
dir_printscreens = os.path.join(dir_home, 'printscreens')
dir_resultados = os.path.join(dir_home, 'resultados')
path_modelo = os.path.join(dir_dataset, 'runs', 'obb', 'train', 'weights/best.pt')

detector_objetos = DetectorObjetos(path_modelo, GPU = True)
print('----------------------------------------------------------------------')

cont = 8
UtilitariosArquivo.esvaziar_pasta(dir_resultados)
while True:
    detectar = str(input("Detectar? [S/N]: ")).strip().upper()[0]
    if detectar == 'S':
        cont+=1
        print(f"Deteccao [{str(cont).zfill(2)}]")
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

        # print("Coords_ab")
        # print(coords_ab)
        # print("inclinacoes")
        # print(inclinacoes)
        # print("Coords yolo")
        # print(coordenadas)
        
        # [print(f"{str(coo):<50}-->>", coo_ab) for coo, coo_ab in zip(coordenadas, coords_ab)]

        # print(mensagem)

        end = time.time()
        print(f"Tempo de Compilacao: {((end - start)*1000):.5f} milisegundos")

        retas_ab = retas(coords_ab, inclinacoes)
        sobrepostos = testa_sobreposicao(retas_ab)
        sobrepostos = sorted(sobrepostos)

        print("Retangulos Sobrepostos: ", end= "")
        [print(s+1, end=" ") for s in sobrepostos]

        print("")

        for ret in coords_ab:
            print(ret)
        print('----------------------------------------------------------------------')
    else:
        break





    