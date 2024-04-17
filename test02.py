import cv2
import os
from ultralytics import YOLO
import numpy as np
import math
import torch
import time
from utils import *

# Subfuncoes
def tensor_para_array(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif hasattr(tensor, "numpy"):
        if "cuda" in str(tensor.device):
            tensor = tensor.cpu()
        return tensor.numpy()
    else:
        raise ValueError("Formato de tensor não reconhecido.")

def converter_confiancas(lst):
    return [round(float(valor), 2) for valor in lst]

def converter_coordenadas(lst):
    nova_lista = []
    for id_sub, sublist in enumerate(lst):
        sublist_int = []
        for id_subsub, subsublist in enumerate(sublist):
            subsublist_int = [int(valor) for valor in subsublist]
            sublist_int.append(subsublist_int)
        nova_lista.append(sublist_int)
    return nova_lista

def filtrar_coordenadas(acuracia, confs, coords):
    confs_filtro = []
    coords_filtro = []
    for indice, valor in enumerate(confs):
        if valor >= acuracia:
            confs_filtro.append(valor)
            coords_filtro.append(coords[indice])
    return confs_filtro, coords_filtro

def calcular_ponto_medio(A, B):
    x1, y1 = A
    x2, y2 = B
    return ((x1 + x2) // 2, (y1 + y2) // 2)
        
def gerar_centros(coords):
    centros = [calcular_ponto_medio(sublista[0], sublista[2]) for sublista in coords]
    return centros

def inverter_eixo_y(coords_int, altura_imagem):
    coordenadas_cartesianas = []
    for sublista in coords_int:
        nova_sublista = []
        for subsublista in sublista:
            novo_ponto = [subsublista[0], altura_imagem - subsublista[1]]
            nova_sublista.append(novo_ponto)
        coordenadas_cartesianas.append(nova_sublista)
    return coordenadas_cartesianas

def calcular_angulo(ponto_a, ponto_b):
    ax, ay = ponto_a
    bx, by = ponto_b

    if ay == by:
        return 0
    elif ax == bx:
        return 90
    elif ax < bx:
        delta_x = bx - ax
        if ay < by:
            delta_y = by - ay
            return round(math.degrees(math.atan2(delta_y, delta_x)), 2)
        else:
            delta_y = ay - by
            return 180 - round(math.degrees(math.atan2(delta_y, delta_x)), 2)
    elif ax > bx:
        delta_x = ax - bx
        if ay > by:
            delta_y = ay - by
            return round(math.degrees(math.atan2(delta_y, delta_x)), 2)
        else:
            delta_y = by - ay
            return 180 - round(math.degrees(math.atan2(delta_y, delta_x)), 2)
            
def gerar_inclinacoes(coordenadas_cartesianas):
    inclinacoes = []
    for lista in coordenadas_cartesianas:
        inclinacoes.append(round(calcular_angulo(lista[0], lista[3]), 2))
    return inclinacoes

def gerar_retas(coordenadas_cartesianas, inclinacoes):
    idx_mais90 = [3,0,2,3,0,1,1,2]
    idx_menos90 = [0,1,3,0,1,2,2,3]
    retas_cartesianas = []
    
    for i, coord in enumerate(coordenadas_cartesianas):
        retangulo = []
        for j in range(len(coord)):   
            if inclinacoes[i] > 90:
                linha = (coord[idx_mais90[j]][0], coord[idx_mais90[j]][1], coord[idx_mais90[j+4]][0], coord[idx_mais90[j+4]][1])
            else:
                linha = (coord[idx_menos90[j]][0], coord[idx_menos90[j]][1], coord[idx_menos90[j+4]][0], coord[idx_menos90[j+4]][1])
            retangulo.append(linha)
        retas_cartesianas.append(retangulo)
    return retas_cartesianas

def detectar_sobreposicoes(retas_cartesianas):
    sobrepostos = []
    for r, retangulo in enumerate(retas_cartesianas):
        for l in range(len(retangulo)):
            for r2 in range(len(retas_cartesianas)):
                for l2 in range(len(retangulo)):
                    if r != r2:
                        if retas_cartesianas[r] not in sobrepostos:   
                            m1 = (retas_cartesianas[r][l][3] - retas_cartesianas[r][l][1]) / (retas_cartesianas[r][l][2] - retas_cartesianas[r][l][0] - 0.01)
                            b1 = round(retas_cartesianas[r][l][1] - m1 * retas_cartesianas[r][l][0])
                            m2 = (retas_cartesianas[r2][l2][3] - retas_cartesianas[r2][l2][1]) / (retas_cartesianas[r2][l2][2] - retas_cartesianas[r2][l2][0]- 0.01)
                            b2 = round(retas_cartesianas[r2][l2][1] - m2 * retas_cartesianas[r2][l2][0])
                            for x1 in range(retas_cartesianas[r][l][0], 1+retas_cartesianas[r][l][2]):
                                y1 = round(m1*x1 + b1)
                                for x2 in range(retas_cartesianas[r2][l2][0], 1+retas_cartesianas[r2][l2][2]):
                                    y2 = round(m2*x2 + b2)
                                    resps_x = [x2-2, x2-1, x2, x2+1, x2+2]
                                    resps_y = [y2-2, y2-1, y2, y2+1, y2+2]
                                    if x1 in resps_x and y1 in resps_y:
                                        if r not in sobrepostos:
                                            sobrepostos.append(r)
                                        if r2 not in sobrepostos:
                                            sobrepostos.append(r2)
                                        break
    return sobrepostos

def marcar_centros(imagem, centros, cor=(0,0,255)):
    for idx, centro in enumerate(centros):
        cv2.putText(imagem, f"{str(idx+1)}", centro, cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

def marcar_caixas(imagem, coords_int, cor=(0,0,255)):
    for pontos in coords_int:
        p1, p2, p3, p4 = tuple(pontos[0]), tuple(pontos[1]), tuple(pontos[2]), tuple(pontos[3])
        cv2.line(imagem, p1, p2, cor, 1)
        cv2.line(imagem, p2, p3, cor, 1)
        cv2.line(imagem, p3, p4, cor, 1)
        cv2.line(imagem, p4, p1, cor, 1)

def gerar_imagem_resultado(imagem, centros, coords_int, resultado="resultado.jpg"):
    marcar_centros(imagem, centros)
    marcar_caixas(imagem, coords_int)
    cv2.imwrite(resultado, imagem)

def gerar_msg(confs, centros, inclinacoes):
    mensagem = ''
    for i, (confianca, inclinacao, centro) in enumerate(zip(confs, inclinacoes, centros)):
        mensagem += f"deteccao: {i+1} - confianca:{confianca * 100}%\n" + f"centro: {centro}\n" + f"inclinacao: {inclinacao}\n\n"
    return mensagem






# Diretorios
dir_home = os.getcwd()
dir_dataset = os.path.join(dir_home, 'datasets')
dir_printscreens = os.path.join(dir_home, 'printscreens')
dir_resultados = os.path.join(dir_home, 'resultados')


# parametros para Deteccao
acuracia = 0.50
path_modelo = os.path.join(dir_dataset, 'runs', 'obb', 'train', 'weights/best.pt')


# Modelo YOLO
modelo = YOLO(path_modelo)

cont = 0
print('----------------------------------------------------------------------')
while True:
    
    #  Inicio da deteccao
    play = str(input("Detectar? [S/N]: ")).strip().upper()[0]
    if play != 'S':break
    
    start = time.time()
    # Predicao
    path_img = os.path.join(dir_printscreens, f'{str(cont).zfill(2)}.jpg')
    imagem_alvo = cv2.imread(path_img)
    modelo.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = modelo.predict(source=path_img, save=False)[0].obb

    # Dados iniciais
    confiancas_obb, coordenadas_obb = results.conf, results.xyxyxyxy

    # Tratamento dos dados [Parte1]
    confs_np = tensor_para_array(confiancas_obb)
    coords_np = tensor_para_array(coordenadas_
                                  obb)

    # Tratamento dos dados [Parte2]
    confs_float = converter_confiancas(confs_np)
    coords_int = converter_coordenadas(coords_np)

    # Tratamento dos dados [Parte3]
    confs, coords = filtrar_coordenadas(acuracia, confs_float, coords_int)

    # Dados Secundarios
    centros = gerar_centros(coords)
    coords_cartesianas = inverter_eixo_y(coords, imagem_alvo.shape[0])
    inclinacoes = gerar_inclinacoes(coords_cartesianas)
    itens_sobrepostos = detectar_sobreposicoes(gerar_retas(coords_cartesianas, inclinacoes))

    # Visualizacao de informacoes
    mensagem = gerar_msg(confs, centros, inclinacoes)

    # Salvamento da imagem detectada
    gerar_imagem_resultado(imagem_alvo, centros, coords, os.path.join(dir_resultados, "runs.jpg"))

    print(f"Coordenadas: {coords}")
    print(f"Confiancas: {confs}")
    print(f"Coordenadas Cartesianas: {coords_cartesianas}")
    print(f"Itens Sobrepostos: {itens_sobrepostos}")

    if cont < 50:
        cont += 1
    else:
        cont = 0
    end = time.time()
    print(f"Tempo de Compilacao: {((end - start)*1000):.5f} milisegundos")
    print('----------------------------------------------------------------------')