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
            for r2 in range(len(retas_ab)):
                for l2 in range(len(retangulo)):
                    if r != r2:
                        if retas_ab[r] not in sobrepostos:   
                            m1 = (retas_ab[r][l][3] - retas_ab[r][l][1]) / (retas_ab[r][l][2] - retas_ab[r][l][0])
                            b1 = round(retas_ab[r][l][1] - m1 * retas_ab[r][l][0])

                            m2 = (retas_ab[r2][l2][3] - retas_ab[r2][l2][1]) / (retas_ab[r2][l2][2] - retas_ab[r2][l2][0])
                            b2 = round(retas_ab[r2][l2][1] - m2 * retas_ab[r2][l2][0])
                            # print(f"retas_ab[{r}][{l}] = {retas_ab[r][l]} m1={m1:.2f} b1={b1:.2f} retas_ab[{r2}][{l2}] = {retas_ab[r2][l2]} m2={m2:.2f} b2={b2:.2f}")

                            
                            for x1 in range(retas_ab[r][l][0], 1+retas_ab[r][l][2]):
                                y1 = round(m1*x1 + b1)
                                for x2 in range(retas_ab[r2][l2][0], 1+retas_ab[r2][l2][2]):
                                    y2 = round(m2*x2 + b2)
                                    
                                    resps_x = [x2-1, x2+1]
                                    resps_y = [y2-1, y2+1]
                                    if x1 in resps_x and y1 in resps_y:
                                        if r not in sobrepostos:
                                            sobrepostos.append(r)
                                        if r2 not in sobrepostos:
                                            sobrepostos.append(r2)
                                        break
    return sobrepostos

coords = [
    [[141, 376], [154, 362], [38, 253], [24, 267]],
    [[128, 204], [144, 212], [217, 71], [201, 63]],
    [[228, 333], [243, 332], [240, 177], [225, 177]],
    [[269, 327], [281, 332], [343, 185], [330, 179]],
    [[85, 261], [98, 261], [97, 110], [84, 111]],
    [[190, 385], [209, 382], [185, 230], [166, 233]],
    [[294, 215], [307, 211], [262, 63], [249, 67]]]

coords_ab = [
    [[141, 24], [154, 38], [38, 147], [24, 133]],
    [[128, 196], [144, 188], [217, 329], [201, 337]],
    [[228, 67], [243, 68], [240, 223], [225, 223]],
    [[269, 73], [281, 68], [343, 215], [330, 221]],
    [[85, 139], [98, 139], [97, 290], [84, 289]],
    [[190, 15], [209, 18], [185, 170], [166, 167]],
    [[294, 185], [307, 189], [262, 337], [249, 333]]]


inclinacoes = [137.03, 62.63, 91.1, 67.6, 90.38, 98.97, 106.91]

coords_ab2 = [[[251, 49], [262, 43], [340, 181], [329, 188]], [[82, 137], [83, 121], [241, 128], [240, 144]], [[122, 40], [137, 39], [155, 199], [141, 201]]]
inclinacoes2 = [60.7, 2.54, 83.27]




retas_ab = retas(coords_ab2, inclinacoes2)
sobrepostos = testa_sobreposicao(retas_ab)
sobrepostos = sorted(sobrepostos)

print("Retangulos Sobrepostos: ", end= "")
[print(s+1, end=" ") for s in sobrepostos]

print("")

for ret in coords_ab2:
    print(ret)
# imagem_teste = np.zeros((400,400, 3), np.uint8)
# while True:

#     imagem_desenho = imagem_teste.copy()
#     desenhar_retangulos(imagem_desenho, coords)

#     cv2.imshow("imagem", imagem_teste)
#     cv2.imshow("imagem desenho", imagem_desenho)


#     if cv2.waitKey(0) == ord("q"):
#         break

# cv2.destroyAllWindows()