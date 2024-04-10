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
start_process = time.time()

for img in range(51):
    start = time.time()

    print(f"Count: {img}")

    imagem = cv2.imread(os.path.join(dir_printscreens, f"{str(cont).zfill(2)}.jpg"))
    confiancas, coordenadas, centros, coords_ab, inclinacoes = detector_objetos.prever_cv2(imagem)
    nome_img_result = f"imagem_predicao_{str(len(os.listdir(dir_resultados)) + 1).zfill(2)}.jpg"
    path_img_result = os.path.join(dir_resultados, nome_img_result)


    CompiladorImagem.gerar_imagem_resultado_2(imagem, centros, coordenadas, path_img_result)

    mensagem = DetectorObjetos.gerar_msg(confiancas, centros, inclinacoes)

    print("",mensagem)
    end = time.time()
    print(f"Tempo de Compilacao: {((end - start)*1000):.3f} milisegundos")
    print('----------------------------------------------------------------------')
end_process = time.time()
print(f"Tempo Total de Processo: {((end_process - start_process)*1000):.3f}", end=" ")
if (end_process - start_process)*1000 >= 1000:
    print("segundos")
else:
    print("milisegundos")
    