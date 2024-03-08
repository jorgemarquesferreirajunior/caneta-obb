from utils import *

dir_home = os.getcwd()
# CAMINHO_CONJUNTO_DADOS = UtilitariosArquivo.descompactar_arquivo(os.path.join(HOME, "caneta-obb.zip"), HOME)
dir_dataset = os.path.join(dir_home, 'datasets')
dir_printscreens = os.path.join(dir_home, 'printscreens')
dir_resultados = os.path.join(dir_home, 'resultados')
path_modelo = os.path.join(dir_dataset, 'runs', 'obb', 'train', 'weights/best.pt')

detector_objetos = DetectorObjetos(path_modelo)

# ip_camera = "http://192.168.0.103:8080/video"
# imagem = CapturaCamera.tirar_foto_cv2(ip_camera)


imagem = cv2.imread(os.path.join(dir_printscreens, '00.jpg'))
confiancas, coordenadas, centros, coords_ab, inclinacoes = detector_objetos.prever_cv2(imagem)
nome_img_result = f"imagem_predicao_{str(len(os.listdir(dir_resultados)) + 1).zfill(2)}.jpg"
path_img_result = os.path.join(dir_resultados, nome_img_result)
CompiladorImagem.gerar_imagem_resultado_cv2(imagem, centros, coordenadas, inclinacoes, path_img_result)

mensagem = DetectorObjetos.gerar_msg(confiancas, centros, inclinacoes)
print(mensagem)

# caminho_resultados = UtilitariosArquivo.criar_pasta('resultados', HOME)
# DIRdataset = os.path.join(HOME, 'datasets')
# DIRmodelo = os.path.join(DIRdataset, 'runs/obb/train/weights/best.pt')
# detector_objetos = DetectorObjetos(DIRmodelo)
# print(f"CAMINHO_CONJUNTO_DADOS: {CAMINHO_CONJUNTO_DADOS}")
# print(f"caminho_resultados: {caminho_resultados}")
# print(f"DIRdataset: {DIRdataset}")
# print(f"DIRmodelo: {DIRmodelo}")
#----------------------------------------------------------------------------------------------------------------------------------------
# prever um pacote de imagens de uma base de dados

# IMAGENS_TESTE = UtilitariosArquivo.listar_imagens_pasta(os.path.join(DIRdataset, 'test', 'images'))
# lista_confiancas, lista_coordenadas, lista_centros, lista_coords_ab, lista_inclinacoes = detector_objetos.prever_lista(IMAGENS_TESTE, detector_objetos, caminho_resultados, limiar_acuracia=0.81, limpar_pasta_resultados=True)
#----------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------------------
#  prever uma imagem de uma captura

# ip_camera = "http://192.168.0.103:8080/video"
# caminho_capturas = UtilitariosArquivo.criar_pasta("capturas", HOME)
# imagem = CapturaCamera.tirar_foto(ip_camera, caminho_capturas)
# confiancas, coordenadas, centros, coords_ab, inclinacoes = detector_objetos.prever(imagem)
# caminho_imagem_resultado = os.path.join(caminho_resultados, f"imagem_compilada_{len(os.listdir(caminho_resultados)) + 1}.jpg")
# CompiladorImagem.gerar_imagem_resultado(cv2.imread(imagem), centros, coordenadas, inclinacoes, caminho_imagem_resultado)
#----------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------------------
#  prever uma imagem de uma base de dados
# imagem = "/home/marks/codes/YOLO/refil-poo/datasets/test/images/37_jpg.rf.a4c23f3c139973b93654f341314c1a44.jpg"
# confiancas, coordenadas, centros, coords_ab, inclinacoes = detector_objetos.prever(imagem, limiar_acuracia=0.81)
# caminho_imagem_resultado = os.path.join(caminho_resultados, f"imagem_compilada_{len(os.listdir(caminho_resultados)) + 1}.jpg")
# CompiladorImagem.gerar_imagem_resultado(cv2.imread(imagem), centros, coordenadas, inclinacoes, caminho_imagem_resultado)

# mensagem = DetectorObjetos.gerar_msg(confiancas, centros, inclinacoes)
# print(mensagem)
#----------------------------------------------------------------------------------------------------------------------------------------

# UtilitariosArquivo.esvaziar_pasta(caminho_resultados)
# UtilitariosArquivo.esvaziar_pasta(caminho_capturas)