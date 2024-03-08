import cv2
import os
from ultralytics import YOLO
import numpy as np
import math
import shutil
from zipfile import ZipFile


# ***************************************************************************************************
class CapturaCamera:
    @staticmethod
    def tirar_foto_jpg(ip_camera, destino, altura_imagem=640, rotacao=cv2.ROTATE_90_CLOCKWISE):
        captura = cv2.VideoCapture(ip_camera)

        if not captura.isOpened():
            print("Erro ao abrir a câmera.")
            exit()

        while True:
            sucesso, frame = captura.read()

            if not sucesso:
                print("Erro ao ler o frame.")
            else:
                largura, altura = frame.shape[1], frame.shape[0]
                proporcao_tela = largura / altura
                largura_imagem = int(proporcao_tela * altura_imagem)

                frame = cv2.resize(frame, (largura_imagem, altura_imagem))
                frame = cv2.rotate(frame, rotacao)
                frame = frame[:altura_imagem, :altura_imagem]
                item = len(os.listdir(destino)) + 1
                cv2.imwrite(os.path.join(destino, f"foto({str(item)}).jpg"), frame)
                print("Foto salva com sucesso!")
            return os.path.join(destino, f"foto({str(item)}).jpg")
    
    @staticmethod
    def tirar_foto_cv2(ip_camera, tamanho_img=640, rotacao=cv2.ROTATE_90_CLOCKWISE):
        captura = cv2.VideoCapture(ip_camera)

        if not captura.isOpened():
            print("Erro ao abrir a câmera.")
            exit()

        while True:
            sucesso, frame = captura.read()

            if not sucesso:
                print("Erro ao ler o frame.")
            else:
                largura, altura = frame.shape[1], frame.shape[0]
                proporcao_tela = largura / altura
                largura_imagem = int(proporcao_tela * tamanho_img)

                frame = cv2.resize(frame, (largura_imagem, tamanho_img))
                frame = cv2.rotate(frame, rotacao)
                frame = frame[:tamanho_img, :tamanho_img]
                return frame
                
        
        
        
# ***************************************************************************************************
class ProcessadorImagem:
    @staticmethod
    def marcar_centros(imagem, centros, raio=5, cor=(0, 0, 255)):
        for centro in centros:
            cv2.circle(imagem, centro, raio, cor, cv2.FILLED)

    @staticmethod
    def enumerar_centros(imagem, centros, cor=(0, 0, 255)):
        for idx, centro in enumerate(centros):
            cv2.putText(imagem, f"{str(idx+1)}", centro, cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

    @staticmethod
    def listar_inclinacoes(imagem, inclinacoes, cor=(0, 0, 255)):
        x_pos = 15
        y_pos = 25

        cor2 = (50, 50, 255)
        cv2.putText(imagem, f"Img | Angulo", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        y_pos += 35
        for idx, inclinacao in enumerate(inclinacoes):
            cv2.putText(imagem, f"  {str(idx+1)}   {str(inclinacao)}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, cor2, 2)
            y_pos += 30

    @staticmethod
    def marcar_caixas(imagem, coordenadas, cor=(0, 0, 255)):
        for pontos in coordenadas:
            p1, p2, p3, p4 = tuple(pontos[0]), tuple(pontos[1]), tuple(pontos[2]), tuple(pontos[3])
            cv2.line(imagem, p1, p2, cor, 1)
            cv2.line(imagem, p2, p3, cor, 1)
            cv2.line(imagem, p3, p4, cor, 1)
            cv2.line(imagem, p4, p1, cor, 1)

    @staticmethod
    def marcar_inclinacoes(imagem, inclinacoes, centros):
        for idx, inclinacao in enumerate(inclinacoes):
            cv2.putText(imagem, f"{str(inclinacao)}", (centros[idx][0] + 10, centros[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            
 
 # ***************************************************************************************************           
class CompiladorImagem:
    @staticmethod
    def gerar_imagem_resultado(imagem, centros, coordenadas, inclinacoes, resultado="resultado.jpg"):
        ProcessadorImagem.enumerar_centros(imagem, centros)
        ProcessadorImagem.marcar_caixas(imagem, coordenadas)
        ProcessadorImagem.listar_inclinacoes(imagem, inclinacoes)
        cv2.imwrite(resultado, imagem)
    
    @staticmethod
    def gerar_imagem_resultado_cv2(imagem, centros, coordenadas, inclinacoes, resultado="resultado.jpg"):
        ProcessadorImagem.enumerar_centros(imagem, centros)
        ProcessadorImagem.marcar_caixas(imagem, coordenadas)
        # ProcessadorImagem.listar_inclinacoes(imagem, inclinacoes)
        cv2.imwrite(resultado, imagem)
        
        
# ***************************************************************************************************
class DetectorObjetos:
    def __init__(self, caminho_modelo):
        self.modelo_yolo = YOLO(caminho_modelo)

    def prever_cv2(self, imagem, limiar_acuracia=0.82):
        
        resultados_obb = self._resultado_previsao(imagem)
        confiancas, coordenadas = self._obter_dados_objeto(resultados_obb, limiar_acuracia)
        centros = self._calcular_centros(coordenadas)
        coords_ab = self._inverter_eixo_y(coordenadas, imagem.shape[0])
        inclinacoes = self._calcular_inclinacoes(coords_ab)
        
        return confiancas, coordenadas, centros, coords_ab, inclinacoes
    
    def prever(self, caminho_imagem, limiar_acuracia=0.82):
        imagem = cv2.imread(caminho_imagem)
        resultados_obb = self._resultado_previsao(imagem)
        confiancas, coordenadas = self._obter_dados_objeto(resultados_obb, limiar_acuracia)
        centros = self._calcular_centros(coordenadas)
        coords_ab = self._inverter_eixo_y(coordenadas, imagem.shape[0])
        inclinacoes = self._calcular_inclinacoes(coords_ab)
        
        return confiancas, coordenadas, centros, coords_ab, inclinacoes

    def prever_lista(self, lista_imagens, detector, pasta_resultados, limiar_acuracia=0.85, limpar_pasta_resultados=False):
        if limpar_pasta_resultados:
            UtilitariosArquivo.esvaziar_pasta(pasta_resultados)
            
        confiancas = []
        coordenadas = []
        centros = []
        coords_ab = []
        inclinacoes = []
        
        for img in lista_imagens:
            confianca, coordenada, centro, coord_ab, inclinacao = detector.prever(img, limiar_acuracia=limiar_acuracia)
            confiancas.append(confianca)
            coordenadas.append(coordenada)
            centros.append(centro)
            coords_ab.append(coord_ab)
            inclinacoes.append(inclinacao)
            
            caminho_imagem_resultado = os.path.join(pasta_resultados, f"imagem_compilada_{len(os.listdir(pasta_resultados)) + 1}.jpg")
            CompiladorImagem.gerar_imagem_resultado(cv2.imread(img), centro, coordenada, inclinacao, caminho_imagem_resultado)
        print(f"Resultados salvis no caminho {pasta_resultados}")
        return confiancas, coordenadas, centros, coords_ab, inclinacoes
    
    def _resultado_previsao(self, imagem, ):
        return self.modelo_yolo.predict(source=imagem, save=False)[0].obb

    def _obter_dados_objeto(self, predicao, acuracia):
        confiancas_obb, coords_obb = predicao.conf, predicao.xyxyxyxy

        confiancas_np = self._tensor_para_array(confiancas_obb)
        coords_np = self._tensor_para_array(coords_obb)

        confiancas_float = self._converter_confiancas(confiancas_np)
        coords_int = self._converter_coordenadas(coords_np)

        confiancas_filtradas = []
        coords_filtradas = []

        for indice, valor in enumerate(confiancas_float):
            if valor >= acuracia:
                confiancas_filtradas.append(valor)
                coords_filtradas.append(coords_int[indice])

        return confiancas_filtradas, coords_filtradas

    def _tensor_para_array(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif hasattr(tensor, "numpy"):
            if "cuda" in str(tensor.device):
                tensor = tensor.cpu()
            return tensor.numpy()
        else:
            raise ValueError("Formato de tensor não reconhecido.")

    def _converter_coordenadas(self, lst):
        nova_lista = []
        for id_sub, sublist in enumerate(lst):
            sublist_int = []
            for id_subsub, subsublist in enumerate(sublist):
                subsublist_int = [int(valor) for valor in subsublist]
                sublist_int.append(subsublist_int)
            nova_lista.append(sublist_int)
        return nova_lista

    def _converter_confiancas(self, lst):
        return [round(float(valor), 2) for valor in lst]

    def _calcular_centros(self, coordenadas_int):
        centros = [self._calcular_ponto_medio(sublista[0], sublista[2]) for sublista in coordenadas_int]
        return centros

    def _inverter_eixo_y(self, coordenadas_int, altura_imagem):
        coordenadas = []
        for sublista in coordenadas_int:
            ponto_a = [sublista[0][0], altura_imagem - sublista[0][1]]
            ponto_b = [sublista[3][0], altura_imagem - sublista[3][1]]
            pontos = [ponto_a, ponto_b]
            coordenadas.append(pontos)
        return coordenadas

    def _calcular_inclinacoes(self, coords_ab):
        inclinacoes = [round(self._calcular_angulo(ponto_a, ponto_b), 2) for ponto_a, ponto_b in coords_ab]
        return inclinacoes

    def _calcular_ponto_medio(self, A, B):
        x1, y1 = A
        x2, y2 = B
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _calcular_angulo(self, ponto_a, ponto_b):
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
    
    def gerar_msg(confiancas, centros, inclinacoes):
        mensagem = ''
        for i, (confianca, inclinacao, centro) in enumerate(zip(confiancas, inclinacoes, centros)):
            mensagem += f"deteccao: {i+1} - confianca:{confianca * 100}%\n" + f"centro: {centro}\n" + f"inclinacao: {inclinacao}\n\n"
        return mensagem


# ***************************************************************************************************
class UtilitariosArquivo:
    @staticmethod
    def criar_pasta(nome, caminho, verbose=False):
        pasta = os.path.join(caminho, nome)
        if not os.path.exists(pasta):
            os.makedirs(pasta)
        else:
            if verbose:
                print(f"A pasta {nome} já existe.")
        return pasta

    @staticmethod
    def deletar_pasta(caminho_pasta):
        try:
            shutil.rmtree(caminho_pasta)
        except Exception as e:
            print(f"Erro ao excluir a pasta {caminho_pasta}: {e}")

    @staticmethod
    def descompactar_arquivo(arquivo_zip, destino, verbose=False):
        destino_zip = os.path.join(destino, "datasets")
        print("Iniciando a descompactacao...")
        if os.path.exists(destino_zip) and verbose:
            print(f"A pasta já existe neste destino: {destino_zip}")
        else:
            try:
                with ZipFile(arquivo_zip, "r") as arquivo_zipado:
                    arquivo_zipado.extractall(destino_zip)
                if verbose:
                    print(f"A pasta {arquivo_zip} foi descompactada com sucesso para {destino_zip}.")
            except Exception as e:
                print(f"Erro ao extrair {arquivo_zip}: {e}")
        return destino_zip

    @staticmethod
    def compactar_pasta(origem, destino, nome_arquivo):
        try:
            destino_zip = os.path.join(destino, nome_arquivo)
            shutil.make_archive(destino_zip, 'zip', origem)
            print(f"A pasta {origem} \n foi compactada com sucesso para {destino_zip}.")
        except Exception as e:
            print(f"Erro ao compactar {origem}: {e}")

    @staticmethod
    def procurar_pasta(caminho_raiz, pasta_alvo):
        caminhos = []
        for pasta_atual, subpastas, arquivos in os.walk(caminho_raiz):
            if pasta_alvo in subpastas:
                caminhos.append(os.path.join(pasta_atual, pasta_alvo))
        return caminhos if caminhos else None

    @staticmethod
    def procurar_arquivo(caminho_raiz, nome_arquivo, verbose=False):
        for pasta_atual, subpastas, arquivos in os.walk(caminho_raiz):
            if nome_arquivo in arquivos:
                caminho_arquivo = os.path.join(pasta_atual, nome_arquivo)
                if verbose:
                    print(f"O arquivo {nome_arquivo} foi encontrado em: {caminho_arquivo}")
                return caminho_arquivo

        print(f"O arquivo {nome_arquivo} não foi encontrado em {caminho_raiz} ou suas subpastas.")
        return None

    @staticmethod
    def esvaziar_pasta(caminho_pasta):
        dados = os.listdir(caminho_pasta)
        if len(dados) > 0:
            print(f"Encontrados {len(dados)} itens.")
            for item in dados:
                caminho_item = os.path.join(caminho_pasta, item)
                if os.path.isfile(caminho_item):
                    os.remove(caminho_item)
            print("Pasta esvaziada com sucesso!")
        else:
            print("A pasta está vazia!")
            
    @staticmethod
    def listar_imagens_pasta(caminho_pasta):
        return [os.path.join(caminho_pasta, f) for f in os.listdir(caminho_pasta) if f.endswith((".jpg", ".png"))]
    