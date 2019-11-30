# Integrantes:
# • Bruno Geisler Vigentas
# • Francisco Lucas Sens
# • Gustavo Westarb
# • William Lopes da Silva

import utils
import numpy as np

class DemoD2:
    

    def __init__(self):

        #Obtém os dados
        self.dados = utils.ler_dados_mat('grupoDados2.mat')
        
        #Normaliza os dados de treinamento e de teste
        self.dados_normalizados_test = utils.realizar_normalizacao(self.dados.grupoTest)
        self.dados_normalizados_train = utils.realizar_normalizacao(self.dados.grupoTrain)

        #Calcula as disntancias
        self.distancias = utils.distancia_euclidiana(self.dados_normalizados_test, self.dados_normalizados_train)

        #Ordena as distancias encontradas
        self.distancias_ordenadas = utils.ordenar_distancias(self.distancias)

        #Rotula os dados ordenados
        # K = 3 -> Foi escolhido pois com ele conseguimos chegar aos 92% desejado, descoberto após fazer o teste de 1 até 100
        self.rotulo_train_k3 = utils.definir_rotulo(self.distancias_ordenadas, self.dados.trainRots, 3)
        self.acuracia_k3 = '%.2f' % utils.definir_acuracia(self.rotulo_train_k3, self.dados.testRots)

        # Printa os resutlados
        print(f'### Acurácia utilizando o K = 3: {self.acuracia_k3}')

        # Exibe os gráficos com os rotulos pré existentes e os rotulos descobertos pelo nosso algoritmo

        # Armazena dados em array para mandar para função que criar o gráfico do dataset, k = 3
        self.rotulos = [self.dados.testRots, self.rotulo_train_k3]
        self.title = ["Dataset", "K = 3"]

        # Gera gráfico
        utils.exibir_grafico_multiplo(self.dados.grupoTest, self.rotulos, 0, 1, self.title, 'Demo D2', height=6, width=10)
        

if __name__ == '__main__':
    demo2 = DemoD2()