# Integrantes:
# • Bruno Geisler Vigentas
# • Francisco Lucas Sens
# • Gustavo Westarb
# • William Lopes da Silva

import utils
import numpy as np

class DemoD1:


    def __init__(self):
        
        #Obtém os dados
        self.dados = utils.ler_dados_mat('grupoDados1.mat')

        #Calcula as disntancias
        self.distancias = utils.distancia_euclidiana(self.dados.grupoTest, self.dados.grupoTrain)

        #Ordena as distancias encontradas
        self.distancias_ordenadas = utils.ordenar_distancias(self.distancias)

        #Rotula os dados ordenados
        self.rotulo_train_k1 = utils.definir_rotulo(self.distancias_ordenadas, self.dados.trainRots, 1)
        self.acuracia_k1 = utils.definir_acuracia(self.rotulo_train_k1, self.dados.testRots)

        #Rotula os dados ordenados
        self.rotulo_train_k10 = utils.definir_rotulo(self.distancias_ordenadas, self.dados.trainRots, 10)
        self.acuracia_k10 = '%.2f' % utils.definir_acuracia(self.rotulo_train_k10, self.dados.testRots)

        #Rotula os dados ordenados
        # K = 3 -> Foi escolhido pois com ele conseguimos chegar aos 92% desejado, descoberto após fazer o teste de 1 até 100
        self.rotulo_train_k3 = utils.definir_rotulo(self.distancias_ordenadas, self.dados.trainRots, 3)
        self.acuracia_k3 = '%.2f' % utils.definir_acuracia(self.rotulo_train_k3, self.dados.testRots)

        # Printa os resutlados
        print(f'### Acurácia utilizando o K = 1: {self.acuracia_k1}')
        print(f'### Acurácia utilizando o K = 10: {self.acuracia_k10}')
        print(f'### Acurácia utilizando o K = 3: {self.acuracia_k3}, acurácia máxima alcançada')
        
        # Exibe os gráficos com os rotulos pré existentes e os rotulos descobertos pelo nosso algoritmo
        utils.exibir_grafico(self.dados.grupoTest, self.dados.testRots, 0, 1)
        # Exibe os gráficos com os rotulos com K = 1
        utils.exibir_grafico(self.dados.grupoTest, self.rotulo_train_k1, 0, 1)
        # Exibe os gráficos com os rotulos com K = 10
        utils.exibir_grafico(self.dados.grupoTest, self.rotulo_train_k10, 0, 1)
        # Exibe os gráficos com os rotulos com K = 3
        utils.exibir_grafico(self.dados.grupoTest, self.rotulo_train_k3, 0, 1)


if __name__ == '__main__':
    demo1 = DemoD1()
    