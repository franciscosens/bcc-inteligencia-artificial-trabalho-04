# Integrantes:
# • Bruno Geisler Vigentas
# • Francisco Lucas Sens
# • Gustavo Westarb
# • William Lopes da Silva

import utils
import numpy as np
import math

class DemoD1:

    def __init__(self):
        
        self.dados = utils.ler_dados_mat_grupo_01()
        self.distancias = []

        self.distancia_euclidiana()

        self.distancias_ordenadas = np.argsort(self.distancias)
        print(self.distancias)
        # print(self.dados)


    def distancia_euclidiana(self):

        np_train = np.array(self.dados.grupoTrain)
        np_test = np.array(self.dados.grupoTest)

        for i_test in range(len(np_test)):

            linha_distancia = []

            for i_train in range(len(np_train)):

                # soma_test_train = np.sqrt(np.sum(np.power(np_test[i_test] - np_train[i_train], 2)))
                subtracao_test_train = np_test[i_test] - np_train[i_train]
                potencia_test_train = np.power(subtracao_test_train, 2)
                soma_test_train = np.sum(potencia_test_train)
                raiz_test_train = np.sqrt(soma_test_train)

                linha_distancia.append(raiz_test_train)
            
            self.distancias.append(linha_distancia)


if __name__ == '__main__':
    demo1 = DemoD1()
    