# Integrantes:
# • Bruno Geisler Vigentas
# • Francisco Lucas Sens
# • Gustavo Westarb
# • William Lopes da Silva

import utils
import numpy as np
import matplotlib.pyplot as plt

class DemoD1:

    def __init__(self):
        
        self.dados = utils.ler_dados_mat_grupo_01('grupoDados1.mat')
        self.distancias = utils.distancia_euclidiana(self.dados)

        self.distancias_ordenadas = utils.ordena_distancias(self.distancias)

        self.rotulo_train_k1 = utils.definir_rotulo(self.distancias_ordenadas, self.distancias, self.dados.trainRots, 1)
        self.acuracia_k1 = utils.definir_acuracia(self.rotulo_train_k1, self.dados.testRots)

        self.rotulo_train_k10 = utils.definir_rotulo(self.distancias_ordenadas, self.distancias, self.dados.trainRots, 10)
        self.acuracia_k10 = utils.definir_acuracia(self.rotulo_train_k10, self.dados.testRots)

        self.rotulo_train_k3 = utils.definir_rotulo(self.distancias_ordenadas, self.distancias, self.dados.trainRots, 3)
        self.acuracia_k3 = utils.definir_acuracia(self.rotulo_train_k3, self.dados.testRots)

        print(f'### Acurácia utilizando o K = 1: {self.acuracia_k1}')
        print(f'### Acurácia utilizando o K = 10: {self.acuracia_k10}')
        print(f'### Acurácia utilizando o K = 3: {self.acuracia_k3}, acurácia máxima alcançada')
        # self.exibir_grafico(self.dados, self.rotulo_train, '', '')


    def exibir_grafico(self, dados, rotulos, x, y):

        dados_1 = []
        dados_2 = []
        dados_3 = []

        for i in range(len(rotulos)):

            valor_rotulo = rotulos[i]

            if (valor_rotulo == 1 ):
                dados_1.append(dados.grupoTest[i])
            elif (valor_rotulo == 2):
                dados_2.append(dados.grupoTest[i])
            elif (valor_rotulo == 3):
                dados_3.append(dados.grupoTest[i])

        plt.plot(np.array(dados_1)[:,0], np.array(dados_1)[:,2], 'r^', dados_2, dados_2, 'b+', dados_3, dados_3, 'go')

        plt.show()


if __name__ == '__main__':
    demo1 = DemoD1()
    