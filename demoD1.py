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
        self.distancias = utils.distancia_euclidiana(self.dados.grupoTest, self.dados.grupoTrain)

        self.distancias_ordenadas = utils.ordena_distancias(self.distancias)

        self.rotulo_train_k1 = utils.definir_rotulo(self.distancias_ordenadas, self.distancias, self.dados.trainRots, 1)
        self.acuracia_k1 = utils.definir_acuracia(self.rotulo_train_k1, self.dados.testRots)

        self.rotulo_train_k10 = utils.definir_rotulo(self.distancias_ordenadas, self.distancias, self.dados.trainRots, 10)
        self.acuracia_k10 = '%.2f' % utils.definir_acuracia(self.rotulo_train_k10, self.dados.testRots)

        self.rotulo_train_k3 = utils.definir_rotulo(self.distancias_ordenadas, self.distancias, self.dados.trainRots, 3)
        self.acuracia_k3 = '%.2f' % utils.definir_acuracia(self.rotulo_train_k3, self.dados.testRots)

        print(f'### Acurácia utilizando o K = 1: {self.acuracia_k1}')
        print(f'### Acurácia utilizando o K = 10: {self.acuracia_k10}')
        print(f'### Acurácia utilizando o K = 3: {self.acuracia_k3}, acurácia máxima alcançada')
        
        self.exibir_grafico(self.dados, self.rotulo_train_k3, '', '')


    def exibir_grafico(self, dados, rotulos, x, y):

        dados_1_x = []
        dados_2_x = []
        dados_3_x = []

        for i in range(len(rotulos)):

            valor_rotulo = rotulos[i]

            if (valor_rotulo == 1 ):
                dados_1_x.append(dados.grupoTest[i])
            elif (valor_rotulo == 2):
                dados_2_x.append(dados.grupoTest[i])
            elif (valor_rotulo == 3):
                dados_3_x.append(dados.grupoTest[i])

        dados_1_y = np.tile(np.repeat(1,4),(len(dados_1_x),1))
        dados_2_y = np.tile(np.repeat(2,4),(len(dados_2_x),1))
        dados_3_y = np.tile(np.repeat(3,4),(len(dados_3_x),1))

        plt.plot(dados_1_x, dados_1_x, 'r^')
        # plt.plot([4.7, 3.2, 1.3, 0.2], [1, 1, 1, 1],'r^')


        plt.show()


if __name__ == '__main__':
    demo1 = DemoD1()
    