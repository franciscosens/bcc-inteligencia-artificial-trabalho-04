# Integrantes:
# • Bruno Geisler Vigentas
# • Francisco Lucas Sens
# • Gustavo Westarb
# • William Lopes da Silva

import utils
import numpy as np
import matplotlib.pyplot as plt

class DemoD2:
    
    def __init__(self):
        
        self.dados = utils.ler_dados_mat_grupo_01('grupoDados2.mat')
        
        self.dados_normalizados_test = utils.realizar_normalizacao(self.dados.grupoTest)
        self.dados_normalizados_train = utils.realizar_normalizacao(self.dados.grupoTrain)

        self.distancias = utils.distancia_euclidiana(self.dados_normalizados_test, self.dados_normalizados_train)

        self.distancias_ordenadas = utils.ordena_distancias(self.distancias)

        self.rotulo_train_k1 = utils.definir_rotulo(self.distancias_ordenadas, self.distancias, self.dados.trainRots, 1)
        self.acuracia_k1 = '%.2f' % utils.definir_acuracia(self.rotulo_train_k1, self.dados.testRots)

        for i in range(100):
            if i > 0:
                self.rotulo_train_k7 = utils.definir_rotulo(self.distancias_ordenadas, self.distancias, self.dados.trainRots, i)
                self.acuracia_k7 = '%.2f' % utils.definir_acuracia(self.rotulo_train_k7, self.dados.testRots)
                print(f'### Acurácia utilizando o K = {i}: {self.acuracia_k7}')

        print(f'### Acurácia utilizando o K = 1: {self.acuracia_k1}')
        print('Ajustando o K para 7, conseguimos chegar a acurácia abaixo, pois passou a considerar 7 elementos mais próximos, assim aumentando nossa acurácia')
        print(f'### Acurácia utilizando o K = 7: {self.acuracia_k7}')

    
if __name__ == '__main__':
    demo2 = DemoD2()