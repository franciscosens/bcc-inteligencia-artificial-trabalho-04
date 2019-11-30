# Integrantes:
# • Bruno Geisler Vigentas
# • Francisco Lucas Sens
# • Gustavo Westarb
# • William Lopes da Silva

import utils
import numpy as np

class DemoD3:
    
    
    def __init__(self):
        
        #Obtém os dados
        self.dados = utils.ler_dados_mat('grupoDados3.mat')
        
        #Calcula as disntancias
        self.distancias = utils.distancia_euclidiana(self.dados.grupoTest, self.dados.grupoTrain)

        #Ordena as distancias encontradas
        self.distancias_ordenadas = utils.ordenar_distancias(self.distancias)

        #Rotula os dados ordenados
        self.rotulo_train_k1 = utils.definir_rotulo(self.distancias_ordenadas, self.dados.trainRots, 1)
        #Define a acuracia com os dados rotulados e o grupo de teste do data set
        self.acuracia_k1 = utils.definir_acuracia(self.rotulo_train_k1, self.dados.testRots)
    
        #Rotula os dados ordenados
        # K = 7 -> Foi escolhido pois com ele conseguimos chegar aos 92% desejado, descoberto após fazer o teste de 1 até 100
        self.rotulo_train_k7 = utils.definir_rotulo(self.distancias_ordenadas, self.dados.trainRots, 7)
        #Define a acuracia com os dados rotulados e o grupo de teste do data set
        self.acuracia_k7 = utils.definir_acuracia(self.rotulo_train_k7, self.dados.testRots)

        # Printa os resutlados
        print(f'### Acurácia utilizando o K = 1: {self.acuracia_k1}')
        print('Ajustando o K para 7, conseguimos chegar a acurácia abaixo, pois passou a considerar 7 elementos mais próximos, assim aumentando nossa acurácia')
        print(f'### Acurácia utilizando o K = 7: {self.acuracia_k7}')



        # Armazena dados em array para mandar para função que criar o gráfico do dataset, k = 7 e k = 1
        self.rotulos = [self.dados.testRots, self.rotulo_train_k7, self.rotulo_train_k1]
        self.title = ["Dataset", "K = 7", "K = 1"]

        # Gera gráfico
        utils.exibir_grafico_multiplo(self.dados.grupoTest, self.rotulos, 0, 1, self.title, 'Demo D3')

        
if __name__ == '__main__':
    demo3 = DemoD3()
    