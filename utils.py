# Integrantes:
# • Bruno Geisler Vigentas
# • Francisco Lucas Sens
# • Gustavo Westarb
# • William Lopes da Silva

from collections import namedtuple
import scipy.io as scipy
import os
import numpy as np
from scipy import stats


def distancia_euclidiana(dados):

    matriz_aux = []
    np_train = np.array(dados.grupoTrain)
    np_test = np.array(dados.grupoTest)

    for i_test in range(len(np_test)):

        linha_distancia = []

        for i_train in range(len(np_train)):

            # soma_test_train = np.sqrt(np.sum(np.power(np_test[i_test] - np_train[i_train], 2)))
            subtracao_test_train = np_test[i_test] - np_train[i_train]
            potencia_test_train = np.power(subtracao_test_train, 2)
            soma_test_train = np.sum(potencia_test_train)
            raiz_test_train = np.sqrt(soma_test_train)

            linha_distancia.append(raiz_test_train)
        
        matriz_aux.append(linha_distancia)

    return matriz_aux


def ler_dados_mat_grupo_01(grupoDados):

    diretorio_corrente = os.getcwd()
    mat = scipy.loadmat(f'{diretorio_corrente}\/assets\Dados\/{grupoDados}')

    __Retorno = namedtuple('Retorno', ['grupoTest', 'grupoTrain', 'testRots', 'trainRots'])
    
    return __Retorno(mat['grupoTest'], mat['grupoTrain'], mat['testRots'], mat['trainRots'])


def ordena_distancias(distancias):
    return np.argsort(distancias)


def definir_rotulo(ordem_matriz, matriz_distancia, trainRots, k):
    
    menor_distancia = ordem_matriz[:, np.arange(k)]
    rotulos_aux = []

    for i in range(len(menor_distancia)):
        rotulos_aux.append(stats.mode(trainRots[menor_distancia[i][0:k]]).mode[0])

    return rotulos_aux


def definir_acuracia(rotulo_train, testRots):
    
    rotulos_corretos = rotulo_train == testRots
    soma_corretos = np.sum(rotulos_corretos)
    acuracia = soma_corretos / len(testRots)

    return acuracia
