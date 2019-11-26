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


def distancia_euclidiana(dados_test, dados_train):

    matriz_aux = []
    np_train = np.array(dados_train)
    np_test = np.array(dados_test)

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


def realizar_normalizacao(matriz_dados):

    # for i in range(len(matriz_dados)):

    matriz_aux = []

    for i in range(len(matriz_dados)):
        parte_cima = matriz_dados[i,:] - np.max(matriz_dados[i,:])
        parte_baixo = np.max(matriz_dados[i,:]) - np.min(matriz_dados[i,:])
        matriz_aux.append(np.absolute(np.array(parte_cima / parte_baixo)))

    # for i in range(len(novas_colunas)):
    #     teste = novas_colunas[i]
    #     np.insert(matriz_aux, [1], teste, axis=0)
    	
        # matriz_aux[i] = np.array(coluna_aux)

    # matriz_aux = np.append(matriz_aux, np.array(novas_colunas).reshape(60,13), axis=1)

    return np.array(matriz_aux)
        

