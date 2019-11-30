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
import matplotlib.pyplot as plt


def distancia_euclidiana(dados_test, dados_train):

    matriz_aux = []
    # Separa os dados de treinamento e teste
    np_train = np.array(dados_train)
    np_test = np.array(dados_test)

    for i_test in range(len(np_test)):

        linha_distancia = []

        for i_train in range(len(np_train)):

            subtracao_test_train = np_test[i_test] - np_train[i_train]
            potencia_test_train = np.power(subtracao_test_train, 2)
            soma_test_train = np.sum(potencia_test_train)
            raiz_test_train = np.sqrt(soma_test_train)

            # Após calcular a distancia de cada elemento, adiciona para uma nova matriz
            linha_distancia.append(raiz_test_train)
        
        matriz_aux.append(linha_distancia)

    # Retorna a matriz com as distancias calculadas
    return matriz_aux


def ler_dados_mat(grupoDados):

    # Lê os dados do dataset passado pela professora
    diretorio_corrente = os.getcwd()
    mat = scipy.loadmat(f'{diretorio_corrente}\/assets\Dados\/{grupoDados}')

    # Define uma tupla com cada grupo de dado do dataset
    __Retorno = namedtuple('Retorno', ['grupoTest', 'grupoTrain', 'testRots', 'trainRots'])
    
    # Retorna os dados separados em seus grupos
    return __Retorno(mat['grupoTest'], mat['grupoTrain'], mat['testRots'], mat['trainRots'])


def ordenar_distancias(distancias):
    #Ordena a matriz de distancia, nessa função é retornada uma matriz, onde seu valor são os indices dos dados ordenados da matriz de entrada 
    return np.argsort(distancias)


def definir_rotulo(ordem_matriz, trainRots, k):
    
    #Obtem os elemento com a menor distancia de acordo com o valor de k
    # K = 1 -> Vai trazer o primeiro elemento mais próximo
    # K = 10 -> Vai trzar os 10 elementos mais próximos
    menor_distancia = ordem_matriz[:, np.arange(k)]
    rotulos_aux = []

    for i in range(len(menor_distancia)):
        # Obtém o rotulo de acordo com o K escolhido
        rotulos_aux.append(stats.mode(trainRots[menor_distancia[i][0:k]]).mode[0])

    # Retorna o rotulo obtido dos indices mais próximos
    return rotulos_aux


def definir_acuracia(rotulo_train, testRots):
    
    # Calcula a acuracia, batendo o rotulo descoberto com o rotulo de treinamento
    rotulos_corretos = rotulo_train == testRots
    soma_corretos = np.sum(rotulos_corretos)
    acuracia = soma_corretos / len(testRots)

    return acuracia


def realizar_normalizacao(matriz_dados):

    # Cria matriz com o shape que desejo ao final do calculo da distancia
    matriz_aux = np.ndarray(shape=matriz_dados.shape)

    # Normaliza o dados para cada coluna "caracteristica" da matriz de entrada
    for i in range(matriz_dados.shape[1]):
        parte_cima = matriz_dados[:,i] - np.min(matriz_dados[:, i])
        parte_baixo = np.max(matriz_dados[:, i]) - np.min(matriz_dados[:, i])
        matriz_aux[:,i] = parte_cima / parte_baixo

    return matriz_aux
        

def getDadosRotulo(dados, rotulos, rotulo, indice):
    
    ret = []

    for idx in range(0, len(dados)):

        if(rotulos[idx] == rotulo):

            ret.append(dados[idx][indice])        

    return ret


def exibir_grafico(dados, rotulos, d1, d2, title = '', figure = 1):

    fig, ax = plt.subplots() 
    plt.figure(figure)
    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), getDadosRotulo(dados, rotulos, 1, d2), c='red' , marker='^')
    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), getDadosRotulo(dados, rotulos, 2, d2), c='blue' , marker='+')
    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), getDadosRotulo(dados, rotulos, 3, d2), c='green', marker='.')
    ax.set_title(title)

    plt.show()

def exibir_grafico_multiplo(dados, rotulos, d1, d2, title, main_title = '', height = 6, width = 16):


    fig, ax = plt.subplots(ncols=len(rotulos), nrows=1, constrained_layout=True)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    fig.suptitle(main_title, fontsize=16)

    for i in range(len(rotulos)):
        ax[i].scatter(getDadosRotulo(dados, rotulos[i], 1, d1), getDadosRotulo(dados, rotulos[i], 1, d2), c='red' , marker='^')
        ax[i].scatter(getDadosRotulo(dados, rotulos[i], 2, d1), getDadosRotulo(dados, rotulos[i], 2, d2), c='blue' , marker='+')
        ax[i].scatter(getDadosRotulo(dados, rotulos[i], 3, d1), getDadosRotulo(dados, rotulos[i], 3, d2), c='green', marker='.')
        ax[i].set_title(title[i])

    plt.show()