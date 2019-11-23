# Integrantes:
# • Bruno Geisler Vigentas
# • Francisco Lucas Sens
# • Gustavo Westarb
# • William Lopes da Silva

from collections import namedtuple
import scipy.io as scipy
import os

def distancia_euclidiana():
    pass


def ler_dados_mat_grupo_01():
    diretorio_corrente = os.getcwd()
    mat = scipy.loadmat(f'{diretorio_corrente}\/assets\Dados\/grupoDados1.mat')

    __Retorno = namedtuple('Retorno', ['grupoTest', 'grupoTrain', 'testRots', 'trainRots'])
    
    return __Retorno(mat['grupoTest'], mat['grupoTrain'], mat['testRots'], mat['trainRots'])




