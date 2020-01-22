
'''
Aluno: Gabriel de Souza Nogueira da Silva
Matricula: 398847
'''
import random 
import math
import re
from scipy import stats
import numpy as np

quant_neuronio_oculto = 10
vet_atributos = []#vetor de entradas
vet_respostas = []#vetor de saidas
one_out = 0 #qual amostra do vetor eu tou removendo
valor_tirado_att = np.ones((1, 4))#amostra retirada para teste
valor_tirado_resp = np.ones((1, 3))#resposta da amostra retirada para teste
cont = 0
x = 0
def normaliza(x):
    #estou separando cada atributo em um vetor diferente e dps os normalizo e crio o
    #vetor com todos os os atributoss normalizados na msm ordem
    vet = list()
    vet1 = list()
    vet2 = list()
    vet3 = list()
    vet4 = list()

    cont1 = 0
    cont2 = 1
    cont3 = 2
    cont4 = 3

    for i in range(0,len(x)):
        if cont1 == i:
            vet1.append(x[i])
            cont1 = cont1 + 4
        if cont2 == i:
            vet2.append(x[i])
            cont2 = cont2 + 4
        if cont3 == i:
            vet3.append(x[i])
            cont3 = cont3 + 4
        if cont4 == i:
            vet4.append(x[i])
            cont4 = cont4 + 4
  
    vet1 = norm(vet1)    
    vet2 = norm(vet2)    
    vet3 = norm(vet3)    
    vet4 = norm(vet4)

    for i  in range(0,len(vet1)):
        vet.append(vet1[i])
        vet.append(vet2[i])
        vet.append(vet3[i])
        vet.append(vet4[i])

    return vet    

def norm(x):
    '''
    #testes usando outros tipo de normalizaçao a que deu resultados melhores foi 
    #a x[i]/max(x)
    vet = list()
    for i in range(0, len(x)):
        #aux = 2*((x[i] - min(x))/max(x)-min(x)) - 1
        aux = x[i]/max(x)
        vet.append(aux)
    return vet'''
    return stats.zscore(x)

dados = open("iris_log.dat", "r")
for line in dados:
    #separando o que é x do que é d 
    line = line.strip()#quebra no \n
    line = re.sub('\s+',',',line)#trocando os espaços vazios por virgula   
    a1,a2,a3,a4,r1,r2,r3 = line.split(",")#quebra nas virgulas e retorna 2 valores
    vet_atributos.append(float(a1))
    vet_atributos.append(float(a2))
    vet_atributos.append(float(a3))
    vet_atributos.append(float(a4))
    vet_respostas.append(float(r1))
    vet_respostas.append(float(r2))
    vet_respostas.append(float(r3))
dados.close()

vet_atributos = normaliza(vet_atributos)

def cria_mat_atributos(vet_atributos):
    #crio a matriz de atributos retirando umas das amostras da base de dados 
    #para ser o teste o valor do teste é salvo na variavel valor_tirado_att 
    # e é retornado o uma matriz com todas as outras amostras

    global one_out
    k=0
    vet = np.ones((int(len(vet_atributos)/4), 4))

    vet_1 = np.ones((int(len(vet_atributos)/4)-1, 4))

    for i in range(0, int(len(vet_atributos)/4)):
        for j in range(0,4):
            vet[i][j] = vet_atributos[k]
            k+=1

    for q in range(0,4):
           valor_tirado_att[0][q] =  vet[one_out][q]

    aux = list()
    for i in range(0, int(len(vet_atributos)/4)):
        for j in range(0,4):
            if i != one_out:
                aux.append(vet[i][j])
    k = 0
    for i in range(0, int(len(aux)/4)):
        for j in range(0,4):
            vet_1[i][j] = aux[k]
            k+=1
    
    return vet_1

def cria_mat_atributos_peso_bias(mat_atributos):
    #adiciona uma linha de 1 para o calculo de W*X
    vet = np.ones((5,149))
    aux = np.transpose(mat_atributos)

    for i in range(1, 5):
        for j in range(0,149):
            vet[i][j]= aux[i-1][j]   
    return vet

def cria_mat_atributos_peso_bias_teste(mat_atributos):
    #adiciona uma linha de 1 para o calculo de W*X

    vet = np.ones((5,1))
    aux = np.transpose(mat_atributos)

    for i in range(1, 5):
        vet[i][0]= aux[i-1][0]   
    return vet

def cria_mat_resposta(vet_resposta):
    #crio a matriz de respostas retirando uma das amostras da base de dados 
    #para ser a resposta do valor do teste é salvo na variavel valor_tirado_resp 
    # e é retornado o uma matriz com todas as outras respostas das amostras
    
    global one_out
    k=0
    vet = np.ones((int(len(vet_resposta)/3), 3))

    vet_1 = np.ones((int(len(vet_resposta)/3)-1, 3))

    for i in range(0, int(len(vet_resposta)/3)):
        for j in range(0,3):
            vet[i][j]= vet_resposta[k]
            k+=1

    for q in range(0,3):
           valor_tirado_resp[0][q] =  vet[one_out][q]
    

    aux = list()
    for i in range(0, int(len(vet_resposta)/3)):
        for j in range(0,3):
            if i != one_out:
                aux.append(vet[i][j])
    k = 0
    for i in range(0, int(len(aux)/3)):
        for j in range(0,3):
            vet_1[i][j] = aux[k]
            k+=1
    #incrementando para na prox iteraçao retirar outra amostra da minha base
    one_out += 1        
    return vet_1

def cria_mat_w():
    vet = np.ones((quant_neuronio_oculto, 5))
    for i in range(0, quant_neuronio_oculto):
        for j in range(0,5):
            #valores aleatorios em uma distribuiçao normal
            vet[i][j]= random.normalvariate(0,0.1)
    return vet

def cria_mat_z(u):
    for i in range(0, quant_neuronio_oculto):
        for j in range(0, 149):
            #aplicando funçao de ativaçao
            u[i][j] = 1 / (1 + math.exp((-1)*u[i][j]))
            #u[i][j] =  (1 - math.exp((-1)*u[i][j])) / (1 + math.exp((-1)*u[i][j]))
    return u

def cria_mat_z_teste(u):
    for i in range(0, quant_neuronio_oculto):
        #aplicando funçao de ativaçao
        #u[i][0] =  (1 - math.exp((-1)*u[i][0])) / (1 + math.exp((-1)*u[i][0]))
        u[i][0] = 1 / (1 + math.exp((-1)*u[i][0]))
    return u

def cria_vetor_z_linha(z):
    vet = np.ones((len(z)+1, 149))
    for i in range(1, len(z)+1):
        for j in range(0, 149):
            vet[i][j] = z[i-1][j]
    return vet 

def cria_vetor_z_linha_teste(z):
    vet = np.ones((len(z)+1, 1))
    for i in range(1, len(z)+1):
        vet[i][0] = z[i-1][0]
    return vet 

while x < 150:
    atributo = cria_mat_atributos(vet_atributos)
   
    respostas = cria_mat_resposta(vet_respostas)
    
    W = cria_mat_w()
    
    atributos_com_peso_bias = cria_mat_atributos_peso_bias(atributo)
  
    U = np.dot(W,atributos_com_peso_bias)
   
    Z = cria_mat_z(U)
    #vetor Z com o bias
    Z_linha = cria_vetor_z_linha(Z)

    #matriz treinada
    M = np.dot(np.dot(np.transpose(respostas),np.transpose(Z_linha)),np.linalg.inv(np.dot(Z_linha,np.transpose(Z_linha))))
    print(np.shape(M))
    #TESTANDO O VALOR RETIRADO

    atributos_com_peso_bias_teste = cria_mat_atributos_peso_bias_teste(valor_tirado_att)
    
    U_teste = np.dot(W,atributos_com_peso_bias_teste)
    
    Z_teste = cria_mat_z_teste(U_teste)
    
    Z_linha_teste = cria_vetor_z_linha_teste(Z_teste)
    
    #resposta da minha rede
    a_teste = np.dot(M, Z_linha_teste)

    #Verificando se acertou ou errou a classe da amostra q foi retirada

    max_t = max(a_teste)

    for i in range(0,3):
        if a_teste[i][0] == max_t:
            indice_max_t = i     
            break

    max_r = max(np.transpose(valor_tirado_resp))
    
    for i in range(0,3):
        aux = valor_tirado_resp[0][i]
        if aux == max_r:
            indice_max_r = i
            break

    if indice_max_r == indice_max_t:
        #print("Acertou!")
        cont += 1 

    else:
        pass
        #print("Errou!")
    break         
    x += 1

print("Acuracia " + str((cont/150)*100) + "% " +"Quant. de amostras acertadas: " +str(cont))    