import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(0)

#OBTENEMOS LOS PRIMEROS CENTROIDES
def centroids(k,data):
    #NUESTRA LISTA DE CENTROIDES, TENDRA 3 COLS(RGB) Y k FILAS
    listCentroids=np.zeros((k,3))
    #ESCOGEMOS VALORES ALEATORIOS DE NUESTROS PIXELES
    for i in range(k):
        cenR=random.choice(data[0])
        cenG=random.choice(data[1])
        cenB=random.choice(data[2])
        listCentroids[i]=[cenR,cenG,cenB]

    return listCentroids

#CALCULAMOS LA DISTANCIA, DE NUESTRO CENTROIDE AL PIXEL
def calDistance(a,b):
    distance=(np.sum((a-b)**2))
    return distance

#DEFINIMOS NUEVOS CENTROIDES
def newCen(valMin,rgb,k, centroids):
    #NUESTRA LISTA DE CENTROIDES, TENDRA 3 COLS(RGB) Y k FILAS
    listCentroids=np.zeros((k,3))
    for i in range(k):
        #SI HAY ALGUN CENTROIDE QUE NO SE LE HAYA ASIGNADO PIXELES
        #SE QUEDA CON LOS CENTROIDES ANTERIORES
        lab = valMin==i
        if np.all(lab==False):
            listCentroids[i] = centroids[i]
        #EN OTRO CASO, SE SACA LA MEDIA DE LOS PIXELES DE CADA CLUSTER
        else:
            listCentroids[i]=np.mean(rgb[valMin==i],axis=0)
    return listCentroids

#ASIGNAMOS LOS CLUSTER FINALES
def asigCluster(valMin,centroids,rgb,k):
    newImage=np.zeros((rgb.shape))
    #PARA CADA PIXEL
    for i in range(rgb.shape[0]):
        #OBTENEMOS A QUE CLUSTER PERTENECE (aux)
        aux=valMin[i]
        #RECORRIENDO CADA CLUSTER
        for j in range(k):
            #SI EL PIXEL DE LA IMAGEN (aux) ES IGUAL AL CLUSTER (j)
            if aux==j:
                #A NUESTRA NUEVA IMAGEN LE ASIGNAMOS EL VALOR DEL CENTROIDE CORRESPONDIENTE
                newImage[i]=centroids[j]

    return newImage

#ALGORITMO K-MEANS
def kMeans(imagen,k):
    print("K-MEANS")
    #REDIMENSIONAMOS NUESTRA IMAGEN A N FILAS x 3 COLUMNAS (RGB)
    rgb=imagen.reshape(-1,3)

    #OBTENEMOS LOS PRIMEROS CENTORIDES
    centros=centroids(k,rgb)
    #CREAMOS NUESTRO ARREGLO DE DISTANCIAS, CADA PIXEL VA A TENR UNA DISTANCIA A CADA CLUSTER
    distances=np.zeros((rgb.shape[0],k))


    #DEFINIMOS EL NUMERO DE ITERACIONES
    iterations=100
    #CONTADOR PARA ITERACIONES
    iter=1

    for i in range(iterations):
        for i in range(distances.shape[0]):
            for j in range(k):
                #CALCULAMOS LAS DISTANCIAS DE CADA PIXEL A CADA CENTRO
                distance=calDistance(centros[j],rgb[i])
                distances[i,j]=distance

        #OBTENEMOS UN ARRGELO CON LOS INDICES DE A QUE CLUSTER PERTENECE CADA PIXEL (LA MENOR DISTANCIA)
        #(ARREGLO DE N PIXELES, 1 COL)
        valMin=np.argmin(distances,axis=1)

        #OBTENEMOS NUEVOS CENTROIDES
        newCentroids=newCen(valMin,rgb,k,centros)
        #SI EN DOS ITERACIONES SEGUIDAS NOS DAN LOS MISMOS CENTORIDES, ROMPE LAS ITERACIONES
        if np.all(np.round(newCentroids)==np.round(centros)):
            break 
        #AHORA centros VAN A SER IGUAL A NUESTROS newCentroids
        centros=newCentroids
        print("iter",iter)
        iter+=1
        
    print("Centroides finales:",centros)
    #ASIGNAMOS LOS CLUSTER 
    newImage=asigCluster(valMin,centros,rgb,k)

    return newImage

#BINARIZADO IMAGEN
def binarizado(imagen):
    imgBin=np.zeros((imagen.shape[0],imagen.shape[1]))
    filas,cols=imagen.shape[0],imagen.shape[1]
    rojos=np.array([142,36,30])
    for i in range(filas):
        for j in range(cols):
            if np.array_equal(imagen[i,j], rojos):
                imgBin[i,j]=255
            else:
                imgBin[i,j]=0
    return imgBin


##########MAIN###################
imagen=cv2.imread("examen2/jit.jpg")
#CONVERTIMOS LA IMAGEN A RGB
imagen=cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

filasOrg,colsOrg=imagen.shape[0],imagen.shape[1]

#k=int(input("Ingrese el valor de k: "))
k=4
imgFinal=kMeans(imagen,k)
print("KM",imgFinal)
#REGRESAMOS LA imgFinal AL TAMAÑO ORIGINAL
imgFinal=imgFinal.reshape((filasOrg,colsOrg,3))
#CONVERTIMOS A uint8
imgFinal=np.uint8(imgFinal)

#BINARIZAMOS LA IMAGEN
imgBin=binarizado(imgFinal)
imgBin=np.uint8(imgBin)
imgBin=cv2.cvtColor(imgBin, cv2.COLOR_BGR2RGB)


cv2.imwrite('examen2/imgKmeans.png',cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB))
# cv2.imwrite('examen2/imgBin.png',cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB))
plt.subplot(1,3,1)
plt.title('Img original')
plt.imshow(imagen)
plt.subplot(1,3,2)
plt.title('Img kMeans, k='+str(k))
plt.imshow(imgFinal)
plt.subplot(1,3,3)
plt.title('Img Bin')
plt.imshow(imgBin)
plt.show()