import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import math
np.set_printoptions(threshold=sys.maxsize)

random.seed(125)

#CONVERTIMOS A ESCALA DE GRISES CON EL METODO NTSC
def rgb2gray(imagen):
    rows,cols,_=imagen.shape #REGRESA UNA TUPLA CON FILAS, COLUMNAS, Y CANALES
    imgGray=np.zeros((rows,cols,1), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            #OBTENEMOS EL VALOR DE CADA COMPONENTE bgr
            b = imagen.item(i, j, 0)
            g = imagen.item(i, j, 1)
            r = imagen.item(i, j, 2)
            #HACEMOS LA OPERACION NTSC
            pixel=0.2989 * r + 0.5870 * g + 0.1140 * b 
            #ASIGNAMOS EL PIXEL A LA IMAGEN imgGray
            imgGray[i,j]=pixel
    imgGray=np.uint8(imgGray)
    return imgGray

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

#K-MEANS PARA OBTENER FIGURAS

#OBTENEMOS UNA LISTA CON LOS PIXELES EN BLANCO
def whites(imagen):
    filas,cols = imagen.shape[0],imagen.shape[1]

    listWhite = []
    for i in range(filas):
        for j in range(cols):
            value=imagen[i,j]
            if value != 0:
                listWhite.append([i,j])
    return listWhite

#OBTENEMOS LOS PRIMEROS CENTROIDES
def centroids(k,data):
    #NUESTRA LISTA DE CENTROIDES, TENDRA 2 COLS(XY) Y k FILAS
    listCentroids=np.zeros((k,2))
    #ESCOGEMOS VALORES ALEATORIOS DE NUESTROS PIXELES
    for i in range(k):
        cenX=random.choice(data[0])
        cenY=random.choice(data[1])
        listCentroids[i]=[cenX,cenY]

    return listCentroids

#CALCULAMOS LA DISTANCIA, DE NUESTRO CENTROIDE AL PIXEL
def calDistance(a,b):
    distance=(np.sum((a-b)**2))
    return distance

#DEFINIMOS NUEVOS CENTROIDES
def newCen(valMin,rgb,k, centroids):
    #NUESTRA LISTA DE CENTROIDES, TENDRA 2 COLS(XY) Y k FILAS
    listCentroids=np.zeros((k,2))
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

def validatePixel(data,clusters):
    c0=[]
    c1=[]
    c2=[]
    c3=[]
    for i in range(clusters.shape[0]):
        if clusters[i]==0:
            c0.append(data[i])
        elif clusters[i]==1:
            c1.append(data[i])
        elif clusters[i]==2:
            c2.append(data[i])
        elif clusters[i]==3:
            c3.append(data[i])
    return c0,c1,c2,c3

def drawFig(data,imagen):
    newData=np.zeros((imagen.shape))
    for i in range(len(data)):
        newData[data[i][0],data[i][1]]=255
    newData=np.uint8(newData)
    return newData

def kMeansPixels(data,k):
    data=data.reshape(-1,2)
    #OBTENEMOS LOS PRIMEROS CENTROIDES
    centros=centroids(k,data)
    #CREAMOS NUESTRO ARREGLO DE DISTANCIAS, CADA PIXEL VA A TENR UNA DISTANCIA A CADA CLUSTER
    distances=np.zeros((data.shape[0],k))

    #DEFINIMOS EL NUMERO DE ITERACIONES
    iterations=100
    #CONTADOR PARA ITERACIONES
    iter=1

    for i in range(iterations):
        for i in range(distances.shape[0]):
            for j in range(k):
                #CALCULAMOS LAS DISTANCIAS DE CADA PIXEL A CADA CENTRO
                distance=calDistance(centros[j],data[i])
                distances[i,j]=distance

        #OBTENEMOS UN ARRGELO CON LOS INDICES DE A QUE CLUSTER PERTENECE CADA PIXEL (LA MENOR DISTANCIA)
        #(ARREGLO DE N PIXELES, 1 COL)
        valMin=np.argmin(distances,axis=1)
        # print("valMin",valMin.shape)
        
        
        #OBTENEMOS NUEVOS CENTROIDES
        newCentroids=newCen(valMin,data,k,centros)
        #SI EN DOS ITERACIONES SEGUIDAS NOS DAN LOS MISMOS CENTORIDES, ROMPE LAS ITERACIONES
        if np.all(np.round(newCentroids)==np.round(centros)):
            break 
        #AHORA centros VAN A SER IGUAL A NUESTROS newCentroids
        centros=newCentroids
        # print("iter",iter)
        iter+=1
    
    c0,c1,c2,c3=validatePixel(data,valMin)

    return c0,c1,c2,c3

def calculateCenter(coord):
    listCenter=np.zeros((1,2))
    coord=np.array(coord)
    listCenter[0]=np.mean(coord,axis=0)
    listCenter=np.round(listCenter)
    return listCenter
    
def ptsRecta(coord):
    varMin=np.amin(coord,axis=0)
    varMax=np.amax(coord,axis=0)
    # print("vMin",varMin)
    # print("vMax",varMax)
    return varMin, varMax

def ptsRecta2(imagen,coord1,coord2):
    coordSup=np.zeros((1,2))
    coordInf=np.zeros((1,2))


    flag2=False
    valX2,valY2=coord2[0],coord2[1]
    while(flag2==False):
        valor2=imagen[valX2,valY2,0]
        if valor2>0:
            coordInf[0,0]=valX2
            coordInf[0,1]=valY2
            flag2=True
        
        valX2+=1
        valY2-=1

    flag1=False
    valX1,valY1=coord1[0],coord1[1]
    while(flag1==False):
        valor1=imagen[valX1,valY1,0]
        if valor1>0:
            coordSup[0,0]=271
            coordSup[0,1]=127   
            flag1=True
        
        valX1+=1
        valY1-=1

    # print("csup",coordSup)
    # print("cinf",coordInf)
    return coordSup,coordInf

def prodCruz(p1,p2):
    line=np.cross(p1,p2)
    return line
    

def f(x,l):
    y=(l[0]*x+l[2])/-l[1]
    return y

def dist(p1,p2):
    distance=np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return distance

imagenOrg=cv2.imread("examen2/jit.jpg")
imagen=cv2.imread("examen2/imgKmeans.png")
#CONVERTIMOS LA IMAGEN A RGB
imagen=cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagenOrg=cv2.cvtColor(imagenOrg, cv2.COLOR_BGR2RGB)

imgBin=binarizado(imagen)
imgBin=np.uint8(imgBin)
imgBin=cv2.cvtColor(imgBin, cv2.COLOR_BGR2RGB)
#CONVERTIMOS A ESCALA DE GRISES PARA QUITAR LOS COMPONENTES RGB
imgGray=rgb2gray(imgBin)
#OBTENEMOS LA LISTA DE PIXELES BLANCOS
listWhites=whites(imgGray)
listWhites=np.array(listWhites)

c0,c1,c2,c3=kMeansPixels(listWhites,4)
# print("c0",np.array(c0))
#OBTENEMOS LOS PIXELES QUE PERTENECEN AL CLUSTER 0 (jit2)
imgC0=drawFig(c0,imgGray)
imgC0=cv2.cvtColor(imgC0, cv2.COLOR_BGR2RGB)
#OBTENEMOS LOS PIXELES QUE PERTENECEN AL CLUSTER 1 (jit4)
imgC1=drawFig(c1,imgGray)
imgC1=cv2.cvtColor(imgC1, cv2.COLOR_BGR2RGB)


#CALCULAMOS LOS CENTROS
centerJ2=calculateCenter(c0)
centerJ4=calculateCenter(c1)

#OBTENEMOS LOS PUNTOS PARA LA RECTA
varMinJ2,varMaxJ2=ptsRecta(c0)
varMinJ4,varMaxJ4=ptsRecta(c1)

#OBTENEMOS PRODUCTO CRUZ DE LOS PUNTOS
#PARA JIT2
p1=[varMaxJ2[1],centerJ2[0,0],1]
p2=[varMinJ2[1],centerJ2[0,0],1]
lineJ2=prodCruz(p1,p2)
# print("line",lineJ2)
x1=range(varMinJ2[1],varMaxJ2[1]+1)
y1=[f(i,lineJ2) for i in x1]
#LINEAS FINALES PARA JIT2
x1P=[varMaxJ2[1],varMinJ2[1]]
y1P=[centerJ2[0,0],centerJ2[0,0]]
#CALCULAMOS LAS DISTANCIAS
disJ2=math.dist(p1,p2)
print("PARA JITOMATE 2:")
print("P1",p1)
print("P2",p2)
print("DISTANCIA ENTRE P1 Y P2 PARA EL JITOMATE 2:",disJ2)


#PARA JIT4
coord1=np.array([varMaxJ4[1],varMinJ4[0]])
coord2=np.array([varMinJ4[1],varMaxJ4[0]])


coordSup,coordInf=ptsRecta2(imgGray,coord1,coord2)
p3=[varMaxJ4[1],varMinJ4[0],1]
p4=[varMinJ4[1],varMaxJ4[0],1]
lineJ4=prodCruz(p3,p4)
lineJ4P=prodCruz(coordSup,coordInf)
# print("line",lineJ4)
x2=range(varMinJ4[1],varMaxJ4[1]+1)
y2=[f(i,lineJ4) for i in x2]
#LINEAS FINALES PARA JIT4
y2P=[coordInf[0,0],coordSup[0,0]]
x2P=[coordInf[0,1],coordSup[0,1]]
#CALCULAMOS LAS DITANCIAS
disJ4=math.dist(p3,p4)
print("PARA JITOMATE 4:")
print("P1",p3)
print("P2",p4)
print("DISTANCIA ENTRE P1 Y P2 PARA EL JITOMATE 4:",disJ4)


plt.subplot(1,3,1)
plt.title('Img Bin')
plt.imshow(imgBin)
plt.subplot(1,3,2)
plt.title('Detection jit2')
plt.imshow(imgC0)
plt.subplot(1,3,3)
plt.title('Detection jit4')
plt.imshow(imgC1)
plt.show()

plt.subplot(1,3,1)
plt.title('Img org')
plt.imshow(imagenOrg)
#JIT2
plt.subplot(1,3,2)
#PUNTOS EXTREMOS
plt.plot(varMaxJ2[1],centerJ2[0,0],marker='o',color='b')
plt.plot(varMinJ2[1],centerJ2[0,0],marker='o',color='b')
#CENTRO
plt.plot(centerJ2[0,1],centerJ2[0,0],marker='o',color='g')
#LINEA DE UNION
plt.plot(x1,y1,color='g')
plt.title('Centre jit2')
plt.imshow(imagenOrg)
#JIT4
plt.subplot(1,3,3)
#PUNTOS EXTREMOS
plt.plot(varMaxJ4[1],varMinJ4[0],marker='o',color='b')
plt.plot(varMinJ4[1],varMaxJ4[0],marker='o',color='b')

plt.plot(coordSup[0,1],coordSup[0,0],marker='o',color='y')
plt.plot(coordInf[0,1],coordInf[0,0],marker='o',color='y')
#CENTRO
plt.plot(centerJ4[0,1],centerJ4[0,0],marker='o',color='g')
#LINEA DE UNION
plt.plot(x2,y2,color='g')
plt.title('Centre jit4')
plt.imshow(imagenOrg)
plt.show()


#IMAGEN FINAL
plt.subplot(1,1,1)
plt.plot(x1P,y1P,color='g',linewidth=3)
# plt.plot(x2P,y2P,color='y')
plt.plot(x2P,y2P, color='y',linewidth=3)
plt.title('Img FINAL')
plt.imshow(imagenOrg)
plt.show()