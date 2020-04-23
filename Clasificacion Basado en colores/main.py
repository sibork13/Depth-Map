## Importar librerias necesarias
import cv2
import Funciones as FC
import numpy as np

# Preparar transmision de realsense a sistemas
FPS = 30 #Numero de fotogramas deseado  (maximo 30)
pipeline,profile = FC.ConfigurarRealSense(1280,720,FPS)#introducir la resolucion a capturar 1280,720  640,480

# Configurar la limitacion de vision
metros = 1 #numero de metros maximo de viion
clipping_distance = FC.LimitarDistancia(profile,metros)

# Eliminar margen de error en imagen a color
align_to,align = FC.Alinear()

# Limites de color en HSV (H-> elige color, S-> que tanto color de blanco a el color vivo, V-> que tanto color de negro al color)
# Los limites fueron elegidos para el filtro Gaussian Blur
Rojo = [(160,100,45),(180,255,255)]
Azul = [[90,100,45],[130,255,255]]
Colores = Rojo+ Azul
Colores = FC.ConvertirToNumpyArray(Colores)


# Bloque de procesamiento
try:
    while True:#bucle infinito

        # Obtencion de imagenes
        depth,color = FC.Obtener_Imagenes(pipeline,align)

        # Condicional, verificar si existe transmision
        if not depth or not color:
            continue

        # Obtenemos la matriz de imagen y profundidad
        depth_image,color_image = FC.Obtener_Datos(depth,color)

        # Eliminar fondo de la imagen a color
        color_image = FC.EliminarFondo(color_image,depth_image,clipping_distance,153)

        ####FILTROS********************
        color_image = cv2.GaussianBlur(color_image,(45,45),0)

        # Conversión  de colres
        hsv_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)#Convertimos BGR  a HSV para deteccion de colores

        # Buscar seccion de la imagen con el color introducido
        ##SECCION DE ROJO
        mascaraR = cv2.inRange(hsv_image,Colores[0],Colores[1])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        # Red_Output = cv2.bitwise_and(color_image,color_image,mask=mascaraR)#la imagen de entrada se filtra con la mascara y se genera la salida (entrada,salida,mascara)

        # #SECCION DEL AZUL
        mascaraB = cv2.inRange(hsv_image,Colores[2],Colores[3])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        # Blue_Output = cv2.bitwise_and(color_image,color_image,mask=mascaraB)#la imagen de entrada se filtra con la mascara y se genera la salida (entrada,salida,mascara)

        # Encontrar pixeles de contornos de objetos
        ContornoRojo = cv2.findContours(mascaraR,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        ContornoAzul = cv2.findContours(mascaraB,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

        # Dibujar contornos
        FC.DibujarContornos(color_image,ContornoRojo,(255,255,255),"Equipo")
        FC.DibujarContornos(color_image,ContornoAzul,(255,255,255),"Enemigo")

        # Mostrar en pantalla
        cv2.imshow('RealSense', color_image)
        # cv2.imshow('Rojo', mascaraR)
        # cv2.imshow('Rojo 2', Red_Output)
        # cv2.imshow('Azul', Blue_Output)
        cv2.waitKey(1)

finally:
    pipeline.stop()
