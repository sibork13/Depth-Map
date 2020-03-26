#Help Suorce
#https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html
#https://medium.com/@gastonace1/detecci%C3%B3n-de-objetos-por-colores-en-im%C3%A1genes-con-python-y-opencv-c8d9b6768ff
#http://docs.ros.org/kinetic/api/librealsense/html/namespacers.html
import pyrealsense2 as rs
import numpy as np
import cv2


#Abrimos
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
# Iniciamos el stream
pipeline.start(config)
try:
    while True:
        frames = pipeline.wait_for_frames()#Obtiene frame
        color = frames.get_color_frame()
        if not color:
            continue
        color_image = np.asanyarray(color.get_data())
        hsv_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)#Convertimos BGR  a HSV para deteccion de colores
        # hsv_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2HSV)
        #Comenzamos a definir los limites del color rojo
        Red_Min = (165,90,80)#HSV
        Red_Max = (180,255,255)#HSV

        #Comenzamos a definir los limites del color Azul
        BLue_Min = (90,90,80)#HSV
        Blue_Max = (135,255,255)#HSV
        #Hacemos la discriminacion de valores que no pertenecen al rojo
        mascara = cv2.inRange(hsv_image,Red_Min,Red_Max)#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        Red_Output = cv2.bitwise_and(color_image,color_image,mask=mascara)#la imagen de entrada se filtra con la mascara y se genera la salida (entrada,salida,mascara)

        #SECCION DEL AZUL
        mascaraB = cv2.inRange(hsv_image,BLue_Min,Blue_Max)#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        Blue_Output = cv2.bitwise_and(color_image,color_image,mask=mascaraB)#la imagen de entrada se filtra con la mascara y se genera la salida (entrada,salida,mascara)
        # images = np.hstack((color_image, res))


        # Columnas,Filas,Dimensiones = color_image.shape
        # Columnas2,Filas2 = depth_image.shape
        cv2.imshow('RGB', color_image)
        cv2.imshow('Rojo', Red_Output)
        cv2.imshow('Azul', Blue_Output)


        cv2.waitKey(1)#muestra la imagen en N milisegundos

finally:
    pipeline.stop()
