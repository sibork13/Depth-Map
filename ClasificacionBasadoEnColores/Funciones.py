import pyrealsense2 as rs
import numpy as np
import cv2

def ConfigurarRealSense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def Obtener_Imagenes(pipeline):
    frames = pipeline.wait_for_frames()#Obtiene frame
    Imagen_Profundidad = frames.get_depth_frame()#obtiene la distancias
    Imagen_Color = frames.get_color_frame()
    return Imagen_Profundidad,Imagen_Color



def Obtener_Datos(Imagen_Profundidad,Imagen_color):
    Datos_Profundidad = np.asanyarray(Imagen_Profundidad.get_data())
    Datos_Color = np.asanyarray(Imagen_color.get_data())
    return Datos_Profundidad,Datos_Color



def EliminarFondo(Imagen_Color,Imagen_Profundidad,Distancia,Color_Contorno):
    Columnas,Filas,Dimensiones = Imagen_Color.shape
    for i in range(0,Dimensiones):
        auxiliar = Imagen_Color[:,:,i]
        np.putmask(auxiliar,Imagen_Profundidad > Distancia,Color_Contorno)
        Imagen_Color[:,:,i] = auxiliar


def ConvertirToNumpyArray(Lista):
    Lista = np.array(Lista, np.uint8)
    return Lista


def DibujarContornos(imagen,contornos,color,Palabra):
    for c in contornos:
        M = cv2.moments(c)
        if (M["m00"]==0):M["m00"]=1
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        cv2.drawContours(imagen,[c],0,color,2)
        epsilon = 0.1*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        cv2.drawContours(imagen,approx,0,color,2)
        cv2.putText(imagen,str(Palabra),(x,y),1,2,(0,0,0),2)
