import cv2
import Funciones as FC

pipeline = FC.ConfigurarRealSense()
try:
    while True:
        depth,color = FC.Obtener_Imagenes(pipeline)
        if not depth or not color:
            continue
        depth_image,color_image = FC.Obtener_Datos(depth,color)

        FC.EliminarFondo(color_image,depth_image,750,255)

        hsv_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)#Convertimos BGR  a HSV para deteccion de colores

        Rojo = [(165,90,80),(180,255,255)]
        Azul = [[90,90,80],[135,255,255]]
        Colores = Rojo+ Azul
        Colores = FC.ConvertirToNumpyArray(Colores)

        mascaraR = cv2.inRange(hsv_image,Colores[0],Colores[1])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        Red_Output = cv2.bitwise_and(color_image,color_image,mask=mascaraR)#la imagen de entrada se filtra con la mascara y se genera la salida (entrada,salida,mascara)
        #SECCION DEL AZUL
        mascaraB = cv2.inRange(hsv_image,Colores[2],Colores[3])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        Blue_Output = cv2.bitwise_and(color_image,color_image,mask=mascaraB)#la imagen de entrada se filtra con la mascara y se genera la salida (entrada,salida,mascara)

        #Contornos Rojos
        ContornoRojo = cv2.findContours(mascaraR,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        FC.DibujarContornos(color_image,ContornoRojo,(255,255,255),"Equipo")
        #Contornos Azules
        ContornoAzul = cv2.findContours(mascaraB,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        FC.DibujarContornos(color_image,ContornoAzul,(255,255,255),"Enemigo")


        cv2.imshow('RealSense', color_image)
        cv2.imshow('Rojo', Red_Output)
        cv2.imshow('Azul', Blue_Output)
        cv2.waitKey(1)

finally:
    pipeline.stop()
