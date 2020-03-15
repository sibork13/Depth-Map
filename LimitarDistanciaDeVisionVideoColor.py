import pyrealsense2 as rs
import numpy as np
import cv2


#Abrimos 
pipeline = rs.pipeline()
config = rs.config()
#En configuracion se guarda la informacion del stream a color y de prfundidad
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Iniciamos el stream
pipeline.start(config)


try:
    # k=1
    while True:
    # while k ==1:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()#Obtiene frame
        depth = frames.get_depth_frame()#obtiene la distancias
        color = frames.get_color_frame()
        if not depth or not color:
            continue
        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())
        #print("La longitud de depth_image es : "+str(color_image.shape))
        Columnas,Filas,Dimensiones = color_image.shape
        Columnas2,Filas2 = depth_image.shape
        print(str(Columnas)+"   "+str(Filas))
        print(str(Columnas2)+"   "+str(Filas2))
        # print("las columnas son: "+str(Columnas)+" y las filas son: "+str(Filas))
        print(depth_image)
        color_image = color_image[:,:,0]
        np.putmask(color_image,depth_image > 750,255)#en cada celda de la m,atriz de color verifica si en esa misma posicion de l otra matriz la distancia es menos a 1 metro, si es asi quta el coor a la matriz de color
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)#muestra la imagen en N milisegundos

finally:
    pipeline.stop()
