import cv2
import numpy
import imutils
import argparse
import argcomplete

from skimage.morphology import skeletonize

def get_corners_debug(original_image, blur=5, rho_step=1, threshold=60):
    image = original_image.copy()

    color=[255,255,255]
    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=color)

    theta_step = numpy.pi / 180
    blur_x = blur_y = blur
    image = cv2.blur(image, (blur_x, blur_y))
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)    
    image = cv2.bitwise_not(image)
    
    skeletonized_image = skeletonize(image > 0)
    skeletonized_image = 255 * (skeletonized_image.astype(numpy.uint8))

    lines = cv2.HoughLines(skeletonized_image, rho_step, theta_step, threshold)

    img_colour = numpy.dstack([image, image, image])
    for rho,theta in lines[:,0]:
        a = numpy.cos(theta)
        b = numpy.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img_colour,(x1,y1),(x2,y2),(0,0,255),2)

    
    points = []
    for i in range(lines.shape[0]):
        (rho1, theta1) = lines[i, 0]
        m1 = -1/numpy.tan(theta1)
        c1 = rho1 / numpy.sin(theta1)
        for j in range(i+1,lines.shape[0]):
            (rho2, theta2) = lines[j,0]
            m2 = -1 / numpy.tan(theta2)
            c2 = rho2 / numpy.sin(theta2)
            if numpy.abs(m1 - m2) <= 1e-8:
                continue
            x = (c2 - c1) / (m1 - m2)
            y = m1*x + c1
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                points.append((int(x), int(y)))

    points = numpy.array(points)
    points = points[:, None]
    output = numpy.dstack([image, image, image])

    for point in points[:, 0]:
        cv2.circle(output, tuple(point), 2, (0, 255, 0), 2)


    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    z = points.copy().astype(numpy.float32)
    compactness, labels, centers = cv2.kmeans(z, 4, None, criteria, 10, flags)

    output_2 = numpy.dstack([image, image, image])

    minX=600
    minY=600
    maxX=0
    maxY=0

    for point in centers:
        if(point[0] < minX):
            minX = point[0]
        if(point[0] > maxX):
            maxX = point[0]
        if(point[1] < minY):
            minY = point[1]
        if(point[1] > maxY):
            maxY = point[1]

    output3 = numpy.dstack([image, image, image])
    output4 = numpy.dstack([image, image, image])    
    output5 = numpy.dstack([image, image, image])
    output6 = numpy.dstack([image, image, image])

    newPoints = []
    for point in points[:, 0]:
        if(point[0] >= minX+50 and point[0] <= maxX-50 or point[1] >= minY+50 and point[1] <= maxY-50):
            cv2.circle(output3, tuple(point), 2, (0, 255, 0), 2)
            continue
        cv2.circle(output4, tuple(point), 2, (0, 255, 0), 2)
        newPoints.append(point)
    
    points = numpy.array(newPoints)
    points = points[:, None]

    newPoints = []
    for point in points[:, 0]:
        if(point[0] >= minX-50 and point[0] <= maxX+50 or point[1] >= minY-50 and point[1] <= maxY+50):
            cv2.circle(output5, tuple(point), 2, (0, 255, 0), 2)
            newPoints.append(point)
            continue
        cv2.circle(output6, tuple(point), 2, (0, 255, 0), 2)
    
    points = numpy.array(newPoints)
    points = points[:, None]

    z = points.copy().astype(numpy.float32)
    compactness, labels, centers = cv2.kmeans(z, 4, None, criteria, 10, flags)

    for point in centers:
        cv2.circle(output_2, tuple(point), 2, (0, 255, 0), 2)

    cv2.imshow('Img_colour', img_colour)
    cv2.imshow('Points', output)
    cv2.imshow('output3', output3)
    cv2.imshow('output4', output4)
    cv2.imshow('output5', output5)
    cv2.imshow('output6', output6)
    cv2.imshow('Centers', output_2)

    centers = cv2.convexHull(centers)[:,0]
    result=[tuple(point) for point in centers]

    for (i, j) in zip(range(4), [1, 2, 3, 0]):
        length = numpy.sqrt(numpy.sum((centers[i] - centers[j]) ** 2.0))

    final_result = numpy.dstack([image, image, image])
    for (i, j) in zip(range(4), [1, 2, 3, 0]):
        cv2.line(final_result, tuple(centers[i]), tuple(centers[j]), (0, 0, 255), 2)

    for point in centers:
        cv2.circle(final_result, tuple(point), 2, (0, 255, 0), 2)

    cv2.imshow('Final result', final_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_corners(original_image, blur=5):
    image = original_image.copy()

    # Añadimos un padding de 100px a la imagen
    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255,255,255])

    blur_x = blur_y = blur
    image = cv2.blur(image, (blur_x, blur_y))
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)    
    
    # Invertimos los colores de la imagen
    image = cv2.bitwise_not(image)
    
    # Extraemos el esqueleto
    skeletonized_image = skeletonize(image > 0)
    skeletonized_image = 255 * (skeletonized_image.astype(numpy.uint8))

    # Usamos la transformada de hough para encontrar las lineas
    lines = cv2.HoughLines(skeletonized_image, 1, numpy.pi / 180, 20)

    # Las lineas vienen en coordenadas polares, hay que convertirlas a coordenadas cartesianas
    # y calculamos los puntos de intercepcion de las lineas
    points = []
    for i in range(lines.shape[0]):
        (rho1, theta1) = lines[i, 0]
        m1 = -1/numpy.tan(theta1)
        c1 = rho1 / numpy.sin(theta1)
        for j in range(i+1,lines.shape[0]):
            (rho2, theta2) = lines[j,0]
            m2 = -1 / numpy.tan(theta2)
            c2 = rho2 / numpy.sin(theta2)
            if numpy.abs(m1 - m2) <= 1e-8:
                continue
            x = (c2 - c1) / (m1 - m2)
            y = m1*x + c1
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                points.append((int(x), int(y)))

    points = numpy.array(points)
    points = points[:, None]

    # Ahora usamos el algoritmo de K-Means para conseguir un promedio de todos los puntos de intercepcion encontrados
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    z = points.copy().astype(numpy.float32)
    compactness, labels, centers = cv2.kmeans(z, 4, None, criteria, 10, flags)

    # K-means devuelve 4 puntos promedio de cada esquina que es donde hay mas concentracion de intercepciones sin embargo
    # como las imagenes son rectangulos muy irregulares, las intercepciones tienden a ser muy dispersas en toda la foto
    # y el resultado de k-means es muy inexacto, necesitamos filtrar los puntos de intercepcion concentrandonos en las esquinas
    # eliminando los puntos de intercepcion dispersos en toda la foto, para ello lo primero es encontrar el minX, MaxX, MinY, MaxY
    minX=600
    minY=600
    maxX=0
    maxY=0

    for point in centers:
        if(point[0] < minX):
            minX = point[0]
        if(point[0] > maxX):
            maxX = point[0]
        if(point[1] < minY):
            minY = point[1]
        if(point[1] > maxY):
            maxY = point[1]

    # Ahora eliminamos los puntos de intercepcion dentro de la region formada por los 4 puntos promedio
    # es importante ver que se dejo una olgura de 40 pixeles para no eliminar los puntos que quedaron en esa region
    newPoints = []
    for point in points[:, 0]:
        if(point[0] >= minX+40 and point[0] <= maxX-40 or point[1] >= minY+40 and point[1] <= maxY-40):
            continue
        newPoints.append(point)
    
    points = numpy.array(newPoints)
    points = points[:, None]

    # Ahora eliminamos los puntos de intercepcion dentro de la region formada por los 4 puntos promedio
    # es importante ver que se dejo una olgura de 40 pixeles para no eliminar los puntos que quedaron en esa region
    newPoints = []
    for point in points[:, 0]:
        if(point[0] >= minX-40 and point[0] <= maxX+40 or point[1] >= minY-40 and point[1] <= maxY+40):
            newPoints.append(point)
    
    points = numpy.array(newPoints)
    points = points[:, None]

    # Calculamos nuevamente k means pero con los puntos de intercepcion agrupados en las esquinas.
    z = points.copy().astype(numpy.float32)
    compactness, labels, centers = cv2.kmeans(z, 4, None, criteria, 10, flags)

    # Aplicamos el casco de hull
    centers = cv2.convexHull(centers)[:,0]

    # Formateamos y retornamos el resultado.
    return [tuple(point) for point in centers]
    

def main(arguments):
    image = cv2.imread(arguments["input"], 0)
    rho_step = arguments["rho"]
    threshold = arguments["threshold"]

    print('Corners found:')
    print(get_corners(image))

    print('Graphically represented:') 
    orientation = get_corners_debug(image, blur=5, rho_step=int(rho_step), threshold=int(threshold))
    
if __name__ == '__main__':
    arguments_parsed = argparse.ArgumentParser()
    arguments_parsed.add_argument("-i", "--input", required=True, help="Path to the image's directory")
    arguments_parsed.add_argument("-r", "--rho", required=False)
    arguments_parsed.add_argument("-t", "--threshold", required=False)
    argcomplete.autocomplete(arguments_parsed)
    arguments = vars(arguments_parsed.parse_args())
    main(arguments)