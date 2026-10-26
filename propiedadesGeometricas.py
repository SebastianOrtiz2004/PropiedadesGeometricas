import cv2
import numpy as np


# ==============================================================================
# 1. FILTRADO GAUSSIANO Y RELLENO MORFOLÓGICO ('imfill' DE MATLAB)
# ==============================================================================
def aplicar_suavizado_gaussiano(imagen, ksize=5, sigma=1.0):
    """Mitiga el ruido de alta frecuencia previo a la binarización (Audio 2)."""
    return cv2.GaussianBlur(imagen, (ksize, ksize), sigma)


def imfill_holes(mascara_binaria):
    """
    Emula textualmente 'imfill(BW, "holes")' de MATLAB mediante floodFill.
    Inunda el exterior desde (0,0) y sella cavidades internas usando operadores lógicos.
    """
    if mascara_binaria.dtype == bool:
        mask_u8 = mascara_binaria.astype(np.uint8) * 255
    else:
        mask_u8 = (mascara_binaria > 0).astype(np.uint8) * 255

    alto, ancho = mask_u8.shape
    flood = mask_u8.copy()
    mascara_auxiliar = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)
    cv2.floodFill(flood, mascara_auxiliar, (0, 0), 255)

    fondo_invertido = (flood == 0)
    return (mascara_binaria > 0) | fondo_invertido


# ==============================================================================
# 2. LOS TRES MOTORES DE SEGMENTACIÓN (PDI CLÁSICO SIN MACHINE LEARNING)
# ==============================================================================
def segmentar_otsu_automatico(img_rgb):
    """
    MODO 1: Umbralización Global Adaptativa de Otsu con Detección de Polaridad.
    Universal para monedas, dados, piezas industriales o figuras sobre fondo homogéneo.
    """
    alto, ancho, _ = img_rgb.shape
    gris = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gris_suavizada = aplicar_suavizado_gaussiano(gris, ksize=5, sigma=1.0)

    _, thresh = cv2.threshold(gris_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inversión automática según el color del fondo perimetral
    bordes = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    bw_base = (thresh == 0) if np.mean(bordes) > 127 else (thresh > 0)

    bw_solida = imfill_holes(bw_base)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw_limpia = cv2.morphologyEx(bw_solida.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

    area_minima = max(50.0, (alto * ancho) * 0.0015)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(bw_limpia.astype(np.uint8))

    componentes = []
    for idx in range(1, num_l):
        if stats[idx, cv2.CC_STAT_AREA] >= area_minima:
            componentes.append(labels == idx)

    # Ordenar de izquierda a derecha según la coordenada X mínima
    componentes.sort(key=lambda m: np.argwhere(m)[:, 1].min() if np.any(m) else 0)

    # Asignar nombres ordenados de izquierda a derecha
    mascaras = [(f"Objeto {i}", m) for i, m in enumerate(componentes, start=1)]
    return mascaras


def segmentar_por_clics(img_rgb, puntos_semilla, tolerancia=45.0):
    """
    MODO 2: Muestreo Espectral Interactivo mediante Semillas de Color (PDI clásico).
    Permite aislar cualquier objeto arbitrario en escenas complejas mediante distancia euclidiana RGB.
    """
    alto, ancho, _ = img_rgb.shape
    img_suavizada = aplicar_suavizado_gaussiano(img_rgb, ksize=5, sigma=1.0)
    I_float = img_suavizada.astype(np.float64)

    mascaras = []
    for idx, (x_seed, y_seed) in enumerate(puntos_semilla, start=1):
        x_c = int(np.clip(x_seed, 0, ancho - 1))
        y_c = int(np.clip(y_seed, 0, alto - 1))

        # Color representativo de la vecindad 5x5 alrededor del clic
        y1, y2 = max(0, y_c - 2), min(alto, y_c + 3)
        x1, x2 = max(0, x_c - 2), min(ancho, x_c + 3)
        color_ref = I_float[y1:y2, x1:x2].mean(axis=(0, 1))

        distancia = np.sqrt(np.sum((I_float - color_ref) ** 2, axis=2))
        mask_cruda = distancia <= tolerancia

        # Morfología matemática e imfill
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_cerrada = cv2.morphologyEx(mask_cruda.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close) > 0
        mask_solida = imfill_holes(mask_cerrada)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_limpia = cv2.morphologyEx(mask_solida.astype(np.uint8), cv2.MORPH_OPEN, kernel_open) > 0

        num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_limpia.astype(np.uint8))
        if num_l > 1:
            distancias = [np.hypot(centroids[i][0] - x_c, centroids[i][1] - y_c) for i in range(1, num_l)]
            idx_mejor = 1 + int(np.argmin(distancias))
            mask_objeto = (labels == idx_mejor)
        else:
            mask_objeto = mask_limpia

        mascaras.append((f"Objeto {idx}", mask_objeto))

    return mascaras


def segmentar_diferencial_multicanal(I_double):
    """
    MODO 3: Segmentación Diferencial Multicanal en Espacio Continuo Double (Caso de Estudio).
    Implementa el desacople por dominancia espectral relativa R - max(G,B) documentado en el informe LaTeX.
    """
    R = I_double[:, :, 0]
    G = I_double[:, :, 1]
    B = I_double[:, :, 2]

    # Objeto Auto: Carrocería con absorción en B y reflectancia en R-G
    carroceria = (R > 0.55) & (G > 0.45) & (B < 0.45)
    ruedas_oscuras = (R < 0.30) & (G < 0.30) & (B < 0.30)

    # Dilatación condicional calibrada (25x25) para capturar la base de las ruedas contiguas al chasis
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    zona_objeto = cv2.dilate(carroceria.astype(np.uint8), kernel_dilatacion) > 0
    base_oscura = ruedas_oscuras & zona_objeto
    bw_a_cruda = carroceria | base_oscura

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bw_a_cerrada = cv2.morphologyEx(bw_a_cruda.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close) > 0
    bw_a_solida = imfill_holes(bw_a_cerrada)

    # Apertura morfológica (5x5) para cortar la línea parásita de 1 px hacia el marco de la puerta
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw_a_limpia = cv2.morphologyEx(bw_a_solida.astype(np.uint8), cv2.MORPH_OPEN, kernel_open) > 0

    num_l_a, labels_a, stats_a, _ = cv2.connectedComponentsWithStats(bw_a_limpia.astype(np.uint8))
    if num_l_a > 1:
        idx_a = 1 + np.argmax(stats_a[1:, cv2.CC_STAT_AREA])
        bwAuto = (labels_a == idx_a)
    else:
        bwAuto = bw_a_limpia

    # Objeto Casa: Dominancia relativa del rojo R - max(G,B) > 0.15 y exclusión del auto
    dominancia_roja = (R - np.maximum(G, B) > 0.15) & (R > 0.45)
    casa_pura = dominancia_roja & (~bwAuto)

    kernel_casa = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bw_c_cerrada = cv2.morphologyEx(casa_pura.astype(np.uint8), cv2.MORPH_CLOSE, kernel_casa) > 0
    bw_c_solida = imfill_holes(bw_c_cerrada)

    num_l_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(bw_c_solida.astype(np.uint8))
    if num_l_c > 1:
        idx_c = 1 + np.argmax(stats_c[1:, cv2.CC_STAT_AREA])
        bwCasa = (labels_c == idx_c)
    else:
        bwCasa = bw_c_solida

    return [("Casa", bwCasa), ("Auto", bwAuto)]


# ==============================================================================
# 3. MOTOR DE PROPIEDADES GEOMÉTRICAS Y ALIAS 'regionprops'
# ==============================================================================
class RegionPropsStruct:
    """Estructura idéntica al struct devuelto por regionprops en MATLAB."""
    def __init__(self):
        self.Nombre = "Objeto"
        self.Area = 0.0
        self.Perimeter = 0.0
        self.Centroid = (0.0, 0.0)
        self.BoundingBox = [0.0, 0.0, 0.0, 0.0]
        self.Image = np.zeros((1, 1), dtype=bool)
        self.Contorno = None
        self.CuadradoEnvolvente = [0.0, 0.0, 0.0, 0.0]
        self.LadoCuadrado = 0.0
        self.CodigoFreeman = []


def extraer_propiedades(mascara_binaria, alto_img, ancho_img, nombre="Objeto"):
    """Calcula analíticamente las medidas geométricas de la región conexa."""
    bw_u8 = mascara_binaria.astype(np.uint8) * 255 if mascara_binaria.dtype == bool else (mascara_binaria > 0).astype(np.uint8) * 255
    contornos, _ = cv2.findContours(bw_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    props = RegionPropsStruct()
    props.Nombre = nombre

    if not contornos:
        return props

    cnt = max(contornos, key=cv2.contourArea)
    props.Contorno = cnt
    props.Area = float(cv2.contourArea(cnt))
    props.Perimeter = float(cv2.arcLength(cnt, True))

    x, y, w, h = cv2.boundingRect(cnt)
    props.BoundingBox = [float(x), float(y), float(w), float(h)]

    # Centroide analítico continuo por momentos espaciales discretos (m10/m00, m01/m00)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2.0, y + h / 2.0
    props.Centroid = (float(cx), float(cy))

    # Subimagen recortada ('Image' en MATLAB)
    props.Image = (mascara_binaria[y:y+h, x:x+w] > 0)

    # Cuadrado simétrico centrado: lado = max(ancho, alto)
    props.CuadradoEnvolvente, props.LadoCuadrado = calcular_cuadrado_envolvente(
        props.BoundingBox, ancho_img, alto_img
    )

    # Código de Freeman en 8-vecindad
    props.CodigoFreeman = freeman_chain_code_8(cnt)
    return props


# Alias formal para garantizar paridad exacta con la sintaxis de MATLAB exigida en clase:
regionprops = extraer_propiedades


# ==============================================================================
# 4. CÓDIGO DE CADENA DE FREEMAN (8-VECINDAD)
# ==============================================================================
def freeman_chain_code_8(contorno):
    """Mapea desplazamientos perimetrales discretos en 8 direcciones (0 a 7)."""
    if contorno is None or len(contorno) < 2:
        return []
    mapa = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    pts = contorno.squeeze()
    codigo = []
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        dx = int(np.clip(p2[0] - p1[0], -1, 1))
        dy = int(np.clip(p2[1] - p1[1], -1, 1))
        if (dx, dy) in mapa:
            codigo.append(mapa[(dx, dy)])
    return codigo


# ==============================================================================
# 5. CUADRADO ENVOLVENTE SIMÉTRICO (lado = max(caja(3:4)))
# ==============================================================================
def calcular_cuadrado_envolvente(caja, ancho_img, alto_img):
    """Construye el cuadrado envolvente simétrico centrado sobre el objeto."""
    lado = max(caja[2], caja[3])
    cx = caja[0] + caja[2] / 2.0
    cy = caja[1] + caja[3] / 2.0

    esq_x = cx - lado / 2.0
    esq_y = cy - lado / 2.0

    esq_x = max(0.0, min(esq_x, ancho_img - lado))
    esq_y = max(0.0, min(esq_y, alto_img - lado))

    return [esq_x, esq_y, lado, lado], lado


# ==============================================================================
# 6. PIPELINE PRINCIPAL DE PROCESAMIENTO
# ==============================================================================
def procesar_imagen_completa(ruta_archivo, modo="otsu", puntos_semilla=None, tolerancia=45.0):
    """Ejecuta el pipeline de procesamiento según el modo de PDI clásico seleccionado."""
    img_bgr = cv2.imread(ruta_archivo)
    if img_bgr is None:
        raise ValueError("No se pudo cargar el archivo como una imagen válida.")

    alto, ancho, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_suavizada = aplicar_suavizado_gaussiano(img_rgb, ksize=5, sigma=1.0)
    I_double = img_suavizada.astype(np.float64) / 255.0

    R = I_double[:, :, 0]
    G = I_double[:, :, 1]
    B = I_double[:, :, 2]

    R_uint8 = img_rgb[:, :, 0]
    G_uint8 = img_rgb[:, :, 1]
    B_uint8 = img_rgb[:, :, 2]

    if modo == "clic" and puntos_semilla and len(puntos_semilla) > 0:
        mascaras = segmentar_por_clics(img_rgb, puntos_semilla, tolerancia=tolerancia)
    elif modo == "multicanal":
        mascaras = segmentar_diferencial_multicanal(I_double)
    else:
        mascaras = segmentar_otsu_automatico(img_rgb)

    bwTotal = np.zeros((alto, ancho), dtype=bool)
    for _, m in mascaras:
        bwTotal = bwTotal | m

    objetos = []
    for nombre, m in mascaras:
        prop = regionprops(m, alto, ancho, nombre=nombre)
        objetos.append(prop)

    img_contornos = np.zeros((alto, ancho), dtype=np.uint8)
    for obj in objetos:
        if obj.Contorno is not None:
            cv2.drawContours(img_contornos, [obj.Contorno], -1, 255, 2)

    return {
        "img_rgb": img_rgb,
        "alto": alto,
        "ancho": ancho,
        "R": R, "G": G, "B": B,
        "R_uint8": R_uint8, "G_uint8": G_uint8, "B_uint8": B_uint8,
        "bwTotal": bwTotal,
        "img_contornos": img_contornos,
        "objetos": objetos,
        "modo_usado": modo
    }