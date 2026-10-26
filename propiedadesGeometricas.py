import cv2
import numpy as np


# ==============================================================================
# 1. MITIGACIÓN DE RUIDO Y RELLENO DE HUECOS ('imfill' DE MATLAB)
# ==============================================================================
def aplicar_suavizado_gaussiano(imagen, ksize=5, sigma=1.0):
    """Mitiga el ruido de alta frecuencia previo a la binarización."""
    return cv2.GaussianBlur(imagen, (ksize, ksize), sigma)


def imfill_holes(mascara_binaria):
    """
    Emula textualmente 'imfill(BW, "holes")' de MATLAB mediante floodFill.
    Inunda el fondo exterior desde (0,0) y sella cavidades internas
    usando operadores lógicos nativos (| y ~).
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
# 2. MÉTODOS DE SEGMENTACIÓN (PDI CLÁSICO SIN MACHINE LEARNING)
# ==============================================================================
def segmentar_otsu_automatico(img_rgb):
    """
    OPCIÓN 1: Segmentación automática por Otsu adaptativo.
    Ideal para monedas, dados, figuras geométricas u objetos sobre fondo contrastado.
    """
    alto, ancho, _ = img_rgb.shape
    gris = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gris_suavizada = aplicar_suavizado_gaussiano(gris, ksize=5, sigma=1.0)

    # Umbralización óptima global de Otsu
    _, thresh = cv2.threshold(gris_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detección automática de polaridad: evalúa el marco perimetral exterior
    bordes = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    bw_base = (thresh == 0) if np.mean(bordes) > 127 else (thresh > 0)

    # Relleno de huecos internos y apertura morfológica suave
    bw_solida = imfill_holes(bw_base)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw_limpia = cv2.morphologyEx(bw_solida.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

    # Filtrar componentes conexas menores al umbral de ruido (0.15% del área total)
    area_minima = max(50.0, (alto * ancho) * 0.0015)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(bw_limpia.astype(np.uint8))

    mascaras = []
    for idx in range(1, num_l):
        if stats[idx, cv2.CC_STAT_AREA] >= area_minima:
            mascaras.append((f"Objeto {len(mascaras) + 1}", labels == idx))

    # Ordenar de izquierda a derecha
    mascaras.sort(key=lambda item: np.argwhere(item[1])[:, 1].min() if np.any(item[1]) else 0)
    return mascaras


def segmentar_por_clics(img_rgb, puntos_semilla, tolerancia=45.0):
    """
    OPCIÓN 2: Segmentación asistida por clic del usuario (PDI clásico).
    Aísla objetos específicos en escenas complejas evaluando la distancia
    euclidiana cromática del píxel seleccionado en espacio continuo RGB.
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

        # Distancia euclidiana en el espacio RGB
        distancia = np.sqrt(np.sum((I_float - color_ref) ** 2, axis=2))
        mask_cruda = distancia <= tolerancia

        # Cierre morfológico, imfill y apertura
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_cerrada = cv2.morphologyEx(mask_cruda.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close) > 0
        mask_solida = imfill_holes(mask_cerrada)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_limpia = cv2.morphologyEx(mask_solida.astype(np.uint8), cv2.MORPH_OPEN, kernel_open) > 0

        # Conservar la componente conexa más cercana a donde el usuario hizo clic
        num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_limpia.astype(np.uint8))
        if num_l > 1:
            distancias = [np.hypot(centroids[i][0] - x_c, centroids[i][1] - y_c) for i in range(1, num_l)]
            idx_mejor = 1 + int(np.argmin(distancias))
            mask_objeto = (labels == idx_mejor)
        else:
            mask_objeto = mask_limpia

        mascaras.append((f"Objeto {idx}", mask_objeto))

    return mascaras


# ==============================================================================
# 3. MOTOR DE PROPIEDADES GEOMÉTRICAS ('regionprops' DE MATLAB)
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
    """Calcula analíticamente las medidas geométricas de la región."""
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

    # Centroide analítico continuo obtenido de momentos espaciales
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2.0, y + h / 2.0
    props.Centroid = (float(cx), float(cy))

    # Subimagen binaria recortada ('Image' en MATLAB)
    props.Image = (mascara_binaria[y:y+h, x:x+w] > 0)

    # Cuadrado simétrico centrado: lado = max(ancho, alto)
    props.CuadradoEnvolvente, props.LadoCuadrado = calcular_cuadrado_envolvente(
        props.BoundingBox, ancho_img, alto_img
    )

    # Código de cadena de Freeman en 8-vecindad
    props.CodigoFreeman = freeman_chain_code_8(cnt)
    return props


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
    """
    Ejecuta el pipeline de procesamiento según la opción seleccionada:
    - modo='otsu': Binarización automática global para monedas o figuras sobre fondo.
    - modo='clic': Binarización espectral interactiva para los objetos señalados por el usuario.
    """
    img_bgr = cv2.imread(ruta_archivo)
    if img_bgr is None:
        raise ValueError("No se pudo cargar el archivo como una imagen válida.")

    alto, ancho, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Conversión formal a double en espacio continuo
    img_suavizada = aplicar_suavizado_gaussiano(img_rgb, ksize=5, sigma=1.0)
    I_double = img_suavizada.astype(np.float64) / 255.0

    R = I_double[:, :, 0]
    G = I_double[:, :, 1]
    B = I_double[:, :, 2]

    R_uint8 = img_rgb[:, :, 0]
    G_uint8 = img_rgb[:, :, 1]
    B_uint8 = img_rgb[:, :, 2]

    # Ejecución de la segmentación según la opción elegida
    if modo == "clic" and puntos_semilla and len(puntos_semilla) > 0:
        mascaras = segmentar_por_clics(img_rgb, puntos_semilla, tolerancia=tolerancia)
    else:
        mascaras = segmentar_otsu_automatico(img_rgb)

    # Máscara combinada para el panel de imagen binarizada
    bwTotal = np.zeros((alto, ancho), dtype=bool)
    for _, m in mascaras:
        bwTotal = bwTotal | m

    # Extracción de propiedades geométricas estilo regionprops
    objetos = []
    for nombre, m in mascaras:
        prop = extraer_propiedades(m, alto, ancho, nombre=nombre)
        objetos.append(prop)

    # Contornos perimetrales sobre lienzo negro
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