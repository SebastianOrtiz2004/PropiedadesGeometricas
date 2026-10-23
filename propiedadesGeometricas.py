import cv2
import numpy as np


# ==============================================================================
# 1. MITIGACIÓN DE RUIDO Y RELLENO DE HUECOS ('imfill' DE MATLAB)
# ==============================================================================
def aplicar_suavizado_gaussiano(imagen, ksize=5, sigma=1.0):
    """
    Mitiga el ruido de alta frecuencia previo a la binarización (Audio 2 del docente).
    """
    return cv2.GaussianBlur(imagen, (ksize, ksize), sigma)


def imfill_holes(mascara_binaria):
    """
    Emula textualmente 'imfill(BW, "holes")' de MATLAB.
    Inunda el fondo exterior desde (0,0) y sella cavidades internas (ventanas, ruedas)
    usando operadores lógicos nativos (~ y |) de NumPy/MATLAB.
    """
    if mascara_binaria.dtype == bool:
        mask_u8 = mascara_binaria.astype(np.uint8) * 255
    else:
        mask_u8 = (mascara_binaria > 0).astype(np.uint8) * 255

    alto, ancho = mask_u8.shape
    flood = mask_u8.copy()
    mascara_auxiliar = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)
    cv2.floodFill(flood, mascara_auxiliar, (0, 0), 255)
    
    # Inversión y unión lógica nativa estilo MATLAB: BW | (~fondo)
    fondo_invertido = (flood == 0)
    return (mascara_binaria > 0) | fondo_invertido


# ==============================================================================
# 2. SEGMENTACIÓN CON CANALES EN DOUBLE Y OPERADORES LÓGICOS NATIVOS (&, |, ~)
# ==============================================================================
def segmentar_objetos_canales(I_double):
    """
    Segmenta bwAuto y bwCasa directamente mediante aritmética de canales en double
    y operadores booleanos vectorizados (&, |, ~), idéntico al Workspace de MATLAB.
    """
    # 1. Descomposición vectorial directa estilo MATLAB: R = I(:,:,1), etc.
    R = I_double[:, :, 0]
    G = I_double[:, :, 1]
    B = I_double[:, :, 2]

    # -------------------------------------------------------------------------
    # A. bwAuto: Carrocería + ruedas oscuras (operadores lógicos nativos)
    # -------------------------------------------------------------------------
    # Carrocería amarilla: R y G altos, B bajo
    carroceria = (R > 0.55) & (G > 0.45) & (B < 0.45)

    # Ruedas oscuras: intensidades bajas en los 3 canales
    ruedas_oscuras = (R < 0.30) & (G < 0.30) & (B < 0.30)

    # Dilatar zona de carrocería para capturar únicamente las ruedas contiguas al auto
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    zona_auto = cv2.dilate(carroceria.astype(np.uint8), kernel_dilatacion) > 0
    ruedas = ruedas_oscuras & zona_auto

    # Unión lógica nativa estilo MATLAB: carroceria | ruedas
    bw_auto_cruda = carroceria | ruedas

    # Cierre morfológico (strel disk/ellipse, radio = 2 como en MATLAB)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bw_auto_cerrada = cv2.morphologyEx(bw_auto_cruda.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close) > 0

    # Relleno de huecos (imfill)
    bw_auto_solida = imfill_holes(bw_auto_cerrada)

    # Apertura morfológica para cortar la línea de 1 px hacia el marco
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw_auto_limpia = cv2.morphologyEx(bw_auto_solida.astype(np.uint8), cv2.MORPH_OPEN, kernel_open) > 0

    # Conservar la componente conexa principal del auto
    num_labels_a, labels_a, stats_a, _ = cv2.connectedComponentsWithStats(bw_auto_limpia.astype(np.uint8))
    if num_labels_a > 1:
        idx_auto = 1 + np.argmax(stats_a[1:, cv2.CC_STAT_AREA])
        bwAuto = (labels_a == idx_auto)
    else:
        bwAuto = bw_auto_limpia

    # -------------------------------------------------------------------------
    # B. bwCasa: Fachada y tejado rojo/ladrillo (Resta de canales R - max(G,B))
    # -------------------------------------------------------------------------
    # En la casa roja, el canal R predomina fuertemente sobre el verde y el azul
    casa_roja = (R - np.maximum(G, B) > 0.15) & (R > 0.45)

    # Negación y exclusión lógica nativa estilo MATLAB: casa_roja & (~bwAuto)
    casa_limpia = casa_roja & (~bwAuto)

    # Cierre morfológico y relleno de puertas/ventanas con imfill
    kernel_casa = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bw_casa_cerrada = cv2.morphologyEx(casa_limpia.astype(np.uint8), cv2.MORPH_CLOSE, kernel_casa) > 0
    bw_casa_solida = imfill_holes(bw_casa_cerrada)

    # Conservar la componente conexa principal de la casa
    num_labels_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(bw_casa_solida.astype(np.uint8))
    if num_labels_c > 1:
        idx_casa = 1 + np.argmax(stats_c[1:, cv2.CC_STAT_AREA])
        bwCasa = (labels_c == idx_casa)
    else:
        bwCasa = bw_casa_solida

    return bwAuto, bwCasa


# ==============================================================================
# 3. FUNCIÓN 'regionprops' IDÉNTICA A MATLAB
# ==============================================================================
class RegionPropsStruct:
    """Estructura idéntica al struct devuelto por regionprops en MATLAB."""
    def __init__(self):
        self.Area = 0.0
        self.Perimeter = 0.0
        self.Centroid = (0.0, 0.0)
        self.BoundingBox = [0.0, 0.0, 0.0, 0.0]
        self.Image = np.zeros((1, 1), dtype=bool)
        self.Contorno = None
        self.NumRegiones = 0


def regionprops(BW, *propiedades):
    """
    Función homóloga a 'regionprops' de MATLAB:
    prop = regionprops(BW, 'Area', 'Perimeter', 'Centroid', 'BoundingBox', 'Image')
    Calcula Área, Perímetro y Centroide continuo (X, Y) mediante Momentos Espaciales discretos:
    m00 = sum(I), m10 = sum(x*I), m01 = sum(y*I) -> Centroide = (m10/m00, m01/m00).
    """
    bw_u8 = BW.astype(np.uint8) * 255 if BW.dtype == bool else (BW > 0).astype(np.uint8) * 255

    contornos, _ = cv2.findContours(bw_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > 50]

    props = RegionPropsStruct()
    props.NumRegiones = len(contornos_validos)

    if props.NumRegiones == 0:
        return props

    cnt = max(contornos_validos, key=cv2.contourArea)
    props.Contorno = cnt
    props.Area = float(cv2.contourArea(cnt))
    props.Perimeter = float(cv2.arcLength(cnt, True))

    x, y, w, h = cv2.boundingRect(cnt)
    props.BoundingBox = [float(x), float(y), float(w), float(h)]

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2.0, y + h / 2.0
    props.Centroid = (float(cx), float(cy))

    # Subimagen binaria recortada ('Image' en MATLAB)
    props.Image = BW[y:y+h, x:x+w]

    return props


# ==============================================================================
# 4. CÓDIGO DE CADENA (8-VECINDAD)
# ==============================================================================
def codigo_cadena_8(contorno):
    """
    Recorre el perfil perimetral mediante 8-vecinos mapeando desplazamientos a [0..7].
    0: Este (+1, 0)       4: Oeste (-1, 0)
    1: Noreste (+1, -1)   5: Suroeste (-1, +1)
    2: Norte (0, -1)      6: Sur (0, +1)
    3: Noroeste (-1, -1)  7: Sureste (+1, +1)
    """
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
# 5. CÁLCULO DEL CUADRADO ENVOLVENTE (lado = max(cajaAuto(3:4)))
# ==============================================================================
def calcular_cuadrado_envolvente(cajaAuto, ancho_img, alto_img):
    """
    Construye el cuadrado envolvente simétrico centrado (Líneas 108-121 de MATLAB).
    """
    lado = max(cajaAuto[2], cajaAuto[3])
    centro_caja_x = cajaAuto[0] + cajaAuto[2] / 2.0
    centro_caja_y = cajaAuto[1] + cajaAuto[3] / 2.0

    esq_x = centro_caja_x - lado / 2.0
    esq_y = centro_caja_y - lado / 2.0

    # Límite en Python (base 0, a diferencia del 0.5 de MATLAB en base 1)
    esq_x = max(0.0, min(esq_x, ancho_img - lado))
    esq_y = max(0.0, min(esq_y, alto_img - lado))

    return [esq_x, esq_y, lado, lado], lado


# ==============================================================================
# 6. PIPELINE PRINCIPAL DE PROCESAMIENTO
# ==============================================================================
def procesar_imagen_completa(ruta_archivo):
    """
    Ejecuta todo el pipeline matemático del deber y devuelve un diccionario con resultados.
    """
    img_bgr = cv2.imread(ruta_archivo)
    if img_bgr is None:
        raise ValueError("No se pudo cargar el archivo como una imagen válida.")

    alto, ancho, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Descomposición en double y uint8 (como en MATLAB: I_double = im2double(I))
    img_suavizada = aplicar_suavizado_gaussiano(img_rgb, ksize=5, sigma=1.0)
    I_double = img_suavizada.astype(np.float64) / 255.0

    R = I_double[:, :, 0]
    G = I_double[:, :, 1]
    B = I_double[:, :, 2]

    R_uint8 = img_rgb[:, :, 0]
    G_uint8 = img_rgb[:, :, 1]
    B_uint8 = img_rgb[:, :, 2]

    # 2. Segmentación usando resta de canales y operadores booleanos
    bwAuto, bwCasa = segmentar_objetos_canales(I_double)
    bwTotal = bwCasa | bwAuto

    # 3. Llamada a regionprops idéntica a MATLAB:
    propCasa = regionprops(bwCasa, 'Area', 'Perimeter', 'Centroid', 'BoundingBox', 'Image')
    propAuto = regionprops(bwAuto, 'Area', 'Perimeter', 'Centroid', 'BoundingBox', 'Image')

    # Validación idéntica a la línea 95 del docente:
    assert propCasa.NumRegiones == 1 and propAuto.NumRegiones == 1, (
        f"Debe existir una sola region conectada por objeto. "
        f"Detectadas: Casa={propCasa.NumRegiones}, Auto={propAuto.NumRegiones}"
    )

    # 4. Código de Cadena (8-vecinos)
    cadena_casa = codigo_cadena_8(propCasa.Contorno)
    cadena_auto = codigo_cadena_8(propAuto.Contorno)

    # 5. Cuadrado envolvente del auto
    cuadradoAuto, lado = calcular_cuadrado_envolvente(propAuto.BoundingBox, ancho, alto)

    # 6. Contornos digitales para visualización
    img_contornos = np.zeros((alto, ancho), dtype=np.uint8)
    if propCasa.Contorno is not None:
        cv2.drawContours(img_contornos, [propCasa.Contorno], -1, 255, 2)
    if propAuto.Contorno is not None:
        cv2.drawContours(img_contornos, [propAuto.Contorno], -1, 255, 2)

    return {
        "img_rgb": img_rgb,
        "alto": alto,
        "ancho": ancho,
        "R": R, "G": G, "B": B,
        "R_uint8": R_uint8, "G_uint8": G_uint8, "B_uint8": B_uint8,
        "bwCasa": bwCasa,
        "bwAuto": bwAuto,
        "bwTotal": bwTotal,
        "img_contornos": img_contornos,
        "propCasa": propCasa,
        "propAuto": propAuto,
        "cadena_casa": cadena_casa,
        "cadena_auto": cadena_auto,
        "cuadradoAuto": cuadradoAuto,
        "lado": lado
    }