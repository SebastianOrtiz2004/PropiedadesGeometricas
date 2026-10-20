import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog


# ==============================================================================
# 1. CARGA DINÁMICA DE LA IMAGEN MEDIANTE VENTANA DE ARCHIVOS
# ==============================================================================
def cargar_imagen():
    """Abre el explorador de archivos del sistema operativo para seleccionar la imagen."""
    raiz = tk.Tk()
    raiz.withdraw()
    raiz.attributes('-topmost', True)
    ruta = filedialog.askopenfilename(
        title="Selecciona la imagen para procesar",
        filetypes=[
            ("Imágenes compatibles", "*.jpg *.jpeg *.png *.bmp *.webp *.tif"),
            ("Todos los archivos", "*.*")
        ]
    )
    raiz.destroy()
    return ruta


# ==============================================================================
# 2. MITIGACIÓN DE RUIDO (SUAVIZADO) Y RELLENO DE HUECOS ('imfill' DE MATLAB)
# ==============================================================================
def aplicar_suavizado_gaussiano(imagen, ksize=5, sigma=1.0):
    """
    Mitiga el ruido de alta frecuencia antes de binarizar para evitar que
    regionprops detecte píxeles aislados como objetos espurios (Audio 2 del docente).
    """
    return cv2.GaussianBlur(imagen, (ksize, ksize), sigma)


def imfill_holes(mascara_binaria):
    """
    Emula textualmente 'imfill(BW, "holes")' de MATLAB.
    Inunda el fondo exterior desde (0,0) e invierte el resultado para sellar
    todas las cavidades internas (ventanas, faros), garantizando una silueta sólida.
    """
    alto, ancho = mascara_binaria.shape
    flood = mascara_binaria.copy()
    mascara_auxiliar = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)
    cv2.floodFill(flood, mascara_auxiliar, (0, 0), 255)
    fondo_invertido = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mascara_binaria, fondo_invertido)


# ==============================================================================
# 3. SEGMENTACIÓN DE MÁSCARAS BINARIAS (bwAuto Y bwCasa)
# ==============================================================================
def segmentar_objetos(img_rgb):
    """
    Genera las máscaras binarias independientes bwAuto y bwCasa en blanco y negro puro (0 y 255).
    Aplica el suavizado gaussiano previo para mitigar ruido y operaciones morfológicas
    para garantizar que cada objeto constituya una sola región conexa sólida.
    """
    alto, ancho, _ = img_rgb.shape

    # 1. Suavizado gaussiano previo para mitigar ruido cromático en bordes
    img_suavizada = aplicar_suavizado_gaussiano(img_rgb, ksize=5, sigma=1.0)
    hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_RGB2HSV)

    # -------------------------------------------------------------------------
    # A. bwAuto: Carrocería amarilla + ruedas oscuras + imfill
    # -------------------------------------------------------------------------
    mask_amarillo = cv2.inRange(hsv, (15, 80, 80), (36, 255, 255))
    mask_oscuros = cv2.inRange(hsv, (0, 0, 0), (180, 255, 75))

    # Dilatación para capturar únicamente las llantas contiguas a la carrocería
    zona_inferior = cv2.dilate(mask_amarillo, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    ruedas = cv2.bitwise_and(mask_oscuros, zona_inferior)
    bw_auto_cruda = cv2.bitwise_or(mask_amarillo, ruedas)

    # Cierre morfológico para sellar la base de las ruedas al chasis
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bw_auto_cerrada = cv2.morphologyEx(bw_auto_cruda, cv2.MORPH_CLOSE, kernel_close)

    # Relleno de ventanas y cavidades interiores (imfill)
    bw_auto_solida = imfill_holes(bw_auto_cerrada)

    # Apertura morfológica: elimina la línea parásita de 1 px hacia el marco de la puerta
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw_auto_limpia = cv2.morphologyEx(bw_auto_solida, cv2.MORPH_OPEN, kernel_open)

    # Conservar la componente conexa del vehículo
    num_labels_a, labels_a, stats_a, _ = cv2.connectedComponentsWithStats(bw_auto_limpia)
    if num_labels_a > 1:
        idx_auto = 1 + np.argmax(stats_a[1:, cv2.CC_STAT_AREA])
        bwAuto = np.uint8(labels_a == idx_auto) * 255
    else:
        bwAuto = bw_auto_limpia

    # -------------------------------------------------------------------------
    # B. bwCasa: Fachada y tejado rojo/ladrillo
    # -------------------------------------------------------------------------
    mask_rojo1 = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
    mask_rojo2 = cv2.inRange(hsv, (165, 70, 70), (180, 255, 255))
    mask_casa_color = cv2.bitwise_or(mask_rojo1, mask_rojo2)

    # Excluir el área que ya pertenece al vehículo
    mask_casa_limpia = cv2.bitwise_and(mask_casa_color, cv2.bitwise_not(bwAuto))

    # Cierre morfológico y sellado de huecos (puerta y ventanas)
    kernel_casa = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bw_casa_cerrada = cv2.morphologyEx(mask_casa_limpia, cv2.MORPH_CLOSE, kernel_casa)
    bw_casa_solida = imfill_holes(bw_casa_cerrada)

    # Conservar la componente conexa de la casa
    num_labels_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(bw_casa_solida)
    if num_labels_c > 1:
        idx_casa = 1 + np.argmax(stats_c[1:, cv2.CC_STAT_AREA])
        bwCasa = np.uint8(labels_c == idx_casa) * 255
    else:
        bwCasa = bw_casa_solida

    return bwAuto, bwCasa


# ==============================================================================
# 4. REGIONPROPS Y GENERACIÓN DE LA ESTRUCTURA ("HIPERMATRIZ")
# ==============================================================================
class RegionPropItem:
    """
    Estructura individual devuelta por regionprops en MATLAB.
    Calcula Área, Perímetro y Centroide continuo (X, Y) mediante Momentos Espaciales discretos:
    m00 = sum(I), m10 = sum(x*I), m01 = sum(y*I) -> Centroide = (m10/m00, m01/m00).
    Registra NumRegiones para la aserción de control de calidad.
    """
    def __init__(self, mascara_binaria, nombre_etiqueta="Objeto"):
        self.Nombre = nombre_etiqueta
        contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filtrar ruido menor a 50 px para evaluar componentes conexas reales
        contornos_validos = [c for c in contornos if cv2.contourArea(c) > 50]
        self.NumRegiones = len(contornos_validos)

        if self.NumRegiones == 0:
            self.Area = 0.0
            self.Perimeter = 0.0
            self.Centroid = (0.0, 0.0)
            self.BoundingBox = [0.0, 0.0, 0.0, 0.0]
            self.Image = np.zeros((1, 1), dtype=np.uint8)
            self.Contorno = None
            return

        cnt = max(contornos_validos, key=cv2.contourArea)
        self.Contorno = cnt
        self.Area = float(cv2.contourArea(cnt))
        self.Perimeter = float(cv2.arcLength(cnt, True))

        # BoundingBox de OpenCV: [x, y, ancho, alto]
        x, y, w, h = cv2.boundingRect(cnt)
        self.BoundingBox = [float(x), float(y), float(w), float(h)]

        # Centroide continuo (X = columna, Y = fila)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + w / 2.0, y + h / 2.0
        self.Centroid = (float(cx), float(cy))

        # Subimagen binaria 2D recortada ('Image' en MATLAB)
        self.Image = mascara_binaria[y:y+h, x:x+w]


# ==============================================================================
# 5. CÓDIGO DE CADENA DE FREEMAN (8-VECINDAD)
# ==============================================================================
def freeman_chain_code_8(contorno):
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
# 6. PIPELINE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
def ejecutar_deber():
    ruta = cargar_imagen()
    if not ruta or not os.path.exists(ruta):
        print("Operación cancelada: No se seleccionó ninguna imagen.")
        return

    # --------------------------------------------------------------------------
    # PASO 1: CARGA RGB Y PRESERVACIÓN INTACTA DE LA IMAGEN ORIGINAL
    # --------------------------------------------------------------------------
    img_bgr = cv2.imread(ruta)
    if img_bgr is None:
        print("Error: El archivo no es una imagen válida.")
        return

    alto, ancho, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Imagen original intacta para el montaje

    # --------------------------------------------------------------------------
    # PASO 2: DESCOMPOSICIÓN EN R, G, B (uint8) Y CONVERSIÓN A DOUBLE (float64)
    # --------------------------------------------------------------------------
    R_uint8, G_uint8, B_uint8 = cv2.split(img_rgb)
    R_double = R_uint8.astype(np.float64) / 255.0
    G_double = G_uint8.astype(np.float64) / 255.0
    B_double = B_uint8.astype(np.float64) / 255.0

    # Conversión a escala de grises y mitigación de ruido (filtro Gaussiano)
    gris = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gris_suavizada = aplicar_suavizado_gaussiano(gris, ksize=5, sigma=1.0)

    print(f"Dimensiones de imagen: {alto} filas x {ancho} columnas")
    print(f"Canales RGB descompuestos en uint8 y convertidos a float64 (double)")

    # --------------------------------------------------------------------------
    # PASO 3: SEGMENTACIÓN DE bwAuto Y bwCasa
    # --------------------------------------------------------------------------
    bwAuto, bwCasa = segmentar_objetos(img_rgb)
    bw_total = cv2.bitwise_or(bwCasa, bwAuto)  # Máscara global (0 y 255)

    # --------------------------------------------------------------------------
    # PASO 4: ARREGLO DE PROPIEDADES GEOMÉTRICAS ("HIPERMATRIZ" / REGIONPROPS)
    # --------------------------------------------------------------------------
    propCasa = RegionPropItem(bwCasa, "Casa")
    propAuto = RegionPropItem(bwAuto, "Auto")
    hipermatriz_propiedades = [propCasa, propAuto]

    # Validación idéntica a la línea 95 del docente:
    # assert(numel(propCasa) == 1 && numel(propAuto) == 1, 'Debe existir una sola region conectada por objeto.')
    assert propCasa.NumRegiones == 1 and propAuto.NumRegiones == 1, (
        f"Debe existir una sola region conectada por objeto. "
        f"Detectadas: Casa={propCasa.NumRegiones}, Auto={propAuto.NumRegiones}"
    )

    # Consola formateada con las propiedades geométricas extraídas
    print("\n" + "=" * 80)
    print("        MEDIDAS DE LAS REGIONES DELIMITADAS (regionprops / Hipermatriz)")
    print("=" * 80)
    print(f"{'Objeto':<10} {'Area_px2':<15} {'Perimetro_px':<15} {'Centroide_X':<18} {'Centroide_Y':<18}")
    print("-" * 80)
    for p in hipermatriz_propiedades:
        print(f"{p.Nombre:<10} {p.Area:<15.1f} {p.Perimeter:<15.2f} {p.Centroid[0]:<18.2f} {p.Centroid[1]:<18.2f}")
    print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # PASO 5: CÓDIGO DE CADENA DE FREEMAN (CASA Y AUTO)
    # --------------------------------------------------------------------------
    cadena_casa = freeman_chain_code_8(propCasa.Contorno)
    cadena_auto = freeman_chain_code_8(propAuto.Contorno)
    print(f"Código Freeman Casa (primeros 30 dígitos): {cadena_casa[:30]} ... Total: {len(cadena_casa)} eslabones")
    print(f"Código Freeman Auto (primeros 30 dígitos): {cadena_auto[:30]} ... Total: {len(cadena_auto)} eslabones\n")

    # --------------------------------------------------------------------------
    # PASO 6: CÁLCULO DEL CUADRADO ENVOLVENTE (lado = max(cajaAuto(3:4)))
    # --------------------------------------------------------------------------
    cajaAuto = propAuto.BoundingBox  # [x, y, ancho, alto]
    lado = max(cajaAuto[2], cajaAuto[3])  # Dimensión dominante para hacer el cuadrado

    # Centro geométrico de la caja rectangular
    centro_caja_x = cajaAuto[0] + cajaAuto[2] / 2.0
    centro_caja_y = cajaAuto[1] + cajaAuto[3] / 2.0

    # Nueva esquina superior izquierda centrada
    esq_x = centro_caja_x - lado / 2.0
    esq_y = centro_caja_y - lado / 2.0

    # En Python (base 0) el límite exterior es [0.0, dimension - lado]
    esq_x = max(0.0, min(esq_x, ancho - lado))
    esq_y = max(0.0, min(esq_y, alto - lado))
    cuadradoAuto = [esq_x, esq_y, lado, lado]

    # Contornos perimetrales para visualización
    img_contornos = np.zeros((alto, ancho), dtype=np.uint8)
    if propCasa.Contorno is not None:
        cv2.drawContours(img_contornos, [propCasa.Contorno], -1, 255, 2)
    if propAuto.Contorno is not None:
        cv2.drawContours(img_contornos, [propAuto.Contorno], -1, 255, 2)

    # --------------------------------------------------------------------------
    # PASO 7: INTERFAZ GRÁFICA EN ORDEN CRONOLÓGICO (2 filas x 3 columnas)
    # --------------------------------------------------------------------------
    plt.rcParams['toolbar'] = 'toolbar2'
    fig = plt.figure(figsize=(16, 8.5), facecolor='#161616')
    fig.canvas.manager.set_window_title('PDI - Determinación de Propiedades Geométricas')
    gs = fig.add_gridspec(2, 3, wspace=0.18, hspace=0.25)

    def estilizar_panel(ax, titulo):
        ax.set_facecolor('#101010')
        ax.set_title(titulo, color='#ffffff', fontsize=9.5, fontweight='bold', pad=6)
        ax.axis('off')

    # Panel 1: Imagen Original RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    estilizar_panel(ax1, "1. Imagen Original (RGB)")

    # Panel 2: Imagen Binarizada Global (Blanco y Negro puro / imfill)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(bw_total, cmap='gray')
    estilizar_panel(ax2, "2. Imagen Binarizada Sólida (0 y 1 / imfill)")

    # Panel 3: Contornos Digitales de 8-Vecindad
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_contornos, cmap='gray')
    estilizar_panel(ax3, "3. Contornos 8-Vecinos (Código de Cadena)")

    # Panel 4: Subimagen Binaria de la Casa (propCasa.Image)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(propCasa.Image, cmap='gray')
    estilizar_panel(ax4, f"4. Subimagen 'Image' Casa (Binaria)\n({propCasa.Image.shape[0]}x{propCasa.Image.shape[1]} px)")

    # Panel 5: Subimagen Binaria del Auto (propAuto.Image / Figure 3 del docente)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(propAuto.Image, cmap='gray')
    estilizar_panel(ax5, f"5. Subimagen 'Image' Auto (Binaria)\n({propAuto.Image.shape[0]}x{propAuto.Image.shape[1]} px)")

    # Panel 6: MONTAJE FINAL SOBRE LA IMAGEN ORIGINAL (Último paso cronológico)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(img_rgb)
    estilizar_panel(ax6, "6. Montaje Final: BoundingBoxes, Cuadrado y Centroides")

    # BoundingBox Casa: LÍNEA RECTA SÓLIDA VERDE
    cajaCasa = propCasa.BoundingBox
    rect_casa = patches.Rectangle(
        (cajaCasa[0], cajaCasa[1]), cajaCasa[2], cajaCasa[3],
        linewidth=1.8, edgecolor='#00ff66', facecolor='none', linestyle='-', label='BoundingBox Casa'
    )
    ax6.add_patch(rect_casa)

    # BoundingBox Auto: LÍNEA RECTA SÓLIDA AZUL/CYAN
    rect_auto = patches.Rectangle(
        (cajaAuto[0], cajaAuto[1]), cajaAuto[2], cajaAuto[3],
        linewidth=1.8, edgecolor='#00d4ff', facecolor='none', linestyle='-', label='BoundingBox Auto'
    )
    ax6.add_patch(rect_auto)

    # Cuadrado Envolvente: LÍNEA RECTA SÓLIDA AMARILLA (lado = max)
    cuad_auto = patches.Rectangle(
        (cuadradoAuto[0], cuadradoAuto[1]), cuadradoAuto[2], cuadradoAuto[3],
        linewidth=1.8, edgecolor='#ffd700', facecolor='none', linestyle='-', label=f'Cuadrado ({int(lado)}x{int(lado)} px)'
    )
    ax6.add_patch(cuad_auto)

    # Centroides con borde de contraste para máxima visibilidad
    ax6.plot(
        propCasa.Centroid[0], propCasa.Centroid[1],
        marker='o', color='#00ff66', markeredgecolor='black', markeredgewidth=1.2,
        markersize=7, label='Centroide Casa'
    )
    ax6.plot(
        propAuto.Centroid[0], propAuto.Centroid[1],
        marker='o', color='#ff3333', markeredgecolor='white', markeredgewidth=1.2,
        markersize=7, label='Centroide Auto'
    )

    # Etiqueta Casa: Arriba del tejado en texto compacto y translúcido
    y_texto_casa = max(16, cajaCasa[1] - 8)
    ax6.text(
        cajaCasa[0], y_texto_casa,
        f"Casa: A={propCasa.Area:.0f} px² | P={propCasa.Perimeter:.1f} px",
        color='#00ff66', fontsize=7, weight='bold',
        bbox=dict(facecolor='#121212', alpha=0.65, edgecolor='#00ff66', boxstyle='round,pad=0.15')
    )

    # Etiqueta Auto: Debajo del vehículo en la calzada, sin tapar la casa
    y_texto_auto = min(alto - 10, cajaAuto[1] + cajaAuto[3] + 16)
    ax6.text(
        cajaAuto[0], y_texto_auto,
        f"Auto: A={propAuto.Area:.0f} px² | P={propAuto.Perimeter:.1f} px",
        color='#ffd700', fontsize=7, weight='bold',
        bbox=dict(facecolor='#121212', alpha=0.65, edgecolor='#ffd700', boxstyle='round,pad=0.15')
    )

    # Leyenda situada TOTALMENTE FUERA de la imagen (abajo del panel) para dejar el lienzo 100% libre
    ax6.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3,
        facecolor='#181818', edgecolor='#383838', labelcolor='white',
        fontsize=6.8, framealpha=0.9, handletextpad=0.4, columnspacing=0.8
    )

    # Ajuste manual de márgenes (reserva espacio inferior para la leyenda exterior de 2 filas)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.09, wspace=0.15, hspace=0.25)
    plt.show()


if __name__ == "__main__":
    ejecutar_deber()