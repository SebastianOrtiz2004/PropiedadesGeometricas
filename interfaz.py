import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog

# Importamos el motor matemático desde el módulo de lógica pura
from propiedadesGeometricas import procesar_imagen_completa


# ==============================================================================
# 1. EXPLORADOR NATIVO DE ARCHIVOS
# ==============================================================================
def seleccionar_archivo_imagen():
    """Abre el explorador de archivos del sistema operativo."""
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
# 2. REPORTE EN CONSOLA (FORMATO DE TABLA DE MATLAB)
# ==============================================================================
def imprimir_reporte_matlab(datos):
    """Muestra en la terminal la tabla de medidas y los códigos de cadena."""
    propCasa = datos["propCasa"]
    propAuto = datos["propAuto"]

    print("\n" + "=" * 80)
    print("        MEDIDAS DE LAS REGIONES DELIMITADAS (regionprops / Hipermatriz)")
    print("=" * 80)
    print(f"{'Objeto':<10} {'Area_px2':<15} {'Perimetro_px':<15} {'Centroide_X':<18} {'Centroide_Y':<18}")
    print("-" * 80)
    print(f"{'Casa':<10} {propCasa.Area:<15.1f} {propCasa.Perimeter:<15.2f} {propCasa.Centroid[0]:<18.2f} {propCasa.Centroid[1]:<18.2f}")
    print(f"{'Auto':<10} {propAuto.Area:<15.1f} {propAuto.Perimeter:<15.2f} {propAuto.Centroid[0]:<18.2f} {propAuto.Centroid[1]:<18.2f}")
    print("=" * 80 + "\n")

    cadena_casa = datos["cadena_casa"]
    cadena_auto = datos["cadena_auto"]
    print(f"Código de Cadena Casa (primeros 30 dígitos): {cadena_casa[:30]} ... Total: {len(cadena_casa)} eslabones")
    print(f"Código de Cadena Auto (primeros 30 dígitos): {cadena_auto[:30]} ... Total: {len(cadena_auto)} eslabones\n")


# ==============================================================================
# 3. INTERFAZ GRÁFICA MULTIPANEL CON TEMA OSCURO
# ==============================================================================
def mostrar_interfaz_grafica(datos):
    """Construye y despliega la cuadrícula 2x3 con tema oscuro y líneas continuas."""
    img_rgb = datos["img_rgb"]
    alto, ancho = datos["alto"], datos["ancho"]
    bwTotal = datos["bwTotal"]
    img_contornos = datos["img_contornos"]
    propCasa = datos["propCasa"]
    propAuto = datos["propAuto"]
    cuadradoAuto = datos["cuadradoAuto"]
    lado = datos["lado"]

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
    ax2.imshow(bwTotal, cmap='gray')
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

    # Panel 6: MONTAJE FINAL SOBRE LA IMAGEN ORIGINAL (Líneas rectas continuas)
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
    cajaAuto = propAuto.BoundingBox
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

    # Centroides con borde de contraste
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

    # Etiqueta Casa: Arriba del tejado
    y_texto_casa = max(16, cajaCasa[1] - 8)
    ax6.text(
        cajaCasa[0], y_texto_casa,
        f"Casa: A={propCasa.Area:.0f} px² | P={propCasa.Perimeter:.1f} px",
        color='#00ff66', fontsize=7, weight='bold',
        bbox=dict(facecolor='#121212', alpha=0.65, edgecolor='#00ff66', boxstyle='round,pad=0.15')
    )

    # Etiqueta Auto: Debajo del vehículo
    y_texto_auto = min(alto - 10, cajaAuto[1] + cajaAuto[3] + 16)
    ax6.text(
        cajaAuto[0], y_texto_auto,
        f"Auto: A={propAuto.Area:.0f} px² | P={propAuto.Perimeter:.1f} px",
        color='#ffd700', fontsize=7, weight='bold',
        bbox=dict(facecolor='#121212', alpha=0.65, edgecolor='#ffd700', boxstyle='round,pad=0.15')
    )

    # Leyenda situada fuera del panel para dejar el lienzo 100% visible
    ax6.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3,
        facecolor='#181818', edgecolor='#383838', labelcolor='white',
        fontsize=6.8, framealpha=0.9, handletextpad=0.4, columnspacing=0.8
    )

    # Ajuste manual de márgenes (evita cualquier UserWarning en la terminal)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.09, wspace=0.15, hspace=0.25)
    plt.show()


# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL
# ==============================================================================
def main():
    ruta = seleccionar_archivo_imagen()
    if not ruta or not os.path.exists(ruta):
        print("Operación cancelada: No se seleccionó ninguna imagen.")
        return

    try:
        datos = procesar_imagen_completa(ruta)
        imprimir_reporte_matlab(datos)
        mostrar_interfaz_grafica(datos)
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")


if __name__ == "__main__":
    main()