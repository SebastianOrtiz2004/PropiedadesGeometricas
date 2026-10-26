import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog, messagebox

from propiedadesGeometricas import procesar_imagen_completa


# ==============================================================================
# 1. VENTANA DE SELECCIÓN DE IMAGEN Y MODO DE OPERACIÓN
# ==============================================================================
def solicitar_modo_y_archivo():
    """Permite elegir el archivo de imagen y la opción de segmentación deseada."""
    config = {"ruta": None, "modo": "otsu"}

    ventana = tk.Tk()
    ventana.title("PDI - Determinación de Propiedades Geométricas")
    ventana.geometry("480x280")
    ventana.configure(bg="#1e1e1e")
    ventana.attributes('-topmost', True)

    tk.Label(
        ventana, text="Procesamiento Digital de Imágenes",
        font=("Arial", 11, "bold"), fg="#ffffff", bg="#1e1e1e"
    ).pack(pady=(15, 3))

    tk.Label(
        ventana, text="Selecciona la técnica de segmentación a aplicar:",
        font=("Arial", 9), fg="#9cdcfe", bg="#1e1e1e"
    ).pack(pady=(0, 10))

    frame_opciones = tk.LabelFrame(
        ventana, text=" Modo de Operación ", font=("Arial", 9, "bold"),
        fg="#4ec9b0", bg="#1e1e1e", padx=12, pady=10
    )
    frame_opciones.pack(fill="x", padx=20, pady=4)

    var_modo = tk.StringVar(value="otsu")

    tk.Radiobutton(
        frame_opciones,
        text="1. Automático con Otsu (Monedas, dados o figuras sobre fondo)",
        variable=var_modo, value="otsu", bg="#1e1e1e", fg="#ffffff",
        selectcolor="#111111", activebackground="#1e1e1e", activeforeground="#4ec9b0"
    ).pack(anchor="w", pady=2)

    tk.Radiobutton(
        frame_opciones,
        text="2. Señalar Objetos con Clic (Escenas complejas, autos, casas)",
        variable=var_modo, value="clic", bg="#1e1e1e", fg="#ffffff",
        selectcolor="#111111", activebackground="#1e1e1e", activeforeground="#4ec9b0"
    ).pack(anchor="w", pady=2)

    def confirmar():
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen para análisis",
            filetypes=[
                ("Imágenes compatibles", "*.jpg *.jpeg *.png *.bmp *.webp *.tif"),
                ("Todos los archivos", "*.*")
            ]
        )
        if ruta:
            config["ruta"] = ruta
            config["modo"] = var_modo.get()
            ventana.destroy()

    tk.Button(
        ventana, text="Cargar Imagen y Continuar", command=confirmar,
        bg="#0e639c", fg="#ffffff", font=("Arial", 10, "bold"),
        padx=15, pady=6, relief="flat", cursor="hand2"
    ).pack(pady=15)

    ventana.mainloop()
    return config["ruta"], config["modo"]


# ==============================================================================
# 2. CAPTURA INTERACTIVA DE CLICS (SOLO PARA MODO 2)
# ==============================================================================
def capturar_clics_usuario(ruta_imagen):
    """Abre la imagen para que el usuario señale los objetos que desea delimitar."""
    img_bgr = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    alto, ancho, _ = img_rgb.shape

    escala = min(1.0, 900.0 / max(alto, ancho))
    ancho_v = int(ancho * escala)
    alto_v = int(alto * escala)
    img_preview = cv2.resize(img_rgb, (ancho_v, alto_v))

    puntos_reales = []

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='#1e1e1e')
    fig.canvas.manager.set_window_title("Haz clic sobre cada objeto que deseas delimitar")
    ax.imshow(img_preview)
    ax.set_facecolor('#1e1e1e')
    ax.set_title(
        "Haz CLIC sobre cada objeto a medir (ej. auto azul, auto rojo).\nAl terminar, presiona ENTER o cierra esta ventana.",
        color='#ffffff', fontsize=10, pad=10
    )
    ax.axis('off')

    def al_hacer_clic(event):
        if event.xdata is not None and event.ydata is not None:
            x_orig = event.xdata / escala
            y_orig = event.ydata / escala
            puntos_reales.append((x_orig, y_orig))
            idx = len(puntos_reales)
            ax.plot(event.xdata, event.ydata, 'o', color='#00ff66', markeredgecolor='black', markersize=8)
            ax.text(event.xdata + 5, event.ydata - 5, f"Obj {idx}", color='#00ff66', fontsize=9, weight='bold')
            fig.canvas.draw()

    def al_presionar_tecla(event):
        if event.key in ['enter', 'escape']:
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', al_hacer_clic)
    fig.canvas.mpl_connect('key_press_event', al_presionar_tecla)
    plt.tight_layout()
    plt.show()

    return puntos_reales


# ==============================================================================
# 3. REPORTE EN TERMINAL (TABLA DE REGIONPROPS ESTILO MATLAB)
# ==============================================================================
def imprimir_reporte_matlab(datos):
    """Muestra en la terminal la tabla de medidas y los códigos de Freeman."""
    objetos = datos["objetos"]

    print("\n" + "=" * 84)
    print(f"   MEDIDAS DE LAS REGIONES DELIMITADAS (regionprops) | Modo: {datos['modo_usado'].upper()}")
    print("=" * 84)
    print(f"{'Objeto':<12} {'Area [px²]':<15} {'Perímetro [px]':<16} {'Centroide (X, Y)':<22} {'Lado Cuadrado'}")
    print("-" * 84)
    for p in objetos:
        cx, cy = p.Centroid
        print(f"{p.Nombre:<12} {p.Area:<15.1f} {p.Perimeter:<16.2f} ({cx:.1f}, {cy:.1f}){'':<7} {p.LadoCuadrado:.1f} px")
    print("=" * 84 + "\n")

    for p in objetos:
        cadena = p.CodigoFreeman
        print(f"Código Freeman {p.Nombre} (primeros 30 dígitos): {cadena[:30]} ... Total: {len(cadena)} eslabones")
    print()


# ==============================================================================
# 4. VISUALIZACIÓN EN MATPLOTLIB (6 PANELES CON TEMA OSCURO)
# ==============================================================================
def mostrar_interfaz_grafica(datos):
    """Despliega los 6 paneles en orden cronológico con líneas rectas continuas."""
    img_rgb = datos["img_rgb"]
    alto, ancho = datos["alto"], datos["ancho"]
    bwTotal = datos["bwTotal"]
    img_contornos = datos["img_contornos"]
    objetos = datos["objetos"]

    plt.rcParams['toolbar'] = 'toolbar2'
    fig = plt.figure(figsize=(16, 8.5), facecolor='#161616')
    fig.canvas.manager.set_window_title(f'PDI - Determinación de Propiedades Geométricas ({len(objetos)} objeto(s))')
    gs = fig.add_gridspec(2, 3, wspace=0.18, hspace=0.25)

    def estilizar_panel(ax, titulo):
        ax.set_facecolor('#101010')
        ax.set_title(titulo, color='#ffffff', fontsize=9.5, fontweight='bold', pad=6)
        ax.axis('off')

    # Panel 1: Imagen Original RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    estilizar_panel(ax1, "1. Imagen Original (RGB)")

    # Panel 2: Imagen Binarizada Global Sólida (imfill)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(bwTotal, cmap='gray')
    estilizar_panel(ax2, "2. Imagen Binarizada Sólida (0 y 1 / imfill)")

    # Panel 3: Contornos Digitales de 8-Vecindad
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_contornos, cmap='gray')
    estilizar_panel(ax3, f"3. Contornos 8-Vecinos ({len(objetos)} detectados)")

    # Panel 4: Subimagen Binaria del Objeto 1
    ax4 = fig.add_subplot(gs[1, 0])
    if len(objetos) >= 1:
        obj1 = objetos[0]
        ax4.imshow(obj1.Image, cmap='gray')
        estilizar_panel(ax4, f"4. Subimagen 'Image' {obj1.Nombre}\n({obj1.Image.shape[0]}x{obj1.Image.shape[1]} px)")
    else:
        estilizar_panel(ax4, "4. Subimagen (Sin objetos)")

    # Panel 5: Subimagen Binaria del Objeto 2
    ax5 = fig.add_subplot(gs[1, 1])
    if len(objetos) >= 2:
        obj2 = objetos[1]
        ax5.imshow(obj2.Image, cmap='gray')
        estilizar_panel(ax5, f"5. Subimagen 'Image' {obj2.Nombre}\n({obj2.Image.shape[0]}x{obj2.Image.shape[1]} px)")
    elif len(objetos) == 1:
        ax5.imshow(objetos[0].Image, cmap='gray')
        estilizar_panel(ax5, f"5. Subimagen {objetos[0].Nombre}")
    else:
        estilizar_panel(ax5, "5. Subimagen (Sin objetos)")

    # Panel 6: MONTAJE FINAL SOBRE LA IMAGEN ORIGINAL (Líneas rectas continuas)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(img_rgb)
    estilizar_panel(ax6, "6. Montaje Final: BoundingBoxes, Cuadrados y Centroides")

    colores = ['#00ff66', '#00d4ff', '#ff9900', '#ff00ff', '#00ffcc', '#ffff00']

    for i, obj in enumerate(objetos):
        color_caja = colores[i % len(colores)]
        caja = obj.BoundingBox
        cuadro = obj.CuadradoEnvolvente

        # BoundingBox rectangular
        rect = patches.Rectangle(
            (caja[0], caja[1]), caja[2], caja[3],
            linewidth=1.8, edgecolor=color_caja, facecolor='none',
            linestyle='-', label=f'Box {obj.Nombre}'
        )
        ax6.add_patch(rect)

        # Cuadrado envolvente simétrico (lado = max) en amarillo
        cuad = patches.Rectangle(
            (cuadro[0], cuadro[1]), cuadro[2], cuadro[3],
            linewidth=1.8, edgecolor='#ffd700', facecolor='none',
            linestyle='-', label=f'Cuadrado ({int(obj.LadoCuadrado)}x{int(obj.LadoCuadrado)} px)' if i == 0 else ""
        )
        ax6.add_patch(cuad)

        # Centroide analítico continuo
        ax6.plot(
            obj.Centroid[0], obj.Centroid[1],
            marker='o', color=color_caja, markeredgecolor='white', markeredgewidth=1.2,
            markersize=6, label=f'Centroide {obj.Nombre}' if i < 3 else ""
        )

        # Etiqueta de texto compacta
        y_txt = max(16, caja[1] - 6) if (i % 2 == 0) else min(alto - 10, caja[1] + caja[3] + 16)
        ax6.text(
            caja[0], y_txt,
            f"{obj.Nombre}: A={obj.Area:.0f} | P={obj.Perimeter:.1f}",
            color=color_caja, fontsize=7, weight='bold',
            bbox=dict(facecolor='#121212', alpha=0.65, edgecolor=color_caja, boxstyle='round,pad=0.15')
        )

    ax6.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3,
        facecolor='#181818', edgecolor='#383838', labelcolor='white',
        fontsize=6.8, framealpha=0.9, handletextpad=0.4, columnspacing=0.8
    )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.09, wspace=0.15, hspace=0.25)
    plt.show()


# ==============================================================================
# 5. EJECUCIÓN PRINCIPAL
# ==============================================================================
def main():
    ruta, modo = solicitar_modo_y_archivo()
    if not ruta or not os.path.exists(ruta):
        print("Operación cancelada: No se seleccionó ninguna imagen.")
        return

    semillas = None
    if modo == "clic":
        semillas = capturar_clics_usuario(ruta)
        # Si el usuario no hizo clics y cerró la ventana, se cambia a modo otsu automáticamente
        if not semillas or len(semillas) == 0:
            modo = "otsu"

    try:
        datos = procesar_imagen_completa(ruta, modo=modo, puntos_semilla=semillas)
        imprimir_reporte_matlab(datos)
        mostrar_interfaz_grafica(datos)
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al procesar:\n{e}")


if __name__ == "__main__":
    main()