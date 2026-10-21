"""
visualizacion.py
----------------
Funciones para DIBUJAR un ArbolExpresion y su funcion con matplotlib.
No usan tkinter: reciben un "ax" de matplotlib, asi se pueden usar en
cualquier interfaz, notebook o script.

Paleta rosita bonita, suave y digerible (no chillona).
Diseno de arbol verdaderamente binario (hijo izquierdo a la izq, derecho a la der,
padre centrado, funciones unarias alineadas verticalmente).
"""
import numpy as np

# Paleta cromatica armoniosa, estética y suave (rosita pastel / rosa empolvado)
ROSA = "#d94a7a"            # Rosa principal elegante (operadores binarios y curva principal)
ROSA_CLARO = "#faebf2"      # Rosa pastel suave (hojas, operandos, fondos de tarjetas)
ROSA_MEDIO = "#e26d96"      # Rosa frambuesa suave (funciones unarias)
ROSA_BORDE = "#c97a9b"      # Borde sutil para nodos
ROSA_LINEA = "#8a667b"      # Tono ciruela / malva para ramas del arbol
BLANCO = "#ffffff"          # Blanco puro para fondos y contraste
FONDO_SUAVE = "#fdf8fb"     # Fondo suave aperlado
NEGRO = "#2d1b28"           # Texto oscuro ciruela/espresso (mucho mas suave que #111111)
TEXTO_MUTED = "#7a596e"     # Texto secundario
AZUL_OBJETIVO = "#486581"   # Azul pizarra / lavanda oscuro para contrastar con f(x) objetivo
GRIS_REJILLA = "#f2e4ec"    # Rejilla suave en tono rosaceo


def posiciones(raiz, nivel=0, min_sep=1.0):
    """
    Calcula coordenadas (x, y) de cada nodo para que el arbol SI PAREZCA BINARIO:
    - El hijo izquierdo siempre queda a la izquierda del padre.
    - El hijo derecho siempre queda a la derecha del padre.
    - Los operadores binarios se centran exactamente sobre sus dos hijos: x = (x_izq + x_der)/2.
    - Las funciones unarias (1 solo argumento en izq) se alinean verticalmente: x = x_izq.
    - Los subarboles respetan contornos para que ninguna rama ni nodo se solape.
    Retorna un diccionario {nodo: (x, y)}.
    """
    if raiz is None:
        return {}

    class NodoAux:
        def __init__(self, nodo, prof=0):
            self.nodo = nodo
            self.prof = prof
            self.x = 0.0
            self.izq = NodoAux(nodo.izq, prof + 1) if nodo.izq else None
            self.der = NodoAux(nodo.der, prof + 1) if nodo.der else None

    root = NodoAux(raiz, nivel)

    def desplazar(aux, dx):
        if not aux:
            return
        aux.x += dx
        desplazar(aux.izq, dx)
        desplazar(aux.der, dx)

    def contorno(aux):
        # Devuelve lista de tuplas (min_x, max_x) por nivel
        res = [(aux.x, aux.x)]
        c_izq = contorno(aux.izq) if aux.izq else []
        c_der = contorno(aux.der) if aux.der else []
        for i in range(max(len(c_izq), len(c_der))):
            if i < len(c_izq) and i < len(c_der):
                res.append((min(c_izq[i][0], c_der[i][0]), max(c_izq[i][1], c_der[i][1])))
            elif i < len(c_izq):
                res.append(c_izq[i])
            else:
                res.append(c_der[i])
        return res

    def pos_rec(aux):
        if not aux.izq and not aux.der:
            aux.x = 0.0
            return

        # Unario: funcion con un solo argumento (en izq)
        if aux.izq and not aux.der:
            pos_rec(aux.izq)
            aux.x = aux.izq.x
            return

        # En caso de solo der (por simetria)
        if not aux.izq and aux.der:
            pos_rec(aux.der)
            aux.x = aux.der.x
            return

        # Binario: tiene hijo izquierdo y derecho
        pos_rec(aux.izq)
        pos_rec(aux.der)

        c_izq = contorno(aux.izq)
        c_der = contorno(aux.der)

        dx = 0.0
        for (min_l, max_l), (min_r, max_r) in zip(c_izq, c_der):
            gap = max_l + min_sep - min_r
            if gap > dx:
                dx = gap

        desplazar(aux.der, dx)
        aux.x = (aux.izq.x + aux.der.x) / 2.0

    pos_rec(root)

    # Convertir a coordenadas finales y normalizar horizontalmente centrado en 0
    pos = {}
    def recolectar(aux):
        pos[aux.nodo] = (aux.x, -float(aux.prof))
        if aux.izq:
            recolectar(aux.izq)
        if aux.der:
            recolectar(aux.der)

    recolectar(root)

    # Centrar horizontalmente respecto al punto medio
    if pos:
        xs = [coord[0] for coord in pos.values()]
        x_centro = (min(xs) + max(xs)) / 2.0
        pos = {nodo: (x - x_centro, y) for nodo, (x, y) in pos.items()}

    return pos


def dibujar_arbol(ax, arbol):
    """
    Dibuja el arbol de expresion con estructura binaria pura,
    con nodos circulares/pildoras estilizados y paleta rosa bonita.
    """
    ax.clear()
    ax.set_facecolor(BLANCO)

    if arbol is None or arbol.raiz is None:
        ax.text(0.5, 0.5, "Sin arbol", ha="center", va="center", color=TEXTO_MUTED)
        ax.axis("off")
        return

    pos = posiciones(arbol.raiz)
    n = len(pos)

    # Tamano de fuente y proporciones dinamicas segun cantidad de nodos
    if n <= 10:
        tamano = 10
        lw_linea = 1.8
    elif n <= 25:
        tamano = 9
        lw_linea = 1.5
    elif n <= 50:
        tamano = 7.5
        lw_linea = 1.2
    else:
        tamano = 6
        lw_linea = 0.9

    # 1. Dibujar lineas (ramas) entre nodos
    for nodo, (x, y) in pos.items():
        for hijo in (nodo.izq, nodo.der):
            if hijo and hijo in pos:
                hx, hy = pos[hijo]
                ax.plot([x, hx], [y, hy], color=ROSA_LINEA, lw=lw_linea, zorder=1)

    # 2. Dibujar nodos
    for nodo, (x, y) in pos.items():
        es_binario = nodo.der is not None
        es_unario = nodo.izq is not None and nodo.der is None
        texto = str(nodo.valor)

        if es_binario:
            # Operador binario: circulo rosa principal
            fc = ROSA
            tc = BLANCO
            b_style = "circle,pad=0.32" if len(texto) <= 2 else "round,pad=0.35,rounding_size=0.4"
            box = dict(boxstyle=b_style, fc=fc, ec=ROSA_BORDE, lw=1.3)
        elif es_unario:
            # Funcion unaria: capsula redondeada rosa frambuesa
            fc = ROSA_MEDIO
            tc = BLANCO
            box = dict(boxstyle="round,pad=0.35,rounding_size=0.45", fc=fc, ec=ROSA_BORDE, lw=1.3)
        else:
            # Operando / hoja (x o numero): circulo o capsula clara
            fc = ROSA_CLARO
            tc = NEGRO
            b_style = "circle,pad=0.30" if len(texto) <= 2 else "round,pad=0.32,rounding_size=0.4"
            box = dict(boxstyle=b_style, fc=fc, ec=ROSA_BORDE, lw=1.1)

        ax.text(x, y, texto, ha="center", va="center", fontsize=tamano,
                fontweight="bold", color=tc, bbox=box, zorder=2)

    # Ajustar limites con un margen comodo
    xs = [coord[0] for coord in pos.values()]
    ys = [coord[1] for coord in pos.values()]
    ancho = max(xs) - min(xs) if len(xs) > 1 else 1.0
    alto = max(ys) - min(ys) if len(ys) > 1 else 1.0
    pad_x = max(0.8, ancho * 0.12)
    pad_y = max(0.8, alto * 0.15)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    ax.set_title("Estructura del Arbol Binario", color=NEGRO, fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")


def dibujar_funcion(ax, arbol, a, b, datos_ref=None):
    """
    Grafica la funcion del arbol f(x) en el intervalo [a, b].
    Si se proporciona datos_ref=(x_ref, y_ref), tambien dibuja los puntos/curva
    de referencia (ej. del CSV o funcion objetivo) con leyenda.
    """
    ax.clear()
    ax.set_facecolor(BLANCO)

    # 1. Grafica de referencia (si existe)
    if datos_ref is not None:
        x_ref, y_ref = datos_ref
        # Si x_ref esta dentro del intervalo solicitado, filtramos
        dentro = (x_ref >= a) & (x_ref <= b)
        if np.any(dentro):
            xr_plot = x_ref[dentro]
            yr_plot = y_ref[dentro]
            # Si hay demasiados puntos (>1200), submuestreamos suavemente para no saturar matplotlib
            if len(xr_plot) > 1200:
                paso_muestreo = len(xr_plot) // 1000
                xr_plot = xr_plot[::paso_muestreo]
                yr_plot = yr_plot[::paso_muestreo]
            ax.plot(xr_plot, yr_plot, color=AZUL_OBJETIVO, lw=2.0, ls="--",
                    label="y Objetivo (CSV)", alpha=0.9, zorder=2)

    # 2. Grafica de f(x) del arbol
    x = np.linspace(a, b, 700)
    with np.errstate(all="ignore"):
        y = np.asarray(arbol.evaluar(x), dtype=float) * np.ones_like(x)
    y[np.abs(y) > 1e6] = np.nan

    etiqueta = "f(x) Árbol" if datos_ref is not None else "f(x)"
    ax.plot(x, y, color=ROSA, lw=2.4, label=etiqueta, zorder=3)

    # Ejes coordenados y rejilla
    ax.axhline(0, color="#bfa2b3", lw=0.9, ls="-", zorder=1)
    ax.axvline(0, color="#bfa2b3", lw=0.9, ls="-", zorder=1)
    ax.grid(True, color=GRIS_REJILLA, linestyle=":", lw=1.1, zorder=0)

    # Estilos de borde y etiquetas
    for spine in ax.spines.values():
        spine.set_color("#e3c9d6")
        spine.set_linewidth(1.0)

    ax.tick_params(colors=TEXTO_MUTED, labelsize=9)
    ax.set_title(f"Gráfica f(x) en [{a}, {b}]", color=NEGRO, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("x", color=NEGRO, fontsize=10, fontweight="bold")
    ax.set_ylabel("f(x)", color=NEGRO, fontsize=10, fontweight="bold")

    if datos_ref is not None:
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor=ROSA_BORDE, fontsize=9, loc="best")
