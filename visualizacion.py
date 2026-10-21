"""
visualizacion.py
----------------
Funciones para DIBUJAR un ArbolExpresion y su funcion con matplotlib.
No usan tkinter: reciben un "ax" de matplotlib, asi se pueden usar en
cualquier interfaz, notebook o script.

Paleta rosita bonita, suave y digerible (no chillona).
Diseno de arbol binario puro:
- El nodo raiz se ubica estrictamente ARRIBA Y EN EL CENTRO.
- Los hijos descienden claramente hacia abajo.
- Los operadores binarios tienen hijo izquierdo a la izquierda, hijo derecho a la derecha,
  y el padre centrado.
- Las funciones unarias tienen su hijo alineado verticalmente hacia abajo.
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
NEGRO = "#2d1b28"           # Texto oscuro ciruela/espresso
TEXTO_MUTED = "#7a596e"     # Texto secundario
AZUL_OBJETIVO = "#486581"   # Azul pizarra / lavanda oscuro para contrastar con f(x) objetivo
GRIS_REJILLA = "#f2e4ec"    # Rejilla suave en tono rosaceo


def posiciones(raiz, min_sep=1.35, dy=1.35):
    """
    Calcula coordenadas (x, y) de cada nodo para que el arbol SI PAREZCA BINARIO:
    - La raiz esta estrictamente ARRIBA Y EN EL CENTRO (x=0, y=0).
    - Los niveles descienden claramente con espaciado vertical constante (-prof * dy).
    - Hijo izquierdo estrictamente a la izquierda, hijo derecho a la derecha.
    - Operadores binarios centrados entre sus dos hijos.
    - Funciones unarias con su argumento directamente abajo.
    """
    if raiz is None:
        return {}

    class Aux:
        def __init__(self, n, prof=0):
            self.node = n
            self.prof = prof
            self.x = 0.0
            self.izq = Aux(n.izq, prof + 1) if n.izq else None
            self.der = Aux(n.der, prof + 1) if n.der else None

    root = Aux(raiz, 0)

    def contorno(aux):
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

    def desplazar(aux, dx):
        if not aux:
            return
        aux.x += dx
        desplazar(aux.izq, dx)
        desplazar(aux.der, dx)

    def pos_rec(aux):
        if not aux.izq and not aux.der:
            aux.x = 0.0
            return

        if aux.izq and not aux.der:
            pos_rec(aux.izq)
            aux.x = aux.izq.x
            return

        if not aux.izq and aux.der:
            pos_rec(aux.der)
            aux.x = aux.der.x
            return

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

    # Centrar estrictamente en la raiz (raiz en x=0, y=0 arriba)
    dx_raiz = -root.x
    desplazar(root, dx_raiz)

    pos = {}
    def recolectar(aux):
        pos[aux.node] = (aux.x, -float(aux.prof) * dy)
        if aux.izq:
            recolectar(aux.izq)
        if aux.der:
            recolectar(aux.der)

    recolectar(root)
    return pos


def dibujar_arbol(ax, arbol, titulo="Estructura del Árbol Binario"):
    """
    Dibuja el arbol de expresion con el nodo raiz estrictamente arriba y centrado,
    hijos hacia abajo de forma jerarquica y paleta rosita bonita.
    """
    ax.clear()
    ax.set_facecolor(BLANCO)

    if arbol is None or arbol.raiz is None:
        ax.text(0.5, 0.5, "Sin árbol", ha="center", va="center", color=TEXTO_MUTED)
        ax.axis("off")
        return

    pos = posiciones(arbol.raiz)
    n = len(pos)

    # Tamano de fuente y proporciones segun cantidad de nodos
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
            fc = ROSA
            tc = BLANCO
            b_style = "circle,pad=0.35" if len(texto) <= 2 else "round,pad=0.38,rounding_size=0.4"
            box = dict(boxstyle=b_style, fc=fc, ec=ROSA_BORDE, lw=1.4)
        elif es_unario:
            fc = ROSA_MEDIO
            tc = BLANCO
            box = dict(boxstyle="round,pad=0.38,rounding_size=0.45", fc=fc, ec=ROSA_BORDE, lw=1.4)
        else:
            fc = ROSA_CLARO
            tc = NEGRO
            b_style = "circle,pad=0.32" if len(texto) <= 2 else "round,pad=0.35,rounding_size=0.4"
            box = dict(boxstyle=b_style, fc=fc, ec=ROSA_BORDE, lw=1.2)

        ax.text(x, y, texto, ha="center", va="center", fontsize=tamano,
                fontweight="bold", color=tc, bbox=box, zorder=2)

    # Ajustar limites asegurando que la raiz este centrada y visible
    xs = [coord[0] for coord in pos.values()]
    ys = [coord[1] for coord in pos.values()]
    ancho = max(xs) - min(xs) if len(xs) > 1 else 2.0
    alto = max(ys) - min(ys) if len(ys) > 1 else 2.0

    pad_x = max(1.0, ancho * 0.15)
    pad_y = max(1.0, alto * 0.15)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    ax.set_title(titulo, color=NEGRO, fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")


def dibujar_funcion(ax, arbol, a, b, datos_ref=None):
    """
    Grafica la funcion del arbol f(x) en el intervalo [a, b].
    Si se proporciona datos_ref=(x_ref, y_ref), tambien dibuja los puntos/curva
    de referencia del CSV con leyenda.
    """
    ax.clear()
    ax.set_facecolor(BLANCO)

    # 1. Grafica de referencia CSV (si existe)
    if datos_ref is not None:
        x_ref, y_ref = datos_ref
        dentro = (x_ref >= a) & (x_ref <= b)
        if np.any(dentro):
            xr_plot = x_ref[dentro]
            yr_plot = y_ref[dentro]
            if len(xr_plot) > 1500:
                paso_muestreo = len(xr_plot) // 1200
                xr_plot = xr_plot[::paso_muestreo]
                yr_plot = yr_plot[::paso_muestreo]
            ax.plot(xr_plot, yr_plot, color=AZUL_OBJETIVO, lw=2.0, ls="--",
                    label="y Objetivo (CSV)", alpha=0.9, zorder=2)

    # 2. Grafica de f(x) del arbol
    x = np.linspace(a, b, 750)
    with np.errstate(all="ignore"):
        y = np.asarray(arbol.evaluar(x), dtype=float) * np.ones_like(x)
    y[np.abs(y) > 1e6] = np.nan

    etiqueta = "f(x) Árbol" if datos_ref is not None else "f(x)"
    ax.plot(x, y, color=ROSA, lw=2.4, label=etiqueta, zorder=3)

    # Ejes coordenados y rejilla
    ax.axhline(0, color="#bfa2b3", lw=0.9, ls="-", zorder=1)
    ax.axvline(0, color="#bfa2b3", lw=0.9, ls="-", zorder=1)
    ax.grid(True, color=GRIS_REJILLA, linestyle=":", lw=1.1, zorder=0)

    for spine in ax.spines.values():
        spine.set_color("#e3c9d6")
        spine.set_linewidth(1.0)

    ax.tick_params(colors=TEXTO_MUTED, labelsize=9)
    ax.set_title(f"Gráfica f(x) en [{a}, {b}]", color=NEGRO, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("x", color=NEGRO, fontsize=10, fontweight="bold")
    ax.set_ylabel("f(x)", color=NEGRO, fontsize=10, fontweight="bold")

    if datos_ref is not None:
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor=ROSA_BORDE, fontsize=9, loc="best")
