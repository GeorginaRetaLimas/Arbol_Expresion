"""
visualizacion.py
----------------
Funciones para DIBUJAR un ArbolExpresion y su funcion con matplotlib.
No usan tkinter: reciben un "ax" de matplotlib, asi se pueden usar en
cualquier interfaz, notebook o script.
"""
import numpy as np

ROSA, ROSA_CLARO, BLANCO, NEGRO = "#ff4fa3", "#ffd6ea", "#ffffff", "#111111"


def posiciones(nodo, nivel=0, pos=None, contador=None):
    """Asigna (x, y) a cada nodo. x = orden de izquierda a derecha, y = nivel."""
    if pos is None:
        pos, contador = {}, [0]
    if nodo.izq:
        posiciones(nodo.izq, nivel + 1, pos, contador)
    pos[nodo] = (contador[0], -nivel)
    contador[0] += 1
    if nodo.der:
        posiciones(nodo.der, nivel + 1, pos, contador)
    return pos


def dibujar_arbol(ax, arbol):
    pos = posiciones(arbol.raiz)
    n = len(pos)
    tamano = 11 if n <= 15 else 9 if n <= 31 else 7 if n <= 63 else 5   # letra segun tamano

    def dibujar(nodo):                                  # preorden: nodo -> izq -> der
        x, y = pos[nodo]
        for hijo in (nodo.izq, nodo.der):
            if hijo:
                hx, hy = pos[hijo]
                ax.plot([x, hx], [y, hy], color=NEGRO, lw=1.2, zorder=1)
        es_operador = nodo.izq is not None              # tiene hijos = operador o funcion
        ax.text(x, y, nodo.valor, ha="center", va="center", fontsize=tamano,
                fontweight="bold", zorder=2,
                color=BLANCO if es_operador else NEGRO,
                bbox=dict(boxstyle="circle,pad=0.35",
                          fc=ROSA if es_operador else ROSA_CLARO, ec=NEGRO))
        for hijo in (nodo.izq, nodo.der):
            if hijo:
                dibujar(hijo)

    dibujar(arbol.raiz)
    ax.set_title("Arbol de expresion", color=NEGRO, fontweight="bold")
    ax.margins(0.1)
    ax.axis("off")


def dibujar_funcion(ax, arbol, a, b):
    x = np.linspace(a, b, 600)
    with np.errstate(all="ignore"):                     # evita avisos (sqrt de negativos, etc.)
        y = np.asarray(arbol.evaluar(x), dtype=float) * np.ones_like(x)
    y[np.abs(y) > 1e6] = np.nan                         # oculta valores gigantes (ej. tan)
    ax.plot(x, y, color=ROSA, lw=2.5)
    ax.axhline(0, color=NEGRO, lw=0.8)
    ax.axvline(0, color=NEGRO, lw=0.8)
    ax.grid(True, color=ROSA_CLARO)
    ax.set_title(f"f(x) en [{a}, {b}]", color=NEGRO, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
