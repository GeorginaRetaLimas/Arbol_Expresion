"""
generador_aleatorio.py
----------------------
Genera arboles de expresion ALEATORIOS con una altura maxima.
Usa la biblioteca de operadores y la clase ArbolExpresion (sin modificarla).

Uso rapido:
    from generador_aleatorio import generar_arbol, altura

    arbol = generar_arbol(altura_max=4)
    print(arbol.expresion)       # ej: (sen((x * 3)) + sqrt(x))
    print(altura(arbol.raiz))    # <= 4

Convencion: un arbol de un solo nodo tiene altura 0.
"""
import random
import numpy as np
from arbol_expresion import ArbolExpresion, Nodo
from biblioteca_operadores import BINARIOS, UNARIOS


# ---------------------------------------------------------------
# Utilidades sobre un arbol
# ---------------------------------------------------------------
def altura(nodo):
    """Altura del arbol (hoja = 0)."""
    if nodo is None:
        return -1
    return 1 + max(altura(nodo.izq), altura(nodo.der))


def nodo_a_texto(nodo):
    """Convierte un arbol de Nodos a texto (con parentesis completos)."""
    if nodo.der:                                      # operador binario
        return f"({nodo_a_texto(nodo.izq)} {nodo.valor} {nodo_a_texto(nodo.der)})"
    if nodo.izq:                                      # funcion
        return f"{nodo.valor}({nodo_a_texto(nodo.izq)})"
    return nodo.valor                                 # hoja (x o numero)


# ---------------------------------------------------------------
# Generacion aleatoria
# ---------------------------------------------------------------
def _hoja():
    """Una hoja: la variable x (60%) o una constante entera 1-9 (40%)."""
    return Nodo("x") if random.random() < 0.6 else Nodo(str(random.randint(1, 9)))


def generar_nodo(altura_max, nivel=0, binarios=None, unarios=None, prob_hoja=0.3):
    """Construye (recursivamente) un arbol aleatorio de Nodos."""
    binarios = list(binarios or BINARIOS)
    unarios = list(unarios or UNARIOS)

    # Limite de altura: al llegar al maximo, solo se pueden poner hojas
    if nivel >= altura_max:
        return _hoja()
    # En niveles intermedios, a veces se corta antes (la raiz nunca es hoja)
    if nivel > 0 and random.random() < prob_hoja:
        return _hoja()

    # Elegir operador: binario (60%) o funcion (40%)
    if random.random() < 0.6:
        op = random.choice(binarios)
        izq = generar_nodo(altura_max, nivel + 1, binarios, unarios, prob_hoja)
        if op == "^":                                 # exponente pequeno: x^2, x^3
            der = Nodo(random.choice(["2", "3"]))
        else:
            der = generar_nodo(altura_max, nivel + 1, binarios, unarios, prob_hoja)
        return Nodo(op, izq, der)

    op = random.choice(unarios)
    return Nodo(op, generar_nodo(altura_max, nivel + 1, binarios, unarios, prob_hoja))


def generar_arbol(altura_max, **opciones):
    """Devuelve un ArbolExpresion aleatorio con altura <= altura_max.
    opciones: binarios=[...], unarios=[...], prob_hoja=0.3"""
    nodo = generar_nodo(altura_max, **opciones)
    return ArbolExpresion(nodo_a_texto(nodo))


def generar_graficable(altura_max, a, b, intentos=200, **opciones):
    """Como generar_arbol, pero repite hasta que la funcion sea 'bonita' de
    graficar en [a, b]: valores reales en al menos la mitad de los puntos,
    sin ser constante y sin valores gigantes."""
    x = np.linspace(a, b, 200)
    arbol = None
    for _ in range(intentos):
        arbol = generar_arbol(altura_max, **opciones)
        with np.errstate(all="ignore"):
            y = np.asarray(arbol.evaluar(x), dtype=float) * np.ones_like(x)
        buenos = np.isfinite(y) & (np.abs(y) < 1e4)
        if buenos.mean() >= 0.5 and np.std(y[buenos]) > 1e-6:
            return arbol
    return arbol                                      # ultimo intento si ninguno sirvio


if __name__ == "__main__":                            # prueba rapida
    for _ in range(3):
        t = generar_arbol(4)
        print(altura(t.raiz), t.expresion)
