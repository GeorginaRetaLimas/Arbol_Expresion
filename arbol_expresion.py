"""
arbol_expresion.py
------------------
Modulo REUTILIZABLE: convierte un texto como "sqrt(2*(x^2)+5*x-3)"
en un arbol binario de expresion, lo recorre en preorden y lo evalua.

No depende de ninguna interfaz. Uso rapido:

    from arbol_expresion import ArbolExpresion

    arbol = ArbolExpresion("sqrt(2 * (x ^2 ) + 5 * x - 3)")
    arbol.preorden()      # ['sqrt', '-', '+', '*', '2', '^', 'x', '2', ...]
    arbol.evaluar(4)      # evalua la expresion con x = 4
"""
import re
import numpy as np

# Operadores binarios (2 hijos) y funciones (1 hijo, siempre el izquierdo)
OPERADORES = {"+": np.add, "-": np.subtract, "*": np.multiply,
              "/": np.divide, "^": np.power}
FUNCIONES = {"sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
             "log": np.log, "exp": np.exp, "abs": np.abs,
             "neg": np.negative}          # "neg" = el signo menos unario (-x)


class Nodo:
    """Un nodo del arbol: un valor y dos hijos (izquierdo y derecho)."""
    def __init__(self, valor, izq=None, der=None):
        self.valor = valor
        self.izq = izq
        self.der = der


class ArbolExpresion:
    """Arbol binario de expresion construido desde un texto."""

    def __init__(self, expresion):
        self.expresion = expresion
        self._tokens = re.findall(r"\d+\.?\d*|[a-zA-Z_]\w*|[-+*/^()]",
                                  expresion.replace(" ", ""))
        self._i = 0                       # posicion actual dentro de los tokens
        self.raiz = self._expresion()     # construye el arbol
        if self._i < len(self._tokens):
            raise ValueError(f"Sobra texto: '{self._tokens[self._i]}'")

    # ---------------------------------------------------------------
    # CONSTRUCCION (de menor a mayor prioridad de operadores)
    #   expresion :  termino (+|- termino)*
    #   termino   :  potencia (*|/ potencia)*
    #   potencia  :  -potencia | base (^ potencia)?
    #   base      :  numero | x | funcion(expresion) | (expresion)
    # ---------------------------------------------------------------
    def _actual(self):
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _avanzar(self):
        token = self._actual()
        self._i += 1
        return token

    def _expresion(self):
        nodo = self._termino()
        while self._actual() in ("+", "-"):
            op = self._avanzar()
            nodo = Nodo(op, nodo, self._termino())
        return nodo

    def _termino(self):
        nodo = self._potencia()
        while self._actual() in ("*", "/"):
            op = self._avanzar()
            nodo = Nodo(op, nodo, self._potencia())
        return nodo

    def _potencia(self):
        if self._actual() == "-":                 # signo menos unario
            self._avanzar()
            return Nodo("neg", self._potencia())
        base = self._base()
        if self._actual() == "^":
            self._avanzar()
            return Nodo("^", base, self._potencia())
        return base

    def _base(self):
        token = self._avanzar()
        if token is None:
            raise ValueError("La expresion termino antes de tiempo")
        if token == "(":
            nodo = self._expresion()
            if self._avanzar() != ")":
                raise ValueError("Falta cerrar un parentesis")
            return nodo
        if token in FUNCIONES and token != "neg":
            if self._avanzar() != "(":
                raise ValueError(f"Falta '(' despues de {token}")
            nodo = self._expresion()
            if self._avanzar() != ")":
                raise ValueError("Falta cerrar un parentesis")
            return Nodo(token, nodo)
        if token == "x" or re.fullmatch(r"\d+\.?\d*", token):
            return Nodo(token)
        raise ValueError(f"Simbolo no valido: '{token}'")

    # ---------------------------------------------------------------
    # RECORRIDO EN PREORDEN: raiz -> izquierdo -> derecho
    # ---------------------------------------------------------------
    def preorden(self, nodo="raiz"):
        if nodo == "raiz":
            nodo = self.raiz
        if nodo is None:
            return []
        return [nodo.valor] + self.preorden(nodo.izq) + self.preorden(nodo.der)

    # ---------------------------------------------------------------
    # EVALUACION: x puede ser un numero o un arreglo de numpy
    # ---------------------------------------------------------------
    def evaluar(self, x, nodo="raiz"):
        if nodo == "raiz":
            nodo = self.raiz
        if nodo.valor == "x":
            return x
        if nodo.valor in OPERADORES:
            return OPERADORES[nodo.valor](self.evaluar(x, nodo.izq),
                                          self.evaluar(x, nodo.der))
        if nodo.valor in FUNCIONES:
            return FUNCIONES[nodo.valor](self.evaluar(x, nodo.izq))
        return float(nodo.valor)                  # es un numero


if __name__ == "__main__":                        # prueba rapida
    a = ArbolExpresion("sqrt(2 * (x ^2 ) + 5 * x - 3)")
    print("Preorden:", a.preorden())
    print("f(4) =", a.evaluar(4))                  # sqrt(2*16+20-3) = sqrt(49) = 7
