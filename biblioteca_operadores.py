"""
biblioteca_operadores.py
------------------------
Biblioteca de operadores VALIDOS. Es el unico lugar donde se decide
con que piezas se pueden armar las funciones.

  - BINARIOS: operadores con 2 hijos (+ - * / ^)
  - UNARIOS : funciones con 1 hijo (sen, cos, tan, sqrt, log, ...)

Para agregar una funcion nueva basta una linea en UNARIOS.
Al importar este modulo, las funciones UNARIAS se registran en
arbol_expresion.py, asi ArbolExpresion("sen(x)") ya las entiende
(sin modificar la clase).
"""
import numpy as np
import arbol_expresion

# Binarios: se toman de arbol_expresion (el parser los conoce de fabrica)
BINARIOS = dict(arbol_expresion.OPERADORES)

# Unarios: nombre -> funcion de numpy
UNARIOS = {
    "sen":  np.sin,
    "cos":  np.cos,
    "tan":  np.tan,
    "sqrt": np.sqrt,
    "log":  np.log,
    "exp":  np.exp,
    "tanh": np.tanh,
}

# Registro: le ensena al ArbolExpresion los nombres nuevos (ej. "sen")
arbol_expresion.FUNCIONES.update(UNARIOS)


def listar():
    """Devuelve todos los operadores validos (para mostrar o consultar)."""
    return list(BINARIOS) + list(UNARIOS)


if __name__ == "__main__":
    print("Operadores validos:", listar())
