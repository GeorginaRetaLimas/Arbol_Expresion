"""
algoritmo_genetico.py
---------------------
Modulo REUTILIZABLE para Regresion Simbolica mediante Programacion Genetica.
Usa los modulos ya creados (arbol_expresion, generador_aleatorio, metricas)
sin alterarlos.

Estructura del Individuo:
    Individuo: {
        gen:   Arbol de expresion (ArbolExpresion)
        feno:  Los valores evaluados f(x) sobre el dataset (fenotipo)
        adap:  Calculo del margen de error (RMSE contra los datos reales)
    }

Caracteristicas:
    - Poblacion inicial de 100 ecuaciones aleatorias
    - Evaluacion con las metricas del proyecto (RMSE, MSE, NUSE, R^2, y_barra)
    - Seleccion por Ruleta (fitness proporcional) o Torneo
    - Cruza de subarboles entre individuos
    - Mutacion con probabilidad del 1% con 4 metodos exclusivos:
        1. Mutacion de hojas
        2. Mutacion de nodos internos
        3. Switching (intercambio de hijos izq y der)
        4. Nueva rama (subarbol aleatorio)
      Reporta exactamente cual metodo ocurrio cuando se activa.
"""
import random
import copy
import numpy as np

from arbol_expresion import ArbolExpresion, Nodo
from generador_aleatorio import generar_arbol, generar_nodo, nodo_a_texto, altura
from biblioteca_operadores import BINARIOS, UNARIOS
import metricas


def clonar_nodo(n):
    """Copia profunda y limpia de una estructura de Nodos."""
    if n is None:
        return None
    return Nodo(str(n.valor), clonar_nodo(n.izq), clonar_nodo(n.der))


def obtener_nodos(n):
    """Retorna una lista plana de todos los nodos en el arbol."""
    if n is None:
        return []
    lista = [n]
    if n.izq:
        lista.extend(obtener_nodos(n.izq))
    if n.der:
        lista.extend(obtener_nodos(n.der))
    return lista


# =====================================================================
# CLASE INDIVIDUO
# =====================================================================
class Individuo:
    """
    Estructura requerida:
        gen:  Arbol de expresion
        feno: Los valores (evaluacion sobre el vector x)
        adap: Calculo del margen de error (RMSE)
    """
    def __init__(self, raiz_o_expr):
        if isinstance(raiz_o_expr, Nodo):
            self.raiz = clonar_nodo(raiz_o_expr)
            self.expresion = nodo_a_texto(self.raiz)
            self.gen = ArbolExpresion(self.expresion)
        elif isinstance(raiz_o_expr, ArbolExpresion):
            self.gen = raiz_o_expr
            self.expresion = raiz_o_expr.expresion
            self.raiz = clonar_nodo(raiz_o_expr.raiz)
        elif isinstance(raiz_o_expr, str):
            self.expresion = raiz_o_expr
            self.gen = ArbolExpresion(raiz_o_expr)
            self.raiz = clonar_nodo(self.gen.raiz)
        else:
            raise TypeError("Tipo no soportado para Individuo")

        self.feno = None        # Fenotipo: valores evaluados sobre x
        self.adap = float("inf")# Adaptabilidad: margen de error (RMSE, menor es mejor)
        self.fitness = 0.0      # Aptitud positiva para ruleta (mayor es mejor)
        self.metricas = {}

    def evaluar(self, x_datos, y_datos):
        """Evalua el fenotipo y calcula la adaptabilidad (margen de error)."""
        with np.errstate(all="ignore"):
            y_pred = np.asarray(self.gen.evaluar(x_datos), dtype=float) * np.ones_like(x_datos)

        # Suprimir desbordamientos numericos
        y_pred[np.abs(y_pred) > 1e12] = np.nan
        self.feno = y_pred

        res_met = metricas.calcular_metricas(y_datos, y_pred)
        self.metricas = res_met

        if res_met["valido"] and np.isfinite(res_met["rmse"]):
            self.adap = float(res_met["rmse"])
            # Fitness positivo no-nulo para seleccion por ruleta
            self.fitness = 1.0 / (1.0 + self.adap)
        else:
            self.adap = 1e9
            self.fitness = 1e-12

        return self.adap


# =====================================================================
# OPERADORES GENETICOS: MUTACIONES (4 METODOS)
# =====================================================================
def mutar_hojas(raiz):
    """1. Mutacion de hojas: cambia una variable x por constante o viceversa."""
    nodos = obtener_nodos(raiz)
    hojas = [n for n in nodos if not n.izq and not n.der]
    if not hojas:
        return "Mutacion de hojas: sin hojas disponibles"
    n = random.choice(hojas)
    ant = n.valor
    if ant == "x":
        n.valor = str(random.choice([1, 2, 3, 5, 7, 10, 11]))
    else:
        n.valor = "x" if random.random() < 0.6 else str(random.choice([1, 2, 3, 5, 7, 10, 11]))
    return f"Mutacion de hojas (cambio '{ant}' -> '{n.valor}')"


def mutar_nodos_internos(raiz):
    """2. Mutacion de nodos internos: cambia un operador o funcion por otro valido."""
    nodos = obtener_nodos(raiz)
    internos = [n for n in nodos if n.izq or n.der]
    if not internos:
        return "Mutacion de nodo interno: sin nodos internos disponibles"
    n = random.choice(internos)
    ant = n.valor
    if n.der is not None:
        ops = [op for op in BINARIOS if op != ant]
        n.valor = random.choice(ops)
    else:
        ops = [op for op in UNARIOS if op != ant]
        n.valor = random.choice(ops)
    return f"Mutacion de nodo interno (cambio '{ant}' -> '{n.valor}')"


def mutar_switching(raiz):
    """3. Switching: intercambia el hijo izquierdo y derecho de un operador binario."""
    nodos = obtener_nodos(raiz)
    binarios = [n for n in nodos if n.izq and n.der]
    if not binarios:
        return "Switching: no hay operadores binarios con 2 hijos"
    n = random.choice(binarios)
    n.izq, n.der = n.der, n.izq
    return f"Switching (intercambio ramas izq <-> der en operador '{n.valor}')"


def mutar_nueva_rama(raiz, altura_max=2):
    """4. Nueva rama: sustituye un subarbol por una rama aleatoria nueva."""
    nodos = obtener_nodos(raiz)
    if not nodos:
        return "Nueva rama: arbol vacio"
    n = random.choice(nodos)
    ant_desc = n.valor
    nueva = generar_nodo(altura_max=altura_max)
    n.valor = nueva.valor
    n.izq = nueva.izq
    n.der = nueva.der
    return f"Nueva rama (insertado subarbol nuevo en lugar de '{ant_desc}')"


def aplicar_mutacion(individuo, prob=0.01):
    """
    Aplica mutacion con la probabilidad dada (por defecto 1%).
    Si ocurre, selecciona exactamente UNO de los 4 metodos y reporta cual fue.
    Retorna (individuo_mutado, detalle_mutacion_o_None).
    """
    if random.random() >= prob:
        return individuo, None

    nueva_raiz = clonar_nodo(individuo.raiz)
    metodo = random.choice(["hojas", "internos", "switching", "nueva_rama"])

    if metodo == "hojas":
        detalle = mutar_hojas(nueva_raiz)
    elif metodo == "internos":
        detalle = mutar_nodos_internos(nueva_raiz)
    elif metodo == "switching":
        detalle = mutar_switching(nueva_raiz)
    else:
        detalle = mutar_nueva_rama(nueva_raiz, altura_max=2)

    try:
        nuevo_ind = Individuo(nueva_raiz)
        return nuevo_ind, detalle
    except Exception:
        # En caso de expresion atipica, conservamos el original
        return individuo, None


# =====================================================================
# OPERADORES GENETICOS: CRUZA Y SELECCION
# =====================================================================
def cruza(ind1, ind2, altura_limite=7):
    """
    Cruza entre dos individuos intercambiando subarboles al azar.
    Retorna dos nuevos individuos hijos.
    """
    r1 = clonar_nodo(ind1.raiz)
    r2 = clonar_nodo(ind2.raiz)

    nodos1 = obtener_nodos(r1)
    nodos2 = obtener_nodos(r2)

    n1 = random.choice(nodos1)
    n2 = random.choice(nodos2)

    # Intercambiar contenidos en los nodos elegidos
    n1.valor, n2.valor = n2.valor, n1.valor
    n1.izq, n2.izq = n2.izq, n1.izq
    n1.der, n2.der = n2.der, n1.der

    # Controlar crecimiento desmedido (bloat)
    h1 = r1 if altura(r1) <= altura_limite else ind1.raiz
    h2 = r2 if altura(r2) <= altura_limite else ind2.raiz

    try:
        hijo1 = Individuo(h1)
    except Exception:
        hijo1 = Individuo(clonar_nodo(ind1.raiz))

    try:
        hijo2 = Individuo(h2)
    except Exception:
        hijo2 = Individuo(clonar_nodo(ind2.raiz))

    return hijo1, hijo2


def seleccion_ruleta(poblacion):
    """Seleccion proporcional a la aptitud (metodo de ruleta)."""
    suma_fitness = sum(ind.fitness for ind in poblacion)
    if suma_fitness <= 0 or not np.isfinite(suma_fitness):
        return random.choice(poblacion)

    tiro = random.uniform(0, suma_fitness)
    acumulado = 0.0
    for ind in poblacion:
        acumulado += ind.fitness
        if acumulado >= tiro:
            return ind
    return poblacion[-1]


def seleccion_torneo(poblacion, k=3):
    """Seleccion por torneo (opcion alternativa requerida)."""
    participantes = random.sample(poblacion, min(k, len(poblacion)))
    # El mejor tiene menor adaptabilidad (menor margen de error)
    return min(participantes, key=lambda ind: ind.adap)


# =====================================================================
# CONTROLADOR DEL ALGORITMO GENETICO
# =====================================================================
class AlgoritmoGenetico:
    """
    Controla el ciclo de vida del algoritmo genetico:
    - Generacion de 100 ecuaciones aleatorias
    - Evaluacion con datos del CSV
    - Seleccion (ruleta o torneo)
    - Cruza y Mutacion (1% con reporte del metodo)
    - Registro de sincronizacion con el CSV
    """
    def __init__(self, x_datos, y_datos, tam_poblacion=100,
                 metodo_seleccion="ruleta", prob_cruza=0.90, prob_mutacion=0.01):
        self.x_datos = x_datos
        self.y_datos = y_datos
        self.tam_poblacion = tam_poblacion
        self.metodo_seleccion = metodo_seleccion  # 'ruleta' o 'torneo'
        self.prob_cruza = prob_cruza
        self.prob_mutacion = prob_mutacion

        self.generacion = 0
        self.poblacion = []
        self.mejor_individuo = None
        self.historial_errores = []
        self.registro_eventos = []

        self.inicializar_poblacion()

    def inicializar_poblacion(self):
        """Genera la poblacion inicial (100 ecuaciones aleatorias)."""
        self.poblacion = []
        self.generacion = 0
        self.historial_errores = []
        self.registro_eventos = []

        intentos = 0
        while len(self.poblacion) < self.tam_poblacion and intentos < self.tam_poblacion * 10:
            intentos += 1
            alt = random.choice([2, 3, 4])
            try:
                arbol = generar_arbol(altura_max=alt)
                ind = Individuo(arbol)
                ind.evaluar(self.x_datos, self.y_datos)
                self.poblacion.append(ind)
            except Exception:
                continue

        # En caso extremo de no completar
        while len(self.poblacion) < self.tam_poblacion:
            ind = Individuo("x")
            ind.evaluar(self.x_datos, self.y_datos)
            self.poblacion.append(ind)

        # Actualizar mejor individuo de la poblacion inicial
        self.mejor_individuo = min(self.poblacion, key=lambda ind: ind.adap)
        self.historial_errores.append(self.mejor_individuo.adap)
        self.registro_eventos.append(
            f"Gen 0: Poblacion inicial ({len(self.poblacion)} individuos). Mejor RMSE: {self.mejor_individuo.adap:.4f}"
        )

    def seleccionar_padre(self):
        if self.metodo_seleccion == "torneo":
            return seleccion_torneo(self.poblacion, k=3)
        return seleccion_ruleta(self.poblacion)

    def paso_generacion(self):
        """
        Ejecuta UNA generacion completa del algoritmo:
        1. Elitismo (conserva el mejor actual)
        2. Seleccion
        3. Cruza
        4. Mutacion (1% con reporte de cual de los 4 metodos ocurrio)
        5. Evaluacion
        """
        self.generacion += 1
        nueva_poblacion = []
        mutaciones_en_generacion = []

        # 1. Elitismo: preservamos los 2 mejores individuos intactos
        ordenados = sorted(self.poblacion, key=lambda ind: ind.adap)
        nueva_poblacion.append(Individuo(clonar_nodo(ordenados[0].raiz)))
        nueva_poblacion.append(Individuo(clonar_nodo(ordenados[1].raiz)))

        # 2. Generar el resto de la nueva poblacion
        while len(nueva_poblacion) < self.tam_poblacion:
            padre1 = self.seleccionar_padre()
            padre2 = self.seleccionar_padre()

            # Cruza
            if random.random() < self.prob_cruza:
                hijo1, hijo2 = cruza(padre1, padre2)
            else:
                hijo1 = Individuo(clonar_nodo(padre1.raiz))
                hijo2 = Individuo(clonar_nodo(padre2.raiz))

            # Mutacion al 1% con reporte
            hijo1, det1 = aplicar_mutacion(hijo1, prob=self.prob_mutacion)
            if det1:
                mutaciones_en_generacion.append(f"[Gen {self.generacion}] {det1}")

            hijo2, det2 = aplicar_mutacion(hijo2, prob=self.prob_mutacion)
            if det2:
                mutaciones_en_generacion.append(f"[Gen {self.generacion}] {det2}")

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < self.tam_poblacion:
                nueva_poblacion.append(hijo2)

        # 3. Evaluar nueva poblacion
        for ind in nueva_poblacion:
            ind.evaluar(self.x_datos, self.y_datos)

        self.poblacion = nueva_poblacion
        mejor_actual = min(self.poblacion, key=lambda ind: ind.adap)

        # Conservar el mejor historico absoluto
        if mejor_actual.adap < self.mejor_individuo.adap:
            self.mejor_individuo = mejor_actual
            self.registro_eventos.append(
                f"Gen {self.generacion}: 🌟 Nuevo mejor ajuste! RMSE: {mejor_actual.adap:.6f} | R²: {mejor_actual.metricas.get('r2', 0):.4f}"
            )
        else:
            # Reinsertar al mejor historico si la generacion actual no lo supero
            self.poblacion[0] = self.mejor_individuo

        self.historial_errores.append(self.mejor_individuo.adap)

        # Registrar mutaciones ocurridas en este ciclo
        for mut_log in mutaciones_en_generacion:
            self.registro_eventos.append(mut_log)

        return {
            "generacion": self.generacion,
            "mejor_rmse": self.mejor_individuo.adap,
            "mejor_mse": self.mejor_individuo.metricas.get("mse", np.nan),
            "mejor_nmse": self.mejor_individuo.metricas.get("nmse", np.nan),
            "mejor_r2": self.mejor_individuo.metricas.get("r2", np.nan),
            "mejor_expr": self.mejor_individuo.expresion,
            "mutaciones_reportadas": mutaciones_en_generacion,
        }
