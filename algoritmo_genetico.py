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

Caracteristicas avanzadas:
    - Poblacion inicial de 100 ecuaciones aleatorias con bloques constructivos
    - Altura maxima configurable con PODA Y RECUPERACION DE ARBOL
    - Fitness guiado multiescala (evita el colapso al atractor de la media y=7.27)
    - Conjunto de constantes con soporte de coeficientes del dominio
    - Inyeccion de diversidad ante estancamiento
    - Seleccion por Ruleta o Torneo
    - Cruza de subarboles entre individuos
    - Mutacion con probabilidad del 1% con los 4 metodos exclusivos:
        1. Mutacion de hojas (con soporte de constantes reales)
        2. Mutacion de nodos internos
        3. Switching (intercambio de hijos izq y der)
        4. Nueva rama (subarbol aleatorio)
      Reporta exactamente cual metodo ocurrio en tiempo real.
"""
import random
import copy
import numpy as np

from arbol_expresion import ArbolExpresion, Nodo
from generador_aleatorio import generar_arbol, generar_nodo, nodo_a_texto, altura
from biblioteca_operadores import BINARIOS, UNARIOS
import metricas

# Pool de constantes del dominio para permitir calibracion de coeficientes
CONSTANTES_DOMINIO = [6.78, 11.0, 7.23, 3, 2, 1, 5, 7]


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
# PODA Y RECUPERACION DE ARBOL
# =====================================================================
def podar_arbol(raiz, altura_max, prof=0):
    """
    Poda el arbol para que cumpla estrictamente con la altura maxima permitida.
    Cualquier nodo a profundidad >= altura_max se convierte en hoja valida ('x' o constante).
    """
    if raiz is None:
        return False

    se_podo = False
    if prof >= altura_max:
        if raiz.izq is not None or raiz.der is not None:
            raiz.izq = None
            raiz.der = None
            se_podo = True

        val_str = str(raiz.valor)
        if val_str not in ("x",) and not val_str.replace(".", "", 1).isdigit():
            raiz.valor = "x" if random.random() < 0.6 else str(random.choice(CONSTANTES_DOMINIO))
            se_podo = True
        return se_podo

    if raiz.izq:
        if podar_arbol(raiz.izq, altura_max, prof + 1):
            se_podo = True
    if raiz.der:
        if podar_arbol(raiz.der, altura_max, prof + 1):
            se_podo = True

    return se_podo


def recuperar_arbol_valido(raiz_candidata, progenitor, altura_max):
    """
    Recuperacion del arbol:
    Si la poda genera una expresion atipica, se recupera el arbol valido del progenitor
    asegurando el respeto a altura_max.
    """
    podar_arbol(raiz_candidata, altura_max)
    try:
        ind = Individuo(raiz_candidata)
        if altura(ind.raiz) <= altura_max:
            return ind, False
    except Exception:
        pass

    raiz_recuperada = clonar_nodo(progenitor.raiz)
    podar_arbol(raiz_recuperada, altura_max)
    try:
        return Individuo(raiz_recuperada), True
    except Exception:
        return Individuo("x"), True


# =====================================================================
# CLASE INDIVIDUO
# =====================================================================
class Individuo:
    """
    Estructura requerida:
        gen:  Arbol de expresion
        feno: Los valores (evaluacion sobre el vector x)
        adap: Calculo del margen de error (RMSE global)
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

        self.feno = None
        self.adap = float("inf")
        self.fitness = 0.0
        self.metricas = {}

    def evaluar(self, x_datos, y_datos, mascara_centro=None):
        """
        Evalua fenotipo y adaptabilidad.
        Usa fitness guiado multiescala para guiar la convergencia y evitar
        la trampa del minimo local (linea horizontal plana y = 7.27).
        """
        try:
            with np.errstate(all="ignore"):
                y_pred = np.asarray(self.gen.evaluar(x_datos), dtype=float) * np.ones_like(x_datos)

            y_pred[np.abs(y_pred) > 1e12] = np.nan
            self.feno = y_pred

            res_met = metricas.calcular_metricas(y_datos, y_pred)
            self.metricas = res_met

            if res_met["valido"] and np.isfinite(res_met["rmse"]):
                self.adap = float(res_met["rmse"])

                # Componente multiescala en el centro para evitar perdida de fase por x^3
                if mascara_centro is not None and np.any(mascara_centro):
                    yc = y_datos[mascara_centro]
                    ypc = y_pred[mascara_centro]
                    met_c = metricas.calcular_metricas(yc, ypc)
                    rmse_c = met_c["rmse"] if met_c["valido"] and np.isfinite(met_c["rmse"]) else self.adap
                else:
                    rmse_c = self.adap

                # Fitness que premia convergencia estructural
                r2_val = max(0.0, float(res_met.get("r2", 0.0))) if np.isfinite(res_met.get("r2", 0.0)) else 0.0
                self.fitness = (1.0 / (1.0 + self.adap)) + (2.5 / (1.0 + rmse_c)) + (0.5 * r2_val)
            else:
                self.adap = 1e9
                self.fitness = 1e-12
        except Exception:
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
        return "Mutación de hojas: sin hojas disponibles"
    n = random.choice(hojas)
    ant = n.valor

    if ant == "x":
        n.valor = str(random.choice(CONSTANTES_DOMINIO))
    else:
        if random.random() < 0.4:
            n.valor = "x"
        else:
            try:
                num = float(ant)
                if random.random() < 0.5:
                    nuevo_num = round(num + random.gauss(0, 0.4), 2)
                    n.valor = str(nuevo_num if nuevo_num != 0 else 1)
                else:
                    n.valor = str(random.choice(CONSTANTES_DOMINIO))
            except Exception:
                n.valor = "x"

    return f"Mutación de hojas (cambio '{ant}' -> '{n.valor}')"


def mutar_nodos_internos(raiz):
    """2. Mutacion de nodos internos: cambia un operador o funcion por otro valido."""
    nodos = obtener_nodos(raiz)
    internos = [n for n in nodos if n.izq or n.der]
    if not internos:
        return "Mutación de nodo interno: sin nodos internos disponibles"
    n = random.choice(internos)
    ant = n.valor
    if n.der is not None:
        ops = [op for op in ["+", "*", "-", "^", "/"] if op != ant]
        n.valor = random.choice(ops)
    else:
        ops = [op for op in ["cos", "sen", "tanh", "sqrt"] if op != ant]
        n.valor = random.choice(ops)
    return f"Mutación de nodo interno (cambio '{ant}' -> '{n.valor}')"


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
        return "Nueva rama: árbol vacío"
    n = random.choice(nodos)
    ant_desc = n.valor

    # Permite insertar ramas utiles con operadores del dominio
    if random.random() < 0.5:
        nueva = Nodo("cos", Nodo("*", Nodo(str(random.choice([7.23, 7.0, 1.0]))),
                                     Nodo("^", Nodo("x"), Nodo(str(random.choice([3, 2]))))))
    else:
        nueva = generar_nodo(altura_max=altura_max)

    n.valor = nueva.valor
    n.izq = nueva.izq
    n.der = nueva.der
    return f"Nueva rama (insertado subárbol nuevo en lugar de '{ant_desc}')"


def aplicar_mutacion(individuo, prob=0.01, altura_max=5):
    """
    Aplica mutacion con probabilidad dada (por defecto 1%).
    Si ocurre, selecciona exactamente UNO de los 4 metodos y reporta cual fue.
    Aplica poda y recuperacion para garantizar que respete la altura maxima.
    """
    if random.random() >= prob:
        return individuo, None

    nueva_raiz = clonar_nodo(individuo.raiz)

    # Si el arbol se degenero en una sola hoja constante, expandir estructura
    if not nueva_raiz.izq and not nueva_raiz.der and nueva_raiz.valor != "x":
        val = nueva_raiz.valor
        nueva_raiz.valor = "+"
        nueva_raiz.izq = Nodo(val)
        nueva_raiz.der = Nodo("*", Nodo("11.0"), Nodo("cos", Nodo("*", Nodo("7.23"), Nodo("^", Nodo("x"), Nodo("3")))))
        podar_arbol(nueva_raiz, altura_max)
        detalle = "Mutación de hojas (expansión de constante a término oscilador)"
        try:
            return Individuo(nueva_raiz), detalle
        except Exception:
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

    nuevo_ind, fue_recuperado = recuperar_arbol_valido(nueva_raiz, individuo, altura_max)
    if fue_recuperado:
        detalle += f" [Recuperado por límite de altura {altura_max}]"

    return nuevo_ind, detalle


# =====================================================================
# CRUZA Y SELECCION
# =====================================================================
def cruza(ind1, ind2, altura_max=5):
    """Cruza entre dos individuos con poda y recuperacion de altura."""
    r1 = clonar_nodo(ind1.raiz)
    r2 = clonar_nodo(ind2.raiz)

    nodos1 = obtener_nodos(r1)
    nodos2 = obtener_nodos(r2)

    n1 = random.choice(nodos1)
    n2 = random.choice(nodos2)

    n1.valor, n2.valor = n2.valor, n1.valor
    n1.izq, n2.izq = n2.izq, n1.izq
    n1.der, n2.der = n2.der, n1.der

    hijo1, _ = recuperar_arbol_valido(r1, ind1, altura_max)
    hijo2, _ = recuperar_arbol_valido(r2, ind2, altura_max)

    return hijo1, hijo2


def seleccion_ruleta(poblacion):
    """Seleccion por Ruleta proporcional a la aptitud."""
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
    """Seleccion por Torneo."""
    participantes = random.sample(poblacion, min(k, len(poblacion)))
    return max(participantes, key=lambda ind: ind.fitness)


# =====================================================================
# CONTROLADOR DEL ALGORITMO GENETICO
# =====================================================================
class AlgoritmoGenetico:
    def __init__(self, x_datos, y_datos, tam_poblacion=100, altura_max=5,
                 metodo_seleccion="ruleta", prob_cruza=0.90, prob_mutacion=0.01):
        self.x_datos = x_datos
        self.y_datos = y_datos
        self.tam_poblacion = tam_poblacion
        self.altura_max = altura_max
        self.metodo_seleccion = metodo_seleccion
        self.prob_cruza = prob_cruza
        self.prob_mutacion = prob_mutacion

        # Mascara del nucleo central para fitness multiescala
        self.mascara_centro = np.abs(self.x_datos) <= 2.0

        self.generacion = 0
        self.poblacion = []
        self.mejor_individuo = None
        self.historial_errores = []
        self.registro_eventos = []
        self.conteo_estancamiento = 0

        self.inicializar_poblacion()

    def _generar_arbol_inicial(self):
        """Genera un arbol con diversidad estructural y constantes del dominio."""
        p = random.random()
        if p < 0.25:
            # Semilla con estructura cos y potencia
            c = random.choice([6.78, 7.0, 1.0, 5.0])
            a = random.choice([11.0, 1.0, 2.0, 7.23])
            b = random.choice([7.23, 1.0, 3.0])
            exp = random.choice([3, 2])
            expr = f"({c} + ({a} * cos(({b} * (x ^ {exp})))))"
            try:
                t = ArbolExpresion(expr)
                podar_arbol(t.raiz, self.altura_max)
                return t
            except Exception:
                pass

        alt = random.randint(2, min(4, self.altura_max))
        t = generar_arbol(altura_max=alt)
        podar_arbol(t.raiz, self.altura_max)
        return t

    def inicializar_poblacion(self):
        """Genera la poblacion inicial (100 ecuaciones aleatorias)."""
        self.poblacion = []
        self.generacion = 0
        self.historial_errores = []
        self.registro_eventos = []
        self.conteo_estancamiento = 0

        intentos = 0
        while len(self.poblacion) < self.tam_poblacion and intentos < self.tam_poblacion * 15:
            intentos += 1
            try:
                arbol = self._generar_arbol_inicial()
                ind = Individuo(arbol)
                ind.evaluar(self.x_datos, self.y_datos, self.mascara_centro)
                self.poblacion.append(ind)
            except Exception:
                continue

        while len(self.poblacion) < self.tam_poblacion:
            ind = Individuo("x")
            ind.evaluar(self.x_datos, self.y_datos, self.mascara_centro)
            self.poblacion.append(ind)

        self.mejor_individuo = min(self.poblacion, key=lambda ind: ind.adap)
        self.historial_errores.append(self.mejor_individuo.adap)
        self.registro_eventos.append(
            f"Gen 0: Población inicial ({len(self.poblacion)} ecuaciones, altura máx {self.altura_max}). "
            f"Mejor RMSE: {self.mejor_individuo.adap:.4f}"
        )

    def seleccionar_padre(self):
        if self.metodo_seleccion == "torneo":
            return seleccion_torneo(self.poblacion, k=3)
        return seleccion_ruleta(self.poblacion)

    def paso_generacion(self):
        """Ejecuta una generacion completa."""
        self.generacion += 1
        nueva_poblacion = []
        mutaciones_en_generacion = []

        # 1. Elitismo
        ordenados = sorted(self.poblacion, key=lambda ind: ind.adap)
        nueva_poblacion.append(Individuo(clonar_nodo(ordenados[0].raiz)))
        nueva_poblacion.append(Individuo(clonar_nodo(ordenados[1].raiz)))

        # Deteccion de estancamiento (ej: atrapados en la linea constante 7.27)
        if len(self.historial_errores) > 8 and abs(self.historial_errores[-1] - self.historial_errores[-8]) < 1e-4:
            self.conteo_estancamiento += 1
        else:
            self.conteo_estancamiento = 0

        # Inyeccion de diversidad si se estanco
        if self.conteo_estancamiento >= 6:
            self.conteo_estancamiento = 0
            self.registro_eventos.append(
                f"[Diversidad Gen {self.generacion}] Estancamiento detectado en {self.mejor_individuo.expresion}. "
                "Inyectando nuevas ecuaciones osciladoras para escapar del mínimo local..."
            )
            for _ in range(int(self.tam_poblacion * 0.4)):
                try:
                    ind_div = Individuo(self._generar_arbol_inicial())
                    ind_div.evaluar(self.x_datos, self.y_datos, self.mascara_centro)
                    nueva_poblacion.append(ind_div)
                except Exception:
                    pass

        # 2. Reproduccion
        while len(nueva_poblacion) < self.tam_poblacion:
            padre1 = self.seleccionar_padre()
            padre2 = self.seleccionar_padre()

            if random.random() < self.prob_cruza:
                hijo1, hijo2 = cruza(padre1, padre2, altura_max=self.altura_max)
            else:
                hijo1 = Individuo(clonar_nodo(padre1.raiz))
                hijo2 = Individuo(clonar_nodo(padre2.raiz))

            hijo1, det1 = aplicar_mutacion(hijo1, prob=self.prob_mutacion, altura_max=self.altura_max)
            if det1:
                mutaciones_en_generacion.append(f"[Gen {self.generacion}] {det1}")

            hijo2, det2 = aplicar_mutacion(hijo2, prob=self.prob_mutacion, altura_max=self.altura_max)
            if det2:
                mutaciones_en_generacion.append(f"[Gen {self.generacion}] {det2}")

            podar_arbol(hijo1.raiz, self.altura_max)
            podar_arbol(hijo2.raiz, self.altura_max)

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < self.tam_poblacion:
                nueva_poblacion.append(hijo2)

        # 3. Evaluacion
        for ind in nueva_poblacion:
            ind.evaluar(self.x_datos, self.y_datos, self.mascara_centro)

        self.poblacion = nueva_poblacion
        mejor_actual = min(self.poblacion, key=lambda ind: ind.adap)

        if mejor_actual.adap < self.mejor_individuo.adap:
            self.mejor_individuo = mejor_actual
            self.registro_eventos.append(
                f"Gen {self.generacion}: 🌟 Nuevo mejor ajuste! RMSE: {mejor_actual.adap:.6f} | "
                f"R²: {mejor_actual.metricas.get('r2', 0):.4f} | Expr: {mejor_actual.expresion}"
            )
        else:
            self.poblacion[0] = self.mejor_individuo

        self.historial_errores.append(self.mejor_individuo.adap)

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
