"""
app.py  -  Interfaz gráfica (Tkinter) que integra todos los módulos.
Ejecutar:  python app.py
Instalar:  pip install numpy matplotlib

Características clave:
1. Pestaña 1 (Analizador & Árbol Binario):
   - Exclusivamente dedicada a escribir expresiones, graficar, árbol aleatorio y preorden.
   - Raíz estrictamente ARRIBA Y EN EL CENTRO con hijos claramente abajo.
   - Botón de Vista Ampliada / Pantalla Completa.
2. Pestaña 2 (Algoritmo Genético - Regresión Simbólica):
   - Contiene la carga y selección de CSV (📁 Seleccionar CSV, 🎯 Objetivo, 📊 Evaluar).
   - Configuración de Población (100), Generaciones, Selección (Ruleta/Torneo), Cruza y Mutación (1%).
   - NUEVA OPCIÓN: Altura máxima del árbol con PODA y RECUPERACIÓN estricta.
   - Sincronización visual en tiempo real de la curva con el CSV.
   - Botón '🌳 Ver Árbol Final vs CSV' para comparar lado a lado ambos árboles.
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from arbol_expresion import ArbolExpresion
from generador_aleatorio import generar_graficable, altura
from visualizacion import (
    ROSA, ROSA_CLARO, ROSA_MEDIO, ROSA_BORDE, BLANCO, NEGRO, TEXTO_MUTED,
    AZUL_OBJETIVO, GRIS_REJILLA, dibujar_arbol, dibujar_funcion
)
from metricas import (
    EXPRESION_OBJETIVO, ARCHIVO_CSV_DEFAULT,
    cargar_csv, evaluar_arbol_contra_datos, generar_csv
)
from algoritmo_genetico import AlgoritmoGenetico, Individuo

# Paleta cromática estética rosita suave
COLOR_FONDO_APP = "#faf3f7"
COLOR_TARJETA = "#ffffff"
COLOR_BORDE_TARJETA = "#edd4e2"
COLOR_HEADER = "#f8e1ec"
COLOR_TITULO = "#381a2c"
COLOR_SUBTITULO = "#754f66"
COLOR_BTN_PRIMARIO = "#d94a7a"
COLOR_BTN_PRIMARIO_HOVER = "#c53b6b"
COLOR_BTN_SECUNDARIO = "#faebf2"
COLOR_BTN_SECUNDARIO_TX = "#a83260"
COLOR_BTN_EVALUAR = "#993d6b"
COLOR_METRICA_BG = "#fcf2f6"
COLOR_METRICA_BORDE = "#ebd0df"


# =====================================================================
# VENTANA MODAL: VISTA AMPLIADA / PANTALLA COMPLETA DE LA GRÁFICA
# =====================================================================
class VentanaVistaAmpliada(tk.Toplevel):
    def __init__(self, parent, arbol, x_csv, y_csv, a=-10.0, b=10.0):
        super().__init__(parent)
        self.title("🔍 Vista Ampliada de Gráfica • Pantalla Completa")
        self.geometry("1280x820")
        self.minsize(950, 600)
        self.configure(bg=COLOR_FONDO_APP)

        self.arbol = arbol
        self.x_csv = x_csv
        self.y_csv = y_csv
        self.a = a
        self.b = b

        self.modo_vista = tk.StringVar(value="ambas")
        self.es_pantalla_completa = False

        self._crear_interfaz()
        self.actualizar_grafica()

    def _crear_interfaz(self):
        barra_sup = tk.Frame(self, bg=COLOR_TARJETA, padx=16, pady=10,
                             highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        barra_sup.pack(fill="x", padx=12, pady=(10, 6))

        tk.Label(barra_sup, text="Vista:", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 8))

        opciones = [
            ("🌸 Ambas funciones juntas", "ambas"),
            ("📈 Solo función f(x) del Árbol", "arbol"),
            ("📊 Solo función del CSV (Objetivo)", "csv"),
        ]
        for texto, valor in opciones:
            rb = tk.Radiobutton(
                barra_sup, text=texto, variable=self.modo_vista, value=valor,
                bg=COLOR_TARJETA, fg=COLOR_TITULO, activebackground=COLOR_TARJETA,
                selectcolor=COLOR_TARJETA, font=("Segoe UI", 9, "bold"),
                command=self.actualizar_grafica
            )
            rb.pack(side="left", padx=10)

        tk.Label(barra_sup, text="Rango [a, b]:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(20, 6))
        self.ent_a = tk.Entry(barra_sup, width=6, font=("Segoe UI", 9), justify="center",
                              bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_a.insert(0, str(self.a))
        self.ent_a.pack(side="left", padx=2)

        tk.Label(barra_sup, text="a", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO).pack(side="left", padx=2)

        self.ent_b = tk.Entry(barra_sup, width=6, font=("Segoe UI", 9), justify="center",
                              bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_b.insert(0, str(self.b))
        self.ent_b.pack(side="left", padx=2)

        btn_act = tk.Button(barra_sup, text="Actualizar Rango", bg=COLOR_BTN_SECUNDARIO,
                            fg=COLOR_BTN_SECUNDARIO_TX, font=("Segoe UI", 8, "bold"),
                            relief="flat", cursor="hand2", padx=8, pady=2,
                            command=self._leer_intervalo_y_actualizar)
        btn_act.pack(side="left", padx=8)

        btn_fs = tk.Button(barra_sup, text="⛶ Pantalla Completa", bg=COLOR_BTN_PRIMARIO,
                           fg=BLANCO, font=("Segoe UI", 8, "bold"),
                           relief="flat", cursor="hand2", padx=10, pady=2,
                           command=self._alternar_pantalla_completa)
        btn_fs.pack(side="right", padx=6)

        self.banner_metricas = tk.Frame(self, bg=COLOR_METRICA_BG, padx=16, pady=8,
                                        highlightbackground=COLOR_METRICA_BORDE, highlightthickness=1)
        self.banner_metricas.pack(fill="x", padx=12, pady=4)

        self.lbl_met_texto = tk.Label(self.banner_metricas, text="Calculando métricas...",
                                      bg=COLOR_METRICA_BG, fg=COLOR_TITULO,
                                      font=("Consolas", 10, "bold"))
        self.lbl_met_texto.pack(side="left")

        frame_plot = tk.Frame(self, bg=COLOR_TARJETA,
                              highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_plot.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        self.fig = Figure(figsize=(12, 7), facecolor=BLANCO)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, frame_plot)
        self.toolbar.config(background=COLOR_TARJETA)
        self.toolbar.update()

    def _alternar_pantalla_completa(self):
        self.es_pantalla_completa = not self.es_pantalla_completa
        self.attributes("-fullscreen", self.es_pantalla_completa)

    def _leer_intervalo_y_actualizar(self):
        try:
            self.a = float(self.ent_a.get().strip())
            self.b = float(self.ent_b.get().strip())
            if self.a >= self.b:
                raise ValueError("a debe ser menor que b")
            self.actualizar_grafica()
        except Exception as e:
            messagebox.showerror("Error de Rango", str(e), parent=self)

    def actualizar_grafica(self):
        self.ax.clear()
        self.ax.set_facecolor(BLANCO)
        modo = self.modo_vista.get()

        if modo in ("csv", "ambas") and self.x_csv is not None and self.y_csv is not None:
            mascara = (self.x_csv >= self.a) & (self.x_csv <= self.b)
            if np.any(mascara):
                x_sub = self.x_csv[mascara]
                y_sub = self.y_csv[mascara]
                if len(x_sub) > 2500:
                    paso = len(x_sub) // 2000
                    x_sub = x_sub[::paso]
                    y_sub = y_sub[::paso]
                self.ax.plot(x_sub, y_sub, color=AZUL_OBJETIVO, lw=2.2, ls="--",
                             label="Función del CSV (Objetivo)", alpha=0.9, zorder=2)

        if modo in ("arbol", "ambas") and self.arbol is not None:
            x_pts = np.linspace(self.a, self.b, 1200)
            with np.errstate(all="ignore"):
                y_pts = np.asarray(self.arbol.evaluar(x_pts), dtype=float) * np.ones_like(x_pts)
            y_pts[np.abs(y_pts) > 1e6] = np.nan
            self.ax.plot(x_pts, y_pts, color=ROSA, lw=2.6, label="Función del Árbol f(x)", zorder=3)

        self.ax.axhline(0, color="#bfa2b3", lw=0.9, ls="-", zorder=1)
        self.ax.axvline(0, color="#bfa2b3", lw=0.9, ls="-", zorder=1)
        self.ax.grid(True, color=GRIS_REJILLA, linestyle=":", lw=1.2, zorder=0)

        for spine in self.ax.spines.values():
            spine.set_color("#e3c9d6")
            spine.set_linewidth(1.2)

        self.ax.tick_params(colors=TEXTO_MUTED, labelsize=10)
        self.ax.set_title(f"Vista Ampliada de Funciones en [{self.a}, {self.b}]",
                          color=NEGRO, fontsize=13, fontweight="bold", pad=12)
        self.ax.set_xlabel("x", color=NEGRO, fontsize=11, fontweight="bold")
        self.ax.set_ylabel("f(x)", color=NEGRO, fontsize=11, fontweight="bold")
        self.ax.legend(frameon=True, facecolor=BLANCO, edgecolor=ROSA_BORDE, fontsize=10, loc="best")

        self.fig.tight_layout()
        self.canvas.draw()

        if self.arbol is not None and self.x_csv is not None and self.y_csv is not None:
            try:
                _, met = evaluar_arbol_contra_datos(self.arbol, self.x_csv, self.y_csv)
                if met["valido"]:
                    txt = (
                        f"Margen de Error:  RMSE = {met['rmse']:.6f}  |  MSE = {met['mse']:.6f}  |  "
                        f"NUSE / NMSE = {met['nmse']:.6f}  |  R² = {met['r2']:.4f}  |  "
                        f"ÿ = {met['y_barra']:.4f}  |  Puntos: {met['n_validos']:,} / {met['n_total']:,}"
                    )
                else:
                    txt = "Margen de Error: Indefinido (valores infinitos o no numéricos)"
            except Exception as e:
                txt = f"Error al calcular métricas: {e}"
        else:
            txt = "Sin dataset CSV cargado para calcular margen de error"

        self.lbl_met_texto.config(text=txt)


# =====================================================================
# VENTANA COMPARATIVA: ÁRBOL FINAL VS ÁRBOL DEL CSV
# =====================================================================
class VentanaComparativaArboles(tk.Toplevel):
    """
    Muestra lado a lado:
    1. Árbol de expresión al que corresponde el CSV (Función Objetivo).
    2. Árbol de expresión final encontrado por el Algoritmo Genético.
    """
    def __init__(self, parent, arbol_objetivo, arbol_final, stats_final=""):
        super().__init__(parent)
        self.title("🌳 Comparación: Árbol Objetivo (CSV) vs Árbol Final (Algoritmo Genético)")
        self.geometry("1350x750")
        self.minsize(950, 600)
        self.configure(bg=COLOR_FONDO_APP)

        header = tk.Frame(self, bg=COLOR_HEADER, padx=16, pady=10,
                          highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        header.pack(fill="x", padx=12, pady=(10, 6))

        tk.Label(header, text="🌳 Comparativa Estructural de Árboles Binarios",
                 bg=COLOR_HEADER, fg=COLOR_TITULO, font=("Segoe UI", 14, "bold")).pack(anchor="w")

        sub_txt = f"Izquierda: Árbol Objetivo del CSV  •  Derecha: Árbol Final Encontrado {stats_final}"
        tk.Label(header, text=sub_txt, bg=COLOR_HEADER, fg=COLOR_SUBTITULO, font=("Segoe UI", 9)).pack(anchor="w")

        frame_canvas = tk.Frame(self, bg=COLOR_TARJETA,
                                highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_canvas.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        fig = Figure(figsize=(14, 6.5), facecolor=BLANCO)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # 1. Dibujar árbol objetivo
        if arbol_objetivo:
            h_obj = altura(arbol_objetivo.raiz)
            tit_obj = f"Árbol Objetivo (CSV)\n{arbol_objetivo.expresion}\n(Altura: {h_obj})"
            dibujar_arbol(ax1, arbol_objetivo, titulo=tit_obj)
        else:
            ax1.text(0.5, 0.5, "Sin árbol objetivo", ha="center", va="center")
            ax1.axis("off")

        # 2. Dibujar árbol final
        if arbol_final:
            h_fin = altura(arbol_final.raiz)
            tit_fin = f"Árbol Final (Algoritmo Genético)\n{arbol_final.expresion}\n(Altura: {h_fin})"
            dibujar_arbol(ax2, arbol_final, titulo=tit_fin)
        else:
            ax2.text(0.5, 0.5, "Sin árbol final", ha="center", va="center")
            ax2.axis("off")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame_canvas)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        canvas.draw()


# =====================================================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# =====================================================================
class App:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("🌸 Árbol de Expresión Binario & Algoritmo Genético")
        self.ventana.geometry("1280x880")
        self.ventana.minsize(1080, 740)
        self.ventana.configure(bg=COLOR_FONDO_APP)

        # Dataset CSV
        self.ruta_csv = ARCHIVO_CSV_DEFAULT
        self.x_csv = None
        self.y_csv = None
        self._cargar_datos_csv()

        # Árbol objetivo del CSV
        try:
            self.arbol_objetivo = ArbolExpresion(EXPRESION_OBJETIVO)
        except Exception:
            self.arbol_objetivo = None

        # Variables de estado del analizador
        self.arbol_actual = None

        # Variables del algoritmo genético
        self.ga = None
        self.ga_corriendo = False
        self.ga_timer_id = None
        self.max_generaciones = 100

        self._crear_interfaz_principal()
        self.graficar()

    def _cargar_datos_csv(self):
        try:
            if not os.path.exists(self.ruta_csv):
                generar_csv(self.ruta_csv)
            self.x_csv, self.y_csv = cargar_csv(self.ruta_csv)
        except Exception as e:
            self.x_csv, self.y_csv = None, None
            print(f"Aviso CSV: {e}")

    def _crear_interfaz_principal(self):
        header = tk.Frame(self.ventana, bg=COLOR_HEADER, pady=10, padx=16,
                          highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        header.pack(fill="x", padx=12, pady=(10, 6))

        titulo_box = tk.Frame(header, bg=COLOR_HEADER)
        titulo_box.pack(side="left")

        tk.Label(titulo_box, text="🌸 Árbol de Expresión Binario & Regresión Genética",
                 bg=COLOR_HEADER, fg=COLOR_TITULO, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(titulo_box,
                 text="Modelado simbólico jerárquico • Algoritmo genético con mutación (1%), poda y recuperación",
                 bg=COLOR_HEADER, fg=COLOR_SUBTITULO, font=("Segoe UI", 9)).pack(anchor="w")

        # Pestañas
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=COLOR_FONDO_APP, borderwidth=0)
        style.configure("TNotebook.Tab", background=COLOR_HEADER, foreground=COLOR_TITULO,
                        font=("Segoe UI", 10, "bold"), padding=[20, 7])
        style.map("TNotebook.Tab",
                  background=[("selected", COLOR_TARJETA)],
                  foreground=[("selected", COLOR_BTN_PRIMARIO)])

        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        # Pestaña 1: Analizador de Árbol
        self.tab_analizador = tk.Frame(self.notebook, bg=COLOR_FONDO_APP)
        self.notebook.add(self.tab_analizador, text="  🌸 Analizador & Árbol Binario  ")

        # Pestaña 2: Algoritmo Genético
        self.tab_genetico = tk.Frame(self.notebook, bg=COLOR_FONDO_APP)
        self.notebook.add(self.tab_genetico, text="  🧬 Algoritmo Genético (Regresión)  ")

        self._construir_pestana_analizador()
        self._construir_pestana_genetico()

    # =========================================================================
    # PESTAÑA 1: ANALIZADOR DE ÁRBOL
    # =========================================================================
    def _construir_pestana_analizador(self):
        panel = tk.Frame(self.tab_analizador, bg=COLOR_TARJETA, padx=14, pady=8,
                         highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        panel.pack(fill="x", pady=(6, 4))

        # Fila 1: Expresión
        f1 = tk.Frame(panel, bg=COLOR_TARJETA)
        f1.pack(fill="x", pady=(2, 6))

        tk.Label(f1, text="Expresión f(x):", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                 font=("Segoe UI", 10, "bold"), width=13, anchor="w").pack(side="left")

        self.entrada_expr = tk.Entry(f1, font=("Consolas", 11), bg="#fffafc", fg=NEGRO,
                                     highlightbackground=COLOR_BORDE_TARJETA,
                                     highlightcolor=COLOR_BTN_PRIMARIO,
                                     highlightthickness=1, relief="flat")
        self.entrada_expr.insert(0, "sqrt(2 * (x ^ 2) + 5 * x - 3)")
        self.entrada_expr.pack(side="left", fill="x", expand=True, padx=(4, 10), ipady=4)

        # Fila 2: Parámetros y botones
        f2 = tk.Frame(panel, bg=COLOR_TARJETA)
        f2.pack(fill="x", pady=2)

        tk.Label(f2, text="Intervalo [a, b]:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.entrada_int = tk.Entry(f2, width=12, font=("Segoe UI", 10),
                                    bg="#fffafc", fg=NEGRO, justify="center",
                                    highlightbackground=COLOR_BORDE_TARJETA,
                                    highlightthickness=1, relief="flat")
        self.entrada_int.insert(0, "-10, 10")
        self.entrada_int.pack(side="left", padx=(6, 16), ipady=2)

        tk.Label(f2, text="Altura máx:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.entrada_alt = tk.Entry(f2, width=5, font=("Segoe UI", 10),
                                    bg="#fffafc", fg=NEGRO, justify="center",
                                    highlightbackground=COLOR_BORDE_TARJETA,
                                    highlightthickness=1, relief="flat")
        self.entrada_alt.insert(0, "4")
        self.entrada_alt.pack(side="left", padx=(6, 20), ipady=2)

        # Botones
        btn_prim = dict(bg=COLOR_BTN_PRIMARIO, fg=BLANCO, activebackground=COLOR_BTN_PRIMARIO_HOVER,
                        activeforeground=BLANCO, font=("Segoe UI", 9, "bold"), relief="flat",
                        cursor="hand2", padx=14, pady=3)
        btn_sec = dict(bg=COLOR_BTN_SECUNDARIO, fg=COLOR_BTN_SECUNDARIO_TX,
                       activebackground="#f3d3e2", activeforeground=COLOR_BTN_SECUNDARIO_TX,
                       font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                       highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1,
                       padx=12, pady=3)

        tk.Button(f2, text="📈 Graficar", command=self.graficar, **btn_prim).pack(side="left", padx=4)
        tk.Button(f2, text="🎲 Aleatoria", command=self.aleatoria, **btn_sec).pack(side="left", padx=4)

        # Botón de vista ampliada
        tk.Button(f2, text="🔍 Vista Ampliada Gráfica", command=self.abrir_vista_ampliada,
                  bg="#6a4c93", fg=BLANCO, activebackground="#583c7f", activeforeground=BLANCO,
                  font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", padx=14, pady=3).pack(side="left", padx=8)

        # Card de Info del Árbol
        card_info = tk.Frame(self.tab_analizador, bg=COLOR_TARJETA, padx=14, pady=6,
                             highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        card_info.pack(fill="x", pady=4)

        info_top = tk.Frame(card_info, bg=COLOR_TARJETA)
        info_top.pack(fill="x")

        self.lbl_altura = tk.Label(info_top, text="Altura: -", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                                   font=("Segoe UI", 10, "bold"))
        self.lbl_altura.pack(side="left")

        self.lbl_preorden = tk.Label(info_top, text="Preorden: -", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                                     font=("Consolas", 9), anchor="w")
        self.lbl_preorden.pack(side="left", padx=(16, 0), fill="x", expand=True)

        # Canvas Matplotlib con 2 subplots (Árbol con raíz arriba centrado, y función f(x))
        frame_canvas = tk.Frame(self.tab_analizador, bg=COLOR_TARJETA,
                                highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_canvas.pack(fill="both", expand=True, pady=(2, 6))

        self.fig = Figure(figsize=(12, 5.2), facecolor=BLANCO)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_canvas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

    # =========================================================================
    # PESTAÑA 2: ALGORITMO GENÉTICO (AQUÍ ESTÁ TODO LO DEL CSV)
    # =========================================================================
    def _construir_pestana_genetico(self):
        # 1. Card superior de Dataset CSV y Evaluación
        card_csv = tk.Frame(self.tab_genetico, bg=COLOR_TARJETA, padx=14, pady=8,
                            highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        card_csv.pack(fill="x", pady=(6, 4))

        f_csv_info = tk.Frame(card_csv, bg=COLOR_TARJETA)
        f_csv_info.pack(fill="x", pady=(0, 4))

        tk.Label(f_csv_info, text="📄 Dataset CSV:", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")

        nom_csv = os.path.basename(self.ruta_csv) if self.ruta_csv else "Sin archivo"
        pts = f"{len(self.x_csv):,} puntos" if self.x_csv is not None else "0 puntos"
        self.lbl_csv_nombre = tk.Label(f_csv_info, text=f"{nom_csv} ({pts})",
                                      bg=COLOR_TARJETA, fg=COLOR_SUBTITULO, font=("Segoe UI", 9))
        self.lbl_csv_nombre.pack(side="left", padx=(6, 16))

        btn_sec = dict(bg=COLOR_BTN_SECUNDARIO, fg=COLOR_BTN_SECUNDARIO_TX, activebackground="#f3d3e2",
                       font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                       highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1, padx=10, pady=3)

        tk.Button(f_csv_info, text="📁 Seleccionar / Cargar CSV...", command=self.seleccionar_csv, **btn_sec).pack(side="left", padx=4)
        tk.Button(f_csv_info, text="🎯 Cargar Función Objetivo CSV", command=self.ga_cargar_objetivo_csv, **btn_sec).pack(side="left", padx=4)
        tk.Button(f_csv_info, text="📊 Evaluar Expresión vs CSV", command=self.ga_evaluar_expresion_manual,
                  bg=COLOR_BTN_EVALUAR, fg=BLANCO, font=("Segoe UI", 9, "bold"), relief="flat",
                  cursor="hand2", padx=12, pady=3).pack(side="left", padx=6)

        # 2. Barra de Parámetros del Algoritmo Genético
        cfg_panel = tk.Frame(self.tab_genetico, bg=COLOR_TARJETA, padx=14, pady=8,
                             highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        cfg_panel.pack(fill="x", pady=4)

        tk.Label(cfg_panel, text="Población:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_pop = tk.Entry(cfg_panel, width=5, font=("Segoe UI", 9), justify="center",
                                   bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_pop.insert(0, "100")
        self.ent_ga_pop.pack(side="left", padx=(4, 10))

        tk.Label(cfg_panel, text="Generaciones:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_gen = tk.Entry(cfg_panel, width=5, font=("Segoe UI", 9), justify="center",
                                   bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_gen.insert(0, "100")
        self.ent_ga_gen.pack(side="left", padx=(4, 10))

        # NUEVO: Opción de altura máxima con poda y recuperación
        tk.Label(cfg_panel, text="Altura máx (Poda):", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_alt = tk.Entry(cfg_panel, width=4, font=("Segoe UI", 9), justify="center",
                                   bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_alt.insert(0, "5")
        self.ent_ga_alt.pack(side="left", padx=(4, 10))

        tk.Label(cfg_panel, text="Selección:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.cbo_ga_sel = ttk.Combobox(cfg_panel, values=["Ruleta", "Torneo"],
                                       width=7, state="readonly", font=("Segoe UI", 9))
        self.cbo_ga_sel.set("Ruleta")
        self.cbo_ga_sel.pack(side="left", padx=(4, 10))

        tk.Label(cfg_panel, text="Cruza:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_pcx = tk.Entry(cfg_panel, width=5, font=("Segoe UI", 9), justify="center",
                                   bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_pcx.insert(0, "0.90")
        self.ent_ga_pcx.pack(side="left", padx=(4, 10))

        tk.Label(cfg_panel, text="Mutación (1%):", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.ent_ga_pmut = tk.Entry(cfg_panel, width=5, font=("Segoe UI", 9), justify="center",
                                    bg="#fffafc", highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.ent_ga_pmut.insert(0, "0.01")
        self.ent_ga_pmut.pack(side="left", padx=(4, 12))

        # Botones de control
        btn_start = dict(bg=COLOR_BTN_PRIMARIO, fg=BLANCO, activebackground=COLOR_BTN_PRIMARIO_HOVER,
                         font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", padx=12, pady=3)

        self.btn_ga_iniciar = tk.Button(cfg_panel, text="▶ Iniciar Evolución",
                                        command=self.ga_iniciar_o_pausar, **btn_start)
        self.btn_ga_iniciar.pack(side="left", padx=3)

        self.btn_ga_detener = tk.Button(cfg_panel, text="⏹ Detener", command=self.ga_detener,
                                        state="disabled", **btn_sec)
        self.btn_ga_detener.pack(side="left", padx=3)

        self.btn_ga_reiniciar = tk.Button(cfg_panel, text="🔄 Reiniciar Población",
                                          command=self.ga_reiniciar_poblacion, **btn_sec)
        self.btn_ga_reiniciar.pack(side="left", padx=3)

        # Botón para ver Árbol Final vs Árbol del CSV
        self.btn_ver_arbol_final = tk.Button(
            cfg_panel, text="🌳 Ver Árbol Final vs CSV", command=self.ga_ver_arbol_final_vs_csv,
            bg="#5a4275", fg=BLANCO, activebackground="#4a3265", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", padx=12, pady=3
        )
        self.btn_ver_arbol_final.pack(side="right", padx=4)

        btn_enviar = tk.Button(
            cfg_panel, text="📋 Enviar a Analizador", command=self.ga_enviar_a_analizador,
            bg=COLOR_TARJETA, fg=COLOR_BTN_PRIMARIO, font=("Segoe UI", 8, "bold"),
            relief="flat", cursor="hand2", highlightbackground=COLOR_BTN_PRIMARIO,
            highlightthickness=1, padx=8, pady=3
        )
        btn_enviar.pack(side="right", padx=4)

        # 3. Card de Métricas en vivo del AG
        card_ga_status = tk.Frame(self.tab_genetico, bg=COLOR_TARJETA, padx=14, pady=6,
                                  highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        card_ga_status.pack(fill="x", pady=4)

        f_status_top = tk.Frame(card_ga_status, bg=COLOR_TARJETA)
        f_status_top.pack(fill="x", pady=(0, 4))

        self.lbl_ga_gen = tk.Label(f_status_top, text="Generación: 0 / 100", bg=COLOR_TARJETA,
                                   fg=COLOR_TITULO, font=("Segoe UI", 10, "bold"))
        self.lbl_ga_gen.pack(side="left")

        self.lbl_ga_expr = tk.Label(f_status_top, text="Mejor Expresión: ---", bg=COLOR_TARJETA,
                                    fg=COLOR_BTN_PRIMARIO, font=("Consolas", 10, "bold"), anchor="w")
        self.lbl_ga_expr.pack(side="left", padx=(20, 0), fill="x", expand=True)

        f_status_met = tk.Frame(card_ga_status, bg=COLOR_TARJETA)
        f_status_met.pack(fill="x")

        self.ga_metricas_labels = {}
        items_ga_met = [
            ("Adap (RMSE)", "lbl_ga_rmse"),
            ("MSE",          "lbl_ga_mse"),
            ("NMSE / NUSE",  "lbl_ga_nmse"),
            ("R² (Ajuste)",  "lbl_ga_r2"),
            ("ÿ (Media y)",  "lbl_ga_y_barra"),
            ("Estado",       "lbl_ga_estado"),
        ]
        for nom, clv in items_ga_met:
            caja = tk.Frame(f_status_met, bg=COLOR_METRICA_BG, padx=8, pady=3,
                            highlightbackground=COLOR_METRICA_BORDE, highlightthickness=1)
            caja.pack(side="left", fill="both", expand=True, padx=2)
            tk.Label(caja, text=nom, bg=COLOR_METRICA_BG, fg=COLOR_SUBTITULO,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            lbl_v = tk.Label(caja, text="---", bg=COLOR_METRICA_BG, fg=COLOR_TITULO,
                             font=("Consolas", 10, "bold"))
            lbl_v.pack(anchor="w")
            self.ga_metricas_labels[clv] = lbl_v

        # 4. Área dividida: Gráfica de sincronización en vivo + Registro de mutaciones
        split_frame = tk.Frame(self.tab_genetico, bg=COLOR_FONDO_APP)
        split_frame.pack(fill="both", expand=True, pady=(2, 6))

        # Panel izquierdo: Gráfica en vivo
        frame_plot_ga = tk.Frame(split_frame, bg=COLOR_TARJETA,
                                 highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_plot_ga.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.fig_ga = Figure(figsize=(7.5, 4.8), facecolor=BLANCO)
        self.ax_ga = self.fig_ga.add_subplot(1, 1, 1)
        self.canvas_ga = FigureCanvasTkAgg(self.fig_ga, master=frame_plot_ga)
        self.canvas_ga.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

        # Panel derecho: Registro de eventos y mutaciones
        frame_log_ga = tk.Frame(split_frame, bg=COLOR_TARJETA, width=380,
                                highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_log_ga.pack(side="right", fill="both", padx=(4, 0))
        frame_log_ga.pack_propagate(False)

        tk.Label(frame_log_ga, text="📝 Registro de Eventos y Mutaciones (1%):",
                 bg=COLOR_TARJETA, fg=COLOR_TITULO, font=("Segoe UI", 9, "bold"), pady=4).pack(anchor="w", padx=6)

        log_scroll = tk.Scrollbar(frame_log_ga)
        log_scroll.pack(side="right", fill="y")

        self.txt_ga_log = tk.Text(frame_log_ga, font=("Consolas", 8), bg="#fffdfb", fg=NEGRO,
                                  yscrollcommand=log_scroll.set, wrap="word", relief="flat")
        self.txt_ga_log.pack(fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        log_scroll.config(command=self.txt_ga_log.yview)

    # =========================================================================
    # ACCIONES DEL ANALIZADOR
    # =========================================================================
    def _intervalo(self):
        txt = self.entrada_int.get().strip()
        partes = txt.split(",")
        if len(partes) != 2:
            raise ValueError("El intervalo debe tener el formato a, b (ejemplo: -10, 10)")
        a, b = float(partes[0]), float(partes[1])
        if a >= b:
            raise ValueError("En el intervalo [a, b], 'a' debe ser menor que 'b'")
        return a, b

    def _altura_max(self):
        val = int(self.entrada_alt.get().strip())
        if val < 0 or val > 12:
            raise ValueError("La altura máxima debe estar entre 0 y 12")
        return val

    def graficar(self):
        try:
            expr = self.entrada_expr.get().strip()
            if not expr:
                raise ValueError("Por favor escribe una expresión matemática")
            self.arbol_actual = ArbolExpresion(expr)
            self.mostrar(self.arbol_actual)
        except Exception as error:
            messagebox.showerror("Error de Expresión", f"Revisa tus datos:\n{error}")

    def aleatoria(self):
        try:
            a, b = self._intervalo()
            self.arbol_actual = generar_graficable(self._altura_max(), a, b)
            self.entrada_expr.delete(0, "end")
            self.entrada_expr.insert(0, self.arbol_actual.expresion)
            self.mostrar(self.arbol_actual)
        except Exception as error:
            messagebox.showerror("Error", f"Revisa tus datos:\n{error}")

    def abrir_vista_ampliada(self):
        try:
            a, b = self._intervalo()
            expr = self.entrada_expr.get().strip()
            arbol = ArbolExpresion(expr) if expr else self.arbol_actual
            VentanaVistaAmpliada(self.ventana, arbol, self.x_csv, self.y_csv, a, b)
        except Exception as error:
            messagebox.showerror("Error", f"No se pudo abrir la vista ampliada:\n{error}")

    def mostrar(self, arbol):
        a, b = self._intervalo()
        h, h_max = altura(arbol.raiz), self._altura_max()

        self.fig.clear()
        ax_arbol = self.fig.add_subplot(1, 2, 1)
        ax_func = self.fig.add_subplot(1, 2, 2)

        dibujar_arbol(ax_arbol, arbol)
        dibujar_funcion(ax_func, arbol, a, b)

        self.fig.tight_layout()
        self.canvas.draw()

        aviso = "  ⚠️ (excede altura máx)" if h > h_max else ""
        self.lbl_altura.config(text=f"Altura: {h} / {h_max}{aviso}")
        tokens_pre = "  ".join(arbol.preorden())
        if len(tokens_pre) > 95:
            tokens_pre = tokens_pre[:92] + "..."
        self.lbl_preorden.config(text=f"Preorden: {tokens_pre}")

    # =========================================================================
    # ACCIONES DEL ALGORITMO GENÉTICO & CSV
    # =========================================================================
    def seleccionar_csv(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo CSV con datos (x, y)",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if ruta:
            try:
                x, y = cargar_csv(ruta)
                self.ruta_csv = ruta
                self.x_csv, self.y_csv = x, y
                nom_csv = os.path.basename(ruta)
                self.lbl_csv_nombre.config(text=f"{nom_csv} ({len(x):,} puntos)")
                messagebox.showinfo("CSV cargado", f"Se cargaron exitosamente {len(x):,} puntos desde:\n{nom_csv}")
                self.ga_reiniciar_poblacion()
            except Exception as e:
                messagebox.showerror("Error al cargar CSV", f"No se pudo leer el archivo:\n{e}")

    def ga_cargar_objetivo_csv(self):
        try:
            self.arbol_objetivo = ArbolExpresion(EXPRESION_OBJETIVO)
            messagebox.showinfo(
                "Función Objetivo del CSV",
                f"Función objetivo de referencia cargada:\n\n{EXPRESION_OBJETIVO}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def ga_evaluar_expresion_manual(self):
        """Evalúa una expresión personalizada contra el CSV en esta pestaña."""
        try:
            expr = self.entrada_expr.get().strip()
            if not expr:
                raise ValueError("Por favor escribe una expresión en la caja de texto")
            arbol = ArbolExpresion(expr)
            if self.x_csv is None or self.y_csv is None:
                self._cargar_datos_csv()
            _, met = evaluar_arbol_contra_datos(arbol, self.x_csv, self.y_csv)
            if met["valido"]:
                self.ga_metricas_labels["lbl_ga_rmse"].config(text=f"{met['rmse']:.6f}")
                self.ga_metricas_labels["lbl_ga_mse"].config(text=f"{met['mse']:.6f}")
                self.ga_metricas_labels["lbl_ga_nmse"].config(text=f"{met['nmse']:.6f}")
                self.ga_metricas_labels["lbl_ga_r2"].config(text=f"{met['r2']:.4f}")
                self.ga_metricas_labels["lbl_ga_y_barra"].config(text=f"{met['y_barra']:.4f}")
                self.lbl_ga_expr.config(text=f"Evaluada: {expr}")
                messagebox.showinfo("Evaluación con CSV",
                                    f"Expresión: {expr}\n\n"
                                    f"RMSE: {met['rmse']:.6f}\n"
                                    f"MSE:  {met['mse']:.6f}\n"
                                    f"NUSE: {met['nmse']:.6f}\n"
                                    f"R²:   {met['r2']:.4f}\n"
                                    f"ÿ:    {met['y_barra']:.4f}")
        except Exception as e:
            messagebox.showerror("Error de Evaluación", str(e))

    def ga_preparar(self):
        if self.x_csv is None or self.y_csv is None:
            self._cargar_datos_csv()
            if self.x_csv is None:
                raise ValueError("No se encontraron datos CSV para ejecutar la evolución")

        tam_pop = int(self.ent_ga_pop.get().strip())
        self.max_generaciones = int(self.ent_ga_gen.get().strip())
        alt_max = int(self.ent_ga_alt.get().strip())
        met_sel = self.cbo_ga_sel.get().strip().lower()
        pcx = float(self.ent_ga_pcx.get().strip())
        pmut = float(self.ent_ga_pmut.get().strip())

        paso_muestreo = max(1, len(self.x_csv) // 1500)
        x_eval = self.x_csv[::paso_muestreo]
        y_eval = self.y_csv[::paso_muestreo]

        self.ga = AlgoritmoGenetico(
            x_eval, y_eval, tam_poblacion=tam_pop, altura_max=alt_max,
            metodo_seleccion=met_sel, prob_cruza=pcx, prob_mutacion=pmut
        )
        self.txt_ga_log.delete("1.0", "end")
        self._ga_log(f"Población inicial generada ({tam_pop} ecuaciones aleatorias, altura máx {alt_max} con poda).")
        self.ga_actualizar_ui()

    def _ga_log(self, mensaje):
        self.txt_ga_log.insert("end", mensaje + "\n")
        self.txt_ga_log.see("end")

    def ga_iniciar_o_pausar(self):
        if not self.ga_corriendo:
            try:
                if self.ga is None:
                    self.ga_preparar()
                self.ga_corriendo = True
                self.btn_ga_iniciar.config(text="⏸ Pausar Evolución", bg="#e08e45")
                self.btn_ga_detener.config(state="normal")
                self.ga_metricas_labels["lbl_ga_estado"].config(text="Evolucionando...", fg=COLOR_BTN_PRIMARIO)
                self.ga_iterar()
            except Exception as e:
                messagebox.showerror("Error AG", str(e))
        else:
            self.ga_pausar()

    def ga_pausar(self):
        self.ga_corriendo = False
        if self.ga_timer_id:
            self.ventana.after_cancel(self.ga_timer_id)
            self.ga_timer_id = None
        self.btn_ga_iniciar.config(text="▶ Reanudar Evolución", bg=COLOR_BTN_PRIMARIO)
        self.ga_metricas_labels["lbl_ga_estado"].config(text="Pausado", fg=COLOR_SUBTITULO)

    def ga_detener(self):
        self.ga_pausar()
        self.btn_ga_iniciar.config(text="▶ Iniciar Evolución", bg=COLOR_BTN_PRIMARIO)
        self.btn_ga_detener.config(state="disabled")
        self.ga_metricas_labels["lbl_ga_estado"].config(text="Detenido", fg=COLOR_SUBTITULO)

    def ga_reiniciar_poblacion(self):
        self.ga_detener()
        self.ga = None
        self.ga_preparar()

    def ga_iterar(self):
        if not self.ga_corriendo or self.ga is None:
            return

        if self.ga.generacion >= self.max_generaciones:
            self.ga_detener()
            self._ga_log(f"🎉 Se completaron las {self.max_generaciones} generaciones!")
            self.ga_metricas_labels["lbl_ga_estado"].config(text="Completado", fg="#2a804d")
            return

        res = self.ga.paso_generacion()

        for mut_detalle in res["mutaciones_reportadas"]:
            self._ga_log(mut_detalle)

        self.ga_actualizar_ui()
        self.ga_timer_id = self.ventana.after(15, self.ga_iterar)

    def ga_actualizar_ui(self):
        if self.ga is None or self.ga.mejor_individuo is None:
            return

        ind = self.ga.mejor_individuo
        met = ind.metricas

        self.lbl_ga_gen.config(text=f"Generación: {self.ga.generacion} / {self.max_generaciones}")
        expr_corta = ind.expresion
        if len(expr_corta) > 75:
            expr_corta = expr_corta[:72] + "..."
        self.lbl_ga_expr.config(text=f"Mejor: {expr_corta}")

        self.ga_metricas_labels["lbl_ga_rmse"].config(text=f"{ind.adap:.6f}")
        self.ga_metricas_labels["lbl_ga_mse"].config(text=f"{met.get('mse', 0):.6f}")
        self.ga_metricas_labels["lbl_ga_nmse"].config(text=f"{met.get('nmse', 0):.6f}")
        self.ga_metricas_labels["lbl_ga_r2"].config(text=f"{met.get('r2', 0):.4f}")
        self.ga_metricas_labels["lbl_ga_y_barra"].config(text=f"{met.get('y_barra', 0):.4f}")

        self.ax_ga.clear()
        self.ax_ga.set_facecolor(BLANCO)

        if self.x_csv is not None and self.y_csv is not None:
            paso = max(1, len(self.x_csv) // 800)
            self.ax_ga.plot(self.x_csv[::paso], self.y_csv[::paso], color=AZUL_OBJETIVO,
                            lw=2.0, ls="--", label="Objetivo y (CSV)", alpha=0.85)

        if ind.gen is not None:
            x_plot = np.linspace(-10, 10, 800)
            with np.errstate(all="ignore"):
                y_pred = np.asarray(ind.gen.evaluar(x_plot), dtype=float) * np.ones_like(x_plot)
            y_pred[np.abs(y_pred) > 1e5] = np.nan
            self.ax_ga.plot(x_plot, y_pred, color=ROSA, lw=2.4,
                            label=f"Mejor f(x) (Gen {self.ga.generacion})")

        self.ax_ga.axhline(0, color="#bfa2b3", lw=0.8, ls="-")
        self.ax_ga.axvline(0, color="#bfa2b3", lw=0.8, ls="-")
        self.ax_ga.grid(True, color=GRIS_REJILLA, linestyle=":", lw=1.1)

        for spine in self.ax_ga.spines.values():
            spine.set_color("#e3c9d6")

        self.ax_ga.tick_params(colors=TEXTO_MUTED, labelsize=9)
        self.ax_ga.set_title(
            f"Sincronización con CSV (Gen {self.ga.generacion}) • RMSE: {ind.adap:.4f} • R²: {met.get('r2', 0):.4f}",
            color=NEGRO, fontsize=11, fontweight="bold", pad=8
        )
        self.ax_ga.set_xlabel("x", color=NEGRO, fontsize=9, fontweight="bold")
        self.ax_ga.set_ylabel("y", color=NEGRO, fontsize=9, fontweight="bold")
        self.ax_ga.legend(frameon=True, facecolor=BLANCO, edgecolor=ROSA_BORDE, fontsize=8, loc="best")

        self.fig_ga.tight_layout()
        self.canvas_ga.draw()

    def ga_ver_arbol_final_vs_csv(self):
        """Muestra lado a lado el árbol objetivo del CSV y el árbol final del AG."""
        if self.ga is None or self.ga.mejor_individuo is None:
            messagebox.showinfo("Árbol Final", "Primero inicializa o corre la evolución para tener un árbol final.")
            return

        ind = self.ga.mejor_individuo
        stats = f"(Gen {self.ga.generacion} • RMSE: {ind.adap:.4f} • R²: {ind.metricas.get('r2', 0):.4f})"
        VentanaComparativaArboles(self.ventana, self.arbol_objetivo, ind.gen, stats_final=stats)

    def ga_enviar_a_analizador(self):
        if self.ga is None or self.ga.mejor_individuo is None:
            messagebox.showinfo("Aviso", "Aún no hay una expresión evolucionada.")
            return

        expr = self.ga.mejor_individuo.expresion
        self.entrada_expr.delete(0, "end")
        self.entrada_expr.insert(0, expr)
        self.notebook.select(self.tab_analizador)
        self.graficar()
        messagebox.showinfo(
            "Expresión Transferida",
            f"La expresión evolucionada fue enviada al analizador principal:\n\n{expr}"
        )


if __name__ == "__main__":
    ventana = tk.Tk()
    app = App(ventana)
    ventana.mainloop()
