import tkinter as tk
from tkinter import ttk
import os
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.ui_components_images import ModernButton, setup_styles, COLOR_BG, COLOR_PRIMARY, COLOR_PRIMARY_LIGHT, COLOR_LIGHT_BG, COLOR_BORDER, COLOR_TEXT, COLOR_TEXT_SECONDARY

class MainView:
    def __init__(self, root):
        self.root = root
        self.main_frame = tk.Frame(root, bg=COLOR_BG)
        self.main_frame.pack(fill='both', expand=True)
        # Inicializar variables
        self.img_tk = None  # Para mantener referencia a la imagen
        self.canvas_error = None
        self.fig_error = None
        self.confusion_matrix_frame = None
        self.error_graph_frame = None
        self.color_percentages = {}  # Para almacenar las etiquetas de porcentajes de colores
        
        # Crear interfaz principal
        self.create_main_interface()
        self.style = setup_styles()
        
        # Inicializar componentes de la interfaz
        self.create_config_panel()
        self.create_graphics_panel()
        self.create_test_panel()
        
        # Configurar evento de redimensionamiento
        self.root.bind("<Configure>", self.on_window_resize)
        
    def create_main_interface(self):
        """Crea la interfaz principal con pestañas"""
        # Crear encabezado
        self.create_header()

        # Crear un contenedor para el contenido principal
        content_container = tk.Frame(self.main_frame, bg=COLOR_BG)
        content_container.pack(fill='both', expand=True, padx=5, pady=10)

        # Usar ttk.Scrollbar estándar en lugar de la implementación personalizada
        self.canvas = tk.Canvas(content_container, bg=COLOR_BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Empaquetar scrollbar y canvas
        self.scrollbar.pack(side=tk.RIGHT, fill='y')
        self.canvas.pack(side=tk.LEFT, fill='both', expand=True)
        
        # Crear frame para el contenido scrollable
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLOR_BG)
        self.scrollable_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configurar eventos de scroll
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Usar bind_all con tag único para evitar conflictos
        self.bind_mousewheel()
        
        # Crear pestañas con estilo mejorado
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=10)
        self.notebook.configure(style='TNotebook')

        # Pestaña para configuración y entrenamiento
        self.config_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.config_frame, text="Configuración y Entrenamiento")

        # Pestaña para gráficas
        self.graphics_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.graphics_frame, text="Matriz de confusión")
        
        # Pestaña para pruebas personalizadas
        self.test_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.test_frame, text="Pruebas Personalizadas")
        
        # Crear pie de página con copyright siempre visible
        self.create_footer()
        
        # NUEVO: Configurar eventos para manejar cambios de pestaña
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    # NUEVO: Método para manejar cambios de pestaña
    def on_tab_changed(self, event=None):
        """Actualiza el scrollbar cuando se cambia de pestaña"""
        # Forzar actualización del canvas y scrollbar
        self.scrollable_frame.update_idletasks()
        self.on_frame_configure()
        
        # Asegurarse de que el scrollbar esté visible si es necesario
        if self.scrollable_frame.winfo_height() > self.canvas.winfo_height():
            self.scrollbar.pack(side=tk.RIGHT, fill='y')
        else:
            self.scrollbar.pack_forget()
        
        # Volver al inicio
        self.canvas.yview_moveto(0)
    
    # NUEVO: Métodos para gestionar el binding del mousewheel
    def bind_mousewheel(self):
        """Vincula el evento de la rueda del ratón de manera más robusta"""
        # Vinculación para Windows (MouseWheel)
        self.root.bind_all("<MouseWheel>", self.on_mousewheel_windows)
        
        # Vinculación para Linux/Unix (Button-4/Button-5)
        self.root.bind_all("<Button-4>", self.on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self.on_mousewheel_linux)
        
        # Vinculación para macOS (Shift-MouseWheel para scroll horizontal)
        self.root.bind_all("<Shift-MouseWheel>", self.on_shift_mousewheel)

    def unbind_mousewheel(self):
        """Desvincula el evento de la rueda del ratón"""
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")
        self.root.unbind_all("<Shift-MouseWheel>")

    def on_mousewheel_windows(self, event):
        """Maneja el evento de la rueda del mouse en Windows"""
        # En Windows, event.delta indica la dirección y cantidad de desplazamiento
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_mousewheel_linux(self, event):
        """Maneja el evento de la rueda del mouse en Linux/Unix"""
        # En Linux, Button-4 es scroll hacia arriba, Button-5 es scroll hacia abajo
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def on_shift_mousewheel(self, event):
        """Maneja el evento de Shift+rueda del mouse para scroll horizontal"""
        # Útil en algunos sistemas para scroll horizontal
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def setup_styles(self):
        """Configurar estilos para ttk widgets con apariencia más profesional"""
        style = ttk.Style()
        style.theme_use('default')
        
        # Estilo para pestañas
        style.configure('TNotebook', background=COLOR_BG)
        style.configure('TNotebook.Tab', background=COLOR_BG, foreground=COLOR_TEXT, 
                        padding=[15, 5], font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab', 
                background=[('selected', COLOR_PRIMARY)], 
                foreground=[('selected', 'white')])
        
        # Estilo para frames
        style.configure('TFrame', background='white')
        
        # Estilo para labels
        style.configure('TLabel', background='white')
        
        # Estilo para combobox
        style.configure('TCombobox', 
                        fieldbackground='white',
                        background=COLOR_PRIMARY,
                        foreground=COLOR_TEXT,
                        arrowcolor=COLOR_PRIMARY)
        style.map('TCombobox',
                fieldbackground=[('readonly', 'white')],
                selectbackground=[('readonly', COLOR_PRIMARY)],
                selectforeground=[('readonly', 'white')])
        
        # Estilo para separadores
        style.configure('TSeparator', background=COLOR_PRIMARY)
        
        # Estilo para barras de progreso
        style.configure("TProgressbar", 
                        troughcolor='white', 
                        background=COLOR_PRIMARY,
                        thickness=8)
        
        # Estilo para barra de progreso verde
        style.configure("Green.Horizontal.TProgressbar",
                        troughcolor='white',
                        background=COLOR_PRIMARY,
                        thickness=10)
        
        # Estilo para scrollbars
        style.configure("TScrollbar",
                        background='white',
                        troughcolor='white',
                        arrowsize=14)
        
        # Estilo para LabelFrame con borde verde
        style.configure('Green.TLabelframe', 
                        bordercolor=COLOR_PRIMARY,
                        background='white')
        style.configure('Green.TLabelframe.Label', 
                        foreground=COLOR_PRIMARY,
                        background='white',
                        font=('Arial', 10, 'bold'))
        
        # Estilo para botones
        style.configure('TButton', 
                        background='white',
                        foreground='COLOR_TEXT')
        
        # Estilo para checkbuttons
        style.configure('TCheckbutton', 
                        background='white')
        
        # Estilo para entradas
        style.configure('TEntry', 
                        fieldbackground='white')
        
        return style

    def create_header(self):
        # Crear un marco para el encabezado con borde inferior sutil
        header_frame = tk.Frame(self.main_frame, bg=COLOR_BG, height=0)  # Reducir altura
        header_frame.pack(fill='x', pady=(0, 0))  # Reducir espacio
        
        # Añadir un borde inferior sutil
        border_frame = tk.Frame(header_frame, bg=COLOR_BG, height=1)
        border_frame.pack(side=tk.BOTTOM, fill='x')
        
        try:
            # Tamaños deseados
            logo_with = 60   # Reducir ancho
            logo_height = 80  # Reducir alto

            # Crear un frame para contener la imagen
            logo_frame = tk.Frame(header_frame, width=logo_with, height=logo_height, bg=COLOR_BG)
            logo_frame.pack(side=tk.LEFT, padx=15, pady=5)  # Reducir padding
            
            try:
                # Obtener la ruta de la imagen de manera segura
                image_path = self.obtener_ruta_relativa(os.path.join("utils", "Images", "escudo_udec.png"))
                
                # Cargar y redimensionar la imagen
                image = Image.open(image_path)
                image = image.resize((logo_with, logo_height), Image.LANCZOS)
                logo_img = ImageTk.PhotoImage(image)

                # Crear un Label con la imagen
                logo_label = tk.Label(logo_frame, image=logo_img, bg=COLOR_BG)
                logo_label.image = logo_img  # Mantener referencia para que no se "pierda" la imagen
                logo_label.pack()

            except Exception as e:
                print(f"Error al cargar la imagen: {e}")
                
                # Como respaldo, dibujamos un canvas con un óvalo verde y texto "UDEC"
                logo_canvas = tk.Canvas(
                    logo_frame, 
                    width=logo_with, 
                    height=logo_height, 
                    bg=COLOR_LIGHT_BG, 
                    highlightthickness=0
                )
                logo_canvas.pack()
                
                logo_canvas.create_oval(
                    5, 5, 
                    logo_with - 5, logo_height - 5, 
                    fill="#006633", 
                    outline=""
                )
                logo_canvas.create_text(
                    logo_with / 2, logo_height / 2, 
                    text="UDEC", 
                    fill="white", 
                    font=("Arial", 10, "bold")  # Reducir tamaño
                )

        except Exception as e:
            print(f"Error en la creación del logo: {e}")
            
        # Título y subtítulo con mejor tipografía
        title_frame = tk.Frame(header_frame, bg=COLOR_BG)
        title_frame.pack(side=tk.LEFT, padx=8, pady=5)  # Reducir padding
        
        # Información del proyecto con mejor alineación
        info_frame = tk.Frame(header_frame, bg=COLOR_BG)
        info_frame.pack(side=tk.RIGHT, padx=15, pady=5)  # Reducir padding
        
        # Crear un marco para la información de autores con estilo mejorado
        authors_frame = tk.Frame(self.main_frame, bg=COLOR_BG, padx=10, pady=5)  # Reducir padding
        authors_frame.pack(fill=tk.X, padx=20, pady=5)
        
        bottom_border = tk.Frame(authors_frame, bg=COLOR_BORDER, height=1)
        bottom_border.pack(side=tk.BOTTOM, fill='x')
        
        title_label = tk.Label(title_frame, text="BACKPROPAGATION", 
                            font=("Arial", 20, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)  # Reducir tamaño
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(title_frame, text="Universidad de Cundinamarca", 
                                font=("Arial", 12), bg=COLOR_BG, fg=COLOR_TEXT_SECONDARY)  # Reducir tamaño
        subtitle_label.pack(anchor='w')
        
        authors_info = tk.Label(authors_frame, text="Autores: Sergio Leonardo Moscoso Ramirez - Miguel Ángel Pardo Lopez",
                                 font=("Segoe UI", 10, "bold"),
                                 bg=COLOR_BG,
                                 fg=COLOR_TEXT)
        authors_info.pack(side=tk.LEFT, padx=5)
        
    def obtener_ruta_relativa(self, ruta_archivo):
        """Obtiene la ruta relativa de un archivo"""
        if getattr(sys, 'frozen', False):  # Si el programa está empaquetado con PyInstaller
            base_path = sys._MEIPASS       # Carpeta temporal donde PyInstaller extrae archivos
        else:
            base_path = os.path.abspath(".")  # Carpeta normal en modo desarrollo

        return os.path.join(base_path, ruta_archivo)
    
    def create_footer(self):
        # Crear un marco para el pie de página con estilo mejorado que siempre sea visible
        footer_frame = tk.Frame(self.root, bg=COLOR_PRIMARY, height=30)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Añadir un borde superior sutil
        top_border = tk.Frame(footer_frame, bg=COLOR_BORDER, height=1)
        top_border.pack(side=tk.TOP, fill='x')
        
        # Texto del pie de página con mejor tipografía
        footer_text = "© Universidad de Cundinamarca - Simulador de BackPropagation"
        footer_label = tk.Label(footer_frame, text=footer_text, 
                                font=("Arial", 8), bg=COLOR_PRIMARY, fg="white")
        footer_label.pack(pady=6)

    def create_config_panel(self):
        # Panel principal dividido en dos columnas
        main_panel = ttk.Frame(self.config_frame)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_panel.configure(style='TFrame')
        
        # Panel izquierdo - Configuración
        left_panel = ttk.Frame(main_panel)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5, ipadx=10)
        left_panel.configure(style='TFrame')
        
        # Panel derecho - Gráfica de error
        right_panel = ttk.Frame(main_panel)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_panel.configure(style='TFrame')
        
        # ===== PANEL DE CONFIGURACIÓN =====
        config_frame = ttk.LabelFrame(left_panel, text="Configuración de la Red")
        config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        config_frame.configure(borderwidth=2, relief="solid")
        config_frame['style'] = 'Green.TLabelframe'
        config_frame.columnconfigure(0, weight=1)

        # ========== ARQUITECTURA DE LA RED ==========
        arch_frame = ttk.LabelFrame(config_frame, text="Arquitectura de la Red")
        arch_frame.pack(fill=tk.X, padx=5, pady=5)
        arch_frame.configure(borderwidth=2, relief="solid")
        arch_frame['style'] = 'Green.TLabelframe'
        
        # Fila 1 - Capa de Entrada
        entrada_frame = ttk.Frame(arch_frame)
        entrada_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(entrada_frame, text="Entrada:").pack(side=tk.LEFT, padx=2)
        self.entrada_input = ttk.Entry(entrada_frame, width=8)
        self.entrada_input.pack(side=tk.LEFT, padx=2)
        self.lbl_entrada1 = ttk.Label(entrada_frame, text="Automático", font=("Arial", 9, "italic"))
        self.lbl_entrada1.pack(side=tk.LEFT, padx=2)

        # Fila 2 - Capa Oculta
        oculta_frame = ttk.Frame(arch_frame)
        oculta_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(oculta_frame, text="Oculta:").pack(side=tk.LEFT, padx=2)
        self.oculta_input = ttk.Entry(oculta_frame, width=8)
        self.oculta_input.pack(side=tk.LEFT, padx=2)
        self.lbl_oculta1 = ttk.Label(oculta_frame, text="Sugerido", font=("Arial", 9, "italic"))
        self.lbl_oculta1.pack(side=tk.LEFT, padx=2)

        # Fila 3 - Capa de Salida
        salida_frame = ttk.Frame(arch_frame)
        salida_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(salida_frame, text="Salida:").pack(side=tk.LEFT, padx=2)
        self.salida_input = ttk.Entry(salida_frame, width=8)
        self.salida_input.pack(side=tk.LEFT, padx=2)
        self.lbl_salida1 = ttk.Label(salida_frame, text="Automático", font=("Arial", 9, "italic"))
        self.lbl_salida1.pack(side=tk.LEFT, padx=2)

        # ========== PARÁMETROS DE ENTRENAMIENTO ==========
        param_frame = ttk.LabelFrame(config_frame, text="Parámetros de Entrenamiento")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        param_frame.configure(borderwidth=2, relief="solid")
        param_frame['style'] = 'Green.TLabelframe'
        
        # Usar grid para organizar en dos columnas
        param_grid = ttk.Frame(param_frame)
        param_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Columna 1
        ttk.Label(param_grid, text="Alpha (α):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.alpha_input = ttk.Entry(param_grid, width=8)
        self.alpha_input.insert(0, "0.01")
        self.alpha_input.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Precisión:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.precision_input = ttk.Entry(param_grid, width=8)
        self.precision_input.insert(0, "0.01")
        self.precision_input.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Momentum:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.momentum_var = tk.BooleanVar()
        self.momentum_check = ttk.Checkbutton(param_grid, variable=self.momentum_var, command=self.toggle_momentum)
        self.momentum_check.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Columna 2
        ttk.Label(param_grid, text="Máx. Épocas:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.max_epocas_input = ttk.Entry(param_grid, width=8)
        self.max_epocas_input.insert(0, "1000000")
        self.max_epocas_input.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Bias:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.bias_input = ttk.Entry(param_grid, width=8)
        self.bias_input.insert(0, "1.0")
        self.bias_input.grid(row=1, column=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(param_grid, text="Beta (β):").grid(row=2, column=2, sticky="w", padx=5, pady=2)
        self.beta_input = ttk.Entry(param_grid, width=8, state='disabled')
        self.beta_input.grid(row=2, column=3, sticky="w", padx=5, pady=2)

        # ========== FUNCIONES DE ACTIVACIÓN ==========
        activ_frame = ttk.LabelFrame(config_frame, text="Funciones de Activación")
        activ_frame.pack(fill=tk.X, padx=5, pady=5)
        activ_frame.configure(borderwidth=2, relief="solid")
        activ_frame['style'] = 'Green.TLabelframe'
        
        # Usar grid para organizar en tres columnas
        activ_grid = ttk.Frame(activ_frame)
        activ_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Columna 1
        ttk.Label(activ_grid, text="Capa Oculta:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.func_oculta = ttk.Combobox(activ_grid, 
                                    values=['Sigmoide', 'ReLU', 'Tanh', 'Leaky ReLU'], 
                                    state='readonly',
                                    width=12)
        self.func_oculta.current(0)
        self.func_oculta.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Columna 2
        ttk.Label(activ_grid, text="Capa Salida:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.func_salida = ttk.Combobox(activ_grid, 
                                    values=['Softmax','Sigmoide', 'Lineal'], 
                                    state='readonly',
                                    width=12)
        self.func_salida.current(0)
        self.func_salida.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        # Columna 3
        ttk.Label(activ_grid, text="Leaky ReLU:").grid(row=0, column=4, sticky="w", padx=5, pady=2)
        self.beta_oculta_input = ttk.Entry(activ_grid, width=8)
        self.beta_oculta_input.insert(0, "0.01")
        self.beta_oculta_input.grid(row=0, column=5, sticky="w", padx=5, pady=2)

        # ===== PANEL DE DATOS Y ACCIONES =====
        data_frame = ttk.LabelFrame(config_frame, text="Datos de Entrenamiento")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        data_frame.configure(borderwidth=2, relief="solid")
        data_frame['style'] = 'Green.TLabelframe'

        # Frame para los botones de carga de datos y entrenamiento
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)

        # Botón: Cargar Entradas
        self.btn_cargar_entrada = ModernButton(
            btn_frame, 
            text="Cargar Entradas", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT
        )
        self.btn_cargar_entrada.pack(side=tk.LEFT, padx=(0, 10))

        # Botón: Entrenar Red
        self.btn_entrenar = ModernButton(
            btn_frame, 
            text="Entrenar Red", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT
        )
        self.btn_entrenar.pack(side=tk.LEFT, padx=5)

        # Barra de progreso para el entrenamiento
        progress_frame = ttk.Frame(data_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progreso:").pack(side=tk.LEFT, padx=2)
        self.progress_bar = ttk.Progressbar(progress_frame, style="Green.Horizontal.TProgressbar", length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=2)
        
        # ===== PANEL DE RESULTADOS DEL ENTRENAMIENTO =====
        results_frame = ttk.LabelFrame(right_panel, text="Resultados del Entrenamiento")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        results_frame.configure(borderwidth=2, relief="solid")
        results_frame['style'] = 'Green.TLabelframe'

        # Estado del entrenamiento
        status_frame = ttk.Frame(results_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Estado: No entrenado", foreground="red", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT)
        
        self.status_indicator = tk.Canvas(status_frame, width=15, height=15, bg='white', highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        self.status_indicator.create_oval(2, 2, 13, 13, fill="red", outline="")
        
        # Indicadores de métricas
        metrics_frame = ttk.LabelFrame(results_frame, text="Indicadores de Métricas")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        metrics_frame.configure(borderwidth=2, relief="solid")
        metrics_frame['style'] = 'Green.TLabelframe'
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(padx=5, pady=5, side=tk.LEFT)
        
        ttk.Label(metrics_grid, text="Épocas:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.epochs_value = ttk.Label(metrics_grid, text="-")
        self.epochs_value.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(metrics_grid, text="Error Final:").grid(row=1, column=0, sticky="w", padx=5, pady=2)   
        self.error_value = ttk.Label(metrics_grid, text="-")
        self.error_value.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(metrics_grid, text="Exactitud:").grid(row=2, column=0, sticky="w", padx=5, pady=2)   
        self.accuracy_value = ttk.Label(metrics_grid, text="-")
        self.accuracy_value.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        # Gráfica de error
        self.error_graph_frame = ttk.LabelFrame(results_frame, text="Evolución del Error vs Épocas")
        self.error_graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.error_graph_frame.configure(borderwidth=2, relief="solid")
        self.error_graph_frame['style'] = 'Green.TLabelframe'
        
        # Mensaje inicial en la gráfica
        msg_frame = ttk.Frame(self.error_graph_frame)
        msg_frame.pack(expand=True)
        
        ttk.Label(msg_frame, 
                 text="La gráfica de error se mostrará aquí después del entrenamiento",
                 font=("Arial", 10, "italic")).pack(pady=50)
    
    def create_graphics_panel(self):
        main_panel = ttk.Frame(self.graphics_frame)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar para que se expanda proporcionalmente
        main_panel.columnconfigure(0, weight=1)
        main_panel.rowconfigure(0, weight=1)
        
        # ===== PANEL DE MATRIZ DE CONFUSIÓN (centrado) =====
        confusion_frame = ttk.LabelFrame(main_panel, text="Matriz de Confusión")
        confusion_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        confusion_frame['style'] = 'Green.TLabelframe'
        confusion_frame.columnconfigure(0, weight=1)
        confusion_frame.rowconfigure(0, weight=1)
        
        # Mensaje inicial en la matriz de confusión
        msg_frame = ttk.Frame(confusion_frame)
        msg_frame.grid(row=0, column=0, sticky="nsew")
        msg_frame.columnconfigure(0, weight=1)
        msg_frame.rowconfigure(0, weight=1)
        
        ttk.Label(
            msg_frame, 
            text="La matriz de confusión se mostrará aquí después del entrenamiento",
            font=("Arial", 10, "italic")
        ).grid(row=0, column=0)
        
        # Guardar referencia al frame de la matriz de confusión
        self.confusion_matrix_frame = confusion_frame
        
    def create_test_panel(self):
        """Crea el panel de pruebas personalizadas con diseño mejorado y responsive"""
        # Marco principal para pruebas con diseño de dos columnas
        main_test_frame = ttk.Frame(self.test_frame)
        main_test_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Configurar para que se expanda proporcionalmente
        main_test_frame.columnconfigure(0, weight=1)  # Panel izquierdo
        main_test_frame.columnconfigure(1, weight=2)  # Panel derecho
        main_test_frame.rowconfigure(0, weight=1)

        # --- Panel Izquierdo: Controles y Configuración ---
        left_panel = ttk.Frame(main_test_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Configurar expansión vertical proporcional
        left_panel.rowconfigure(0, weight=0)  # Instrucciones
        left_panel.rowconfigure(1, weight=0)  # Acciones
        left_panel.rowconfigure(2, weight=1)  # Info panel
        left_panel.columnconfigure(0, weight=1)

        # Panel de instrucciones
        instruction_panel = ttk.LabelFrame(left_panel, text="Instrucciones")
        instruction_panel.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        instruction_panel['style'] = 'Green.TLabelframe'
        
        instructions = ttk.Label(
            instruction_panel, 
            text="Cargue una imagen de prueba para clasificar la vocal. El sistema analizará la imagen y determinará qué vocal representa basándose en el modelo entrenado.",
            wraplength=350, 
            justify='left', 
            font=("Arial", 9),
            padding=(15, 10)
        )
        instructions.pack(pady=5, padx=5, fill=tk.X)

        # Panel de acciones
        actions_panel = ttk.LabelFrame(left_panel, text="Acciones")
        actions_panel.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        actions_panel['style'] = 'Green.TLabelframe'
        
        # Contenedor para los botones con mejor espaciado
        btn_container = ttk.Frame(actions_panel)
        btn_container.pack(pady=15, padx=15, fill=tk.X)
        btn_container.columnconfigure(0, weight=1)
        
        # Botón para cargar imagen con icono
        btn_frame1 = ttk.Frame(btn_container)
        btn_frame1.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        btn_frame1.columnconfigure(0, weight=1)
        
        self.btn_cargar_prueba = ModernButton(
            btn_frame1, 
            text=" Cargar Imagen de Prueba", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=15,
            pady=8,
            font=("Arial", 10)
        )
        self.btn_cargar_prueba.grid(row=0, column=0, sticky="ew")
        
        # Botón para cargar pesos con icono
        btn_frame2 = ttk.Frame(btn_container)
        btn_frame2.grid(row=1, column=0, sticky="ew")
        btn_frame2.columnconfigure(0, weight=1)
        
        self.btn_cargar_pesos = ModernButton(
            btn_frame2, 
            text=" Cargar Pesos", 
            bg=COLOR_PRIMARY, 
            fg='white',
            hover_bg=COLOR_PRIMARY_LIGHT,
            padx=15,
            pady=8,
            font=("Arial", 10)
        )
        self.btn_cargar_pesos.grid(row=0, column=0, sticky="ew")
        
        # Panel de información
        info_panel = ttk.LabelFrame(left_panel, text="Información del Modelo")
        info_panel.grid(row=2, column=0, sticky="nsew")
        info_panel['style'] = 'Green.TLabelframe'
        
        info_content = ttk.Frame(info_panel)
        info_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        info_content.columnconfigure(1, weight=1)
        
        # Información sobre el modelo cargado
        ttk.Label(info_content, text="Estado del Modelo:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.model_status = ttk.Label(info_content, text="No cargado", foreground="red")
        self.model_status.grid(row=0, column=1, sticky="w", pady=(0, 10))
        
        ttk.Label(info_content, text="Archivo de Pesos:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.weights_file = ttk.Label(info_content, text="Ninguno", font=("Arial", 9, "italic"))
        self.weights_file.grid(row=1, column=1, sticky="w", pady=(0, 10))
        
        ttk.Label(info_content, text="Arquitectura:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(0, 10))
        self.architecture_info = ttk.Label(info_content, text="No disponible", font=("Arial", 9, "italic"))
        self.architecture_info.grid(row=2, column=1, sticky="w", pady=(0, 10))
        
        # Separador visual
        ttk.Separator(info_content, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Información sobre la imagen cargada
        ttk.Label(info_content, text="Imagen Cargada:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w", pady=(0, 10))
        self.image_status = ttk.Label(info_content, text="Ninguna", font=("Arial", 9, "italic"))
        self.image_status.grid(row=4, column=1, sticky="w", pady=(0, 10))
        
        ttk.Label(info_content, text="Dimensiones:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(0, 10))
        self.image_dimensions = ttk.Label(info_content, text="N/A", font=("Arial", 9, "italic"))
        self.image_dimensions.grid(row=5, column=1, sticky="w", pady=(0, 10))

        # --- Panel Derecho: Visualización y Resultados ---
        right_panel = ttk.Frame(main_test_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Configurar expansión vertical proporcional
        right_panel.rowconfigure(0, weight=1)  # Visualización
        right_panel.rowconfigure(1, weight=1)  # Resultados
        right_panel.columnconfigure(0, weight=1)

        # Panel superior: Visualización de la imagen
        image_panel = ttk.LabelFrame(right_panel, text="Visualización")
        image_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 15))
        image_panel['style'] = 'Green.TLabelframe'
        
        image_container = ttk.Frame(image_panel)
        image_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Marco decorativo para la imagen
        image_frame = ttk.Frame(image_container)
        image_frame.pack(expand=True)
        
        # Canvas para mostrar la imagen con borde elegante
        self.canvas_imagen = tk.Canvas(
            image_frame, 
            width=250, 
            height=250, 
            bg='white',
            highlightthickness=2, 
            highlightbackground=COLOR_PRIMARY
        )
        self.canvas_imagen.pack(fill=tk.BOTH, expand=True)
        
        # Texto de placeholder para el canvas
        self.canvas_imagen.create_text(
            125, 125, 
            text="Cargue una imagen para visualizarla", 
            font=("Arial", 10, "italic"),
            fill=COLOR_TEXT_SECONDARY
        )

        # Panel inferior: Resultados de la clasificación
        results_panel = ttk.LabelFrame(right_panel, text="Resultados de la Clasificación")
        results_panel.grid(row=1, column=0, sticky="nsew")
        results_panel['style'] = 'Green.TLabelframe'
        
        results_container = ttk.Frame(results_panel)
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        results_container.columnconfigure(0, weight=1)
        results_container.rowconfigure(0, weight=0)  # Resultado principal
        results_container.rowconfigure(1, weight=0)  # Separador
        results_container.rowconfigure(2, weight=1)  # Niveles de activación
        
        # Sección superior: Resultado principal
        result_header = ttk.Frame(results_container)
        result_header.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        result_header.columnconfigure(0, weight=1)
        
        # Resultado de la clasificación con estilo mejorado
        result_box = ttk.Frame(result_header, padding=10)
        result_box.grid(row=0, column=0, sticky="ew")
        result_box.columnconfigure(0, weight=1)
        
        # Crear un marco decorativo para el resultado
        result_display = tk.Frame(result_box, bg=COLOR_LIGHT_BG, padx=15, pady=15)
        result_display.grid(row=0, column=0, sticky="ew")
        result_display.columnconfigure(0, weight=1)
        
        # Contenedor para la vocal y el color
        result_content = tk.Frame(result_display, bg=COLOR_LIGHT_BG)
        result_content.pack()
        
        # Vocal detectada
        vocal_frame = tk.Frame(result_content, bg=COLOR_LIGHT_BG)
        vocal_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Label(
            vocal_frame, 
            text="VOCAL DETECTADA", 
            font=("Arial", 9, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT_SECONDARY
        ).pack(anchor="center")
        
        self.vocal_label = tk.Label(
            vocal_frame, 
            text="?", 
            font=("Arial", 36, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        self.vocal_label.pack(anchor="center", pady=5)
        
        # Separador vertical
        tk.Frame(result_content, width=1, bg=COLOR_BORDER).pack(side=tk.LEFT, fill="y", padx=10)
        
        # Color dominante - MODIFICADO para mostrar porcentajes de cada color
        color_frame = tk.Frame(result_content, bg=COLOR_LIGHT_BG)
        color_frame.pack(side=tk.LEFT)
        
        tk.Label(
            color_frame, 
            text="COLOR DOMINANTE", 
            font=("Arial", 9, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT_SECONDARY
        ).pack(anchor="center")
        
        # Contenedor para el indicador de color y los porcentajes
        color_info_frame = tk.Frame(color_frame, bg=COLOR_LIGHT_BG)
        color_info_frame.pack(pady=5)
        
        # Indicador de color (cuadrado coloreado)
        self.color_indicator = tk.Canvas(
            color_info_frame, 
            width=30, 
            height=30, 
            highlightthickness=1,
            highlightbackground=COLOR_BORDER
        )
        self.color_indicator.pack(side=tk.LEFT)
        self.color_indicator.create_rectangle(0, 0, 30, 30, fill="white", outline="")
        
        # Contenedor para los porcentajes de colores
        percentages_frame = tk.Frame(color_info_frame, bg=COLOR_LIGHT_BG)
        percentages_frame.pack(side=tk.LEFT, padx=10)
        
        # Etiquetas para los porcentajes de cada color
        colors = ['Rojo', 'Verde', 'Azul']
        self.color_percentages = {}
        
        for i, color_name in enumerate(colors):
            color_label = tk.Label(
                percentages_frame,
                text=f"{color_name}: 0.0%",
                font=("Arial", 9),
                bg=COLOR_LIGHT_BG,
                fg=COLOR_TEXT,
                anchor="w"
            )
            color_label.pack(anchor="w")
            self.color_percentages[color_name] = color_label
        
        # Separador horizontal
        ttk.Separator(results_container, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=10)
        
        # Sección inferior: Niveles de activación
        activation_frame = ttk.Frame(results_container)
        activation_frame.grid(row=2, column=0, sticky="nsew")
        activation_frame.columnconfigure(0, weight=1)
        
        ttk.Label(
            activation_frame, 
            text="Niveles de Activación por Vocal", 
            font=("Arial", 10, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Barras de activación con diseño mejorado
        self.barras_similitud = {}
        self.lbl_porcentajes = {}
        vocales = ['A', 'E', 'I', 'O', 'U']
        
        # Crear un contenedor para todas las barras
        bars_container = ttk.Frame(activation_frame)
        bars_container.grid(row=1, column=0, sticky="ew")
        bars_container.columnconfigure(0, weight=1)
        
        for idx, vocal in enumerate(vocales):
            # Marco para cada barra con espaciado mejorado
            bar_frame = ttk.Frame(bars_container)
            bar_frame.grid(row=idx, column=0, sticky="ew", pady=5)
            bar_frame.columnconfigure(1, weight=1)
            
            # Etiqueta de vocal con estilo mejorado
            vocal_label = ttk.Label(
                bar_frame, 
                text=f"{vocal}:", 
                width=3, 
                font=("Arial", 10, "bold")
            )
            vocal_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
            
            # Barra de progreso con estilo mejorado
            self.barras_similitud[vocal] = ttk.Progressbar(
                bar_frame, 
                length=150, 
                style="Green.Horizontal.TProgressbar"
            )
            self.barras_similitud[vocal].grid(row=0, column=1, sticky="ew")
            
            # Etiqueta de porcentaje con estilo mejorado
            self.lbl_porcentajes[vocal] = ttk.Label(
                bar_frame, 
                text="0.0%", 
                width=8,
                font=("Arial", 9)
            )
            self.lbl_porcentajes[vocal].grid(row=0, column=2, sticky="e", padx=(10, 0))
    
    def log(self, mensaje):
        print(mensaje)
        
    def toggle_momentum(self):
        """Habilita o deshabilita el campo beta de momentum"""
        if self.momentum_var.get():
            self.beta_input.config(state='normal')
        else:
            self.beta_input.config(state='disabled')
    
    def mostrar_imagen_en_canvas(self, imagen):
        """Muestra una imagen en el canvas"""
        # Guardar referencia a la imagen original para posibles redimensionamientos
        self.current_image = imagen
        
        # Limpiar canvas
        self.canvas_imagen.delete("all")
        
        # Obtener dimensiones actuales del canvas
        ancho_canvas = self.canvas_imagen.winfo_width()
        alto_canvas = self.canvas_imagen.winfo_height()
        
        # Si el canvas aún no tiene tamaño (primera carga), usar valores predeterminados
        if ancho_canvas <= 1:
            ancho_canvas = 250
        if alto_canvas <= 1:
            alto_canvas = 250
        
        # Calcular el tamaño para mantener la proporción
        ancho_img, alto_img = imagen.size
        ratio = min(ancho_canvas/ancho_img, alto_canvas/alto_img)
        nuevo_ancho = int(ancho_img * ratio)
        nuevo_alto = int(alto_img * ratio)
        
        # Redimensionar la imagen
        imagen_resized = imagen.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
        
        # Convertir la imagen para Tkinter
        self.img_tk = ImageTk.PhotoImage(imagen_resized)
        
        # Calcular posición para centrar la imagen
        x = (ancho_canvas - nuevo_ancho) // 2
        y = (alto_canvas - nuevo_alto) // 2
        
        # Mostrar la imagen en el canvas
        self.canvas_imagen.create_image(ancho_canvas//2, alto_canvas//2, image=self.img_tk)
        
        # Añadir un borde decorativo alrededor de la imagen
        padding = 5
        self.canvas_imagen.create_rectangle(
            x - padding, 
            y - padding, 
            x + nuevo_ancho + padding, 
            y + nuevo_alto + padding, 
            outline=COLOR_BORDER, 
            width=1
        )
    
    def actualizar_indicador_color(self, color, porcentajes=None):
        """
        Actualiza el indicador de color y los porcentajes
        
        Args:
            color: Color dominante ('Rojo', 'Verde', 'Azul')
            porcentajes: Diccionario con los porcentajes de cada color {'Rojo': 25.5, 'Verde': 60.2, 'Azul': 14.3}
        """
        # Si no se proporcionan porcentajes, usar valores predeterminados
        if porcentajes is None:
            porcentajes = {'Rojo': 0.0, 'Verde': 0.0, 'Azul': 0.0}
            # Asignar 100% al color dominante
            if color in porcentajes:
                porcentajes[color] = 100.0
        
        # Mapear nombre de color a valor RGB
        color_map = {
            'Verde': '#00FF00',
            'Azul': '#0000FF',
            'Rojo': '#FF0000'
        }
        
        # Actualizar canvas de color con el color dominante
        self.color_indicator.delete("all")
        self.color_indicator.create_rectangle(0, 0, 30, 30, fill=color_map.get(color, 'white'), outline="black")
        
        # Actualizar las etiquetas de porcentaje para cada color
        for color_name, percentage in porcentajes.items():
            if color_name in self.color_percentages:
                # Formatear el porcentaje con un decimal
                percentage_text = f"{color_name}: {percentage:.1f}%"
                
                # Destacar el color dominante con negrita y color primario
                if color_name == color:
                    self.color_percentages[color_name].config(
                        text=percentage_text,
                        font=("Arial", 9, "bold"),
                        fg=COLOR_PRIMARY
                    )
                else:
                    self.color_percentages[color_name].config(
                        text=percentage_text,
                        font=("Arial", 9),
                        fg=COLOR_TEXT
                    )
                    
    def actualizar_barras(self, activaciones):
        """Actualiza las barras de progreso con los niveles de activación de cada vocal"""
        # Encontrar la vocal con mayor nivel de activación
        max_vocal = max(activaciones, key=activaciones.get)
        
        for vocal, barra in self.barras_similitud.items():
            valor = activaciones.get(vocal, 0)
            barra['value'] = valor
            
            # Formatear el texto del porcentaje
            texto_porcentaje = f"{valor:.1f}%"
            
            # Destacar la vocal con mayor nivel de activación
            if vocal == max_vocal:
                self.lbl_porcentajes[vocal].config(
                    text=texto_porcentaje,
                    font=("Arial", 9, "bold"),
                    foreground=COLOR_PRIMARY
                )
            else:
                self.lbl_porcentajes[vocal].config(
                    text=texto_porcentaje,
                    font=("Arial", 9),
                    foreground=COLOR_TEXT
                )
    
    def mostrar_matriz_confusion(self, cm, labels):
        """Muestra la matriz de confusión en el panel de gráficas centrada"""
        # Limpiar frame anterior
        for widget in self.confusion_matrix_frame.winfo_children():
            widget.destroy()
        
        # Configurar el frame para expandirse
        self.confusion_matrix_frame.columnconfigure(0, weight=1)
        self.confusion_matrix_frame.rowconfigure(0, weight=1)
        
        # Crear figura para la matriz de confusión
        fig_cm = plt.figure(figsize=(6, 5), constrained_layout=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Greens, ax=fig_cm.add_subplot(111))
        plt.title("Matriz de Confusión del Entrenamiento")
        
        # Crear un frame intermedio para centrar el canvas
        canvas_frame = ttk.Frame(self.confusion_matrix_frame)
        canvas_frame.grid(row=0, column=0)  # Se coloca en el centro del frame principal
        
        # Integrar gráfica en Tkinter
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=canvas_frame)
        canvas_cm.draw()
        
        # Usar pack con anchor='center' para centrar la figura
        canvas_cm.get_tk_widget().pack(anchor="center", padx=10, pady=10)
        
        # Ajustar la figura al redimensionar (si tienes implementada esa función)
        self.bind_figure_resize(canvas_cm, fig_cm)

    def mostrar_grafica_error(self, errores):
        # Limpiar frame anterior
        for widget in self.error_graph_frame.winfo_children():
            widget.destroy()

        # Crear figura
        self.fig_error = plt.figure(figsize=(6, 4))
        plt.plot(errores, color='#004d25', linewidth=2)
        plt.title("Evolución del Error en Backpropagation", fontsize=12)
        plt.xlabel("Épocas", fontsize=10)
        plt.ylabel("Error Promedio", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Integrar gráfica en Tkinter
        self.canvas_error = FigureCanvasTkAgg(self.fig_error, master=self.error_graph_frame)
        self.canvas_error.draw()
        self.canvas_error.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def bind_figure_resize(self, canvas, figure):
        """Configura un evento para redimensionar la figura cuando cambia el tamaño del canvas"""
        def on_resize(event):
            # Actualizar el tamaño de la figura cuando el canvas cambia de tamaño
            figure.set_size_inches(event.width/100, event.height/100)
            canvas.draw_idle()
        
        # Vincular el evento de redimensionamiento
        canvas.get_tk_widget().bind("<Configure>", on_resize)

    def on_window_resize(self, event=None):
        """Maneja el redimensionamiento de la ventana"""
        # Solo procesar eventos de la ventana principal, no de widgets internos
        if event and event.widget == self.root:
            # Actualizar canvas y otros elementos que necesitan ajustarse
            self.adjust_canvas_sizes()
            self.on_frame_configure()  # Actualizar scrollbar
    
    def adjust_canvas_sizes(self):
        """Ajusta los tamaños de los canvas y otros elementos según el tamaño de la ventana"""
        # Obtener el tamaño actual de la ventana
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        # Ajustar el tamaño del canvas de imagen si existe
        if hasattr(self, 'canvas_imagen') and self.canvas_imagen.winfo_exists():
            # Calcular un nuevo tamaño proporcional al tamaño de la ventana
            # pero con un mínimo y máximo razonables
            new_size = min(max(int(window_width * 0.2), 200), 400)
            
            # Actualizar el tamaño del canvas
            self.canvas_imagen.config(width=new_size, height=new_size)
            
            # Si hay una imagen cargada, redimensionarla y mostrarla de nuevo
            if hasattr(self, 'current_image') and self.current_image:
                self.mostrar_imagen_en_canvas(self.current_image)

    def on_frame_configure(self, event=None):
        """Configura el canvas para scrollear todo el contenido"""
        # Actualizar la región de desplazamiento para incluir todo el contenido
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Mostrar u ocultar el scrollbar según sea necesario
        if self.scrollable_frame.winfo_height() > self.canvas.winfo_height():
            self.scrollbar.pack(side=tk.RIGHT, fill='y')
        else:
            self.scrollbar.pack_forget()

    def on_canvas_configure(self, event=None):
        """Ajusta el ancho del frame scrollable al canvas"""
        if event:
            canvas_width = event.width
            self.canvas.itemconfig(self.scrollable_window, width=canvas_width)

    def update_scrollbar(self):
        """Este método ya no es necesario con la nueva implementación"""
        pass

    def analizar_colores_imagen(self, imagen):
        """
        Analiza los colores de una imagen y devuelve el color dominante y los porcentajes
        
        Args:
            imagen: Imagen PIL a analizar
            
        Returns:
            tuple: (color_dominante, porcentajes)
                color_dominante: 'Rojo', 'Verde' o 'Azul'
                porcentajes: Diccionario con los porcentajes de cada color
        """
        # Convertir a RGB si no lo está
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        
        # Obtener datos de píxeles
        pixels = list(imagen.getdata())
        total_pixels = len(pixels)
        
        # Inicializar contadores para cada canal
        r_total = g_total = b_total = 0
        
        # Sumar valores de cada canal
        for r, g, b in pixels:
            r_total += r
            g_total += g
            b_total += b
        
        # Calcular la suma total de todos los canales
        total_sum = r_total + g_total + b_total
        
        # Evitar división por cero
        if total_sum == 0:
            return 'Rojo', {'Rojo': 100.0, 'Verde': 0.0, 'Azul': 0.0}
        
        # Calcular porcentajes
        r_percent = (r_total / total_sum) * 100
        g_percent = (g_total / total_sum) * 100
        b_percent = (b_total / total_sum) * 100
        
        # Crear diccionario de porcentajes
        porcentajes = {
            'Rojo': r_percent,
            'Verde': g_percent,
            'Azul': b_percent
        }
        
        # Determinar color dominante
        if r_percent >= g_percent and r_percent >= b_percent:
            color_dominante = 'Rojo'
        elif g_percent >= r_percent and g_percent >= b_percent:
            color_dominante = 'Verde'
        else:
            color_dominante = 'Azul'
        
        return color_dominante, porcentajes

# Ejemplo de cómo usar estas funciones cuando se carga una imagen
    def cargar_imagen(self, ruta_imagen):
        """Carga una imagen desde una ruta y actualiza la interfaz"""
        try:
            # Cargar la imagen
            imagen = Image.open(ruta_imagen)
            
            # Mostrar la imagen en el canvas
            self.mostrar_imagen_en_canvas(imagen)
            
            # Analizar los colores de la imagen
            color_dominante, porcentajes = self.analizar_colores_imagen(imagen)
            
            # Actualizar el indicador de color y los porcentajes
            self.actualizar_indicador_color(color_dominante, porcentajes)
            
            # Actualizar información de la imagen en la interfaz
            self.image_status.config(text=os.path.basename(ruta_imagen))
            self.image_dimensions.config(text=f"{imagen.width} x {imagen.height}")
            
            # Aquí iría el código para procesar la imagen con la red neuronal
            # y actualizar las barras de activación
            
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            # Mostrar mensaje de error en la interfaz