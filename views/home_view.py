import tkinter as tk
from tkinter import ttk
import webbrowser
from PIL import Image, ImageTk
import os
import sys

from utils.ui_components import COLOR_BG, COLOR_PRIMARY, COLOR_TEXT, COLOR_TEXT_LIGHT, COLOR_LIGHT_BG, ModernButton, create_header_frame

class HomeView:
    def __init__(self, root):
        self.root = root
        self.create_widgets()
        
    def create_widgets(self):
        """Crea los widgets para la vista de inicio"""
        # Contenedor principal
        self.main_container = tk.Frame(self.root, bg=COLOR_BG)
        self.main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Crear encabezado con logo y título
        self.header_frame = create_header_frame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))
        
        # Crear sección de información de la universidad y desarrolladores
        self.create_university_info()
        
        # Separador
        separator = ttk.Separator(self.main_container, orient='horizontal')
        separator.pack(fill='x', padx=20, pady=20)
        
        # Descripción de la red neuronal
        self.create_description()
        
    def create_university_info(self):
        """Crea la sección de información de la universidad y desarrolladores"""
        info_frame = tk.Frame(self.main_container, bg=COLOR_LIGHT_BG, padx=20, pady=15)
        info_frame.pack(fill='x', pady=(0, 20))
        
        # Información de la universidad
        university_label = tk.Label(
            info_frame, 
            text="Universidad de Cundinamarca", 
            font=("Arial", 14, "bold"), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_PRIMARY
        )
        university_label.pack(anchor='w')
        
        program_label = tk.Label(
            info_frame, 
            text="Ingeniería de Sistemas", 
            font=("Arial", 12), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT
        )
        program_label.pack(anchor='w')
        
        course_label = tk.Label(
            info_frame, 
            text="Inteligencia Artificial - 2025-1", 
            font=("Arial", 12), 
            bg=COLOR_LIGHT_BG, 
            fg=COLOR_TEXT
        )
        course_label.pack(anchor='w', pady=(0, 10))
        
    def create_description(self):
        """Crea la descripción de la red neuronal backpropagation"""
        # Título de la sección
        section_title = tk.Label(
            self.main_container, 
            text="¿Qué es una Red Neuronal Backpropagation?", 
            font=("Arial", 14, "bold"), 
            bg=COLOR_BG, 
            fg=COLOR_TEXT
        )
        section_title.pack(anchor='w', pady=(0, 10))
        
        # Texto de descripción
        description_text = """
        El algoritmo de Backpropagation (propagación hacia atrás) es un método de entrenamiento supervisado para redes neuronales artificiales. Es uno de los métodos más utilizados en el aprendizaje de redes neuronales multicapa.

        Características principales:

        • Aprendizaje supervisado: Requiere un conjunto de datos de entrenamiento con entradas y salidas deseadas.
        
        • Propagación hacia adelante: La información fluye desde la capa de entrada, a través de las capas ocultas, hasta la capa de salida.
        
        • Cálculo del error: Se compara la salida generada con la salida deseada para calcular el error.
        
        • Propagación hacia atrás: El error se propaga desde la capa de salida hacia las capas anteriores para ajustar los pesos.
        
        • Descenso de gradiente: Los pesos se ajustan en la dirección que reduce el error, utilizando la derivada del error respecto a cada peso.

        Aplicaciones:

        • Reconocimiento de patrones
        • Clasificación de imágenes
        • Procesamiento de lenguaje natural
        • Predicción de series temporales
        • Sistemas de recomendación
        
        En este laboratorio, implementaremos una red neuronal backpropagation para dos aplicaciones diferentes:
        
        1. Laboratorio #3: Implementación básica de una red backpropagation
        2. Laboratorio #3a: Clasificación de imágenes de vocales
        """
        
        # Crear un widget Text para mostrar la descripción con formato
        description = tk.Text(
            self.main_container, 
            wrap=tk.WORD, 
            bg=COLOR_BG, 
            fg=COLOR_TEXT,
            font=("Arial", 11),
            height=20,
            width=80,
            bd=0,
            padx=10,
            pady=10,
            highlightthickness=0
        )
        description.pack(fill='both', expand=True)
        
        # Insertar el texto
        description.insert(tk.END, description_text)
        
        # Hacer el widget de solo lectura
        description.config(state=tk.DISABLED)
        
        # Agregar un diagrama simple de una red neuronal
        self.create_network_diagram()
        
    def create_network_diagram(self):
        """Crea un diagrama simple de una red neuronal"""
        diagram_frame = tk.Frame(self.main_container, bg=COLOR_BG)
        diagram_frame.pack(pady=20)
        
        # Título del diagrama
        diagram_title = tk.Label(
            diagram_frame, 
            text="Estructura de una Red Neuronal Multicapa", 
            font=("Arial", 12, "bold"), 
            bg=COLOR_BG, 
            fg=COLOR_TEXT
        )
        diagram_title.pack(pady=(0, 10))
        
        # Canvas para dibujar el diagrama
        canvas = tk.Canvas(diagram_frame, width=600, height=300, bg=COLOR_BG, highlightthickness=0)
        canvas.pack()
        
        # Dibujar capas
        # Capa de entrada (3 neuronas)
        for i in range(3):
            y = 50 + i * 80
            canvas.create_oval(50, y, 90, y+40, fill="#4fc3f7", outline="")
            canvas.create_text(70, y+20, text=f"x{i+1}", font=("Arial", 10, "bold"), fill="white")
        
        # Capa oculta (4 neuronas)
        for i in range(4):
            y = 30 + i * 60
            canvas.create_oval(250, y, 290, y+40, fill="#7e57c2", outline="")
            canvas.create_text(270, y+20, text=f"h{i+1}", font=("Arial", 10, "bold"), fill="white")
        
        # Capa de salida (2 neuronas)
        for i in range(2):
            y = 80 + i * 100
            canvas.create_oval(450, y, 490, y+40, fill="#66bb6a", outline="")
            canvas.create_text(470, y+20, text=f"y{i+1}", font=("Arial", 10, "bold"), fill="white")
        
        # Dibujar conexiones
        # Conexiones entre capa de entrada y capa oculta
        for i in range(3):
            y1 = 70 + i * 80
            for j in range(4):
                y2 = 50 + j * 60
                canvas.create_line(90, y1, 250, y2, fill="#bbbbbb", width=1)
        
        # Conexiones entre capa oculta y capa de salida
        for i in range(4):
            y1 = 50 + i * 60
            for j in range(2):
                y2 = 100 + j * 100
                canvas.create_line(290, y1, 450, y2, fill="#bbbbbb", width=1)
        
        # Etiquetas de las capas
        canvas.create_text(70, 280, text="Capa de Entrada", font=("Arial", 10), fill=COLOR_TEXT)
        canvas.create_text(270, 280, text="Capa Oculta", font=("Arial", 10), fill=COLOR_TEXT)
        canvas.create_text(470, 280, text="Capa de Salida", font=("Arial", 10), fill=COLOR_TEXT)
