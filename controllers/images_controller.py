"""
Controlador principal de la aplicación
Universidad de Cundinamarca
"""

import os
import numpy as np
import threading
import tkinter as tk  # Añadir esta importación
from tkinter import filedialog
from PIL import Image
from sklearn.metrics import confusion_matrix

from models.Red_BP import RedBP
from models.data_processor import normalize_image, save_normalized_data, load_training_data, determine_dominant_color, process_test_image
from views.main_view_Images import MainView

class AppController:
    def __init__(self, root):
        """Inicializa el controlador de la aplicación"""
        # Inicializar la vista
        self.view = MainView(root)
        
        # Inicializar variables del modelo
        self.red = None
        self.datos_entrenamiento = None
        self.datos_salida = None
        self.pesos_archivo = None
        self.entrenamiento_en_progreso = False
        
        # Conectar eventos de la vista
        self.conectar_eventos()
    
    def conectar_eventos(self):
        """Conecta los eventos de la vista con los métodos del controlador"""
        # Eventos de configuración y entrenamiento
        self.view.btn_cargar_entrada.config(command=self.cargar_carpeta_entrada)
        self.view.btn_entrenar.config(command=self.entrenar_red)
        self.view.func_oculta.bind("<<ComboboxSelected>>", self.actualizar_interfaz_activacion)
        
        # Eventos de pruebas
        self.view.btn_cargar_prueba.config(command=self.cargar_imagen_prueba)
        self.view.btn_cargar_pesos.config(command=self.cargar_pesos)
    
    def actualizar_interfaz_activacion(self, event=None):
        """Actualiza la interfaz según la función de activación seleccionada"""
        func_oculta = self.view.func_oculta.get().lower()
        
        # Habilitar o deshabilitar el campo beta de Leaky ReLU
        if func_oculta == 'leaky relu':
            self.view.beta_oculta_input.config(state='normal')
        else:
            self.view.beta_oculta_input.config(state='disabled')
    
    def actualizar_sugerencias(self):
        """Actualiza las sugerencias de capas en la interfaz"""
        if (self.datos_entrenamiento is not None and 
            self.datos_salida is not None and
            len(self.datos_entrenamiento) > 0 and 
            len(self.datos_salida) > 0):

            # Calcular valores
            entrada = len(self.datos_entrenamiento[0])
            salida = len(self.datos_salida[0])
            oculta_sugerida = int(np.sqrt(entrada * salida))

            # Actualizar labels
            self.view.lbl_entrada1.config(text=f"Automático ({entrada})")
            self.view.lbl_oculta1.config(text=f"Sugerido ({oculta_sugerida})")
            self.view.lbl_salida1.config(text=f"Automático ({salida})")

            # Actualizar campos editables
            self.view.entrada_input.delete(0, tk.END)
            self.view.entrada_input.insert(0, str(entrada))
            self.view.oculta_input.delete(0, tk.END)
            self.view.oculta_input.insert(0, str(oculta_sugerida))
            self.view.salida_input.delete(0, tk.END)
            self.view.salida_input.insert(0, str(salida))
    
    def get_config(self):
        """Obtiene la configuración de la red desde la interfaz"""
        try:
            # Obtener valores de la arquitectura
            capa_entrada = int(self.view.entrada_input.get())
            capa_oculta = int(self.view.oculta_input.get())
            capa_salida = int(self.view.salida_input.get())
            
            # Obtener valores de los parámetros de entrenamiento
            alfa = float(self.view.alpha_input.get())
            max_epocas = int(self.view.max_epocas_input.get())
            precision = float(self.view.precision_input.get())
            
            # Obtener valor de bias
            bias = float(self.view.bias_input.get()) > 0
            
            # Obtener funciones de activación
            func_oculta = self.view.func_oculta.get().lower()
            func_salida = self.view.func_salida.get().lower()
            
            # Obtener beta para Leaky ReLU si es necesario
            beta_leaky_relu = float(self.view.beta_oculta_input.get())
            
            # Obtener valor de momentum si está habilitado
            momentum = self.view.momentum_var.get()
            beta = 0.0
            if momentum:
                beta = float(self.view.beta_input.get())
        
            # Crear diccionario de configuración
            config = {
                'capa_entrada': capa_entrada,
                'capa_oculta': capa_oculta,
                'capa_salida': capa_salida,
                'alfa': alfa,
                'max_epocas': max_epocas,
                'precision': precision,
                'bias': bias,
                'funciones_activacion': [func_oculta, func_salida],
                'beta_leaky_relu': beta_leaky_relu,
                'momentum': momentum,
                'beta': beta
            }
        
            return config
    
        except ValueError as e:
            self.view.log(f"Error en la configuración: {str(e)}")
            self.view.log("Verifique que todos los campos numéricos contengan valores válidos")
            return None
        except Exception as e:
            self.view.log(f"Error inesperado: {str(e)}")
            return None
    
    def cargar_carpeta_entrada(self):
        """Carga imágenes de entrenamiento desde una carpeta"""
        carpeta = filedialog.askdirectory(title="Seleccionar carpeta con imágenes de entrenamiento")
        if carpeta:
            try:
                self.view.log(f"Cargando imágenes de la carpeta: {carpeta}")
                
                # Crear archivo temporal para datos normalizados
                archivo_temp = "temp_normalized_data.txt"
                save_normalized_data(carpeta, archivo_temp)
                
                # Cargar datos normalizados
                self.datos_entrenamiento, self.datos_salida = load_training_data(archivo_temp)
                
                if len(self.datos_entrenamiento) > 0:
                    self.view.log(f"Cargadas {len(self.datos_entrenamiento)} imágenes válidas")
                    
                    # Actualizar sugerencias de arquitectura
                    self.actualizar_sugerencias()
                else:
                    self.view.log("No se encontraron imágenes válidas en la carpeta")
                
            except Exception as e:
                self.view.log(f"Error al cargar imágenes: {str(e)}")
    
    def entrenar_red(self):
        """Entrena la red neuronal con los datos cargados"""
        # Evitar múltiples entrenamientos simultáneos
        if self.entrenamiento_en_progreso:
            self.view.log("Ya hay un entrenamiento en curso. Espere a que termine.")
            return
            
        try:
            # Validar que los datos estén cargados
            if self.datos_entrenamiento is None or self.datos_salida is None:
                self.view.log("Error: Cargue los datos de entrenamiento y salida primero")
                return

            # Obtener configuración
            config = self.get_config()
            if config is None:
                return
                
            # Marcar inicio del entrenamiento
            self.entrenamiento_en_progreso = True
            self.view.btn_entrenar.config(state='disabled')
            
            # Reiniciar barra de progreso
            self.view.progress_bar['value'] = 0
            self.view.progress_label.config(text="0%")
            
            # Mostrar parámetros de entrenamiento
            self.view.log(f"Parámetros de entrenamiento:")
            self.view.log(f"- Alfa: {float(config['alfa'])}")
            self.view.log(f"- Épocas máximas: {int(config['max_epocas'])}")
            self.view.log(f"- Precisión objetivo: {float(config['precision'])}")
            self.view.log(f"- Funciones de activación: {config['funciones_activacion'][0]} (oculta), {config['funciones_activacion'][1]} (salida)")
            if 'leaky relu' in config['funciones_activacion']:
                self.view.log(f"- Beta Leaky ReLU: {float(config['beta_leaky_relu'])}")
            if config['momentum']:
                self.view.log(f"- Momentum habilitado con Beta: {float(config['beta'])}")

            # Crear red neuronal
            self.red = RedBP(config)
            
            # Iniciar entrenamiento en un hilo separado para no bloquear la interfaz
            self.thread_entrenamiento = threading.Thread(target=self.ejecutar_entrenamiento)
            self.thread_entrenamiento.daemon = True
            self.thread_entrenamiento.start()
            
            # Iniciar actualización periódica de la interfaz
            self.view.root.after(100, self.actualizar_progreso_entrenamiento)

        except Exception as e:
            self.view.log(f"Error al iniciar entrenamiento: {str(e)}")
            self.entrenamiento_en_progreso = False
            self.view.btn_entrenar.config(state='normal')
            import traceback
            self.view.log(traceback.format_exc())
    
    def actualizar_progreso_callback(self, epoca, max_epocas, error):
        """Callback para actualizar el progreso del entrenamiento"""
        self.epoca_actual = epoca
        self.error_actual = error
        
        # Calcular porcentaje de progreso
        if max_epocas > 0:
            self.progreso = min(100, int((epoca / max_epocas) * 100))
    
    def ejecutar_entrenamiento(self):
        """Ejecuta el entrenamiento en un hilo separado"""
        try:
            # Entrenar y obtener errores
            self.view.log("Iniciando entrenamiento con backpropagation...")
            
            # Inicializar variables para seguimiento
            self.epoca_actual = 0
            self.error_actual = float('inf')
            self.progreso = 0
            
            # Entrenar la red con callback para actualizar progreso
            self.errores_entrenamiento, self.exactitud = self.red.entrenar(
                np.array(self.datos_entrenamiento),
                np.array(self.datos_salida),
                callback=self.actualizar_progreso_callback
            )
            
            # Guardar pesos automáticamente
            carpeta = "Pesos_entrenados"
            if not os.path.exists(carpeta):
                os.makedirs(carpeta)
            self.pesos_archivo = os.path.join(carpeta, "pesos_actuales_Images.json")
            self.red.guardar_pesos(self.pesos_archivo)
            
            # Generar matriz de confusión
            self.view.root.after(0, self.generar_matriz_confusion())
            
            # Actualizar información del modelo en el panel de pruebas
            self.actualizar_info_modelo()
            
            # Marcar finalización del entrenamiento
            self.entrenamiento_completado = True
            
        except Exception as e:
            self.view.log(f"Error durante el entrenamiento: {str(e)}")
            import traceback
            self.view.log(traceback.format_exc())
        finally:
            # Asegurarse de que la interfaz se actualice al finalizar
            self.entrenamiento_en_progreso = False
    
    def generar_matriz_confusion(self):
        """Genera la matriz de confusión para el entrenamiento"""
        try:
            # Preparar datos para la matriz de confusión
            X = np.array(self.datos_entrenamiento).T
            Y = np.array(self.datos_salida).T
            
            # Obtener predicciones
            y_true = []
            y_pred = []
            
            for i in range(X.shape[1]):
                x = X[:, [i]]
                y = Y[:, [i]]
                pred = self.red.predecir([X[:, i]])[0]
                y_true.append(np.argmax(y))
                y_pred.append(np.argmax(pred))
            
            # Calcular matriz de confusión
            cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
            
            # Mostrar matriz de confusión en la vista
            self.view.root.after(0, lambda: self.view.mostrar_matriz_confusion(cm, ['A', 'E', 'I', 'O', 'U']))
            
            # Cambiar a la pestaña de gráficas para mostrar los resultados
            self.view.notebook.select(1)  # Índice 1 corresponde a la pestaña "Gráficas"
            
        except Exception as e:
            self.view.log(f"Error al generar matriz de confusión: {str(e)}")
            import traceback
            self.view.log(traceback.format_exc())
    
    def actualizar_progreso_entrenamiento(self):
        """Actualiza la interfaz durante el entrenamiento"""
        if not self.entrenamiento_en_progreso:
            # El entrenamiento ha terminado
            if hasattr(self, 'entrenamiento_completado') and self.entrenamiento_completado:
                # Actualizar estado de entrenamiento
                self.view.status_label.config(text="Estado: Entrenado Exitosamente", foreground="green")
                self.view.status_indicator.delete("all")
                self.view.status_indicator.create_oval(2, 2, 13, 13, fill="green", outline="")
                
                # Actualizar métricas
                self.view.epochs_value.config(text=str(len(self.errores_entrenamiento)))
                self.view.error_value.config(text=f"{float(self.errores_entrenamiento[-1]):.6f}")
                self.view.accuracy_value.config(text=f"{self.exactitud:.2f}%")
                
                # Mostrar gráfica
                self.view.mostrar_grafica_error(self.errores_entrenamiento)
                self.view.log(f"Entrenamiento completado exitosamente en {len(self.errores_entrenamiento)} épocas")
                self.view.log(f"Pesos guardados automáticamente en: {self.pesos_archivo}")
                
                # Actualizar barra de progreso al 100%
                self.view.progress_bar['value'] = 100
                self.view.progress_label.config(text="100%")
                
                # Limpiar flag
                self.entrenamiento_completado = False
            
            # Habilitar botón de entrenamiento
            self.view.btn_entrenar.config(state='normal')
            return
            
        # Actualizar barra de progreso
        self.view.progress_bar['value'] = self.progreso
        self.view.progress_label.config(text=f"{self.progreso}%")
        
        # Actualizar log cada 100 épocas
        if hasattr(self, 'epoca_actual') and self.epoca_actual % 100 == 0 and self.epoca_actual > 0:
            self.view.log(f"Entrenando... Época {self.epoca_actual}/{self.red.max_epocas} ({self.progreso}%)")
        
        # Programar la próxima actualización
        self.view.root.after(100, self.actualizar_progreso_entrenamiento)
    
    def cargar_pesos(self):
        """Carga los pesos desde un archivo JSON"""
        archivo = filedialog.askopenfilename(filetypes=[("Archivos JSON", "*.json")])
        if archivo:
            try:
                # Crear una instancia de la red si no existe
                if self.red is None:
                    # Obtener configuración básica
                    config = self.get_config()
                    if config is None:
                        return
                    self.red = RedBP(config)
                
                # Cargar pesos
                self.red.cargar_pesos(archivo)
                self.pesos_archivo = archivo
                self.view.log(f"Pesos cargados exitosamente desde: {archivo}")
                
                # Actualizar interfaz
                nombre_archivo = os.path.basename(archivo)
                self.view.log(f"Red lista para clasificar con pesos: {nombre_archivo}")
                
                # Actualizar estado de entrenamiento
                self.view.status_label.config(text="Estado: Pesos Cargados", foreground="green")
                self.view.status_indicator.delete("all")
                self.view.status_indicator.create_oval(2, 2, 13, 13, fill="green", outline="")
                
                # Actualizar información del modelo en el panel de pruebas
                self.actualizar_info_modelo()
                
            except Exception as e:
                self.view.log(f"Error al cargar pesos: {str(e)}")
    
    def cargar_imagen_prueba(self):
        """Carga una imagen para prueba y la clasifica"""
        archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if archivo:
            try:
                # Cargar la imagen
                imagen = Image.open(archivo)
                
                # Mostrar la imagen en el canvas
                self.view.mostrar_imagen_en_canvas(imagen)
                
                # Actualizar información de la imagen
                self.actualizar_info_imagen(archivo, imagen)
                
                # Procesar la imagen para obtener el vector de características
                r, g, b = normalize_image(archivo)
                input_vec = np.concatenate([r, g, b])
                
                # Determinar el color dominante y los porcentajes
                color_dominante, porcentajes = determine_dominant_color(r, g, b)
                
                # Actualizar indicador de color con los porcentajes
                self.view.actualizar_indicador_color(color_dominante, porcentajes)
                
                # Mostrar los porcentajes en el log
                self.view.log(f"Color dominante: {color_dominante}")
                self.view.log(f"Porcentajes: Rojo: {porcentajes['Rojo']:.1f}%, Verde: {porcentajes['Verde']:.1f}%, Azul: {porcentajes['Azul']:.1f}%")
                
                # Si hay una red entrenada, clasificar la imagen
                if self.red is not None:
                    # Realizar clasificación
                    salida = self.red.predecir([input_vec])[0]
                    
                    # Calcular porcentajes de activación
                    activaciones = {vocal: float(salida[i])*100 for i, vocal in enumerate(['A', 'E', 'I', 'O', 'U'])}
                    
                    # Actualizar barras de activación
                    self.view.actualizar_barras(activaciones)
                    
                    # Determinar la vocal clasificada
                    vocal_clasificada = max(activaciones, key=activaciones.get)
                    nivel_activacion = activaciones[vocal_clasificada]
                    
                    # Actualizar etiqueta de vocal
                    self.view.vocal_label.config(text=vocal_clasificada)
                    
                    # Actualizar resultados en texto
                    self.view.log(f"Imagen cargada: {os.path.basename(archivo)}")
                    self.view.log(f"Clasificación: {vocal_clasificada} ({nivel_activacion:.2f}%)")
                else:
                    self.view.log("Error: Primero debe entrenar la red o cargar pesos")
                    # Limpiar el canvas y mostrar mensaje de error
                    self.view.canvas_imagen.delete("all")
                    self.view.canvas_imagen.create_text(
                        125, 125, 
                        text="Se requiere cargar un modelo primero", 
                        font=("Arial", 10, "italic"),
                        fill="red"
                    )
                
            except Exception as e:
                self.view.log(f"Error al cargar imagen: {str(e)}")
                import traceback
                self.view.log(traceback.format_exc())
    
    def actualizar_info_modelo(self):
        """Actualiza la información del modelo en la interfaz de pruebas"""
        if self.red is not None:
            # Actualizar estado del modelo
            self.view.model_status.config(text="Cargado", foreground="green")
            
            # Actualizar archivo de pesos
            if self.pesos_archivo:
                nombre_archivo = os.path.basename(self.pesos_archivo)
                self.view.weights_file.config(text=nombre_archivo)
            else:
                self.view.weights_file.config(text="Generado en memoria")
            
            # Actualizar información de arquitectura
            arquitectura = f"{self.red.capa_entrada}-{self.red.capa_oculta}-{self.red.capa_salida}"
            activaciones = f"{self.red.funciones_activacion[0]}/{self.red.funciones_activacion[1]}"
            self.view.architecture_info.config(text=f"{arquitectura} ({activaciones})")
        else:
            # Restablecer valores predeterminados
            self.view.model_status.config(text="No cargado", foreground="red")
            self.view.weights_file.config(text="Ninguno")
            self.view.architecture_info.config(text="No disponible")
    
    def actualizar_info_imagen(self, imagen_path=None, imagen=None):
        """Actualiza la información de la imagen en la interfaz de pruebas"""
        if imagen_path:
            # Actualizar estado de la imagen
            nombre_archivo = os.path.basename(imagen_path)
            self.view.image_status.config(text=nombre_archivo)
            
            # Actualizar dimensiones
            if imagen:
                ancho, alto = imagen.size
                self.view.image_dimensions.config(text=f"{ancho} x {alto} px")
            else:
                self.view.image_dimensions.config(text="N/A")
        else:
            # Restablecer valores predeterminados
            self.view.image_status.config(text="Ninguna")
            self.view.image_dimensions.config(text="N/A")