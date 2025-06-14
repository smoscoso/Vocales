import os
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix

from models.backpropagation import RedBP
from views.main_view import MainView

class BackpropController:
    def __init__(self, root):
        """Inicializa el controlador para el laboratorio de backpropagation"""
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
        
        # Inicializar la aplicación
        self.view.log("Laboratorio #3 inicializado correctamente")
        self.view.log("Utilice este laboratorio para implementar una red neuronal backpropagation básica")
    
    def conectar_eventos(self):
        """Conecta los eventos de la vista con los métodos del controlador"""
        # Eventos de configuración
        self.view.func_oculta.bind("<<ComboboxSelected>>", self.actualizar_interfaz_activacion)
        
        # Eventos de datos
        self.view.btn_cargar_entrada.config(command=self.cargar_datos_entrenamiento)
        
        # Eventos de entrenamiento
        self.view.btn_entrenar.config(command=self.entrenar_red)
        
        # Eventos de prueba
        self.view.btn_cargar_pesos.config(command=self.cargar_pesos)
        self.view.btn_clasificar.config(command=self.clasificar_patron)
    
    def actualizar_interfaz_activacion(self, event=None):
        """Actualiza la interfaz según la función de activación seleccionada"""
        func_oculta = self.view.func_oculta.get().lower()
        
        # Habilitar o deshabilitar el campo Beta para Leaky ReLU
        if func_oculta == 'leaky relu':
            self.view.beta_oculta_input.config(state='normal')
        else:
            self.view.beta_oculta_input.config(state='disabled')
    
    def cargar_datos_entrenamiento(self):
        """Carga los datos de entrenamiento desde un archivo"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if not archivo:
            return
            
        try:
            self.view.log(f"Cargando datos desde: {archivo}")
            
            # Leer el archivo
            datos_entrada = []
            datos_salida = []
            
            with open(archivo, 'r') as f:
                for linea in f:
                    # Formato esperado: x1 x2 ... xn | y1 y2 ... ym
                    partes = linea.strip().split('|')
                    if len(partes) != 2:
                        continue
                        
                    # Procesar entrada
                    entrada = list(map(float, partes[0].strip().split()))
                    
                    # Procesar salida
                    salida = list(map(float, partes[1].strip().split()))
                    
                    datos_entrada.append(entrada)
                    datos_salida.append(salida)
            
            # Verificar que se cargaron datos
            if len(datos_entrada) == 0:
                self.view.log("Error: No se encontraron datos válidos en el archivo")
                return
                
            # Guardar los datos
            self.datos_entrenamiento = datos_entrada
            self.datos_salida = datos_salida
            
            # Actualizar la interfaz
            self.view.patrones_label.config(text=str(len(datos_entrada)))
            self.view.log(f"Se cargaron {len(datos_entrada)} patrones de entrenamiento")
            
            # Actualizar sugerencias de arquitectura
            self.actualizar_sugerencias()
            
        except Exception as e:
            self.view.log(f"Error al cargar datos: {str(e)}")
    
    def actualizar_sugerencias(self):
        """Actualiza las sugerencias de arquitectura basadas en los datos cargados"""
        if (self.datos_entrenamiento is not None and 
            self.datos_salida is not None and
            len(self.datos_entrenamiento) > 0 and 
            len(self.datos_salida) > 0):

            # Calcular valores
            entrada = len(self.datos_entrenamiento[0])
            salida = len(self.datos_salida[0])
            oculta_sugerida = int(np.sqrt(entrada * salida))

            # Actualizar etiquetas
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
            # Obtener valores de arquitectura
            capa_entrada = int(self.view.entrada_input.get())
            capa_oculta = int(self.view.oculta_input.get())
            capa_salida = int(self.view.salida_input.get())
            
            # Obtener valores de parámetros de entrenamiento
            alfa = float(self.view.alpha_input.get())
            max_epocas = int(self.view.max_epocas_input.get())
            precision = float(self.view.precision_input.get())
            
            # Obtener valor de bias
            bias = self.view.bias_var.get() > 0
            
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
    
    def entrenar_red(self):
        """Entrena la red neuronal con los datos cargados"""
        # Evitar múltiples entrenamientos simultáneos
        if self.entrenamiento_en_progreso:
            self.view.log("Ya hay un entrenamiento en curso. Espere a que termine.")
            return
            
        try:
            # Validar que los datos estén cargados
            if self.datos_entrenamiento is None or self.datos_salida is None:
                self.view.log("Error: Cargue los datos de entrenamiento primero")
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
            
            # Inicializar variables de seguimiento
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
            
            # Usar archivos diferentes para diferentes laboratorios
            self.pesos_archivo = os.path.join(carpeta, "pesos_lab3.json")
            self.red.guardar_pesos(self.pesos_archivo)
            
            # Generar matriz de confusión
            self.generar_matriz_confusion()
            
            # Actualizar información del modelo en el panel de prueba
            self.actualizar_info_modelo()
            
            # Marcar finalización del entrenamiento
            self.entrenamiento_completado = True
            
        except Exception as e:
            self.view.log(f"Error durante el entrenamiento: {str(e)}")
            import traceback
            self.view.log(traceback.format_exc())
        finally:
            # Asegurar que la interfaz se actualice al finalizar
            self.entrenamiento_en_progreso = False
    
    def generar_matriz_confusion(self):
        """Genera la matriz de confusión para el entrenamiento"""
        try:
            # Preparar datos para la matriz de confusión
            X = np.array(self.datos_entrenamiento)
            Y = np.array(self.datos_salida)
            
            # Obtener predicciones
            y_true = []
            y_pred = []
            
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                pred = self.red.predecir([x])[0]
                
                # Para clasificación binaria
                if len(y) == 1:
                    y_true.append(1 if y[0] > 0.5 else 0)
                    y_pred.append(1 if pred[0] > 0.5 else 0)
                # Para clasificación multiclase
                else:
                    y_true.append(np.argmax(y))
                    y_pred.append(np.argmax(pred))
            
            # Calcular matriz de confusión
            # Determinar el número de clases
            if len(self.datos_salida[0]) == 1:
                # Clasificación binaria
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                etiquetas = ['Clase 0', 'Clase 1']
            else:
                # Clasificación multiclase
                num_clases = len(self.datos_salida[0])
                cm = confusion_matrix(y_true, y_pred, labels=range(num_clases))
                etiquetas = [f'Clase {i}' for i in range(num_clases)]
            
            # Mostrar matriz de confusión en la vista
            self.view.mostrar_matriz_confusion(cm, etiquetas)
            
            # Cambiar a la pestaña de gráficas para mostrar resultados
            self.view.notebook.select(1)  # Índice 1 corresponde a la pestaña "Gráficas"
            
        except Exception as e:
            self.view.log(f"Error al generar matriz de confusión: {str(e)}")
            import traceback
            self.view.log(traceback.format_exc())
    
    def actualizar_progreso_entrenamiento(self):
        """Actualiza la interfaz durante el entrenamiento"""
        if not self.entrenamiento_en_progreso:
            # El entrenamiento ha finalizado
            if hasattr(self, 'entrenamiento_completado') and self.entrenamiento_completado:
                # Actualizar estado del entrenamiento
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
                
                # Limpiar bandera
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
        
        # Programar próxima actualización
        self.view.root.after(100, self.actualizar_progreso_entrenamiento)
    
    def cargar_pesos(self):
        """Carga los pesos desde un archivo JSON"""
        archivo = filedialog.askopenfilename(filetypes=[("Archivos JSON", "*.json")])
        if archivo:
            try:
                # Crear una instancia de red si no existe
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
                
                # Actualizar estado del entrenamiento
                self.view.status_label.config(text="Estado: Pesos Cargados", foreground="green")
                self.view.status_indicator.delete("all")
                self.view.status_indicator.create_oval(2, 2, 13, 13, fill="green", outline="")
                
                # Actualizar información del modelo en el panel de prueba
                self.actualizar_info_modelo()
                
            except Exception as e:
                self.view.log(f"Error al cargar pesos: {str(e)}")
    
    def clasificar_patron(self):
        """Clasifica un patrón de entrada"""
        if self.red is None:
            self.view.log("Error: Primero debe entrenar la red o cargar pesos")
            return
            
        try:
            # Obtener patrón de entrada desde la interfaz
            patron_texto = self.view.pattern_input.get("1.0", tk.END).strip()
            
            # Convertir a lista de números
            patron = []
            for valor in patron_texto.split():
                patron.append(float(valor))
                
            # Verificar dimensiones
            if len(patron) != self.red.config['capa_entrada']:
                self.view.log(f"Error: El patrón debe tener {self.red.config['capa_entrada']} valores")
                return
                
            # Clasificar el patrón
            resultado = self.red.predecir([patron])[0]
            
            # Mostrar resultado en la interfaz
            self.view.output_text.delete("1.0", tk.END)
            
            # Para clasificación binaria
            if len(resultado) == 1:
                self.view.output_text.insert(tk.END, f"Salida: {resultado[0]:.6f}\n")
                self.view.output_text.insert(tk.END, f"Clase: {1 if resultado[0] > 0.5 else 0}")
            # Para clasificación multiclase
            else:
                for i, valor in enumerate(resultado):
                    self.view.output_text.insert(tk.END, f"Clase {i}: {valor:.6f}\n")
                
                clase_predicha = np.argmax(resultado)
                self.view.output_text.insert(tk.END, f"\nClase predicha: {clase_predicha}")
                
            self.view.log(f"Patrón clasificado exitosamente")
            
        except ValueError:
            self.view.log("Error: Formato de patrón inválido. Ingrese valores numéricos separados por espacios")
        except Exception as e:
            self.view.log(f"Error al clasificar patrón: {str(e)}")
    
    def actualizar_info_modelo(self):
        """Actualiza la información del modelo en la interfaz de prueba"""
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
            arquitectura = f"{self.red.config['capa_entrada']}-{self.red.config['capa_oculta']}-{self.red.config['capa_salida']}"
            activaciones = f"{self.red.config['funciones_activacion'][0]}/{self.red.config['funciones_activacion'][1]}"
            self.view.architecture_info.config(text=f"{arquitectura} ({activaciones})")
        else:
            # Restablecer valores predeterminados
            self.view.model_status.config(text="No cargado", foreground="red")
            self.view.weights_file.config(text="Ninguno")
            self.view.architecture_info.config(text="No disponible")
