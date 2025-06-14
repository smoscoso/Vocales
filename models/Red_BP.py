import numpy as np
import json
import time

class RedBP:
    def __init__(self, config):
        """Inicializa la red neuronal con la configuración proporcionada"""
        self.capa_entrada = config['capa_entrada']
        self.capa_oculta = config['capa_oculta']
        self.capa_salida = config['capa_salida']
        self.alfa = config['alfa']
        self.max_epocas = config['max_epocas']
        self.precision = config['precision']
        self.bias = config['bias']
        self.funciones_activacion = config['funciones_activacion']
        self.beta_leaky_relu = config.get('beta_leaky_relu', 0.01)
        self.momentum = config.get('momentum', False)
        self.beta = config.get('beta', 0.0)
        
        # Variables para seguimiento del entrenamiento
        self.epoca_actual = 0
        self.error_actual = float('inf')
        
        # Inicializar pesos aleatoriamente
        self.inicializar_pesos()
        
        # Inicializar variables para momentum si está habilitado
        if self.momentum:
            self.delta_w_oculta_prev = np.zeros((self.capa_oculta, self.capa_entrada))
            self.delta_w_salida_prev = np.zeros((self.capa_salida, self.capa_oculta))
            self.delta_b_oculta_prev = np.zeros((self.capa_oculta, 1))
            self.delta_b_salida_prev = np.zeros((self.capa_salida, 1))
    
    def inicializar_pesos(self):
        """Inicializa los pesos y bias de la red con valores aleatorios pequeños"""
        # Inicialización de Xavier/Glorot para mejorar la convergencia
        limite_oculta = np.sqrt(6 / (self.capa_entrada + self.capa_oculta))
        limite_salida = np.sqrt(6 / (self.capa_oculta + self.capa_salida))
        
        self.w_oculta = np.random.uniform(-limite_oculta, limite_oculta, (self.capa_oculta, self.capa_entrada))
        self.w_salida = np.random.uniform(-limite_salida, limite_salida, (self.capa_salida, self.capa_oculta))
        
        # Inicializar bias con ceros
        self.b_oculta = np.zeros((self.capa_oculta, 1))
        self.b_salida = np.zeros((self.capa_salida, 1))
    
    def activacion(self, x, funcion, derivada=False):
        """Aplica la función de activación especificada"""
        if funcion == 'sigmoide':
            if derivada:
                return x * (1 - x)
            return 1 / (1 + np.exp(-x))
        
        elif funcion == 'tanh':
            if derivada:
                return 1 - np.power(np.tanh(x), 2)
            return np.tanh(x)
        
        elif funcion == 'relu':
            if derivada:
                return np.where(x > 0, 1, 0)
            return np.maximum(0, x)
        
        elif funcion == 'leaky relu':
            if derivada:
                return np.where(x > 0, 1, self.beta_leaky_relu)
            return np.where(x > 0, x, x * self.beta_leaky_relu)
        
        elif funcion == 'lineal':
            if derivada:
                return np.ones_like(x)
            return x
        
        elif funcion == 'softmax':
            if derivada:
                # La derivada de softmax se maneja de manera especial en el algoritmo
                return x
            exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)
        
        else:
            raise ValueError(f"Función de activación no reconocida: {funcion}")
    
    def forward(self, X):
        """Propagación hacia adelante"""
        # Capa oculta
        self.z_oculta = np.dot(self.w_oculta, X) + self.b_oculta
        self.a_oculta = self.activacion(self.z_oculta, self.funciones_activacion[0])
        
        # Capa de salida
        self.z_salida = np.dot(self.w_salida, self.a_oculta) + self.b_salida
        self.a_salida = self.activacion(self.z_salida, self.funciones_activacion[1])
        
        return self.a_salida
    
    def backward(self, X, Y, salida):
        """Propagación hacia atrás y actualización de pesos"""
        m = X.shape[1]  # Número de ejemplos
        
        # Cálculo del error en la capa de salida
        if self.funciones_activacion[1] == 'softmax':
            # Para softmax, el error es simplemente la diferencia
            delta_salida = salida - Y
        else:
            # Para otras funciones, multiplicamos por la derivada
            delta_salida = (salida - Y) * self.activacion(self.a_salida, self.funciones_activacion[1], derivada=True)
        
        # Cálculo del error en la capa oculta
        delta_oculta = np.dot(self.w_salida.T, delta_salida) * self.activacion(self.a_oculta, self.funciones_activacion[0], derivada=True)
        
        # Cálculo de gradientes
        dw_salida = np.dot(delta_salida, self.a_oculta.T) / m
        db_salida = np.sum(delta_salida, axis=1, keepdims=True) / m
        dw_oculta = np.dot(delta_oculta, X.T) / m
        db_oculta = np.sum(delta_oculta, axis=1, keepdims=True) / m
        
        # Actualización de pesos con momentum si está habilitado
        if self.momentum:
            # Calcular deltas con momentum
            delta_w_salida = -self.alfa * dw_salida + self.beta * self.delta_w_salida_prev
            delta_b_salida = -self.alfa * db_salida + self.beta * self.delta_b_salida_prev
            delta_w_oculta = -self.alfa * dw_oculta + self.beta * self.delta_w_oculta_prev
            delta_b_oculta = -self.alfa * db_oculta + self.beta * self.delta_b_oculta_prev
            
            # Guardar deltas para la siguiente iteración
            self.delta_w_salida_prev = delta_w_salida
            self.delta_b_salida_prev = delta_b_salida
            self.delta_w_oculta_prev = delta_w_oculta
            self.delta_b_oculta_prev = delta_b_oculta
        else:
            # Actualización estándar sin momentum
            delta_w_salida = -self.alfa * dw_salida
            delta_b_salida = -self.alfa * db_salida
            delta_w_oculta = -self.alfa * dw_oculta
            delta_b_oculta = -self.alfa * db_oculta
        
        # Aplicar actualizaciones
        self.w_salida += delta_w_salida
        self.b_salida += delta_b_salida
        self.w_oculta += delta_w_oculta
        self.b_oculta += delta_b_oculta
    
    def calcular_error(self, Y, salida):
        """Calcula el error cuadrático medio"""
        return np.mean(np.sum((Y - salida) ** 2, axis=0)) / 2
    
    def entrenar(self, X, Y, callback=None):
        """Entrena la red neuronal con los datos proporcionados"""
        # Convertir a matrices numpy si no lo son ya
        X = np.array(X).T  # Transponer para tener ejemplos en columnas
        Y = np.array(Y).T  # Transponer para tener ejemplos en columnas
        
        # Lista para almacenar errores durante el entrenamiento
        errores = []
        
        # Iniciar entrenamiento
        inicio = time.time()
        for epoca in range(self.max_epocas):
            self.epoca_actual = epoca + 1
            
            # Forward pass
            salida = self.forward(X)
            
            # Calcular error
            error = self.calcular_error(Y, salida)
            self.error_actual = error
            errores.append(error)
            
            # Llamar al callback si existe
            if callback:
                callback(epoca + 1, self.max_epocas, error)
            
            # Verificar condición de parada
            if error <= self.precision:
                break
            
            # Backward pass y actualización de pesos
            self.backward(X, Y, salida)
            
            # Mostrar progreso cada 1000 épocas o en la última
            if (epoca + 1) % 1000 == 0 or epoca == self.max_epocas - 1:
                tiempo_transcurrido = time.time() - inicio
                print(f"Época {epoca + 1}/{self.max_epocas}, Error: {error:.6f}, Tiempo: {tiempo_transcurrido:.2f}s")
        
        # Calcular exactitud final
        salida_final = self.forward(X)
        predicciones = np.argmax(salida_final, axis=0)
        etiquetas = np.argmax(Y, axis=0)
        exactitud = np.mean(predicciones == etiquetas) * 100
        
        print(f"Entrenamiento completado en {self.epoca_actual} épocas")
        print(f"Error final: {self.error_actual:.6f}")
        print(f"Exactitud: {exactitud:.2f}%")
        
        return errores, exactitud
    
    def predecir(self, X):
        """Realiza predicciones para los datos de entrada"""
        # Convertir a matriz numpy si no lo es ya
        X = np.array(X)
        
        # Si X es un solo ejemplo (vector), convertirlo a matriz columna
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 2 and X.shape[0] != self.capa_entrada:
            # Si X tiene múltiples ejemplos en filas, transponerlo
            X = X.T
        
        # Forward pass
        return self.forward(X).T  # Transponer de vuelta para tener ejemplos en filas
    
    def guardar_pesos(self, archivo):
        """Guarda los pesos y bias de la red en un archivo JSON"""
        datos = {
            'w_oculta': self.w_oculta.tolist(),
            'b_oculta': self.b_oculta.tolist(),
            'w_salida': self.w_salida.tolist(),
            'b_salida': self.b_salida.tolist(),
            'config': {
                'capa_entrada': self.capa_entrada,
                'capa_oculta': self.capa_oculta,
                'capa_salida': self.capa_salida,
                'funciones_activacion': self.funciones_activacion,
                'beta_leaky_relu': self.beta_leaky_relu
            }
        }
        
        with open(archivo, 'w') as f:
            json.dump(datos, f)
    
    def cargar_pesos(self, archivo):
        """Carga los pesos y bias de la red desde un archivo JSON"""
        with open(archivo, 'r') as f:
            datos = json.load(f)
        
        # Cargar pesos y bias
        self.w_oculta = np.array(datos['w_oculta'])
        self.b_oculta = np.array(datos['b_oculta'])
        self.w_salida = np.array(datos['w_salida'])
        self.b_salida = np.array(datos['b_salida'])
        
        # Verificar y actualizar configuración si es necesario
        config = datos.get('config', {})
        if config:
            # Verificar dimensiones
            if config.get('capa_entrada') != self.capa_entrada:
                print(f"Advertencia: La capa de entrada en el archivo ({config.get('capa_entrada')}) no coincide con la configuración actual ({self.capa_entrada})")
            
            if config.get('capa_oculta') != self.capa_oculta:
                print(f"Advertencia: La capa oculta en el archivo ({config.get('capa_oculta')}) no coincide con la configuración actual ({self.capa_oculta})")
                self.capa_oculta = config.get('capa_oculta')
            
            if config.get('capa_salida') != self.capa_salida:
                print(f"Advertencia: La capa de salida en el archivo ({config.get('capa_salida')}) no coincide con la configuración actual ({self.capa_salida})")
                self.capa_salida = config.get('capa_salida')
            
            # Actualizar funciones de activación si están presentes
            if 'funciones_activacion' in config:
                self.funciones_activacion = config['funciones_activacion']
            
            # Actualizar beta para Leaky ReLU si está presente
            if 'beta_leaky_relu' in config:
                self.beta_leaky_relu = config['beta_leaky_relu']
