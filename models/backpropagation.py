import numpy as np
import json
import os

class RedBP:
    def __init__(self, config):
        """
        Inicializa la red neuronal con la configuración proporcionada.
        
        Args:
            config: Diccionario con la configuración de la red
        """
        self.config = config
        self.inicializar_red()
        
    def inicializar_red(self):
        """
        Inicializa los pesos y umbrales de la red neuronal.
        """
        # Obtener dimensiones de la red
        n = self.config['capa_entrada']
        l = self.config['capa_oculta']
        m = self.config['capa_salida']
        self.bias = self.config.get('bias', True)
        
        # Inicializar pesos de la capa oculta (Wh) con valores aleatorios entre -0.5 y 0.5
        self.W_h = np.random.rand(l, n) - 0.5
        
        # Inicializar pesos de la capa de salida (Wo) con valores aleatorios entre -0.5 y 0.5
        self.W_o = np.random.rand(m, l) - 0.5
        
        # Inicializar umbrales (bias) si están habilitados
        if self.bias:
            self.Th = np.random.rand(l, 1) - 0.5
            self.To = np.random.rand(m, 1) - 0.5
        
        # Diccionario de funciones de activación y sus derivadas
        self.funciones = {
            'sigmoide': [self.sigmoide, self.sigmoide_derivada],
            'tanh': [self.tanh, self.tanh_derivada],
            'relu': [self.relu, self.relu_derivada],
            'leaky relu': [self.leaky_relu, self.leaky_relu_derivada],
            'lineal': [self.lineal, self.lineal_derivada],
            'softmax': [self.softmax, self.softmax_derivada]
        }
        
    def sigmoide(self, x):
        """Función de activación sigmoide"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoide_derivada(self, x):
        """Derivada de la función sigmoide"""
        s = self.sigmoide(x)
        return s * (1 - s)
    
    def tanh(self, x):
        """Función de activación tangente hiperbólica"""
        return np.tanh(x)
    
    def tanh_derivada(self, x):
        """Derivada de la función tangente hiperbólica"""
        return 1 - np.tanh(x)**2
    
    def relu(self, x):
        """Función de activación ReLU"""
        return np.maximum(0, x)
    
    def relu_derivada(self, x):
        """Derivada de la función ReLU"""
        return np.where(x > 0, 1, 0)
    
    def leaky_relu(self, x):
        """Función de activación Leaky ReLU"""
        beta = self.config['beta_leaky_relu']
        return np.where(x > 0, x, beta * x)
    
    def leaky_relu_derivada(self, x):
        """Derivada de la función Leaky ReLU"""
        beta = self.config['beta_leaky_relu']
        return np.where(x > 0, 1, beta)
    
    def lineal(self, x):
        """Función de activación lineal"""
        return x
    
    def lineal_derivada(self, x):
        """Derivada de la función lineal"""
        return np.ones_like(x)
    
    def softmax(self, x):
        """Función de activación softmax"""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def softmax_derivada(self, x):
        """Derivada de la función softmax (simplificada para backpropagation)"""
        s = self.softmax(x)
        return s * (1 - s)
    
    def entrenar(self, X, Yd):
        """
        Entrena la red neuronal utilizando el algoritmo de backpropagation.
        
        Args:
            X: Matriz de patrones de entrada, cada fila es un patrón
            Yd: Matriz de salidas deseadas, cada fila corresponde a un patrón de entrada
            
        Returns:
            Lista de errores por época
        """
        # Parámetros de entrenamiento - usar exactamente los valores definidos por el usuario
        alfa = float(self.config['alfa'])
        max_epocas = int(self.config['max_epocas'])
        precision = float(self.config['precision'])
        
        # Obtener funciones de activación
        func_oculta = self.config['funciones_activacion'][0].lower()
        func_salida = self.config['funciones_activacion'][1].lower()
        
        # Obtener las funciones de activación y sus derivadas
        f_oculta, f_oculta_derivada = self.funciones.get(func_oculta, [self.sigmoide, self.sigmoide_derivada])
        f_salida, f_salida_derivada = self.funciones.get(func_salida, [self.sigmoide, self.sigmoide_derivada])
        
        # Obtener dimensiones
        P = X.shape[0]  # Número de patrones
        n = X.shape[1]  # Número de entradas
        l = self.W_h.shape[0]  # Número de neuronas en capa oculta
        m = self.W_o.shape[0]  # Número de neuronas en capa de salida
        
        # Lista para almacenar errores por época
        errores = []
        
        # Inicializar error
        Et = float('inf')
        epoca = 0
        
        # Variables para momentum si está habilitado
        dW_h_prev = np.zeros_like(self.W_h)
        dW_o_prev = np.zeros_like(self.W_o)
        dTh_prev = np.zeros_like(self.Th) if self.bias else None
        dTo_prev = np.zeros_like(self.To) if self.bias else None
        
        # Bucle principal de entrenamiento
        while Et > precision and epoca < max_epocas:
            Et = 0  # Error total de la época
            
            # Para cada patrón
            for p in range(P):
                # Obtener patrón de entrada y salida deseada
                x_p = X[p].reshape(-1, 1)  # Convertir a vector columna
                yd_p = Yd[p].reshape(-1, 1)  # Convertir a vector columna
                
                # ===== PROPAGACIÓN HACIA ADELANTE (FORWARD PASS) =====
                
                # Calcular entradas netas para la capa oculta
                Neth = np.dot(self.W_h, x_p)
                if self.bias:
                    Neth += self.Th
                
                # Calcular salidas de la capa oculta usando la función de activación seleccionada
                Yh = f_oculta(Neth)
                
                # Calcular entradas netas para la capa de salida
                Neto = np.dot(self.W_o, Yh)
                if self.bias:
                    Neto += self.To
                
                # Calcular salidas de la capa de salida usando la función de activación seleccionada
                Yo = f_salida(Neto)
                
                # ===== RETROPROPAGACIÓN DEL ERROR (BACKWARD PASS) =====
                
                # Calcular error para la capa de salida
                error_o = yd_p - Yo
                
                # Calcular deltas para la capa de salida
                if func_salida == 'sigmoide':
                    delta_o = error_o * Yo * (1 - Yo)
                else:
                    delta_o = error_o * f_salida_derivada(Neto)
                
                # Calcular deltas para la capa oculta
                error_h = np.dot(self.W_o.T, delta_o)
                if func_oculta == 'sigmoide':
                    delta_h = error_h * Yh * (1 - Yh)
                else:
                    delta_h = error_h * f_oculta_derivada(Neth)
                
                # Calcular cambios en los pesos
                dW_o = alfa * np.dot(delta_o, Yh.T)
                dW_h = alfa * np.dot(delta_h, x_p.T)
                
                # Aplicar momentum si está habilitado
                if self.config.get('momentum', False):
                    beta = float(self.config['beta'])
                    dW_o = dW_o + beta * dW_o_prev
                    dW_h = dW_h + beta * dW_h_prev
                    dW_o_prev = dW_o.copy()
                    dW_h_prev = dW_h.copy()
                
                # Actualizar pesos
                self.W_o += dW_o
                self.W_h += dW_h
                
                # Actualizar bias si está habilitado
                if self.bias:
                    dTo = alfa * delta_o
                    dTh = alfa * delta_h
                    
                    # Aplicar momentum al bias si está habilitado
                    if self.config.get('momentum', False):
                        dTo = dTo + beta * dTo_prev
                        dTh = dTh + beta * dTh_prev
                        dTo_prev = dTo.copy()
                        dTh_prev = dTh.copy()
                    
                    self.To += dTo
                    self.Th += dTh
                
                # Calcular error para este patrón (error cuadrático medio)
                Ep = 0.5 * np.sum(error_o**2)
                Et += Ep
            
            # Calcular error promedio de la época
            Et = Et / P
            errores.append(float(Et))
            
            # Incrementar contador de épocas
            epoca += 1
            
            # Imprimir progreso cada 100 épocas o en la última época
            if epoca % 100 == 0 or Et <= precision:
                print(f"Época {epoca}: Error = {float(Et)}")
        
        print(f"Entrenamiento completado en {epoca} épocas con error final {float(Et)}")
        return errores
    
    def predecir(self, X):
        """
        Realiza la clasificación de los patrones de entrada.
        
        Args:
            X: Matriz de patrones de entrada, cada fila es un patrón
            
        Returns:
            Matriz de salidas de la red, cada fila corresponde a un patrón de entrada
        """
        # Obtener funciones de activación
        func_oculta = self.config['funciones_activacion'][0].lower()
        func_salida = self.config['funciones_activacion'][1].lower()
        
        # Obtener las funciones de activación
        f_oculta, _ = self.funciones.get(func_oculta, [self.sigmoide, None])
        f_salida, _ = self.funciones.get(func_salida, [self.sigmoide, None])
        
        salidas = []
        for x in X:
            # Asegurarse de que x sea un vector columna
            x_col = x.reshape(-1, 1)
            
            # Propagación hacia adelante - capa oculta
            Neth = np.dot(self.W_h, x_col)
            if self.bias:
                Neth += self.Th
            
            # Calcular salidas de la capa oculta
            Yh = f_oculta(Neth)
            
            # Propagación hacia adelante - capa de salida
            Neto = np.dot(self.W_o, Yh)
            if self.bias:
                Neto += self.To
            
            # Calcular salidas de la capa de salida
            Yo = f_salida(Neto)
            
            salidas.append(Yo.flatten())
        
        return np.array(salidas)
    
    def guardar_pesos(self, archivo):
        """
        Guarda los pesos de la red en un archivo JSON.
        
        Args:
            archivo: Ruta del archivo donde se guardarán los pesos
        """
        datos = {
            'W_h': self.W_h.tolist(),
            'W_o': self.W_o.tolist(),
            'config': self.config
        }
        
        if self.bias:
            datos['Th'] = self.Th.tolist()
            datos['To'] = self.To.tolist()
        
        with open(archivo, 'w') as f:
            json.dump(datos, f, indent=4)
    
    def cargar_pesos(self, archivo):
        """
        Carga los pesos de la red desde un archivo JSON.
        
        Args:
            archivo: Ruta del archivo desde donde se cargarán los pesos
        """
        with open(archivo, 'r') as f:
            datos = json.load(f)
        
        self.W_h = np.array(datos['W_h'])
        self.W_o = np.array(datos['W_o'])
        
        if 'Th' in datos and 'To' in datos:
            self.Th = np.array(datos['Th'])
            self.To = np.array(datos['To'])
            self.bias = True
        else:
            self.bias = False
        
        # Actualizar configuración si está disponible
        if 'config' in datos:
            self.config = datos['config']