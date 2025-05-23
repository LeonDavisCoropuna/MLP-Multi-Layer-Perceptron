# MLP para Clasificación XOR en C++ by Leon Davis

Este proyecto implementa una red neuronal Multicapa Perceptrón (MLP) desde cero en C++ para resolver el problema de clasificación XOR. El proyecto utiliza CMake para la construcción y gestión de dependencias. 

Código fuente en: https://github.com/LeonDavisCoropuna/MLP-Multi-Layer-Perceptron.git

## Requisitos
- Compilador C++ compatible con C++17 (g++, clang, etc.)
- CMake (versión 3.10 o superior)
- Git

## Instalación y Ejecución

1. Clona el repositorio:
```bash
git git remote add origin https://github.com/LeonDavisCoropuna/MLP-Multi-Layer-Perceptron.git
cd MLP-Multi-Layer-Perceptron
```

2. Dale permisos de ejecución al script de configuración:
```bash
chmod +x run.sh
```

3. Finalmente ejecute el script para compilar y ejecutar el proyecto (por defecto se ejecuta ./xor):
```bash
./run.sh
```


## Estructura del Proyecto
```
.
├── images/                  # Resultados visuales
│   └── result.png           # Gráfico de resultados
├── models/                  # Implementaciones de modelos
│   ├── MLP.hpp              # Red neuronal multicapa
│   ├── perceptron.hpp       # Perceptrón básico
│   └── singleLayerPerceptron.hpp # Perceptrón de una capa
├── utils/                   # Utilidades auxiliares
│   ├── activations.hpp      # Funciones de activación
│   ├── load_dataset.hpp     # Carga de datos
│   ├── loss.hpp            # Funciones de pérdida
│   └── optimizer.hpp       # Algoritmos de optimización
├── CMakeLists.txt           # Configuración de CMake
├── main.cpp                # Punto de entrada principal
├── README_XOR.md           # Documentación adicional
├── run.sh                  # Script de compilación automática
├── training_outputs.txt    # Registro de salidas del entrenamiento
└── xor.cpp                 # Implementación específica para la actividad propuesta
```

## Implementación 

### 1. Clase `Perceptron`

Un `Perceptron` es la unidad más básica de tu red: recibe un vector de entradas, calcula una salida lineal y almacena gradientes para el entrenamiento.

```cpp
class Perceptron {
public:
  // Parámetros entrenables
  float bias;
  vector<float> weights;
  float learning_rate;

  // Estado temporal durante forward/backward
  float output;     
  float delta;      // ∂L/∂z local
  vector<float> grad_w;  // Acumulador de gradientes para cada peso

  // Constructor: inicializa pesos con distribución normal y bias pequeño
  Perceptron(int num_inputs, float _learning_rate);

  // Cálculo lineal z = w·x + b (sin activación)
  float forward(const vector<float>& inputs);

  // Getters/Setters de delta y salida
  void set_delta(float d);
  float get_delta() const;
  float get_output() const;

  // Serialización binaria de pesos+bias
  void serialize(ofstream& file) const;
  void deserialize(ifstream& file);

  // (Opcional) imprime pesos y bias a consola
  void print_weights();
};
```

**Flujo de datos:**

1. **Inicialización**
   * Los pesos `weights` se muestrean de una Normal(0, √(2/`num_inputs`)).
   * El `bias` se fija en 0.01.
2. **Forward**
   * Suma ponderada de entradas + bias.
   * El valor `z` se guarda en `output` para la fase de retropropagación.
3. **Backward**
   * El delta local `δ = ∂L/∂z` se calcula en la capa (no aquí).
   * `grad_w[i] = δ * input[i]`.
4. **Optimización**
   * `optimizer->update(weights, grad_w, bias, δ)` ajusta pesos/bias según tu regla (SGD, Adam, etc.).
---

### 2. Clase `SingleLayerPerceptron`

Representa una capa completa: un vector de `Perceptron` seguido de una **función de activación**.

```cpp
class SingleLayerPerceptron {
public:
  vector<Perceptron*> list_perceptrons;  // neuronas de la capa
  ActivationFunction* activation;        // p.ej. Sigmoid, ReLU, Softmax
  float learning_rate;
  Optimizer* optimizer;

  // Durante forward/backward:
  vector<float> inputs_layer;   // entradas X de la capa
  vector<float> outputs_layer;  // activaciones A

  SingleLayerPerceptron(int num_neurons,
                       int num_inputs,
                       ActivationFunction* _activation,
                       float _learning_rate,
                       Optimizer* _optimizer);

  // Propagación hacia adelante para todo el batch (un vector de features)
  vector<float> forward(vector<float> batch_inputs);

  // Cálculo de deltas en capa de salida (cross-entropy+softmax o MSE)
  void backward_output_layer(const vector<float>& targets);

  // Cálculo de deltas en capas ocultas
  void backward_hidden_layer(SingleLayerPerceptron* next_layer);

  // Actualizar pesos de cada perceptrón con el optimizador
  void update_weights();

  // Poner a cero todos los deltas (antes de un nuevo paso)
  void zero_grad();
};
```
**Pasos clave en una capa:**

1. **forward()**

   * Guarda `batch_inputs` en `inputs_layer`.
   * Para cada `Perceptron`:

     * Calcula `z = w·x + b`.
     * Aplica la activación:

       * **Softmax** en vector completo.
       * **Otras** (Sigmoid, ReLU…) individual.
   * Almacena el vector `outputs_layer` para la siguiente capa.

2. **backward\_output\_layer()**

   * Para la capa final, calcula `δᵢ = outputᵢ – targetᵢ` (softmax+cross-entropy)
     o `δᵢ = (outputᵢ – targetᵢ)·σ′(zᵢ)` (MSE+sigmoid/lineal).

3. **backward\_hidden\_layer()**

   * Recolecta los deltas de la **siguiente** capa (`next_layer`).
   * Para cada neurona i en la capa actual:
    `δ_i = ( Σ_j w_j_i * δ⁽next⁾_j ) * activation′(z_i)`

4. **update\_weights()**

   * Para cada neurona, construye `grad_w[i] = δ * inputs_layer[i]`.
   * Llama a `optimizer->update(...)` para ajustar `weights` y `bias`.

## 3. Clase `MLP` (Multi-Layer Perceptron)

Encapsula varias `SingleLayerPerceptron`, define la topología, controla entrenamiento, evaluación y serialización.

```cpp
class MLP {
private:
  float learning_rate;
  vector<int> num_layers;                           // neuronas por capa
  int num_inputs;                                   // dimensión de entrada
  vector<SingleLayerPerceptron*> layers;            // capas
  vector<vector<float>> output_layers;              // activaciones intermedias
  vector<ActivationFunction*> activations;          // activaciones por capa
  Loss* loss_function;                              // MSE, BCELoss, etc.
  Optimizer* optimizer;                             // SGD, Adam…
  int last_output_size = -1;

public:
  // Constructor “estático”: define toda la topología y funciones de activación
  MLP(float _learning_rate,
      vector<int> _num_layers,
      vector<ActivationFunction*> _activations,
      Loss* _loss_function);

  // Constructor dinámico: añadir capas manualmente
  MLP(float _learning_rate, Optimizer* _optimizer);
  void add_input_layer(int input_size, int num_neurons, ActivationFunction*);
  void add_layer(int num_neurons, ActivationFunction*);

  void set_loss(Loss* _loss_function);

  // Propagación completa
  vector<float> forward(vector<float> batch_inputs);

  // Devuelve la etiqueta más probable
  int predict(const vector<float>& input);

  // Entrenamiento con retropropagación
  void train(int num_epochs,
             const vector<vector<float>>& X,
             const vector<float>& Y);

  // Cálculo de accuracy sobre test set
  float evaluate(const vector<vector<float>>& X_test,
                 const vector<float>& Y_test);

  // Guardar y cargar pesos (serialización)
  void save_model_weights(const string& filename);
  void load_model_weights(const string& filename);

  ~MLP();  // libera memoria de capas y activaciones
};
```

### Flujo de entrenamiento (`train()`)

1. **Epoch loop**
2. **Batch loop** (aquí batch de tamaño 1)

   * `forward(X[i])` → predicción
   * Construir `target_vec` (uno-hot o escalar para BCELoss).
   * `loss += loss_function->compute(outputs, target_vec)`
   * `backward_output_layer(target_vec)` en última capa.
   * Iterar hacia atrás `backward_hidden_layer(...)`.
   * `update_weights()` en cada capa.
3. Imprimir **loss** promedio y **accuracy**.

---

### Resumen de responsabilidades

| Componente                | Rol principal                                                                   |
| ------------------------- | ------------------------------------------------------------------------------- |
| **Perceptron**            | Cálculo lineal, almacenamiento de pesos y delta.                                |
| **SingleLayerPerceptron** | Conjunto de perceptrones + activación + backprop.                               |
| **MLP**                   | Orquestación de capas, estrategia de entrenamiento, evaluación y serialización. |

## Actividad

Completar la siguiente lista de ejercicios:

1. **Arquitectura MLP para XOR**  
   Implementar una red con la siguiente estructura:  
   - 2 neuronas de entrada
   - 2 neuronas en capa oculta
   - 1 neurona de salida
   
   ![Arquitectura MLP](images/xor_tanh_sigmoid.png)  

   - La arquitectura crea un instancia de MLP y se le envía el learning rate junto con el optimizador, luego se añaden las capas según lo indicado. El primer parametro representa el número de entradas a la neurona, en este caso 2 (pares de {0,0}, {0,1}, etc). Luego se añaden las capas con una función de activación y se escoje una función de pérdida, para este ejemplo se uso BCE (Binary cross-entropy).

2. **Entrenamiento del modelo XOR**  
   - Implementar la función de propagación hacia adelante (forward pass)
   - Implementar el algoritmo de backpropagation para ajuste de pesos
   - Entrenar el modelo hasta convergencia
   
   ![Arquitectura MLP](images/xor_tanh_sigmoid_console.png)  

3. **Pruebas con compuertas lógicas**  
   - Entrenamiento de AND
    ![Arquitectura MLP](images/and_tanh_sigmoid_console.png)  
   - Entrenamiento de OR
    ![Arquitectura MLP](images/or_tanh_sigmoid_console.png)  

   - En ambos casos se alcanza un accuracy del 100%. Tanto la implementación anterior como esta son efectivas para clasificar un problema linealmente separable (AND y OR), la diferencia esta cuando el problema es más complejo; el perceptron implementado anteriormente no puede clasificar un XOR mientras que un MLP si puede.
4. **Funciones de Activación**  
   En este caso se esta optando por una capa de 4 neuronas debido a que en algunos casos con solo 2 neuronas no se llega a una convergencia ideal o se requieren demasiados epochs. Se usó un learning rate de 0.05 y MSE (Mean Square Error) como función de pérdida.
   - **Sigmoide**: Implementación clásica para problemas binarios, la menos eficiente en este caso, pero cambiando de parametros mejora significatvamente.

   ![Comparación Activaciones](images/xor_sigmoid_sigmoid.png)  
   ![Comparación Activaciones](images/xor_sigmoid_sigmoid_console.png)  

   - **Tanh**: Versión centrada en cero del sigmoide, basado en estos parametros es la función que menos epochs requiere.

   ![Comparación Activaciones](images/xor_tanh_tanh.png)  
   ![Comparación Activaciones](images/xor_tanh_tanh_console.png)  

   - **ReLU**: Muy útil en la mayoría de los casos sobretodo cuando hay muchas capas conectadas y/o con muchas neuronas por capa, sin embargo, si se opta por una configuración de solo 2 neuronas en la capa oculta no se llega a una convergencia ideal.
   
   ![Comparación Activaciones](images/xor_relu_relu.png)  
   ![Comparación Activaciones](images/xor_relu_relu_console.png)  

## Curvas de aprendizaje (loss y accuracy)
