# GNT

[![Licencia MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Versión de Python](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![Issues](https://img.shields.io/github/issues/CAMILORR23/GNTLibrary.svg)](https://github.com/CAMILORR23/GNTLibrary/issues)
[![Estrellas en GitHub](https://img.shields.io/github/stars/CAMILORR23/GNTLibrary.svg?style=social&label=Stars)](https://github.com/CAMILORR23/GNTLibrary)

GNT es una **biblioteca de Python** diseñada para implementar **algoritmos genéticos** en proyectos de optimización. Es ideal para resolver problemas de **maximización** y **minimización** de funciones mediante procesos evolutivos. Esta biblioteca facilita el desarrollo de soluciones basadas en algoritmos genéticos de forma intuitiva y flexible.

## Tabla de Contenidos

- [Características](#características)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Ejemplo de Código](#ejemplo-de-código)
  - [Parámetros Principales](#parámetros-principales)
- [Ejemplos](#ejemplos)
- [Contribuir](#contribuir)
  - [Reportar Problemas](#reportar-problemas)
- [Licencia](#licencia)
- [Contacto](#contacto)

## Características

- **Optimización de Funciones**: Resuelve problemas de maximización y minimización de manera eficiente.
- **Flexible y Personalizable**: Ajusta parámetros clave como tamaño de población, tasa de mutación, tasa de cruce, entre otros.
- **Fácil de Usar**: Interfaces intuitivas que simplifican la implementación de algoritmos genéticos.
- **Extensible**: Permite agregar nuevas funcionalidades y personalizaciones según las necesidades del usuario.
- **Documentación Completa**: Guías y ejemplos que facilitan el inicio rápido y el uso avanzado.

## Instalación

Puedes instalar la biblioteca directamente desde GitHub utilizando `pip`. Ejecuta el siguiente comando en tu terminal:

```bash
pip install -i https://test.pypi.org/simple/ GNT
```

También puedes clonar el repositorio e instalar localmente:

```bash
git clone https://github.com/CAMILORR23/GNTLibrary.git
cd GNTLibrary
pip install .
```

## Uso

GNT proporciona una interfaz sencilla para definir y ejecutar algoritmos genéticos. A continuación, se muestra cómo comenzar a utilizar la biblioteca.

### Ejemplo de Código

```python
from GNT import GeneticAlgorithm  # Importación de la clase GA

# ---------------------------------------------------------------------------
# Optimización de 1 Variable: Minimización
# ---------------------------------------------------------------------------

# Definición de la función objetivo para optimización de 1 variable (Minimización)
def f1_min(x):
    """
    Función a minimizar: (x - 3)^2
    
    Parámetros:
    - x (float): Valor de la variable x.
    
    Retorna:
    - float: Valor de la función evaluada en x.
    
    Descripción:
    - Esta función tiene un mínimo global en x = 3 con un valor de 0.
    """
    return (x - 3) ** 2  # Objetivo: Minimizar

# Límites de la variable para f1_min (1 variable)
v1_min = [(-10, 10)]  # Límites para x

# Creación y configuración del GA para la función f1_min (1 variable, Minimización)
ga_1_min = GeneticAlgorithm(
    population_size=50,             # Tamaño de la población
    variable_limits=v1_min,          # Límites para x
    generations=100,                 # Número máximo de generaciones
    crossover_rate=0.7,              # Tasa de cruce
    mutation_rate=0.1,               # Tasa de mutación
    elitism_size=1,                  # Número de individuos de élite
    objective_function=f1_min,        # Función objetivo a minimizar
    selection_method="tournament",    # Método de selección (torneo)
    maximize=False,                   # Objetivo de minimización
    convergence_threshold=1e-6,       # Umbral de convergencia
    max_stagnant_generations=10       # Máximo de generaciones sin mejora
)

# Ejecución del algoritmo genético para f1_min
final_population_1_min, stats_history_1_min = ga_1_min.run()

# Identificación de la mejor solución en la población final para f1_min
best_solution_1_min = min(final_population_1_min, key=lambda ind: f1_min(*ind))  # Para minimizar f1_min
best_fitness_1_min = f1_min(*best_solution_1_min)

# Presentación de los resultados para f1_min
print("=== Optimización de 1 Variable: Minimización ===")
print(f"Mejor solución encontrada: {best_solution_1_min}")
print(f"Mejor fitness: {best_fitness_1_min}\n")

# Visualización de las estadísticas del GA para f1_min
ga_1_min.plot_stats()
```

### Parámetros Principales

- **population_size** (`int`): Número de individuos en la población.	
- **variable_limits** (`List[Tuple[float, float]]`): Límites para cada variable, definidos como una lista de tuplas (mínimo, máximo).
- **generations** (`int`): Número de generaciones a evolucionar.	
- **crossover_rate** (`float`): Probabilidad de cruzamiento entre individuos.
- **mutation_rate** (`float`): Probabilidad de mutación de genes.
- **elitism_size** (`int`): Número de los mejores individuos que se conservan en cada generación.
- **objective_function** (`Callable`): Función que evalúa la aptitud de cada individuo.
- **selection_method** (`str`): Método de selección de padres. Puede ser 'tournament' o 'roulette'.
- **maximize** (`bool`): Tipo de optimización, puede ser True para maximización o False para minimización.
- **convergence_threshold** (`float`): Umbral mínimo de cambio en el mejor fitness para considerar la convergencia.	
- **max_stagnant_generations** (`int`): Máximo de generaciones sin mejora en el mejor fitness antes de detener el algoritmo.	
- **seed** (`int, opcional`): Semilla para la generación de números aleatorios, permitiendo reproducibilidad de resultados.	

## Ejemplos

### Maximización de una Función Matemática

Supongamos que queremos encontrar el valor máximo de la función \( f(x,y,z) = -(x^2 + y^2 + z^2) + 50 \) en el rango de (-5, 5), (-10, 10), (-3, 3) respectivamente para cada variable.

```python
from GNT import GeneticAlgorithm  # Importación de la clase GA

# ---------------------------------------------------------------------------
# Optimización de 3 Variables: Maximización
# ---------------------------------------------------------------------------

# Definición de la función objetivo para optimización de 3 variables (Maximización)
def f3_max(x, y, z):
    """
    Función a maximizar: -(x^2 + y^2 + z^2) + 50.

    Parámetros:
    - x (float): Valor de la variable x.
    - y (float): Valor de la variable y.
    - z (float): Valor de la variable z.

    Retorna:
    - float: Valor de la función evaluada en (x, y, z).
    """
    return -(x**2 + y**2 + z**2) + 50  # Objetivo: Maximizar

# Límites de las variables para f3_max (3 variables)
v3_max = [(-7, 7), (-7, 7), (-7, 7)]  # Límites para x, y y z

# Creación y configuración del GA para la función f3_max (3 variables, Maximización)
ga_3_max = GeneticAlgorithm(
    population_size=120,             # Tamaño de la población
    variable_limits=v3_max,          # Límites para x, y y z
    generations=250,                 # Número máximo de generaciones
    crossover_rate=0.85,             # Tasa de cruce
    mutation_rate=0.08,              # Tasa de mutación
    elitism_size=4,                  # Número de individuos de élite
    objective_function=f3_max,        # Función objetivo a maximizar
    selection_method="roulette",      # Método de selección (ruleta)
    maximize=True,                    # Objetivo de maximización
    convergence_threshold=1e-5,       # Umbral de convergencia
    max_stagnant_generations=25        # Máximo de generaciones sin mejora
)

# Ejecución del algoritmo genético para f3_max
final_population_3_max, stats_history_3_max = ga_3_max.run()

# Identificación de la mejor solución en la población final para f3_max
best_solution_3_max = max(final_population_3_max, key=lambda ind: f3_max(*ind))  # Para maximizar f3_max
best_fitness_3_max = f3_max(*best_solution_3_max)

# Presentación de los resultados para f3_max
print("=== Optimización de 3 Variables: Maximización ===")
print(f"Mejor solución encontrada: {best_solution_3_max}")
print(f"Mejor fitness: {best_fitness_3_max}\n")

# Visualización de las estadísticas del GA para f3_max
ga_3_max.plot_stats()

```

### Resolución de un Problema de Minimización

Buscar el valor mínimo de la función \( f(x,y) = abs(x) + abs(y) \) en el rango de (-20, 20), (-20, 20) respectivamente.

```python
from GNT import GeneticAlgorithm  # Importación de la clase GA

# ---------------------------------------------------------------------------
# Optimización de 2 Variables: Minimización
# ---------------------------------------------------------------------------

# Definición de la función objetivo para optimización de 2 variables (Minimización)
def f2_min(x, y):
    """
    Función a minimizar: suma de los valores absolutos de x e y.

    Parámetros:
    - x (float): Valor de la variable x.
    - y (float): Valor de la variable y.

    Retorna:
    - float: Valor de la función evaluada en (x, y).
    """
    return abs(x) + abs(y)  # Objetivo: Minimizar

# Límites de las variables para f2_min (2 variables)
v2_min = [(-20, 20), (-20, 20)]  # Límites para x e y

# Creación y configuración del GA para la función f2_min (2 variables, Minimización)
ga_2_min = GeneticAlgorithm(
    population_size=80,              # Tamaño de la población
    variable_limits=v2_min,          # Límites para x e y
    generations=150,                 # Número máximo de generaciones
    crossover_rate=0.75,             # Tasa de cruce
    mutation_rate=0.07,              # Tasa de mutación
    elitism_size=2,                   # Número de individuos de élite
    objective_function=f2_min,        # Función objetivo a minimizar
    selection_method="tournament",     # Método de selección (torneo)
    maximize=False,                    # Objetivo de minimización
    convergence_threshold=1e-4,        # Umbral de convergencia
    max_stagnant_generations=15         # Máximo de generaciones sin mejora
)

# Ejecución del algoritmo genético para f2_min
final_population_2_min, stats_history_2_min = ga_2_min.run()

# Identificación de la mejor solución en la población final para f2_min
best_solution_2_min = min(final_population_2_min, key=lambda ind: f2_min(*ind))  # Para minimizar f2_min
best_fitness_2_min = f2_min(*best_solution_2_min)

# Presentación de los resultados para f2_min
print("=== Optimización de 2 Variables: Minimización ===")
print(f"Mejor solución encontrada: {best_solution_2_min}")
print(f"Mejor fitness: {best_fitness_2_min}\n")

# Visualización de las estadísticas del GA para f2_min
ga_2_min.plot_stats()
```

## Contribuir

¡Las contribuciones son bienvenidas! Si deseas mejorar GNT, sigue estos pasos para contribuir de manera efectiva.

### Reportar Problemas

Si encuentras algún error o tienes una sugerencia de mejora, por favor, abre un [issue](https://github.com/CAMILORR23/GNTLibrary/issues) en el repositorio de GitHub. Asegúrate de proporcionar información detallada para que podamos reproducir y resolver el problema de manera eficiente.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](https://github.com/CAMILORR23/GNTLibrary/blob/main/LICENSE) para obtener más información.

## Contacto

Si tienes alguna pregunta o comentario, puedes contactarme a través de:

- **Correo Electrónico**: carodriguezreyes@ucundinamarca.edu.co
- **GitHub**: [CAMILORR23](https://github.com/CAMILORR23)

¡Gracias por utilizar GNT! Esperamos que esta biblioteca te sea de gran ayuda en tus proyectos de optimización con algoritmos genéticos.