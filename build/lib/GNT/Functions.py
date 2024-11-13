import random
import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt

@dataclass
class GAStats:
    """Statistics for tracking GA performance"""
    generation: int
    best_fitness: float
    average_fitness: float
    best_individual: List[float]
    diversity: float

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        variable_limits: List[Tuple[float, float]],
        generations: int,
        crossover_rate: float,
        mutation_rate: float,
        elitism_size: int,
        objective_function: Callable,
        selection_method: str = "tournament",
        crossover_function: Optional[Callable] = None,
        mutation_function: Optional[Callable] = None,
        tournament_size: int = 3,
        maximize: bool = False,
        convergence_threshold: float = 1e-6,
        max_stagnant_generations: int = 20,
        seed: Optional[int] = None
    ):
        """
        Initialize the genetic algorithm with the specified parameters.
        
        Parameters:
        -----------
        population_size : int
            Size of the population
        variable_limits : List[Tuple[float, float]]
            List of (min, max) tuples specifying limits for each variable
        generations : int
            Maximum number of generations to run
        crossover_rate : float
            Probability of crossover occurring (0-1)
        mutation_rate : float
            Probability of mutation occurring (0-1)
        elitism_size : int
            Number of best individuals to preserve in each generation
        objective_function : Callable
            Function to optimize
        selection_method : str
            Method of selection ('tournament' or 'roulette')
        crossover_function : Optional[Callable]
            Custom crossover function
        mutation_function : Optional[Callable]
            Custom mutation function
        tournament_size : int
            Size of tournament for tournament selection
        maximize : bool
            Whether to maximize (True) or minimize (False) the objective
        convergence_threshold : float
            Minimum change in best fitness to consider convergence
        max_stagnant_generations : int
            Maximum generations without improvement before stopping
        seed : Optional[int]
            Random seed for reproducibility
        """
        # Input validation
        self._validate_inputs(population_size, crossover_rate, mutation_rate, 
                            elitism_size, tournament_size)
        
        # Initialize attributes
        self.population_size = population_size
        self.variable_limits = variable_limits
        self.num_variables = len(variable_limits)
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        self.objective_function = objective_function
        self.selection_method = selection_method
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.tournament_size = tournament_size
        self.maximize = maximize
        self.convergence_threshold = convergence_threshold
        self.max_stagnant_generations = max_stagnant_generations
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize population and statistics
        self.population = self.create_population()
        self.stats_history: List[GAStats] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _validate_inputs(population_size, crossover_rate, mutation_rate, 
                        elitism_size, tournament_size):
        """Validate input parameters"""
        if population_size < 2:
            raise ValueError("Population size must be at least 2")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if elitism_size >= population_size:
            raise ValueError("Elitism size must be less than population size")
        if tournament_size < 2:
            raise ValueError("Tournament size must be at least 2")

    def create_individual(self) -> List[float]:
        """Creates an individual within specified variable limits"""
        return [random.uniform(min_val, max_val) 
                for min_val, max_val in self.variable_limits]

    def create_population(self) -> List[List[float]]:
        """Generates initial population of individuals"""
        return [self.create_individual() for _ in range(self.population_size)]

    def calculate_population_fitness(self) -> List[float]:
        """Calculates fitness for entire population"""
        try:
            return [self.objective_function(*individual) for individual in self.population]
        except Exception as e:
            self.logger.error(f"Error calculating fitness: {str(e)}")
            raise

    def roulette_selection(self, population: List[List[float]], 
                          scores: List[float]) -> List[float]:
        """Selects individual using roulette wheel selection"""
        if self.maximize:
            # Shift scores to be positive for maximization
            min_score = min(scores)
            adjusted_scores = [s - min_score + 1e-10 for s in scores]
        else:
            # Invert scores for minimization
            max_score = max(scores)
            adjusted_scores = [max_score - s + 1e-10 for s in scores]
        
        total_fitness = sum(adjusted_scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for individual, score in zip(population, adjusted_scores):
            current += score
            if current > pick:
                return individual
        
        return population[-1]  # Fallback

    def tournament_selection(self, population: List[List[float]], 
                           scores: List[float]) -> List[float]:
        """Selects individual through tournament selection"""
        tournament_indices = random.sample(range(self.population_size), 
                                        self.tournament_size)
        tournament_scores = [scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[
            tournament_scores.index(max(tournament_scores) if self.maximize 
                                 else min(tournament_scores))
        ]
        return population[winner_idx]

    def crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Performs crossover between two parents"""
        if random.random() < self.crossover_rate:
            if self.crossover_function:
                return self.crossover_function(parent1, parent2)
            
            # Default: Simulated Binary Crossover (SBX)
            eta = 20  # Distribution index
            child = []
            for i in range(self.num_variables):
                if random.random() < 0.5:
                    beta = 2. * random.random()
                else:
                    beta = 1. / (2. * (1. - random.random()))
                beta = beta ** (1. / (eta + 1.))
                
                child.append(0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i]))
                
                # Ensure within bounds
                min_val, max_val = self.variable_limits[i]
                child[i] = max(min_val, min(max_val, child[i]))
            
            return child
        
        return parent1 if random.random() < 0.5 else parent2

    def mutation(self, individual: List[float]) -> List[float]:
        """Mutates an individual using polynomial mutation"""
        if self.mutation_function:
            return self.mutation_function(individual)
        
        eta = 20  # Distribution index
        mutated = individual.copy()
        
        for i in range(self.num_variables):
            if random.random() < self.mutation_rate:
                min_val, max_val = self.variable_limits[i]
                delta1 = (mutated[i] - min_val) / (max_val - min_val)
                delta2 = (max_val - mutated[i]) / (max_val - min_val)
                
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                mutated[i] = mutated[i] + delta_q * (max_val - min_val)
                mutated[i] = max(min_val, min(max_val, mutated[i]))
        
        return mutated

    def select_individual(self, population: List[List[float]], 
                         scores: List[float]) -> List[float]:
        """Selects an individual based on selection method"""
        if self.selection_method == "tournament":
            return self.tournament_selection(population, scores)
        elif self.selection_method == "roulette":
            return self.roulette_selection(population, scores)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def calculate_diversity(self, population: List[List[float]]) -> float:
        """Calculate population diversity using average pairwise distance"""
        if len(population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = sum((a - b) ** 2 for a, b in 
                             zip(population[i], population[j])) ** 0.5
                distances.append(distance)
        
        return sum(distances) / len(distances)

    def run(self) -> Tuple[List[List[float]], List[GAStats]]:
        """
        Executes the genetic algorithm.
        
        Returns:
        --------
        Tuple[List[List[float]], List[GAStats]]
            Final population and history of statistics
        """
        best_fitness = float('-inf') if self.maximize else float('inf')
        stagnant_generations = 0
        
        for generation in range(self.generations):
            try:
                # Calculate fitness scores
                scores = self.calculate_population_fitness()
                
                # Sort population by fitness
                population_scores = list(zip(self.population, scores))
                population_scores.sort(key=lambda x: x[1], reverse=self.maximize)
                sorted_population = [ind for ind, _ in population_scores]
                
                # Update statistics
                current_best_fitness = population_scores[0][1]
                print(population_scores[0][0])
                avg_fitness = sum(scores) / len(scores)
                diversity = self.calculate_diversity(self.population)
                
                self.stats_history.append(GAStats(
                    generation=generation,
                    best_fitness=current_best_fitness,
                    average_fitness=avg_fitness,
                    best_individual=sorted_population[0].copy(),
                    diversity=diversity
                ))
                
                # Check for improvement
                if ((self.maximize and current_best_fitness > best_fitness + self.convergence_threshold) or
                    (not self.maximize and current_best_fitness < best_fitness - self.convergence_threshold)):
                    best_fitness = current_best_fitness
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1
                
                # Check termination conditions
                if stagnant_generations >= self.max_stagnant_generations:
                    self.logger.info(f"Converged after {generation + 1} generations")
                    break
                
                # Create next generation
                next_population = sorted_population[:self.elitism_size]
                
                while len(next_population) < self.population_size:
                    parent1 = self.select_individual(self.population, scores)
                    parent2 = self.select_individual(self.population, scores)
                    
                    while parent1 == parent2:
                        parent2 = self.select_individual(self.population, scores)
                    
                    child = self.crossover(parent1, parent2)
                    child = self.mutation(child)
                    next_population.append(child)
                
                self.population = next_population
                
                # Log progress
                self.logger.info(
                    f"Generation {generation + 1}: "
                    f"Best fitness = {current_best_fitness:.6f}, "
                    f"Avg fitness = {avg_fitness:.6f}, "
                    f"Diversity = {diversity:.6f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in generation {generation + 1}: {str(e)}")
                raise
        
        return self.population, self.stats_history
    
    def plot_stats(self):
        """Genera y muestra las gráficas de rendimiento del GA"""
        generations = [stat.generation for stat in self.stats_history]
        best_fitnesses = [stat.best_fitness for stat in self.stats_history]
        average_fitnesses = [stat.average_fitness for stat in self.stats_history]
        diversities = [stat.diversity for stat in self.stats_history]
        best_individuals = [stat.best_individual for stat in self.stats_history]

        # Gráfica de Best Fitness y Average Fitness
        plt.figure(figsize=(10, 5))
        plt.plot(generations, best_fitnesses, label='Best Fitness')
        plt.plot(generations, average_fitnesses, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness over Generations')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Gráfica de Diversity
        plt.figure(figsize=(10, 5))
        plt.plot(generations, diversities, label='Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.title('Diversity over Generations')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Gráficas de Best Individual Variables
        num_variables = len(self.variable_limits)
        for var_index in range(num_variables):
            variable_values = [ind[var_index] for ind in best_individuals]
            plt.figure(figsize=(10, 5))
            plt.plot(generations, variable_values, label=f'Variable {var_index + 1}')
            plt.xlabel('Generation')
            plt.ylabel(f'Variable {var_index + 1} Value')
            plt.title(f'Best Individual Variable {var_index + 1} over Generations')
            plt.legend()
            plt.grid(True)
            plt.show()