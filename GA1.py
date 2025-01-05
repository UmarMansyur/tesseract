import random
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool
from functools import lru_cache
import json
import logging
from datetime import datetime
from pathlib import Path
from Tesseract import Tesseract
from Rule import Rule

class GeneticAlgorithm:
    def __init__(self, 
                 input_paths: List[str], 
                 output_paths: List[str], 
                 ground_truth_paths: Optional[List[str]] = None,
                 population_size: int = 80,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elite_size: int = 2,
                 n_rules: int = 9,
                 sequence_length: int = 9,
                 early_stopping_patience: int = 15,
                 n_islands: int = 3,
                 migration_interval: int = 10):
        
        # Setup logging
        self.setup_logging()
        
        # Basic parameters
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.ground_truth_paths = ground_truth_paths
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.n_rules = n_rules
        self.sequence_length = sequence_length
        self.early_stopping_patience = early_stopping_patience
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        
        # Load images and ground truths
        self.load_data()
        
        # Initialize cache and best sequence
        self.fitness_cache = {}
        self.best_sequence = None
        self.best_fitness = float('-inf')
        
        logging.info("Genetic Algorithm initialized with parameters:")
        logging.info(f"Population size: {population_size}")
        logging.info(f"Generations: {generations}")
        logging.info(f"Mutation rate: {mutation_rate}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/ga_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self):
        """Load all input images and ground truths"""
        self.current_images = []
        for path in self.input_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not load image from {path}")
            self.current_images.append(img)
        
        self.ground_truths = []
        if self.ground_truth_paths:
            for path in self.ground_truth_paths:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.ground_truths.append(f.read())
                except FileNotFoundError:
                    if path != 'dummy':
                        raise
    
    def adaptive_mutation_rate(self, generation: int) -> float:
        """Calculate adaptive mutation rate based on generation"""
        start_rate = self.mutation_rate
        end_rate = self.mutation_rate / 5
        return start_rate - (start_rate - end_rate) * (generation / self.generations)
    
    def adaptive_population_size(self, generation: int) -> int:
        """Calculate adaptive population size based on generation"""
        start_size = self.population_size
        end_size = self.population_size // 2
        return int(start_size - (start_size - end_size) * (generation / self.generations))
    
    def create_individual(self) -> List[int]:
        """Create a valid individual"""
        return random.sample(range(1, self.n_rules + 1), self.sequence_length)
    
    def create_population(self, size: Optional[int] = None) -> List[List[int]]:
        """Create initial population with unique individuals"""
        if size is None:
            size = self.population_size
            
        unique_individuals = set()
        population = []
        max_attempts = size * 3
        attempts = 0
        
        while len(population) < size and attempts < max_attempts:
            new_individual = self.create_individual()
            new_tuple = tuple(new_individual)
            
            if new_tuple not in unique_individuals:
                unique_individuals.add(new_tuple)
                population.append(new_individual)
            
            attempts += 1
        
        while len(population) < size:
            population.append(self.create_individual())
            
        return population
    
    @lru_cache(maxsize=1000)
    def calculate_fitness(self, individual_tuple: tuple) -> float:
        """Calculate fitness with caching"""
        individual = list(individual_tuple)
        total_fitness = 0
        
        for idx, current_image in enumerate(self.current_images):
            current_image = current_image.copy()
            
            for rule_num in individual:
                rule_processor = Rule(None, None)
                current_image = rule_processor.run(rule_num, current_image)
            
            tesseract = Tesseract(None, self.output_paths[idx])
            if self.ground_truths:
                tesseract.set_ground_truth(self.ground_truths[idx])
            tesseract.tesseract(current_image)
            
            wer, cer = tesseract.get_test()
            error_rate = 0.3 * wer + 0.7 * cer
            total_fitness += np.exp(-error_rate)
        
        return total_fitness / len(self.current_images)

    def fitness(self, individual: List[int]) -> float:
        """Wrapper for cached fitness calculation"""
        return self.calculate_fitness(tuple(individual))

    def evaluate_population(self, population: List[List[int]]) -> List[float]:
        """Parallel fitness evaluation"""
        with Pool() as pool:
            return pool.map(self.fitness, population)
    
    def calculate_diversity(self, population: List[List[int]]) -> float:
        """Calculate population diversity"""
        return len(set(tuple(ind) for ind in population)) / len(population)
    
    def fitness_sharing(self, population: List[List[int]], 
                       fitness_scores: List[float], 
                       sigma: float = 0.1) -> List[float]:
        """Implement fitness sharing"""
        shared_fitness = []
        for i, ind1 in enumerate(population):
            niche_count = 0
            for ind2 in population:
                distance = sum(abs(a-b) for a, b in zip(ind1, ind2)) / len(ind1)
                if distance < sigma:
                    niche_count += 1
            shared_fitness.append(fitness_scores[i] / niche_count)
        return shared_fitness
    
    def tournament_selection(self, population: List[List[int]], 
                           fitness_scores: List[float]) -> List[int]:
        """Tournament selection"""
        tournament_idx = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_idx]
        return population[tournament_idx[np.argmax(tournament_fitness)]]
    
    def uniform_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Uniform crossover implementation"""
        child1, child2 = [], []
        for p1, p2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(p1)
                child2.append(p2)
            else:
                child1.append(p2)
                child2.append(p1)
                
        if len(set(child1)) != self.sequence_length or len(set(child2)) != self.sequence_length:
            return parent1[:], parent2[:]
            
        return child1, child2
    
    def mutate(self, individual: List[int], mutation_rate: float) -> List[int]:
        """Enhanced mutation operator"""
        if random.random() < mutation_rate:
            pos1, pos2 = random.sample(range(len(individual)), 2)
            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        return individual
    
    def migrate_individuals(self, islands: List[List[List[int]]]):
        """Implement migration between islands"""
        n_migrants = max(1, self.population_size // 20)
        
        for i in range(self.n_islands):
            next_island = (i + 1) % self.n_islands
            
            # Select best individuals to migrate
            migrants = sorted(islands[i], 
                            key=lambda x: self.fitness(x), 
                            reverse=True)[:n_migrants]
            
            # Replace worst individuals in next island
            islands[next_island].sort(key=lambda x: self.fitness(x))
            islands[next_island][:n_migrants] = migrants
    
    def evolve_island(self, population: List[List[int]], 
                      generation: int) -> List[List[int]]:
        """Evolve a single island population"""
        fitness_scores = self.evaluate_population(population)
        shared_fitness = self.fitness_sharing(population, fitness_scores)
        
        # Elitism
        elite = sorted(zip(population, shared_fitness), 
                      key=lambda x: x[1], 
                      reverse=True)[:self.elite_size]
        new_population = [ind for ind, _ in elite]
        
        # Calculate adaptive parameters
        mutation_rate = self.adaptive_mutation_rate(generation)
        
        # Create rest of new population
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population, shared_fitness)
            parent2 = self.tournament_selection(population, shared_fitness)
            
            child1, child2 = self.uniform_crossover(parent1, parent2)
            
            child1 = self.mutate(child1, mutation_rate)
            child2 = self.mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def evolve(self) -> Tuple[List[int], Dict]:
        """Main evolution process using island model"""
        # Initialize islands
        islands = [self.create_population() for _ in range(self.n_islands)]
        
        stagnation_counter = 0
        
        for generation in range(self.generations):
            logging.info(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evolve each island
            for i in range(self.n_islands):
                islands[i] = self.evolve_island(islands[i], generation)
            
            # Periodic migration
            if generation % self.migration_interval == 0:
                self.migrate_individuals(islands)
            
            # Find best solution across all islands
            all_individuals = [ind for island in islands for ind in island]
            fitness_scores = self.evaluate_population(all_individuals)
            
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_sequence = all_individuals[current_best_idx]
            
            logging.info(f"Current best fitness: {current_best_fitness:.4f}")
            
            # Update best solution
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_sequence = current_best_sequence
                stagnation_counter = 0
                logging.info(f"New best solution found: {self.best_sequence}")
            else:
                stagnation_counter += 1
            
            # Early stopping
            if stagnation_counter >= self.early_stopping_patience:
                logging.info(f"Early stopping at generation {generation + 1}")
                break
            
            # Calculate and log diversity
            diversity = np.mean([self.calculate_diversity(island) for island in islands])
            logging.info(f"Population diversity: {diversity:.4f}")
        
        # Calculate final metrics
        final_metrics = self.calculate_final_metrics()
        
        return self.best_sequence, final_metrics
    
    def calculate_final_metrics(self) -> Dict:
        """Calculate final performance metrics"""
        if self.best_sequence is None:
            raise ValueError("No solution found yet")
            
        current_image = None
        for i, rule in enumerate(self.best_sequence):
            if i == 0:
                rule_processor = Rule(self.input_paths[0], f"temp_final_{i}.jpg")
            else:
                rule_processor = Rule(f"temp_final_{i-1}.jpg", f"temp_final_{i}.jpg")
            current_image = rule_processor.run(rule)
            
        tesseract = Tesseract(self.input_paths[0], self.output_paths[0])
        if self.ground_truths:
            tesseract.set_ground_truth(self.ground_truths[0])
        tesseract.tesseract(current_image)
        wer, cer = tesseract.get_test()
        
        return {
            'wer': wer * 100,
            'cer': cer * 100,
            'fitness': self.best_fitness,
            'sequence': self.best_sequence
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.best_sequence is None:
            raise ValueError("No solution found yet")
            
        model_data = {
            'best_sequence': self.best_sequence,
            'n_rules': self.n_rules,
            'sequence_length': self.sequence_length,
            'fitness': self.best_fitness
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
            
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'r') as f:
            model_data = json.dump(f)
            
        self.best_sequence = model_data['best_sequence']
        self.n_rules = model_data['n_rules']
        self.sequence_length = model_data['sequence_length']
        self.best_fitness = model_data['fitness']
        
        logging.info(f"Model loaded from {filepath}")
    
    def predict(self, image_path: str, output_path: Optional[str] = None) -> str:
        """Use model to predict on new image"""
        if self.best_sequence is None:
            raise ValueError("No solution found yet")
            
        current_image = None
        for i, rule in enumerate(self.best_sequence):
            if i == 0:
                rule_processor = Rule(image_path, f"temp_pred_{i}.jpg")
            else:
                rule_processor = Rule(f"temp_pred_{i-1}.jpg", f"temp_pred_{i}.jpg")
            current_image = rule_processor.run(rule)
        
        # Use provided output path or generate default
        if output_path is None:
            output_path = f"prediction_{Path(image_path).stem}.txt"
        
        # Run OCR on processed image
        tesseract = Tesseract(None, output_path)
        tesseract.tesseract(current_image)
        
        # Clean up temporary files
        for i in range(len(self.best_sequence)):
            temp_file = Path(f"temp_pred_{i}.jpg")
            if temp_file.exists():
                temp_file.unlink()
        
        return output_path
      
      
if __name__ == "__main__":
    # Training dengan multiple gambar
    input_paths = [
        'input_path/image1.png',
        'input_path/image2.png',
        'input_path/image3.png'
    ]
    
    output_paths = [
        'output_path/result1.txt',
        'output_path/result2.txt',
        'output_path/result3.txt'
    ]
    
    ground_truth_paths = [
        'ground_truth/truth1.txt',
        'ground_truth/truth2.txt',
        'ground_truth/truth3.txt'
    ]
    
    # Inisialisasi dan jalankan algoritma genetika
    ga = GeneticAlgorithm(
        input_paths=input_paths,
        output_paths=output_paths,
        ground_truth_paths=ground_truth_paths,
        population_size=80,
        generations=100,
        mutation_rate=0.1,
        n_rules=9,
        sequence_length=9
    )
    
    # Evolusi dan dapatkan hasil terbaik
    best_sequence, metrics = ga.evolve()
    
    # Tampilkan hasil
    print("\nHasil Optimasi:")
    print(f"Sequence terbaik: {best_sequence}")
    print(f"WER: {metrics['wer']:.2f}%")
    print(f"CER: {metrics['cer']:.2f}%")
    print(f"Fitness: {metrics['fitness']:.4f}")
    
    # Simpan model
    ga.save_model('model_terbaik.json')