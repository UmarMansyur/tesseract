import random
import numpy as np
from Tesseract import Tesseract
from Rule import Rule
from typing import List, Tuple, Dict
import cv2
from multiprocessing import Pool
from functools import lru_cache

class GeneticAlgorithm:
    def __init__(self, input_path: str, output_path: str, ground_truth_path: str = None,
                 population_size: int = 20, generations: int = 50):
        self.input_path = input_path
        self.output_path = output_path
        
        # Load input image
        self.current_image = cv2.imread(input_path)
        if self.current_image is None:
            raise ValueError(f"Could not load image from {input_path}")
        
        # Load ground truth
        self.ground_truth = None
        if ground_truth_path:
            try:
                with open(ground_truth_path, 'r', encoding='utf-8') as f:
                    self.ground_truth = f.read()
            except FileNotFoundError:
                if ground_truth_path != 'dummy':  # Ignore error for dummy path
                    raise

        self.population_size = population_size
        self.generations = generations
        self.n_rules = 9
        self.sequence_length = 9
        self.fitness_cache = {}
        self.best_sequence = None
        
    def is_valid_sequence(self, sequence: List[int]) -> bool:
        """Memeriksa apakah urutan rule valid (tidak ada rule yang berurutan)"""
        for i in range(len(sequence) - 1):
            if sequence[i] == sequence[i + 1]:
                return False
        return True
    
    def int_to_binary(self, number: int) -> str:
        """Mengkonversi integer ke binary string 4-bit"""
        return format(number, '04b')  # 4 bit karena max rule adalah 15 (butuh 4 bit)
    
    def binary_to_int(self, binary: str) -> int:
        """Mengkonversi binary string ke integer"""
        return int(binary, 2)
    
    def create_individual(self) -> List[int]:
        """Membuat individu dengan representasi biner"""
        # Membuat sequence seperti sebelumnya
        sequence = random.sample(range(1, self.n_rules + 1), self.sequence_length)
        return sequence  # Tetap return dalam bentuk decimal untuk kemudahan penggunaan
    
    def create_population(self) -> List[List[int]]:
        """Membuat populasi awal yang unik"""
        unique_individuals = set()
        population = []
        
        max_attempts = self.population_size * 3  # Batasi jumlah percobaan
        attempts = 0
        
        while len(population) < self.population_size and attempts < max_attempts:
            new_individual = self.create_individual()
            new_tuple = tuple(new_individual)
            
            if new_tuple not in unique_individuals:
                unique_individuals.add(new_tuple)
                population.append(new_individual)
            
            attempts += 1
        
        # Jika masih kurang, isi sisa populasi dengan individu random
        while len(population) < self.population_size:
            population.append(self.create_individual())
        print(f"Populasi: {population} ke {len(population)}")
        return population
    
    @lru_cache(maxsize=1000)
    def calculate_fitness(self, individual_tuple: tuple) -> float:
        """Cached version of fitness calculation for single image"""
        individual = list(individual_tuple)
        current_image = self.current_image.copy()
        
        # Apply rules sequentially without saving intermediate files
        for rule_num in individual:
            rule_processor = Rule(None, None)
            current_image = rule_processor.run(rule_num, current_image)
        
        # Perform OCR
        tesseract = Tesseract(None, self.output_path)
        if self.ground_truth:
            tesseract.set_ground_truth(self.ground_truth)
        tesseract.tesseract(current_image)
        
        metrics = tesseract.get_test()
        # Use WER and CER from metrics dictionary
        error_rate = 0.3 * metrics['WER'] + 0.7 * metrics['CER']
        return np.exp(-error_rate)

    def fitness(self, individual: List[int]) -> float:
        """Wrapper for cached fitness calculation"""
        return self.calculate_fitness(tuple(individual))

    def evaluate_population(self, population: List[List[int]]) -> List[float]:
        """Parallel fitness evaluation"""
        with Pool() as pool:
            return pool.map(self.fitness, population)
    
    def select_parents(self, population: List[List[int]], fitness_scores: List[float]) -> Tuple[List[int], List[int]]:
        """Memilih parents menggunakan tournament selection"""
        tournament_size = 3
        
        def tournament_select() -> List[int]:
            tournament_idx = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            return population[winner_idx]
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        return parent1, parent2
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Melakukan crossover di level angka individual"""
        log_entries = []  # Untuk menyimpan log
        log_entries.append(f"Parent 1: {parent1}")
        log_entries.append(f"Parent 2: {parent2}")
        
        max_attempts = 10
        for attempt in range(max_attempts):
            pos = random.randint(0, self.sequence_length - 1)
            log_entries.append(f"\nAttempt {attempt + 1}")
            log_entries.append(f"Crossover position: {pos}")
            
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            # Konversi ke binary dan log
            p1_binary = self.int_to_binary(parent1[pos])
            p2_binary = self.int_to_binary(parent2[pos])
            log_entries.append(f"Parent 1 binary at pos {pos}: {p1_binary}")
            log_entries.append(f"Parent 2 binary at pos {pos}: {p2_binary}")
            
            bit_pos = random.randint(1, 3)
            log_entries.append(f"Bit crossover position: {bit_pos}")
            
            c1_binary = p1_binary[:bit_pos] + p2_binary[bit_pos:]
            c2_binary = p2_binary[:bit_pos] + p1_binary[bit_pos:]
            log_entries.append(f"Child 1 binary: {c1_binary}")
            log_entries.append(f"Child 2 binary: {c2_binary}")
            
            c1_val = self.binary_to_int(c1_binary)
            c2_val = self.binary_to_int(c2_binary)
            log_entries.append(f"Child 1 value: {c1_val}")
            log_entries.append(f"Child 2 value: {c2_val}")
            
            if (1 <= c1_val <= self.n_rules and 1 <= c2_val <= self.n_rules):
                child1[pos] = c1_val
                child2[pos] = c2_val
                
                if (len(set(child1)) == self.sequence_length and 
                    len(set(child2)) == self.sequence_length):
                    log_entries.append("\nValid children found!")
                    log_entries.append(f"Final Child 1: {child1}")
                    log_entries.append(f"Final Child 2: {child2}")
                    
                    # Simpan log ke file
                    with open('crossover_log.txt', 'a', encoding='utf-8') as f:
                        f.write('\n' + '-'*50 + '\n')
                        f.write('\n'.join(log_entries))
                    
                    return child1, child2
            
            log_entries.append("Invalid result, trying again...")
        
        log_entries.append("\nNo valid children found after max attempts")
        log_entries.append("Returning original parents")
        
        # Simpan log ke file untuk kasus gagal
        with open('crossover_log.txt', 'a', encoding='utf-8') as f:
            f.write('\n' + '-'*50 + '\n')
            f.write('\n'.join(log_entries))
        
        return parent1[:], parent2[:]
    
    def mutate(self, individual: List[int], mutation_rate: float = 0.1) -> List[int]:
        """Melakukan mutasi pada level biner"""
        if random.random() < mutation_rate:
            # Konversi ke biner
            binary = ''.join([self.int_to_binary(x) for x in individual])
            binary_list = list(binary)
            
            # Pilih posisi untuk mutasi (flip bit)
            pos = random.randint(0, len(binary) - 1)
            binary_list[pos] = '1' if binary_list[pos] == '0' else '0'
            
            # Konversi kembali ke integer
            mutated = []
            for i in range(0, len(binary_list), 4):
                val = self.binary_to_int(''.join(binary_list[i:i+4]))
                if 1 <= val <= self.n_rules:
                    mutated.append(val)
                else:
                    return individual[:]  # Jika hasil tidak valid, kembalikan original
            
            # Validasi hasil mutasi
            if len(mutated) == self.sequence_length and len(set(mutated)) == self.sequence_length:
                return mutated
            
        return individual[:]
    
    def save_model(self, filepath: str):
        """Menyimpan model (sequence terbaik) ke file"""
        if self.best_sequence is None:
            raise ValueError("Model belum dilatih. Jalankan evolve() terlebih dahulu.")
        
        import json
        model_data = {
            'best_sequence': self.best_sequence,
            'n_rules': self.n_rules,
            'sequence_length': self.sequence_length
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Memuat model dari file"""
        import json
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.best_sequence = model_data['best_sequence']
        self.n_rules = model_data['n_rules']
        self.sequence_length = model_data['sequence_length']
    
    def predict(self, image_path: str, output_path: str = None) -> Dict:
        """Menggunakan model untuk memprediksi gambar baru"""
        if self.best_sequence is None:
            raise ValueError("Model belum dilatih atau dimuat.")
        
        current_image = None
        for i, rule in enumerate(self.best_sequence):
            if i == 0:
                rule_processor = Rule(image_path, f"temp_pred_{i}.jpg")
            else:
                rule_processor = Rule(image_path_result, f"temp_pred_{i}.jpg")
            current_image = rule_processor.run(rule)
            image_path_result = f"temp_pred_{i}.jpg"
            cv2.imwrite(image_path_result, current_image)
        
        # Lakukan OCR pada gambar hasil
        output_path = output_path if output_path else "result_predict.txt"
        tesseract = Tesseract(image_path, output_path)
        ocr_result = tesseract.tesseract(current_image)
        
        # Calculate WER and CER if ground truth is available
        metrics = {}
        if hasattr(self, 'ground_truth') and self.ground_truth:
            tesseract.set_ground_truth(self.ground_truth)
            test_metrics = tesseract.get_test()
            metrics = {
                'text': ocr_result,
                'wer': round(test_metrics['WER'], 4),
                'cer': round(test_metrics['CER'], 4)
            }
        else:
            metrics = {'text': ocr_result}
        
        # Save metrics to file
        metrics_path = output_path.rsplit('.', 1)[0] + '_metrics.txt'
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        return metrics

    def evolve(self) -> Tuple[List[int], Dict]:
        """Optimized evolution process"""
        population = self.create_population()
        best_sequence = None
        best_fitness = float('-inf')
        
        print("\nStarting Genetic Algorithm Evolution...")
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Parallel fitness evaluation
            print("Evaluating population fitness...")
            fitness_scores = self.evaluate_population(population)
            
            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[max_fitness_idx]
            current_best_sequence = population[max_fitness_idx]
            
            print(f"Current generation best fitness: {current_best_fitness:.4f}")
            print(f"Current generation best sequence: {current_best_sequence}")
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_sequence = current_best_sequence
                print(f"New overall best fitness found: {best_fitness:.4f}")
                print(f"New overall best sequence: {best_sequence}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(population[max_fitness_idx])
            print("\nCreating new population...")
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = self.select_parents(population, fitness_scores)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim population to original size
            population = new_population[:self.population_size]
            print(f"New population size: {len(population)}")

        print("\nEvolution completed!")
        print(f"Final best sequence: {best_sequence}")
        print(f"Final best fitness: {best_fitness:.4f}")

        # Calculate final metrics for best sequence
        current_image = self.current_image.copy()
        for rule in best_sequence:
            rule_processor = Rule(None, None)
            current_image = rule_processor.run(rule, current_image)
        
        tesseract = Tesseract(self.input_path, self.output_path)
        if self.ground_truth:
            tesseract.set_ground_truth(self.ground_truth)
        tesseract.tesseract(current_image)
        test_metrics = tesseract.get_test()
        
        metrics = {
            'wer': round(test_metrics['WER'], 4),
            'cer': round(test_metrics['CER'], 4),
            'fitness': round(best_fitness, 4)
        }
        print(metrics)

        # Save best sequence
        self.best_sequence = best_sequence
        
        return best_sequence, metrics

if __name__ == "__main__":
    # Training dengan single gambar
    # ga = GeneticAlgorithm(
    #     input_path='input_path/image1.png',
    #     output_path='output_path/result.txt',
    #     ground_truth_path='ground_truth/image1.txt',
    #     population_size=20,
    #     generations=50
    # )
    # best_sequence, metrics = ga.evolve()
    # ga.save_model('ga_model.json')

    # Prediksi
    predictor = GeneticAlgorithm(
        input_path='input_path/image5.png',
        output_path='output_path/result.txt',
        ground_truth_path='ground_truth/image5.txt'
    )
    predictor.load_model('ga_model.json')
    result = predictor.predict('input_path/image5.png')
    with open('output_path/result_predict.txt', 'w') as f:
        f.write(result['text'])

    print(result)
