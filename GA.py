import random
import numpy as np
from Tesseract import Tesseract
from Rule import Rule
from typing import List, Tuple, Dict
import cv2

class GeneticAlgorithm:
    def __init__(self, input_path: str, output_path: str, ground_truth_path: str,
                 population_size: int = 80, generations: int = 100):
        self.input_path = input_path
        self.output_path = output_path
        # Baca ground truth dari file
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.ground_truth = f.read()
        self.population_size = population_size
        self.generations = generations
        self.n_rules = 9  # Jumlah total rule yang tersedia
        self.sequence_length = 6  # Panjang tetap untuk setiap sequence
        
    def is_valid_sequence(self, sequence: List[int]) -> bool:
        """Memeriksa apakah urutan rule valid (tidak ada rule yang berurutan)"""
        for i in range(len(sequence) - 1):
            if sequence[i] == sequence[i + 1]:
                return False
        return True
    
    def int_to_binary(self, number: int) -> str:
        """Mengkonversi integer ke binary string 4-bit"""
        return format(number, '04b')  # 4 bit karena max rule adalah 9 (butuh 4 bit)
    
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
    
    def fitness(self, individual: List[int]) -> float:
        """Menghitung fitness berdasarkan WER dan CER"""
        current_image = None
        image_path_result = None
        
        # Aplikasikan setiap rule secara berurutan
        for i, rule in enumerate(individual):
            if i == 0:
                rule_processor = Rule(self.input_path, f"temp_image_{i}.jpg")
            else:
                rule_processor = Rule(image_path_result, f"temp_image_{i}.jpg")
            
            current_image = rule_processor.run(rule)
            
            # Simpan hasil intermediate
            if i < len(individual) - 1:
                image_path_result = f"temp_image_{i}.jpg"
                cv2.imwrite(image_path_result, current_image)
        
        # Lakukan OCR pada hasil akhir
        tesseract = Tesseract(self.input_path, self.output_path)
        tesseract.set_ground_truth(self.ground_truth)
        tesseract.tesseract(current_image)
        
        # Hitung metrics
        wer, cer = tesseract.get_test()
        
        # Fitness adalah inverse dari rata-rata WER dan CER
        # Semakin kecil error, semakin besar fitness
        # nilai wer itu 0.3 dan cer itu 0.7
        error_rate = 0.3 * wer + 0.7 * cer
        fitness_score = np.exp(-error_rate)
        return fitness_score
    
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
    
    def evolve(self) -> Tuple[List[int], Dict]:
        """Menjalankan proses evolusi"""
        population = self.create_population()
        best_sequence = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.fitness(ind) for ind in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_sequence = population[max_fitness_idx]
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(population[max_fitness_idx])
            
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
        
        # Calculate final metrics for best sequence
        current_image = None
        for i, rule in enumerate(best_sequence):
            if i == 0:
                rule_processor = Rule(self.input_path, f"temp_image_final_{i}.jpg")
            else:
                rule_processor = Rule(image_path_result, f"temp_image_final_{i}.jpg")
            current_image = rule_processor.run(rule)
            image_path_result = f"temp_image_final_{i}.jpg"
            cv2.imwrite(image_path_result, current_image)
        
        tesseract = Tesseract(self.input_path, self.output_path)
        tesseract.set_ground_truth(self.ground_truth)
        tesseract.tesseract(current_image)
        wer, cer = tesseract.get_test()
        
        metrics = {
            'wer': wer * 100,  # Convert to percentage
            'cer': cer * 100,  # Convert to percentage
            'fitness': best_fitness
        }
        print(metrics)
        return best_sequence, metrics

if __name__ == "__main__":
    # Contoh penggunaan
    ga = GeneticAlgorithm(
        input_path='input_path/image1.png',
        output_path='output_path/result.txt',
        ground_truth_path='ground_truth/image1.txt',  # Path ke file ground truth
        population_size=20,
        generations=50
    )

    best_sequence, metrics = ga.evolve()
    print(f"Best sequence: {best_sequence}")
    print(f"Metrics: WER={metrics['wer']:.2f}%, CER={metrics['cer']:.2f}%")