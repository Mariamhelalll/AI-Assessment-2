""" BaselineGA.py
#
# Your GA code. 
#
# This is the file you will modify.
#
# Modified by: YOUR NAME
# Last Modified: DD/MM/25
"""

from numpy.random import randint   # for random integers
from numpy.random import rand      # for random probabilities
import random
import math

from abstractGA import AbstractGA
import config
           
""" A GA to solve the TSP
    This class extends the AbstractGA class.
"""
class BaselineGA(AbstractGA):    
        
    """ Creates a new population and returns the best individual found so far.
        self.population stores the current population 
        self.fitnesses stores the fitness of each member of the population.
    """
    def produce_new_generation(self):
        new_population = []

        # (Optional) elitism: keep the current best individual
        if self.best_individual is not None:
            new_population.append(self.best_individual)

        # Keep creating children until the new population is full
        while len(new_population) < config.POPULATION_SIZE:

            # ---- 1. SELECT PARENTS (TOURNAMENT) ----
            parent1 = self.tournament_selection(k=3)
            parent2 = self.tournament_selection(k=3)

            # ---- 2. CROSSOVER (ORDER-BASED) ----
            if rand() < config.CROSSOVER_RATE:
                child1, child2 = self.order_crossover(parent1, parent2)
            else:
                # no crossover → just copies
                child1 = parent1.copy()
                child2 = parent2.copy()

            # ---- 3. MUTATION (SWAP) ----
            child1 = self.swap_mutation(child1)
            child2 = self.swap_mutation(child2)

            # ---- 4. ADD CHILDREN TO NEW POPULATION ----
            new_population.append(child1)
            if len(new_population) < config.POPULATION_SIZE:
                new_population.append(child2)

        # Replace old population with new one
        self.population = new_population

        # calculate the new fitness and return the best individual
        self.calculate_fitness_of_population()  # method in AbstractGA
        return (self.best_individual, self.best_fitness)   
    
    
    """ Fitness = total Euclidean distance of the route.
        The chromosome is a list of City objects in visiting order.
    """
    def calculate_fitness(self, chromosome):    
        cities = self.convert_chromosome_to_city_list(chromosome)   
         
        total = 0.0

        # distance between each consecutive pair of cities
        for i in range(len(cities) - 1):
            total += self.euclidean_distance(cities[i], cities[i + 1])

        # add distance from last city back to first (make it a cycle)
        total += self.euclidean_distance(cities[-1], cities[0])
                   
        return total 
    
    # ================= HELPER METHODS =================
    
    def euclidean_distance(self, city_a, city_b):
        """Return Euclidean distance between two City objects."""
        # IMPORTANT: adjust attribute names if your City is different.
        # Most versions use: city.x and city.y
        x1, y1 = city_a.x, city_a.y
        x2, y2 = city_b.x, city_b.y

        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    # ---------- Selection: Tournament ----------
    def tournament_selection(self, k=3):
        """Pick k random individuals and return the best (lowest fitness)."""
        # choose k random indices into population
        indices = random.sample(range(len(self.population)), k)

        best_index = indices[0]
        best_fitness = self.fitnesses[best_index]

        for idx in indices[1:]:
            if self.fitnesses[idx] < best_fitness:
                best_index = idx
                best_fitness = self.fitnesses[idx]

        # return a copy so we don't mutate the original parent
        return self.population[best_index].copy()


    # ---------- Crossover: Order-based (as in brief) ----------
    def order_crossover(self, parent1, parent2):
        """
        Implements the crossover described in the brief:
        1. Choose a random crossover point.
        2. child1 gets all genes from parent1 before crossover point.
        3. Then fill remaining positions in the order genes appear in parent2,
           skipping any that are already in child1.
        4. Repeat with parents swapped to get child2.
        """
        length = len(parent1)
        # crossover point between 1 and length-1 (avoid empty prefix)
        cp = randint(1, length)

        # --- child 1: prefix from parent1, fill from parent2 ---
        child1 = parent1[:cp]
        for gene in parent2:
            if gene not in child1:
                child1.append(gene)

        # --- child 2: prefix from parent2, fill from parent1 ---
        child2 = parent2[:cp]
        for gene in parent1:
            if gene not in child2:
                child2.append(gene)

        return child1, child2


    # ---------- Mutation: Swap ----------
    def swap_mutation(self, chromosome):
        """
        With probability MUTATION_RATE:
        choose two random positions and swap the cities.
        """
        if rand() < config.MUTATION_RATE:
            length = len(chromosome)
            i = randint(0, length)
            j = randint(0, length)
            while j == i:
                j = randint(0, length)

            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

        return chromosome
           
           
    """ The stopping criteria. When this returns true, the GA will stop producing new generations.
    """
    def finished(self):
        return self.number_of_generations >= config.MAX_NUMBER_OF_GENERATIONS
    
       
    #-------------------
    # Representation conversion methods
    #-------------------
    
    def convert_city_list_to_chromosome(self, cities):        
        return cities   
        
    def convert_chromosome_to_city_list(self, chromosome):
        return chromosome
    
# End of BaselineGA class
