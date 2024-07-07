import random
import math
import itertools
from lib.graph import Graph
from lib.utils.calculate_tour_distance import calculate_tour_distance

class Individual():
    '''
    Create an 'individual' who holds a chromosome consisting of 4 parameters: temperature, cooling rate, approval rate and tour nodes
    '''

    def __init__(self,temperature,cooling_rate,approval_rate, nodes):
        '''
        Initializes the individual using the 4 gene parameters
        :param temperature:
        :param cooling_rate:
        :param approval_rate:
        :param nodes:
        '''
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.approval_rate = approval_rate
        self.nodes = nodes

    def return_chromosome(self):
        '''
        Returns the 4 genes of the individual as a list
        '''
        chromosome = [self.temperature,self.cooling_rate,self.approval_rate,self.nodes]
        return chromosome
    
    def set_nodes(self, nodes):
        '''
        Sets the individual's nodes
        param nodes:
        '''
        self.nodes = nodes

def check_and_update_all_chromosomes_with_new_nodes(population, best_chromosome, best_chromosome_overall):
    '''
    Updates the population with the shortest distance found so far
    param population: list of objects of class Individual
    param best_chromosome: a list of the best_chromosome of the current generation
    param best_chromosome_overall: a list of the best_chromosome globally
    '''
    nodes = None

    #If the best global chromosome is not set or the best chromsome of the generation has a shorter tour, set the best global chromosome to the best one from the current generation
    if best_chromosome_overall == None or calculate_tour_distance(best_chromosome_overall[3]) > calculate_tour_distance(best_chromosome[3]):
        best_chromosome_overall = best_chromosome[:]
        nodes = best_chromosome[3]
    else:
        #The best chromosome overall has a shorter distance than the best one of the current generation
        nodes = best_chromosome_overall[3]

    #Set the population with the shortest distance
    for chromosome in population:
        chromosome.set_nodes(nodes)
    
    return best_chromosome_overall

def find_and_set_best_chromosome_nodes(gen_num,population, best_chromosome_overall):
    '''
    Given a generation number, a population and the best chromosome overall, computes the maximum fitness value and returns
    the chromosome with the shortest tour distance
    :param gen_num: integer number of the generation
    :param population: list of objects of class Individual
    :param best_chromosome_overall: list
    :return: the best chromosome globally
    '''
    global best_distance #Not used

    chromosomes = get_chromosomes(population)
    fitness_scores_parents,fitness_scores_children = compute_fitness_scores(chromosomes,[])
    if fitness_scores_parents != []:
        max_fitness = max(fitness_scores_parents)
        index = fitness_scores_parents.index(max_fitness)
        best_chromosome = chromosomes[index]

        best_chromosome_overall = check_and_update_all_chromosomes_with_new_nodes(population, best_chromosome, best_chromosome_overall)

        #print("For generation: ", gen_num, " | Best Parameters: ", best_chromosome_overall[:3], " | Best Fitness: ", max_fitness, " | Distance: ", str(calculate_tour_distance(best_chromosome_overall[3])))
    else:
        max_fitness = None
        best_chromosome = None
        #print("For generation: ", gen_num, " | Best Parameters: ", best_chromosome, " | Best Fitness: ", max_fitness)
    
    return best_chromosome_overall

def initialize_gene_possibilities(setNumber):
    '''
    Creates a list of various temperatures, cooling rates and approval rates
    :param setNumber: integer for the range of values for temp, cooling rate and approval rate
    :return: Return 3 hard-coded lists, 1 for list of possible temperature alleles, 1 for list of possible cooling
    rate alleles and 1 for list of possible approval rate alleles.
    '''
    if setNumber == 1:
        temperature_alleles = [100+10*i for i in range(20)] #Store temperatures as 100-290 degrees in increments of 10 degrees
        cooling_rate_alleles = [0.9+0.005*i for i in range(20)] #Store cooling rates from 0.9-0.995 in increments of 0.005
        approval_rate_alleles = [0.5+0.05*i for i in range(10)] #Store approval rates from 0.5-0.95 in increments of 0.5
    else:
        temperature_alleles = [50+10*i for i in range(495)] 
        cooling_rate_alleles = [0.005+0.005*i for i in range(198)] 
        approval_rate_alleles = [0.5+0.05*i for i in range(10)] #Store approval rates from 0.5-0.95 in increments of 0.5

    return temperature_alleles, cooling_rate_alleles, approval_rate_alleles

def randomly_initialize_genes(temp_list, cr_list, apprv_list):
    '''
    Given a temperature list, cooling rate list and approval rate list, randomly initialize and return a temperature, cooling
    rate and approval rate from each of the 3 lists.
    :param temp_list: list of possible temperature values (alleles)
    :param cr_list: list of possible cooling rate values
    :param apprv_list: list of possible approval rate values
    :return: return 3 parameters: temperature, cooling_rate, approval_rate
    '''

    #Apply uniform sampling on each allele
    temp = random.choice(temp_list)
    cooling_rate = random.choice(cr_list)
    approval_rate = random.choice(apprv_list)

    return temp, cooling_rate, approval_rate

def getNextCombination():
    '''
    Generate the next combination of inital conditions
    :return: a list that contains [max_pop, num_children, mutation_rate, step_sizes, cross_over_rate_list, initialTempSet]
    '''
    global allCombinations

    if allCombinations == None:
        max_pop_list = [10, 30, 50]
        num_children_list = [1, 3, 5]
        mutation_rate_list = [0.1, 0.3, 0.5]
        step_sizes_list = [[1, 1, 1], [3, 3, 3], [5, 5, 5]]
        cross_over_rate_list = [0.1, 0.3, 0.5]
        initalTempSet = [1, 2]

        allCombinations = [max_pop_list] + [num_children_list] + [mutation_rate_list] + [step_sizes_list] + [cross_over_rate_list] + [initalTempSet]
        allCombinations = itertools.product(*allCombinations) # Creates all possible combinations of every list
    
    nextCombo = next(allCombinations, None) # Gets the next combination
    return nextCombo

def initialize_pop(nodes, nextCombo):
    '''
    Initialize a hard-coded population and return parameters that will be used in running the genetic algorithm.
    :param nodes: nodes
    :param nextCombo: list
    :return: temperature list, cooling rate list, approval rate list, max population, number of children,
    mutation rate, cross-over rate, list of Individuals (population), list of step sizes, combination of all the factors as a list
    '''

    nextCombo = getNextCombination()
    if nextCombo == None: # There are no more combinations left
        return None, None, None, None, None, None, None, None, None, False

    max_pop = nextCombo[0]
    num_children = nextCombo[1]
    mutation_rate = nextCombo[2]
    step_sizes = nextCombo[3] #Maximum step size for mutation in each dimension: temp, cr, ar
    cross_over_rate = nextCombo[4]
    temp_list, cr_list, apprv_list = initialize_gene_possibilities(nextCombo[5])

    population = []

    #Initialize initial parents in 'generation 0'
    for i in range(max_pop):
        temp,cr,ar = randomly_initialize_genes(temp_list,cr_list,apprv_list)
        new_individual = Individual(temp,cr,ar, nodes)
        population.append(new_individual)

    return temp_list, cr_list, apprv_list, max_pop, num_children, mutation_rate,cross_over_rate, population, step_sizes, nextCombo

def mating_generation(population, num_children, mutation_rate, cross_over_rate, temp_list, cr_list, apprv_list, step_sizes):
    '''
    Simulate the mating process by applying crossover and mutation to generate new children.
    :param population: list of Individuals
    :param num_children: number of children (integer)
    :param mutation_rate: rate of mutation (float)
    :param cross_over_rate: rate of crossover (float)
    :param temp_list: list of possible temperature alleles (list)
    :param cr_list: possible cooling rate alleles (list)
    :param apprv_list: possible approval rate alleles (list)
    :param step_sizes: maximal step sizes in each dimension (list)
    :return:
    '''

    #print(population)
    len_pop = len(population)

    num_pairs = math.floor(len_pop/2) #since it takes 2 to make a pair

    child_chromosomes = []

    for i in range(num_pairs):
        #print("pair:",i)

        #First a pair
        random.shuffle(population) #Reorder the population randomly
        pop_pair = population[0:2]
        population = population[2:]

        #Mate the pair: assume 2 parents produce 2 children
        child_pair_chromosomes = mating_pairwise(pop_pair,num_children,mutation_rate,cross_over_rate,temp_list,cr_list,apprv_list, step_sizes)
        #print(child_pair_chromosomes)

        child_chromosomes.extend(child_pair_chromosomes)

    return child_chromosomes

def get_chromosomes(population):
    '''
    Given a list of Individuals, return a list of chromosomes associated with these individuals
    :param population: individuals (list)
    :return: chromosomes (list)
    '''

    parent_chromosomes = []
    for parent in population:
        chromosome = parent.return_chromosome()
        parent_chromosomes.append(chromosome)

    return parent_chromosomes

def mating_pairwise(pop_pair,num_children,mutation_rate,cross_over_rate,temp_list,cr_list,apprv_list, step_sizes):
    '''
    Apply the mating process but for each pair of parents chosen from the parent-pool
    :param pop_pair: 2 individuals (list)
    :param num_children: fixed at 2 (not needed yet, integer)
    :param mutation_rate: float
    :param cross_over_rate: float
    :param temp_list: list
    :param cr_list: list
    :param apprv_list: list
    :param step_sizes: list
    :return:
    '''

    #print(pop_pair)

    #Compute parent chromosomes
    parent_chromosomes = []
    for parent in pop_pair:
        chromosome = parent.return_chromosome()
        parent_chromosomes.append(chromosome)

    #Create baseline child chromosomes
    child_chromosomes = parent_chromosomes[:]

    #Check if Cross-Over will occur for the Pair. if so, apply cross-over
    cross_over_prob = random.random()
    if cross_over_prob < cross_over_rate:
        #print("------------- APPLYING CROSSOVER ---------------")
        #print("Children Before Crossover: ", child_chromosomes)
        child_chromosomes = apply_crossover(child_chromosomes) #Single point crossover
        #print("Children After Crossover: ", child_chromosomes)

    #Check if Mutation will occur for each child chromosome
    for i in range(len(child_chromosomes)):

        mutation_prob = random.random()
        if mutation_prob < mutation_rate:
            #print("------------- APPLYING MUTATION ---------------")
            #print("Children Before Mutation: ", child_chromosomes)
            chromosome = apply_mutation(child_chromosomes[i], temp_list, cr_list, apprv_list, step_sizes) #Single gene mutation
            child_chromosomes[i] = chromosome
            #print("Children After Mutation: ", child_chromosomes)

    return child_chromosomes

def apply_crossover(child_chromosomes):
    '''
    Given a list of child chromosomes, apply cross over at a random index along the gene, and return the new chromosome list
    :param child_chromosomes: list
    :return: list
    '''

    #Assume the following representation: 0->before the first gene,len_chromosome->after the last gene
    # then: 1->Between first and second genes
    # So if the cross-over point is 1, then it would occur between gene 1 and gene 2.
    len_chromosome = len(child_chromosomes[0])
    cross_over_points = list(range(1,len_chromosome))
    cross_over_point = random.choice(cross_over_points)

    old_chromo_1,old_chromo_2 = child_chromosomes
    new_chromo_1 = old_chromo_1[:cross_over_point] + old_chromo_2[cross_over_point:]
    new_chromo_2 = old_chromo_2[:cross_over_point] + old_chromo_1[cross_over_point:]

    new_chromosomes = [new_chromo_1,new_chromo_2]

    return new_chromosomes

def apply_mutation(chromosome, temp_list, cr_list, apprv_list,step_sizes):
    '''
    Given a chromosome, and lists of allele choices for each gene, along with the maximum possible step sizes for each
    gene, select a single random gene to mutate, and mutate that gene along the available step_sizes.
    :param chromosome: list
    :param temp_list: list
    :param cr_list: list
    :param apprv_list: list
    :param step_sizes: list
    :return: list
    '''

    #First pick which gene to mutate
    gene_choices = [0,1,2]
    gene_choice = random.choice(gene_choices)
    allele_list = None

    if gene_choice == 0:
        allele_list = temp_list[:]
    elif gene_choice == 1:
        allele_list = cr_list[:]
    else:
        allele_list = apprv_list[:]

    step_size = step_sizes[gene_choice]
    current_allele = chromosome[gene_choice]
    allele_index = allele_list.index(current_allele)
    start_index = max(allele_index-step_size,0)
    end_index = min(allele_index+step_size,len(allele_list))
    #print("Allele List Before: ",allele_list)
    #print("Allele Index: ",allele_index)
    #print("Start Index: ",start_index)
    #print("End Index: ",end_index)
    allele_list.remove(current_allele)
    allele_list = allele_list[start_index:end_index] #Want to reduce allele choices to a smaller set around the original allele location
    #print("Allele List After: ",allele_list)


    #Remove current allele from list of possible mutations
    #print("Allele List: ", allele_list)
    #print("Gene Choice: ", gene_choice)
    #print("Current Chromosome: ",chromosome)

    #print("Current Allele: ", current_allele)

    #print("New Allele List: ", allele_list)

    #Make a random choice of mutation
    mutation = random.choice(allele_list)

    #Create and return new chromosome
    new_chromosome = chromosome[:]
    new_chromosome[gene_choice] = mutation

    return new_chromosome

def evaluate_individual(individual):
    '''
    Score an individual according to the ratio between (SA / Brute force) distances
    :param individual: list
    :return: float
    '''
    temp, cr, ar, current_nodes = individual

    #Score the individual based on how they perform at the *task*
    score = -1.0
    global best_nodes
    global best_distance
    global bruteForceDistance

    current_distance = calculate_tour_distance(current_nodes)

    ##
    ## Our main loop
    ##
    while temp > 1:
        i, j = sorted(random.sample(range(len(current_nodes)), 2))
        new_nodes = (
            current_nodes[:i] +
            current_nodes[i:j][::-1] +
            current_nodes[j:]
        )

        new_distance = calculate_tour_distance(new_nodes)
        #tempScore = math.pow(math.e, -1)*abs(bruteForceDistance-new_distance)
        tempScore = bruteForceDistance / float(new_distance)

        if tempScore > score or score == -1.0:
            score = tempScore

            if new_distance < current_distance:
                #best_nodes = new_nodes[:]
                #best_distance = new_distance
                current_nodes = new_nodes
                current_distance = new_distance

            elif new_distance > current_distance:
                if (math.exp((current_distance - new_distance) / temp) > random.random()):
                    current_nodes = new_nodes
                    current_distance = new_distance

        temp *= cr

    return score

def evaluate_individual_obj(individual):
    '''
    Score an individual according to the ratio between (SA / Brute force) distances
    :param individual: Individual
    :return: float
    '''
    temp = individual.temperature 
    cr = individual.cooling_rate
    ar = individual.approval_rate
    current_nodes = individual.nodes

    current_distance = calculate_tour_distance(current_nodes)

    #Score the individual based on how they perform at the *task*
    score = -1.0
    global best_nodes
    global best_distance 
    global bruteForceDistance

    ##
    ## Our main loop
    ##
    while temp > 1:
        #Create a random list
        i, j = sorted(random.sample(range(len(current_nodes)), 2))
        new_nodes = (
            current_nodes[:i] +
            current_nodes[i:j][::-1] +
            current_nodes[j:]
        )

        #Calculate the distance
        new_distance = calculate_tour_distance(new_nodes)
        
        #Calculate the score
        #tempScore = math.pow(math.e, -1)*abs(bruteForceDistance-new_distance)
        tempScore = bruteForceDistance / float(new_distance)

        if tempScore > score or score == -1.0:
            score = tempScore
            individual.nodes = new_nodes[:] #Store these nodes to calculate the distance

            if new_distance < current_distance:
                #best_nodes = new_nodes[:]
                #best_distance = new_distance
                current_nodes = new_nodes
                current_distance = new_distance

            elif new_distance > current_distance:
                if (math.exp((current_distance - new_distance) / temp) > random.random()):
                    current_nodes = new_nodes
                    current_distance = new_distance

        temp *= cr

    return score #Largest score for this chromosone

def compute_fitness_scores(parent_chromosomes,child_chromosomes):
    '''
    Given a set of parent and child chromosomes, compute the fitness scores of each and return 2 lists
    :param parent_chromosomes: list
    :param child_chromosomes: list
    :return: list,list
    '''

    parent_scores = []
    child_scores = []

    for chromosome in parent_chromosomes:
        score = evaluate_individual(chromosome)
        parent_scores.append(score)

    for chromosome in child_chromosomes:
        score = evaluate_individual(chromosome)
        child_scores.append(score)

    return parent_scores, child_scores

def compute_fitness_scores_new(population, child_individuals):
    '''
    Given a set of parent and child chromosomes, compute the fitness scores of each and return 2 lists
    :param parent_chromosomes: list of Individuals
    :param child_chromosomes: list of Individuals
    :return: list,list
    '''
    parent_scores = []
    child_scores = []

    for chromosome in population:
        score = evaluate_individual_obj(chromosome)
        parent_scores.append(score)

    for chromosome in child_individuals:
        score = evaluate_individual_obj(chromosome)
        child_scores.append(score)

    return parent_scores, child_scores

def compute_diversity_all_elems(source_set,dest_set):
    '''
    Compute the diversity for each element in source set against elements of the destination set
    :param source_set: list
    :param dest_set:list
    :return: list
    '''

    diversity_scores = []

    for elem in source_set:

        diversity = compute_diversity_one_elem(elem,dest_set)
        diversity_scores.append(diversity)

    #Post-process diversity scores
    sum_scores = max(sum(diversity_scores),0.01) #Avoid the div/0 error

    diversity_scores_rel = [diversity_scores[i]/sum_scores for i in range(len(diversity_scores))]

    return diversity_scores_rel

def compute_diversity_one_elem(element,set):
    '''
    Compute the diversity of one element with respect to a set.
    We prefer a metric that is low when one of the distances to one of the elements is low, but is higher when there
    is a balance of distances between element and elements of the set.
    :param element: list
    :param set: list
    :return: float
    '''
    if set == [] or set == None:
        return 0 #If the destination set contains no elements, then diversity of source element is irrelevant

    sum_scores = 0
    prod_scores = 0
    elem_0, elem_1, elem_2, elem_3 = element

    for set_elem in set:
        set_elem_0, set_elem_1, set_elem_2, set_elem_3 = set_elem #Unpack the set element into its 3 constituents

        #Euclidean Distance
        score_incr = abs((elem_0-set_elem_0)**2) + abs((elem_1-set_elem_1)**2) + abs((elem_2-set_elem_2)**2)
        sum_scores += score_incr
        prod_scores *= score_incr

    sum_scores = max(sum_scores,0.01) #Avoid div by 0 error

    diversity = prod_scores/sum_scores

    return diversity

def selection(parent_chromosomes,child_chromosomes,max_pop,fitness_scores_parents,fitness_scores_children):
    '''
    Given a set of parent and child chromosomes, a maximum population, and fitness scores of parents and children,
    apply selection to generate the subsequent generation of the population.
    :param parent_chromosomes: list
    :param child_chromosomes: list
    :param max_pop: int
    :param fitness_scores_parents: list
    :param fitness_scores_children: list
    :return: list
    '''

    #Initialize Next Gen variables/structures
    next_gen_chromosomes = []

    #Compute fitness scores and compile list of chromosomes for 1st iteration (parent batch)
    chromosomes_all = parent_chromosomes + child_chromosomes
    fitness_scores_all = fitness_scores_parents + fitness_scores_children
    sum_fitness_scores = max(sum(fitness_scores_all),0.01) #Lowerbound the sum at 0.01 to avoid div/0 error
    fitness_scores_all_rel = [fitness_scores_all[i]/sum_fitness_scores for i in range(len(fitness_scores_all))]
    #Compute diversity score
    diversity_scores_all_rel = compute_diversity_all_elems(chromosomes_all,next_gen_chromosomes)
    #diversity_scores_all_rel = [0]*len(fitness_scores_all_rel)
    #Compute sum of diversity scores and fitness scores as a total score
    total_scores_all_rel = [fitness_scores_all_rel[i] + diversity_scores_all_rel[i] for i in range(len(fitness_scores_all_rel))]

    #Select a population of next generation chromosomes
    for i in range(max_pop):

        #Compute choice of chromosome to pass into next generation
        indices = list(range(len(total_scores_all_rel)))
        index_choice = (random.choices(indices,weights=total_scores_all_rel,k=1))[0] #Since choices returns a list of size 1 here
        #print("Index Choice:",index_choice)
        chromosome_choice = chromosomes_all[index_choice]
        next_gen_chromosomes.append(chromosome_choice)

        #Modify the list of unselected individuals to prepare the choice of the next individual
        fitness_scores_all_rel = fitness_scores_all_rel[:index_choice] + fitness_scores_all_rel[index_choice+1:] #Reduce in size
        diversity_scores_all_rel = compute_diversity_all_elems(chromosomes_all,next_gen_chromosomes) #Recompute
        #diversity_scores_all_rel = [0]*len(fitness_scores_all_rel)
        fitness_scores_all_rel = [fitness_scores_all_rel[i] + diversity_scores_all_rel[i] for i in range(len(fitness_scores_all_rel))] #Recompute

    #Make a population of individuals out of the set of chromosomes for the next generation and return it
    next_gen_pop = []

    for i in range(len(next_gen_chromosomes)):
        chromosome = next_gen_chromosomes[i]
        tmp,cr,ar,nodes = chromosome
        individual = Individual(tmp,cr,ar,nodes)
        next_gen_pop.append(individual)

    return next_gen_pop

def getChildIndividuals (child_chromosomes):
    '''
    Given a list of child chromosomes, return a list of each chromosome as an Individual
    :param child_chromosomes: list
    :return: list of Individuals
    '''
    individualsList = []

    for i in range(len(child_chromosomes)):
        chromosome = child_chromosomes[i]
        tmp,cr,ar,nodes = chromosome
        individual = Individual(tmp,cr,ar,nodes)
        individualsList.append(individual)

    return individualsList

def run_generations(num_gens=100, bruteForce=0.1, nodes=[], bestDistance=0.0):
    '''
    Algorithm runs n generations of the genetic algorithm
    :param n: int (optional)
    :param bruteForce: float
    :param nodes: list
    :param bestDistance: float
    :return: None
    '''
    global best_nodes
    global best_distance
    global bruteForceDistance
    global allCombinations

    bruteForceDistance = bruteForce
    allCombinations = None
    generateNewCombo = True
    best_chromosome_overall = None
    #counter = 1

    #Initialize Population, set all parameters: pop max size, num_children, mutation rate, cross-over rate, next combination
    temp_list, cr_list, apprv_list, max_pop, num_children, mutation_rate,cross_over_rate, population, step_sizes, generateNewCombo = initialize_pop(nodes, generateNewCombo)

    while generateNewCombo:
        #Reset the global variables
        best_chromosome_overall = None
        best_nodes = nodes 
        best_distance = bestDistance

        find_and_set_best_chromosome_nodes(0,[],best_chromosome_overall) # Generation 0
        
        for i in range(num_gens):
            print("GENERATION: ",i+1)

            #Produce new population's genes: Simulate Mating
                #Introduce Mutation
                #Introduce Cross-over
            child_chromosomes = mating_generation(population, num_children, mutation_rate, cross_over_rate, temp_list, cr_list, apprv_list, step_sizes)
            
            #Conver the list of child chromosomes into child objects (Individuals)
            childObjs = getChildIndividuals(child_chromosomes)
            
            #Compute fitness score for each individual
            parent_chromosomes = get_chromosomes(population)
            fitness_scores_parents,fitness_scores_children = compute_fitness_scores_new(population, childObjs)

            #Convert the child objects back into a list of child chromosomes
            child_chromosomes = get_chromosomes(childObjs)

            child_chromosomes = mating_generation(population, num_children, mutation_rate, cross_over_rate, temp_list, cr_list, apprv_list, step_sizes)
            #Perform 'selection' by incorporating diversity to determine next generation
            population = selection(parent_chromosomes,child_chromosomes,max_pop,fitness_scores_parents,fitness_scores_children)

            best_chromosome_overall = find_and_set_best_chromosome_nodes(i+1,population,best_chromosome_overall) 
            
        #Print the best chromosome to a file
        f = open("GeAnResults127.txt", "a")
        f.write(str(generateNewCombo) + ", " + str(calculate_tour_distance(best_chromosome_overall[3])) + ", " + str(best_chromosome_overall[:3]) + "\n")
        f.close()

        #print(str(counter))
        #counter += 1
        
        #Initialize Population, set all parameters: pop max size, num_children, mutation rate, cross-over rate for the next generation
        temp_list, cr_list, apprv_list, max_pop, num_children, mutation_rate,cross_over_rate, population, step_sizes, generateNewCombo = initialize_pop(nodes, generateNewCombo)
    
    #Return something to the GUI
    return best_chromosome_overall[3], calculate_tour_distance(best_chromosome_overall[3])

if __name__ == '__main__':

    run_generations()

    goal_temp = 210
    goal_cr = 0.925
    goal_ar = 0.75

    print("Goal Parameters: ",(goal_temp,goal_cr,goal_ar))