import random
from deap import base, creator, tools

# Функція оцінювання
def eval_func(individual):
    target_sum = 45
    return len(individual) - abs(sum(individual) - target_sum),
# Створіть панель інструментів із правильними параметрами def

def create_toolbox(num_bits):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list,fitness=creator.FitnessMax)

    # Ініціалізація панелі інструментів
    toolbox = base.Toolbox()
    # Створення атрибутів
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Ініціалізація структур
    toolbox.register("individual", tools.initRepeat,
    creator.Individual,toolbox.attr_bool, num_bits)
    # Визначте сукупність як список індивідуумів
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Реєстрація оператора оцінки
    toolbox.register("evaluate", eval_func)
    # Реєстрація оператора кросовера
    toolbox.register("mate", tools.cxTwoPoint)
    # Реєстрація оператора мутації
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # Оператор по відбору особин для розмноження
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

if __name__ == "__main__":
    # Визначити кількість бітів
    num_bits = 75
    # Створіть панель інструментів,
    # використовуючи наведений вище параметр
    toolbox = create_toolbox(num_bits)
    # Задати генератор випадкових чисел
    random.seed(7)
    # Створіть початкову популяцію з 500 особин
    population = toolbox.population(n=500)
    # Визначте ймовірність схрещування та мутації
    probab_crossing, probab_mutating = 0.5, 0.2
    # Визначте кількість поколінь
    num_generations = 60

    print('\nStarting the evolution process')
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    print('\nEvaluated', len(population), 'individuals')

    # Iterate through generations
    for g in range(num_generations):
        print("\n===== Generation", g)
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Cross two individuals
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)
                # "Forget" the fitness values of the children
                del child1.fitness.values
                del child2.fitness.values
        # Apply mutation
        for mutant in offspring:
            # Mutate an individual
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print('Evaluated', len(invalid_ind), 'individuals')

        # The population is entirely replaced by the offspring
        population[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        print('Min =', min(fits), ', Max =', max(fits))
        print('Average =', round(mean, 2), ', Standard deviation =', round(std, 2))

    print("\n==== End of evolution")

    best_ind = tools.selBest(population, 1)[0]
    print('\nBest individual:\n', best_ind)
    print('\nNumber of ones:', sum(best_ind))