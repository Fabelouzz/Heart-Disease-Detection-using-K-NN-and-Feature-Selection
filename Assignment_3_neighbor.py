from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from multiprocessing import Process, Manager
import random
import itertools

# The dataset is uploaded
f = open("Assignment 3 medical_dataset.DATA")
dataset_X = []
dataset_y = []
line = " "
while line != "":
    line = f.readline()
    line = line[:-1]
    if line != "":
        line = line.split(",")
        floatList = []
        for i in range(len(line)):
            if i < len(line)-1:
                floatList.append(float(line[i]))
            else:
                value = float(line[i])
                if value == 0:
                    dataset_y.append(0)
                else:
                    dataset_y.append(1)
        dataset_X.append(floatList)
f.close()

# The dataset is splited into training and test.
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size = 0.25, random_state = 0)

# The dataset is scaled
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
fitness_cache = {} # key: solution tuple, value: fitness
# The model is created
model = KNeighborsClassifier(n_neighbors = 3)

# Function that calculates the fitness of a solution
def calculateFitness(solution):
    solution_tuple = tuple(solution)
    # If the solution is already in the cache, return the cached fitness
    if solution_tuple in fitness_cache:
        return fitness_cache[solution_tuple]



    fitness = 0

    # The features are selected according to solution
    X_train_Fea_selc = []
    X_test_Fea_selc = []
    for example in X_train:
        X_train_Fea_selc.append([a*b for a,b in zip(example,solution)])
    for example in X_test:
        X_test_Fea_selc.append([a*b for a,b in zip(example,solution)])

    model.fit(X_train_Fea_selc, y_train)

    # We predict the test cases
    y_pred = model.predict(X_test_Fea_selc)

    # We calculate the Accuracy
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0] # True positives
    FP = cm[0][1] # False positives
    TN = cm[1][1] # True negatives
    FN = cm[1][0] # False negatives

    fitness = (TP + TN) / (TP + TN + FP + FN)
    fitness_cache[solution_tuple] = round(fitness * 100, 2)
    return round(fitness *100,2)

MAX_FITNESS_CALCULATIONS = 5000

FITNESS_CALCULATIONS_COUNTER = 0

def random_restart_variable_neighbor_search():

    global FITNESS_CALCULATIONS_COUNTER
    bestSolutionFitness = 0
    bestSolution = []


    for _ in range(10):  # Run 10 times
        FITNESS_CALCULATIONS_COUNTER = 0
        # Random-restart variable neighbor search
        # This line generates a random solution by choosing a random bit (0 or 1) for each feature in the dataset.


        # manage the flipping of multiple bits in the solution vector to generate different neighbors.
        #  the algorithm keeps searching through the neighborhoods of better solutions found, going from smaller neighborhoods
        # to larger ones and back to smaller ones each time a better solution is found
        # Without this loop, the algorithm would only flip k bits once for each value of k from 1 to n, and then stop.
        while FITNESS_CALCULATIONS_COUNTER < MAX_FITNESS_CALCULATIONS:
            currentSolution = [random.choice([0, 1]) for _ in range(len(dataset_X[0]))]
            currentSolutionFitness = calculateFitness(currentSolution)
            FITNESS_CALCULATIONS_COUNTER += 1
            k = 1
            # manage the process of flipping k bits in the solution to generate a neighbor.
            while k <= len(currentSolution) and FITNESS_CALCULATIONS_COUNTER < MAX_FITNESS_CALCULATIONS:
                bestNeighbor = None
                # any neighbor with a fitness higher than that of the current solution will be considered better.
                bestNeighborFitness = currentSolutionFitness


                # creates an iterable of all indices in the current solution, k is the number of indices we want to select to flip.
                # generates all possible combinations of k indices to flip in the current solution
                # produces all combinations of k elements from the input.
                for indices_to_flip in itertools.combinations(range(len(currentSolution)), k):
                    neighbor = currentSolution.copy()
                    #  generates all subsets of these indices that have k elements.
                    #  Each subset represents a unique set of k indices to flip in the solution to generate a neighbor.
                    # for each combination of indices to flip, flip the bits at those indices
                    # in the current solution to generate a neighbor.
                    for index in indices_to_flip:
                        neighbor[index] = 1 - neighbor[index]

                    # Calculate fitness of the neighbor
                    neighborFitness = calculateFitness(neighbor)
                    FITNESS_CALCULATIONS_COUNTER += 1

                    # If the neighbor's fitness is better, update the bestNeighbor
                    if neighborFitness > bestNeighborFitness:
                        bestNeighbor = neighbor
                        bestNeighborFitness = neighborFitness
                    if FITNESS_CALCULATIONS_COUNTER >= MAX_FITNESS_CALCULATIONS:
                        break

                # If the best neighbor's fitness is better, update the current solution and reset k
                if bestNeighborFitness > currentSolutionFitness:
                    currentSolution = bestNeighbor
                    currentSolutionFitness = bestNeighborFitness
                    #  is reset to 1 because the search now begins in the neighborhood of a new solution,
                    #  starting with the neighbors that differ by only one bit.
                    k = 1 # reference point
                # incremented by 1 to expand the neighborhood by considering neighbors that differ by one additional bit
                else:
                    # This continues until a better solution is found or k equals the size of the solution,
                    # at which point all possible neighbors have been considered.
                    k += 1

                # Update the best solution if the current solution is better
                if currentSolutionFitness > bestSolutionFitness:
                    bestSolution = currentSolution
                    bestSolutionFitness = currentSolutionFitness
                    print("Best solution fitness ( ", FITNESS_CALCULATIONS_COUNTER, "/", MAX_FITNESS_CALCULATIONS, "):",
                          bestSolutionFitness)

                # Randomly restart the search if the best neighbor's fitness is not better than the current solution's
                """
                if bestNeighborFitness <= currentSolutionFitness:
                    currentSolution = [random.choice([0, 1]) for _ in range(len(dataset_X[0]))]
                    currentSolutionFitness = calculateFitness(currentSolution)
                    FITNESS_CALCULATIONS_COUNTER += 1
                """
            if FITNESS_CALCULATIONS_COUNTER >= MAX_FITNESS_CALCULATIONS:
                break

        return bestSolution, bestSolutionFitness


bestSolution_VNS, bestSolutionFitness_VNS = random_restart_variable_neighbor_search()
#print(fitness_cache)
print("Variable Neighbor Search Best Solution:", bestSolution_VNS)
print("Variable Neighbor Search Best Solution Fitness:", bestSolutionFitness_VNS)
