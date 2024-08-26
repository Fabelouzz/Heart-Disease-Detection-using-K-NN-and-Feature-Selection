from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import random

# The dataset is uploaded
f = open("Assignment 3 medical_dataset.DATA") # The dataset is uploaded
# to store the feature vectors and target values, respectively.
dataset_X = []
dataset_y = []
line = " "
while line != "":
    #
    line = f.readline()
    #  The last character is removed
    line = line[:-1]
    # The line is split into a list
    if line != "":
        line = line.split(",")
        floatList = []
        for i in range(len(line)):
            if i < len(line)-1:
                floatList.append(float(line[i]))
                # all the feature values are converted to floats and stored in floatList
            else:
                value = float(line[i])
                if value == 0:
                    #This step classifies the target variable into two classes, 0 and 1.
                    # the target variable is stored in the dataset_y list, and it is a binary variable
                    # indicating the presence (1) or absence (0) of a particular medical condition in a patient.
                    dataset_y.append(0)
                else:
                    dataset_y.append(1)
        # Appending the feature vector to dataset_X:
        # Each feature vector in dataset_X is a
        # list of 13 numerical values representing the medical characteristics of a patient
        dataset_X.append(floatList)
f.close()

# The dataset is splited into training and test.
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size = 0.25, random_state = 0)

# The dataset is scaled
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# The model is created
model = KNeighborsClassifier(n_neighbors = 3)
fitness_cache = {} # key: solution tuple, value: fitness
# Function that calculates the fitness of a solution
def calculateFitness(solution):
    # Convert the solution to a tuple so it can be used as a dictionary key
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

def random_restart_hill_climbing():
    global FITNESS_CALCULATIONS_COUNTER # global variable to keep track of the number of fitness calculations
    # initialize the best solution fitness to be 0
    bestSolutionFitness = 0
    # initialize the best solution to be an empty list
    bestSolution = []

    for _ in range(10):  # Run the algorithm 10 times
        # Reset the fitness calculations counter every iteration
        FITNESS_CALCULATIONS_COUNTER = 0
        # Random-restart hill climbing
        # Generate a random solution with the same length as
        # the feature vectors
        currentSolution = [random.choice([0, 1]) for _ in range(len(dataset_X[0]))]
        # calculate the fitness of the current solution and increment the fitness calculations counter by 1
        currentSolutionFitness = calculateFitness(currentSolution)
        FITNESS_CALCULATIONS_COUNTER += 1
        #the main loop of the hill climning algorithm, each iteration is an attempt to find a better solution in the
        #neighborhood of the current solution
        while FITNESS_CALCULATIONS_COUNTER < MAX_FITNESS_CALCULATIONS:
            bestNeighbor = None
            bestNeighborFitness = currentSolutionFitness
            # ach iteration of this loop corresponds to generating a single neighbor.
            for i in range(len(currentSolution)):
                #when we modify neighbor, it doesn't affect the original currentSolution.
                neighbor = currentSolution.copy()
                # Flip the bit at index i, checking all possible neighbors in the current solution
                neighbor[i] = 1 - neighbor[i]

                # Calculate fitness of the current neighbor
                neighborFitness = calculateFitness(neighbor)
                FITNESS_CALCULATIONS_COUNTER += 1

                # If the neighbor's fitness is better, update the bestNeighbor
                if neighborFitness > bestNeighborFitness:
                    bestNeighbor = neighbor
                    bestNeighborFitness = neighborFitness
                if FITNESS_CALCULATIONS_COUNTER >= MAX_FITNESS_CALCULATIONS:
                    break



            # If the best neighbor's fitness is better, update the current solution
            if bestNeighborFitness > currentSolutionFitness:
                currentSolution = bestNeighbor
                currentSolutionFitness = bestNeighborFitness
            # Calculate fitness of the neighbor

            # Update the best solution if the current solution is better

            if currentSolutionFitness > bestSolutionFitness:
                bestSolution = currentSolution
                bestSolutionFitness = currentSolutionFitness
                print("Best solution fitness ( ", FITNESS_CALCULATIONS_COUNTER, "/", MAX_FITNESS_CALCULATIONS, "):", bestSolutionFitness)

            # If the current solution is at a local optimum, generate a new random solution
            if currentSolutionFitness > bestNeighborFitness:
                currentSolution = [random.choice([0, 1]) for _ in range(len(dataset_X[0]))]
                currentSolutionFitness = calculateFitness(currentSolution)
                FITNESS_CALCULATIONS_COUNTER += 1

            if FITNESS_CALCULATIONS_COUNTER >= MAX_FITNESS_CALCULATIONS:
                break

    return bestSolution, bestSolutionFitness

bestSolution_HC, bestSolutionFitness_HC = random_restart_hill_climbing()
#print(fitness_cache)
print("Hill Climbing Best Solution:", bestSolution_HC)
print("Hill Climbing Best Solution Fitness:", bestSolutionFitness_HC)
print(f"best solution found: used {FITNESS_CALCULATIONS_COUNTER} / {MAX_FITNESS_CALCULATIONS} calculations in 10 iterations and found the best solution to have the fitness accuarcy of: {bestSolutionFitness_HC} %")