
import java.io.File;
import java.util.*;

public class Knapsack {

    // for holding problem data
    static class Data {

        int n;
        double capacity;
        double[] values;
        double[] weights;
    }

    public static void main(String args[]) {
        Scanner scanner = new Scanner(System.in);
        scanner.useLocale(Locale.US); //fix bc of decimal point issues with f5
        System.out.println("Enter a seed value: ");
        long seed;
        try {
            seed = scanner.nextLong();
            scanner.nextLine();
        } catch (Exception e) {
            System.out.println("Invalid input. Please enter a valid long integer for the seed value.");
            scanner.close();
            return;
        }

        File currentDir = new File(".");
        File[] files = currentDir.listFiles((dir, name) -> name.startsWith("f") || name.startsWith("knapP"));
        if (files == null || files.length == 0) {
            System.out.println("No input files found in the current directory.");
            scanner.close();
            return;
        }

        Arrays.sort(files); //making sure theyre actually in the correct order 

        // results table setup
        System.out.println("\n" + "=".repeat(85));
        System.out.printf("%-25s | %-10s | %-10s | %-15s | %-15s\n", "Problem Instance", "Algorithm", "Seed", "Best Solution", "Runtime (s)");
        System.out.println("-".repeat(85));

        for (File file : files) {
            Data data = loadData(file); // function that converts files into data objects
            if (data == null) {
                System.out.println("Failed to load instance from file: " + file.getName());
                continue;
            }

            long startILS = System.currentTimeMillis();
            double ILSResult = ILS(data, seed);
            long endILS = System.currentTimeMillis();
            double timeILS = (endILS - startILS) / 1000.0;
            System.out.printf("%-25s | %-10s | %-10d | %-15.2f | %-15.3f\n", file.getName(), "ILS", seed, ILSResult, timeILS);

            long startGA = System.currentTimeMillis();
            double GAResult = GA(data, seed);
            long endGA = System.currentTimeMillis();
            double timeGA = (endGA - startGA) / 1000.0;
            System.out.printf("%-25s | %-10s | %-10d | %-15.2f | %-15.3f\n","", "GA", seed, GAResult, timeGA);
            System.out.println("-".repeat(80));
        }
        scanner.close();
    }

    //stubbed out functions for algos
    static double ILS(Data data, long seed) {
        Random random = new Random(seed);
        int maxIterations = 100;
        int perturbationStrength = Math.min(2, data.n);
        int[] initialSolution = generateInitialSolution(data, random);
        int[] currentLocalOptimum = localSearch(initialSolution, data);
        double currentValue = calculateValue(data, currentLocalOptimum);
        double bestValue = currentValue;
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            int[] perturbed = perturbation(currentLocalOptimum, random, data, perturbationStrength, 50);
            int[] candidateLocalOptimum = localSearch(perturbed, data); 
            currentLocalOptimum = acceptanceCriterion(currentLocalOptimum, candidateLocalOptimum, data);
            currentValue = calculateValue(data, currentLocalOptimum);
            if (currentValue > bestValue) {
                bestValue = currentValue;
            }
        }
        return bestValue;
    }

    static int[] generateInitialSolution(Data data, Random random) {
        int[] solution = new int[data.n];
        for (int i = 0; i < data.n; i++) {
            if(random.nextBoolean()){
                solution[i] = 1;
            }
            else{
                solution[i] = 0;
            }
        }
        if (!isFeasible(data, solution)) {
            repair(solution, data, random);
        }
        return solution;
    }

    static int[] localSearch(int[] start, Data data) {
        int[] current = Arrays.copyOf(start, start.length);
        double currentValue = calculateValue(data, current);
        boolean improved = true;
        while (improved) {
            improved = false;
            int[] bestNeighbor = current;
            double bestNeighborValue = currentValue;
            for (int i = 0; i < data.n; i++) {
                int[] neighbor = Arrays.copyOf(current, current.length);
                neighbor[i] = 1 - neighbor[i];
                if (!isFeasible(data, neighbor)) {
                    continue;
                }
                double neighborValue = calculateValue(data, neighbor);
                if (neighborValue > bestNeighborValue) {
                    bestNeighbor = neighbor;
                    bestNeighborValue = neighborValue;
                }
            }
            if (bestNeighborValue > currentValue) {
                current = bestNeighbor;
                currentValue = bestNeighborValue;
                improved = true;
            }
        }
        return current;
    }

    static int[] perturbation(int[] base, Random random, Data data, int flips, int maxAttempts) {
        if (data.n == 0) {
            return Arrays.copyOf(base, base.length);
        }
        int actualFlips = Math.max(1, Math.min(flips, data.n));
        for (int attempt = 0; attempt < maxAttempts; attempt++) {
            int[] candidate = Arrays.copyOf(base, base.length);
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < data.n; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices, random);
            for (int i = 0; i < actualFlips; i++) {
                int index = indices.get(i);
                candidate[index] = 1 - candidate[index];
            }
            if (isFeasible(data, candidate)) {
                return candidate;
            }
        }
        return Arrays.copyOf(base, base.length);
    }

    static int[] acceptanceCriterion(int[] current, int[] candidate, Data data) {
        double currentValue = calculateValue(data, current);
        double candidateValue = calculateValue(data, candidate);
        if (candidateValue > currentValue) {
            return candidate;
        }
        return current;
    }

    static boolean isFeasible(Data data, int[] solution) {
        return calculateWeight(data, solution) <= data.capacity;
    }

    static double calculateWeight(Data data, int[] solution) {
        double totalWeight = 0;
        for (int i = 0; i < data.n; i++) {
            if (solution[i] == 1) {
                totalWeight += data.weights[i];
            }
        }
        return totalWeight;
    }

    static double calculateValue(Data data, int[] solution) {
        double totalValue = 0;
        for (int i = 0; i < data.n; i++) {
            if (solution[i] == 1) {
                totalValue += data.values[i];
            }
        }
        return totalValue;
    }

    static double GA(Data data, long seed) {
        Random rand = new Random(seed);
        int popSize = 100;
        int maxGenerations = 1000;
        double mutationRate = 1.0 / data.n;
        int tornamentSize = 5;

        int[][] population = new int[popSize][data.n];
        for (int i = 0; i < popSize; i++) {
            for (int j = 0; j < data.n; j++) {
                if (rand.nextBoolean()) {
                    population[i][j] = 1;
                } else {
                    population[i][j] = 0;
                }
            }
            repair(population[i], data, rand);
        }
        double bestGlobalFitness = 0;

        for (int gen = 0; gen < maxGenerations; gen++) {
            double[] fitnesses = new double[popSize];
            for (int i = 0; i < popSize; i++) {
                fitnesses[i] = evaluteFitness(population[i], data);
                if (fitnesses[i] > bestGlobalFitness) {
                    bestGlobalFitness = fitnesses[i];
                }
            }
            int[][] newPopulation = new int[popSize][data.n];

            for (int j = 0; j < popSize; j++) {
                int[] parent1 = tournamentSelection(population, fitnesses, tornamentSize, rand);
                int[] parent2 = tournamentSelection(population, fitnesses, tornamentSize, rand);
                int[] offspring = new int[data.n];
                for (int k = 0; k < data.n; k++) {
                    if (rand.nextBoolean()) {
                        offspring[k] = parent1[k];
                    } else {
                        offspring[k] = parent2[k];
                    }
                }
                if (rand.nextDouble() < mutationRate) {
                    applyPointMutation(offspring, rand);
                }

                repair(offspring, data, rand);
                newPopulation[j] = offspring;
            }
            population = newPopulation;
        }

        return bestGlobalFitness;
    }

    static void applyPointMutation(int[] individual, Random rand) {
        int targetIndex = rand.nextInt(individual.length);
        individual[targetIndex] = 1 - individual[targetIndex];
    }

    static int[] tournamentSelection(int[][] population, double[] fitnesses, int tournamentSize, Random rand) {
        int bestIndex = rand.nextInt(population.length);
        for (int i = 0; i < tournamentSize; i++) {
            int index = rand.nextInt(population.length);
            if (fitnesses[index] > fitnesses[bestIndex]) {
                bestIndex = index;
            }
        }
        return population[bestIndex];
    }

    static double evaluteFitness(int[] individual, Data data) {
        double totalWeight = 0;
        double totalValue = 0;
        for (int i = 0; i < data.n; i++) {
            if (individual[i] == 1) {
                totalWeight += data.weights[i];
                totalValue += data.values[i];
            }
        }
        if (totalWeight > data.capacity) {
            return 0;
        }
        return totalValue;
    }

    static void repair(int[] individual, Data data, Random rand) {
        double totalWeight = 0;
        for (int i = 0; i < data.n; i++) {
            if (individual[i] == 1) {
                totalWeight += data.weights[i];
            }
        }
        while (totalWeight > data.capacity) {
            int randomIndex = rand.nextInt(data.n);
            if (individual[randomIndex] == 1) {
                individual[randomIndex] = 0;
                totalWeight -= data.weights[randomIndex];
            }
        }
    }

    // converts file data to data objs
    static Data loadData(File file) {
        try (Scanner scanner = new Scanner(file)) {
            scanner.useLocale(Locale.US); // same fix in main
            Data data = new Data();
            data.n = scanner.nextInt();
            data.capacity = scanner.nextDouble();
            data.values = new double[data.n];
            data.weights = new double[data.n];
            for (int i = 0; i < data.n; i++) {
                data.values[i] = scanner.nextDouble();
                data.weights[i] = scanner.nextDouble();
            }
            scanner.close();
            return data;
        } catch (Exception e) {
            System.err.println("Error reading " + file.getName() + ": " + e.getMessage());
            return null;
        }
    }
}
