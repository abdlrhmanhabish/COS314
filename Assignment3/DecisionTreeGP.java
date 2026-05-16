import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.Scanner;

public class DecisionTreeGP {

    private static String trainFilePath = "Breast_train.csv";
    private static String testFilePath = "Breast_test.csv";
    private static String modelFilePath = "decision_tree_gp.model";

    private static final int POPULATION_SIZE = 200;
    private static final int MAX_GENERATIONS = 100;
    private static final int INITIAL_TREE_DEPTH = 4;
    private static final int MAX_OFFSPRING_DEPTH = 6;
    private static final int TOURNAMENT_SIZE = 5;
    private static final double CROSSOVER_RATE = 0.90;
    private static final double MUTATION_RATE = 0.10;
    private static final int ELITE_COUNT = 1;
    private static final int POSITIVE_CLASS = 1;

    private static class Instance implements Serializable {
        final int label;
        final int[] features;

        Instance(int label, int[] features) {
            this.label = label;
            this.features = features;
        }
    }

    private static class Dataset implements Serializable {
        final String[] featureNames;
        final List<Instance> instances;
        final int featureCount;
        final int[] minValues;
        final int[] maxValues;
        final Set<Integer>[] observedValues;

        Dataset(String[] featureNames, List<Instance> instances, int featureCount, int[] minValues, int[] maxValues,
                Set<Integer>[] observedValues) {
            this.featureNames = featureNames;
            this.instances = instances;
            this.featureCount = featureCount;
            this.minValues = minValues;
            this.maxValues = maxValues;
            this.observedValues = observedValues;
        }
    }

    private interface Node extends Serializable {
        int predict(Instance instance);

        Node deepCopy();

        int depth();

        int size();

        String pretty(String indent, String[] featureNames);
    }

    private static class LeafNode implements Node {
        int prediction;

        LeafNode(int prediction) {
            this.prediction = prediction;
        }

        @Override
        public int predict(Instance instance) {
            return prediction;
        }

        @Override
        public Node deepCopy() {
            return new LeafNode(prediction);
        }

        @Override
        public int depth() {
            return 1;
        }

        @Override
        public int size() {
            return 1;
        }

        @Override
        public String pretty(String indent, String[] featureNames) {
            return indent + "The prediction is:  " + prediction + "\n";
        }
    }

    private static class DecisionNode implements Node {
        int featureIndex;
        int threshold;
        Node left;
        Node right;

        // Creates a decision split node.
        DecisionNode(int featureIndex, int threshold, Node left, Node right) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }

        @Override
        public int predict(Instance instance) {
            if (instance.features[featureIndex] <= threshold)
                return left.predict(instance);
            return right.predict(instance);
        }

        @Override
        public Node deepCopy() {
            return new DecisionNode(featureIndex, threshold, left.deepCopy(), right.deepCopy());
        }

        @Override
        public int depth() {
            return 1 + Math.max(left.depth(), right.depth());
        }

        @Override
        public int size() {
            return 1 + left.size() + right.size();
        }

        @Override
        // print function to make the tree
        public String pretty(String indent, String[] featureNames) {
            StringBuilder builder = new StringBuilder();
            builder.append(indent).append("if ").append(featureNames[featureIndex])
                    .append(" <= ").append(threshold).append("\n");
            builder.append(left.pretty(indent + "  ", featureNames));
            builder.append(indent).append("else\n");
            builder.append(right.pretty(indent + "  ", featureNames));
            return builder.toString();
        }
    }

    private static class Individual implements Serializable {
        Node root;
        double fitness;
        double accuracy;

        Individual(Node root) {
            this.root = root;
        }

        // Copies the individual and its tree.
        Individual deepCopy() {
            Individual copy = new Individual(root.deepCopy());
            copy.fitness = fitness;
            copy.accuracy = accuracy;
            return copy;
        }
    }

    private static class Metrics {
        double accuracy;
        double precision;
        double recall;
        double fMeasure;
        int[][] confusion;
    }

    private static class NodePath {
        final Node node;
        final List<Boolean> path;

        NodePath(Node node, List<Boolean> path) {
            this.node = node;
            this.path = path;
        }
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter mode (train/test): ");
        String mode = scanner.nextLine().trim().toLowerCase(Locale.ROOT);

        if ("train".equals(mode)) {
            System.out.print("Enter seed value: ");
            long seed = parseSeed(scanner.nextLine());
            
            System.out.print("Enter training filepath (e.g., Breast_train.csv): ");
            String inputTrain = scanner.nextLine().trim();
            if (!inputTrain.isEmpty()) trainFilePath = inputTrain;
            
            System.out.print("Enter test filepath (e.g., Breast_test.csv): ");
            String inputTest = scanner.nextLine().trim();
            if (!inputTest.isEmpty()) testFilePath = inputTest;

            runTraining(seed);
        } else if ("test".equals(mode)) {
            System.out.print("Enter test filepath (e.g., Breast_test.csv): ");
            String inputTest = scanner.nextLine().trim();
            if (!inputTest.isEmpty()) testFilePath = inputTest;
            
            System.out.print("Enter model filepath (e.g., decision_tree_gp.model): ");
            String inputModel = scanner.nextLine().trim();
            if (!inputModel.isEmpty()) modelFilePath = inputModel;

            runTesting();
        } else {
            System.out.println("Invalid mode. Please use 'train' or 'test'.");
        }
        scanner.close();
    }

    // Parses the random seed
    private static long parseSeed(String value) {
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException ex) {
            return 45L;
        }
    }

    // Runs the training part, population initialisation and evolution etc.
    private static void runTraining(long seed) throws Exception {
        File trainFile = resolveExistingFile(trainFilePath);
        File testFile = resolveExistingFile(testFilePath);
        if (trainFile == null)
            throw new IOException("Training file not found: " + trainFilePath);

        Dataset train = loadDataset(trainFile);
        Dataset test = testFile != null ? loadDataset(testFile) : null;
        Random random = new Random(seed);

        List<Individual> population = initializePopulation(train, random);
        Individual globalBest = null;

        long start = System.nanoTime();
        for (int generation = 1; generation <= MAX_GENERATIONS; generation++) {
            evaluatePopulation(population, train);
            population.sort(Comparator.comparingDouble((Individual individual) -> individual.fitness).reversed());

            Individual bestOfGeneration = population.get(0).deepCopy();
            if (globalBest == null || bestOfGeneration.fitness > globalBest.fitness)
                globalBest = bestOfGeneration.deepCopy();

                System.out.println("Generation " + generation);
                System.out.printf(Locale.US, "  Best Individual: accuracy=%.4f, fitness=%.4f%n", 
                    bestOfGeneration.accuracy, bestOfGeneration.fitness);
                System.out.println("  Tree Structure:");
                System.out.println(bestOfGeneration.root.pretty("    ", train.featureNames));
                System.out.println();

            if (generation < MAX_GENERATIONS)
                population = nextGeneration(population, train, random);
        }
        long end = System.nanoTime();

        saveModel(globalBest);

        Metrics trainMetrics = evaluate(globalBest, train);
        Metrics testMetrics = test != null ? evaluate(globalBest, test) : null;
        double runtimeSeconds = (end - start) / 1_000_000_000.0;

        System.out.printf(Locale.US, "Training accuracy: %.4f%%%n", trainMetrics.accuracy * 100.0);
        if (testMetrics != null) {
            System.out.printf(Locale.US, "Test accuracy: %.4f%%%n", testMetrics.accuracy * 100.0);
            System.out.printf(Locale.US, "F-measure: %.4f%n", testMetrics.fMeasure);
        }
        System.out.printf(Locale.US, "Runtime: %.3f s%n", runtimeSeconds);
    }

    // Runs the testing wpart, loading the model and evaluating on the test set
    private static void runTesting() throws Exception {
        File modelFile = resolveExistingFile(modelFilePath);
        File testFile = resolveExistingFile(testFilePath);
        if (modelFile == null)
            throw new IOException("Model file not found: " + modelFilePath + ". Run training first.");
        if (testFile == null)
            throw new IOException("Test file not found: " + testFilePath);

        Individual model = loadModel(modelFile);
        Dataset test = loadDataset(testFile);
        Metrics metrics = evaluate(model, test);

        System.out.printf(Locale.US, "Test accuracy: %.4f%%%n", metrics.accuracy * 100.0);
        System.out.printf(Locale.US, "F-measure: %.4f%n", metrics.fMeasure);
        printConfusionMatrix(metrics.confusion);
    }

    // makes the initial population
    private static List<Individual> initializePopulation(Dataset dataset, Random random) {
        List<Individual> population = new ArrayList<>(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            boolean full = i % 2 == 0;
            int targetDepth = 2 + random.nextInt(Math.max(1, INITIAL_TREE_DEPTH - 1));
            population.add(new Individual(generateTree(dataset, random, targetDepth, 1, full)));
        }
        return population;
    }

    // Generates a random tree
    private static Node generateTree(Dataset dataset, Random random, int maxDepth, int currentDepth, boolean full) {
        if (currentDepth >= maxDepth)
            return new LeafNode(random.nextBoolean() ? POSITIVE_CLASS : 0);
        if (!full && currentDepth > 1 && random.nextDouble() < 0.35)
            return new LeafNode(random.nextBoolean() ? POSITIVE_CLASS : 0);

        int featureIndex = random.nextInt(dataset.featureCount);
        int threshold = randomThresholdForFeature(dataset, featureIndex, random);
        Node left = generateTree(dataset, random, maxDepth, currentDepth + 1, full);
        Node right = generateTree(dataset, random, maxDepth, currentDepth + 1, full);
        return new DecisionNode(featureIndex, threshold, left, right);
    }

    // Selects a threshold for one feature
    private static int randomThresholdForFeature(Dataset dataset, int featureIndex, Random random) {
        List<Integer> values = new ArrayList<>(dataset.observedValues[featureIndex]);
        Collections.sort(values);

        if (values.isEmpty())
            return 0;
        if (values.size() == 1)
            return values.get(0);

        return values.get(random.nextInt(values.size() - 1));
    }

    // gets the fitness of the population
    private static void evaluatePopulation(List<Individual> population, Dataset dataset) {
        for (Individual individual : population) {
            Metrics metrics = evaluate(individual, dataset);
            individual.fitness = metrics.accuracy;
            individual.accuracy = metrics.accuracy;
        }
    }

    // Creates the next generation
    private static List<Individual> nextGeneration(List<Individual> population, Dataset dataset, Random random) {
        List<Individual> next = new ArrayList<>(POPULATION_SIZE);
        for (int i = 0; i < ELITE_COUNT; i++)
            next.add(population.get(i).deepCopy());

        while (next.size() < POPULATION_SIZE) {
            Individual parent1 = tournamentSelect(population, random);
            Individual child;

            if (random.nextDouble() < CROSSOVER_RATE) {
                Individual parent2 = tournamentSelect(population, random);
                child = crossover(parent1, parent2, random);
            } else {
                child = parent1.deepCopy();
            }

            if (random.nextDouble() < MUTATION_RATE)
                mutate(child, dataset, random);
            if (child.root.depth() > MAX_OFFSPRING_DEPTH)
                child.root = trimToDepth(child.root, MAX_OFFSPRING_DEPTH);

            next.add(child);
        }

        return next;
    }

    // we use tournamentselection for selection method
    private static Individual tournamentSelect(List<Individual> population, Random random) {
        Individual best = null;
        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            Individual candidate = population.get(random.nextInt(population.size()));
            if (best == null || candidate.fitness > best.fitness)
                best = candidate;
        }
        return best.deepCopy();
    }

    // Crossover function,we use subtree crossover
    private static Individual crossover(Individual parent1, Individual parent2, Random random) {
        Node root = parent1.root.deepCopy();
        List<NodePath> firstNodes = collectNodes(root, new ArrayList<>(), new ArrayList<>());
        List<NodePath> secondNodes = collectNodes(parent2.root, new ArrayList<>(), new ArrayList<>());

        if (firstNodes.isEmpty() || secondNodes.isEmpty())
            return parent1.deepCopy();

        NodePath firstTarget = firstNodes.get(random.nextInt(firstNodes.size()));
        NodePath secondTarget = secondNodes.get(random.nextInt(secondNodes.size()));
        Node replacement = secondTarget.node.deepCopy();

        if (firstTarget.path.isEmpty())
            root = replacement;
        else
            root = replaceAtPath(root, firstTarget.path, replacement);

        return new Individual(root);
    }

    // get all nodes
    private static List<NodePath> collectNodes(Node node, List<Boolean> path, List<NodePath> result) {
        result.add(new NodePath(node, new ArrayList<>(path)));
        if (node instanceof DecisionNode) {
            DecisionNode decision = (DecisionNode) node;
            path.add(Boolean.TRUE);
            collectNodes(decision.left, path, result);
            path.remove(path.size() - 1);
            path.add(Boolean.FALSE);
            collectNodes(decision.right, path, result);
            path.remove(path.size() - 1);
        }
        return result;
    }

    // helper to replaces a subtree
    private static Node replaceAtPath(Node current, List<Boolean> path, Node replacement) {
        if (path.isEmpty())
            return replacement;
        if (!(current instanceof DecisionNode))
            return current.deepCopy();

        DecisionNode decision = (DecisionNode) current;
        boolean goLeft = path.get(0);
        List<Boolean> remainder = path.subList(1, path.size());
        Node newLeft = decision.left;
        Node newRight = decision.right;

        if (goLeft)
            newLeft = replaceAtPath(decision.left, remainder, replacement);
        else
            newRight = replaceAtPath(decision.right, remainder, replacement);

        return new DecisionNode(decision.featureIndex, decision.threshold, newLeft, newRight);
    }

    // mutation function,
    private static void mutate(Individual individual, Dataset dataset, Random random) {
        List<NodePath> nodes = collectNodes(individual.root, new ArrayList<>(), new ArrayList<>());
        if (nodes.isEmpty())
            return;

        NodePath target = nodes.get(random.nextInt(nodes.size()));
        Node mutated;

        if (target.node instanceof LeafNode) {
            LeafNode leaf = (LeafNode) target.node.deepCopy();
            leaf.prediction = leaf.prediction == POSITIVE_CLASS ? 0 : POSITIVE_CLASS;
            mutated = leaf;
        } else {
            DecisionNode decision = (DecisionNode) target.node.deepCopy();
            if (random.nextBoolean())
                decision.featureIndex = random.nextInt(dataset.featureCount);
            else
                decision.threshold = randomThresholdForFeature(dataset, decision.featureIndex, random);

            mutated = decision;
        }

        if (target.path.isEmpty())
            individual.root = mutated;
        else
            individual.root = replaceAtPath(individual.root, target.path, mutated);
    }

    private static Node trimToDepth(Node node, int maxDepth) {
        return trimToDepth(node, maxDepth, 1);
    }

    private static Node trimToDepth(Node node, int maxDepth, int currentDepth) {
        if (node instanceof LeafNode)
            return node.deepCopy();
        if (currentDepth >= maxDepth)
            return new LeafNode(majorityPrediction(node));

        DecisionNode decision = (DecisionNode) node;
        Node left = trimToDepth(decision.left, maxDepth, currentDepth + 1);
        Node right = trimToDepth(decision.right, maxDepth, currentDepth + 1);
        return new DecisionNode(decision.featureIndex, decision.threshold, left, right);
    }

    // we select the majority class among the leaf nodes for prediction
    private static int majorityPrediction(Node node) {
        List<Integer> predictions = new ArrayList<>();
        gatherLeafPredictions(node, predictions);
        int positive = 0;
        for (int prediction : predictions) {
            if (prediction == POSITIVE_CLASS)
                positive++;
        }
        return positive * 2 >= Math.max(1, predictions.size()) ? POSITIVE_CLASS : 0;
    }

    // gets all leaf predictions in the subtree
    private static void gatherLeafPredictions(Node node, List<Integer> predictions) {
        if (node instanceof LeafNode) {
            predictions.add(((LeafNode) node).prediction);
            return;
        }
        DecisionNode decision = (DecisionNode) node;
        gatherLeafPredictions(decision.left, predictions);
        gatherLeafPredictions(decision.right, predictions);
    }

    // Evaluates accuracy and classification metrics.
    private static Metrics evaluate(Individual individual, Dataset dataset) {
        Metrics metrics = new Metrics();
        metrics.confusion = new int[2][2];

        int correct = 0;
        for (Instance instance : dataset.instances) {
            int prediction = individual.root.predict(instance);
            int actual = instance.label;
            if (prediction == actual)
                correct++;
            int actualIndex = actual == POSITIVE_CLASS ? 1 : 0;
            int predictedIndex = prediction == POSITIVE_CLASS ? 1 : 0;
            metrics.confusion[actualIndex][predictedIndex]++;
        }

        metrics.accuracy = dataset.instances.isEmpty() ? 0.0 : (double) correct / dataset.instances.size();
        int tp = metrics.confusion[1][1];
        int fp = metrics.confusion[0][1];
        int fn = metrics.confusion[1][0];
        metrics.precision = tp + fp == 0 ? 0.0 : (double) tp / (tp + fp);
        metrics.recall = tp + fn == 0 ? 0.0 : (double) tp / (tp + fn);
        metrics.fMeasure = metrics.precision + metrics.recall == 0.0 ? 0.0
                : (2.0 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
        return metrics;
    }

    private static void printConfusionMatrix(int[][] confusion) {
        System.out.println("Confusion matrix [actual x predicted]:");
        System.out.println("            predicted 0   predicted 1");
        System.out.printf(Locale.US, "actual 0    %10d   %10d%n", confusion[0][0], confusion[0][1]);
        System.out.printf(Locale.US, "actual 1    %10d   %10d%n", confusion[1][0], confusion[1][1]);
    }

    private static Dataset loadDataset(File file) throws IOException {
        List<String> rawLines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
        List<String> lines = normalizeCsvLines(rawLines);
        if (lines.isEmpty()) {
            throw new IOException("Empty dataset: " + file.getName());
        }

        String[] headerParts = splitCsvLine(lines.get(0));
        if (headerParts.length < 2 || isNumeric(headerParts[0])) {
            throw new IOException("Unexpected header format in file: " + file.getName());
        }

        String[] featureNames = Arrays.copyOfRange(headerParts, 1, headerParts.length);
        int featureCount = featureNames.length;
        List<Instance> instances = new ArrayList<>();
        int[] minValues = new int[featureCount];
        int[] maxValues = new int[featureCount];
        Arrays.fill(minValues, Integer.MAX_VALUE);
        Arrays.fill(maxValues, Integer.MIN_VALUE);

        @SuppressWarnings("unchecked")
        Set<Integer>[] observedValues = new Set[featureCount];
        for (int i = 0; i < featureCount; i++) {
            observedValues[i] = new HashSet<>();
        }

        for (int i = 1; i < lines.size(); i++) {
            String[] parts = splitCsvLine(lines.get(i));
            if (parts.length < featureCount + 1)
                continue;

            int label = parseIntSafe(parts[0]);
            int[] features = new int[featureCount];
            for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
                int value = parseIntSafe(parts[featureIndex + 1]);
                features[featureIndex] = value;
                observedValues[featureIndex].add(value);
                minValues[featureIndex] = Math.min(minValues[featureIndex], value);
                maxValues[featureIndex] = Math.max(maxValues[featureIndex], value);
            }
            instances.add(new Instance(label, features));
        }

        return new Dataset(featureNames, instances, featureCount, minValues, maxValues, observedValues);
    }

    // Normalizes CSV lines before parsing.
    private static List<String> normalizeCsvLines(List<String> rawLines) {
        List<String> lines = new ArrayList<>();
        for (String rawLine : rawLines) {
            if (rawLine == null) {
                continue;
            }
            String line = rawLine.trim();
            if (line.isEmpty()) {
                continue;
            }
            if (!lines.isEmpty() && line.startsWith(",")) {
                int lastIndex = lines.size() - 1;
                lines.set(lastIndex, lines.get(lastIndex) + line);
            } else {
                lines.add(line);
            }
        }
        return lines;
    }

    private static String[] splitCsvLine(String line) {
        return line.split(",", -1);
    }

    private static boolean isNumeric(String value) {
        try {
            Integer.parseInt(value.trim());
            return true;
        } catch (NumberFormatException ex) {
            return false;
        }
    }

    // Parses an integer safely
    private static int parseIntSafe(String value) {
        String trimmed = value == null ? "" : value.trim();
        if (trimmed.isEmpty()) {
            return 0;
        }
        try {
            return Integer.parseInt(trimmed);
        } catch (NumberFormatException ex) {
            return 0;
        }
    }

    private static File resolveExistingFile(String... candidates) {
        for (String candidate : candidates) {
            if (candidate == null)
                continue;
            File file = new File(candidate);
            if (file.isFile())
                return file;
        }
        return null;
    }

    private static void saveModel(Individual best) throws IOException {
        File modelFile = new File(modelFilePath);
        File parent = modelFile.getParentFile();
        if (parent != null && !parent.exists())
            parent.mkdirs();
        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(modelFile))) {
            output.writeObject(best);
        }
    }

    private static Individual loadModel(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream input = new ObjectInputStream(new FileInputStream(file))) {
            return (Individual) input.readObject();
        }
    }
}