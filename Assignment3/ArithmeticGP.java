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
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class ArithmeticGP {

    private static final String TRAIN_FILE = "Assignment3/Breast_train.csv";
    private static final String TEST_FILE = "Assignment3/Breast_test.csv";
    private static final String MODEL_FILE = "Assignment3/arithmetic_gp.model";

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

        Dataset(String[] featureNames, List<Instance> instances, int featureCount, int[] minValues, int[] maxValues) {
            this.featureNames = featureNames;
            this.instances = instances;
            this.featureCount = featureCount;
            this.minValues = minValues;
            this.maxValues = maxValues;
        }
    }

    private interface Node extends Serializable {
        double evaluate(Instance instance);

        Node deepCopy();

        int depth();

        int size();

        String pretty(String[] featureNames);
    }

    private enum Operator {
        ADD("+"),
        SUB("-"),
        MUL("*"),
        DIV("/");

        private final String symbol;

        Operator(String symbol) {
            this.symbol = symbol;
        }

        String symbol() {
            return symbol;
        }
    }

    private static class ConstantNode implements Node {
        double value;

        ConstantNode(double value) {
            this.value = value;
        }

        @Override
        public double evaluate(Instance instance) {
            return value;
        }

        @Override
        public Node deepCopy() {
            return new ConstantNode(value);
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
        public String pretty(String[] featureNames) {
            return String.format(Locale.US, "%.3f", value);
        }
    }

    private static class FeatureNode implements Node {
        int featureIndex;

        FeatureNode(int featureIndex) {
            this.featureIndex = featureIndex;
        }

        @Override
        public double evaluate(Instance instance) {
            return instance.features[featureIndex];
        }

        @Override
        public Node deepCopy() {
            return new FeatureNode(featureIndex);
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
        public String pretty(String[] featureNames) {
            return featureNames[featureIndex];
        }
    }

    private static class OperatorNode implements Node {
        Operator operator;
        Node left;
        Node right;

        OperatorNode(Operator operator, Node left, Node right) {
            this.operator = operator;
            this.left = left;
            this.right = right;
        }

        @Override
        public double evaluate(Instance instance) {
            double leftValue = left.evaluate(instance);
            double rightValue = right.evaluate(instance);
            switch (operator) {
                case ADD:
                    return leftValue + rightValue;
                case SUB:
                    return leftValue - rightValue;
                case MUL:
                    return leftValue * rightValue;
                case DIV:
                    return safeDivide(leftValue, rightValue);
                default:
                    return 0.0;
            }
        }

        @Override
        public Node deepCopy() {
            return new OperatorNode(operator, left.deepCopy(), right.deepCopy());
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
        public String pretty(String[] featureNames) {
            return "(" + left.pretty(featureNames) + " " + operator.symbol() + " " + right.pretty(featureNames) + ")";
        }
    }

    private static class Individual implements Serializable {
        Node root;
        double fitness;
        double accuracy;

        Individual(Node root) {
            this.root = root;
        }

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
        String mode = args.length > 0 ? args[0].trim().toLowerCase(Locale.ROOT) : "train";
        long seed = args.length > 1 ? parseSeed(args[1]) : 45L;

        if ("train".equals(mode)) {
            runTraining(seed);
            return;
        }
        if ("test".equals(mode)) {
            runTesting();
            return;
        }

        System.out.println("Usage:");
        System.out.println("  java ArithmeticGP train [seed]");
        System.out.println("  java ArithmeticGP test");
    }

    // Parse the seed argument with a safe fallback.
    private static long parseSeed(String value) {
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException ex) {
            return 45L;
        }
    }

    // Run the GP evolution loop and persist the best model.
    private static void runTraining(long seed) throws Exception {
        File trainFile = resolveExistingFile(TRAIN_FILE, "Breast_train.csv");
        File testFile = resolveExistingFile(TEST_FILE, "Breast_test.csv");
        if (trainFile == null)
            throw new IOException("Training file not found: " + TRAIN_FILE);

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
            if (globalBest == null || bestOfGeneration.fitness > globalBest.fitness) {
                globalBest = bestOfGeneration.deepCopy();
            }

            System.out.println("Generation " + generation);
            System.out.printf(Locale.US, "  Best accuracy=%.4f, fitness=%.4f%n", bestOfGeneration.accuracy,
                    bestOfGeneration.fitness);
            System.out.println("  Best expression: " + bestOfGeneration.root.pretty(train.featureNames));
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

    // Load the saved model and evaluate it on the test set.
    private static void runTesting() throws Exception {
        File modelFile = resolveExistingFile(MODEL_FILE);
        File testFile = resolveExistingFile(TEST_FILE, "Breast_test.csv");
        if (modelFile == null)
            throw new IOException("Model file not found: " + MODEL_FILE + ". Run training first.");
        if (testFile == null)
            throw new IOException("Test file not found: " + TEST_FILE);

        Individual model = loadModel(modelFile);
        Dataset test = loadDataset(testFile);
        Metrics metrics = evaluate(model, test);

        System.out.printf(Locale.US, "Test accuracy: %.4f%%%n", metrics.accuracy * 100.0);
        System.out.printf(Locale.US, "F-measure: %.4f%n", metrics.fMeasure);
        printConfusionMatrix(metrics.confusion);
    }

    // Ramped half-and-half initialization of the population.
    private static List<Individual> initializePopulation(Dataset dataset, Random random) {
        List<Individual> population = new ArrayList<>(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            boolean full = i % 2 == 0;
            int targetDepth = 2 + random.nextInt(Math.max(1, INITIAL_TREE_DEPTH - 1));
            population.add(new Individual(generateTree(dataset, random, targetDepth, 1, full)));
        }
        return population;
    }

    // build a random syntax tree using full/grow logic
    private static Node generateTree(Dataset dataset, Random random, int maxDepth, int currentDepth, boolean full) {
        if (currentDepth >= maxDepth)
            return randomTerminal(dataset, random);
        if (!full && currentDepth > 1 && random.nextDouble() < 0.35)
            return randomTerminal(dataset, random);

        Operator operator = randomOperator(random);
        Node left = generateTree(dataset, random, maxDepth, currentDepth + 1, full);
        Node right = generateTree(dataset, random, maxDepth, currentDepth + 1, full);
        return new OperatorNode(operator, left, right);
    }

    // Pick a terminal node (feature or constant).
    private static Node randomTerminal(Dataset dataset, Random random) {
        if (random.nextBoolean())
            return new FeatureNode(random.nextInt(dataset.featureCount));
        return new ConstantNode(randomConstant(dataset, random));
    }

    // Sample a constant using the observed feature ranges.
    private static double randomConstant(Dataset dataset, Random random) {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < dataset.featureCount; i++) {
            min = Math.min(min, dataset.minValues[i]);
            max = Math.max(max, dataset.maxValues[i]);
        }
        int range = Math.max(1, Math.max(Math.abs(min), Math.abs(max)));
        return (random.nextDouble() * 2.0 - 1.0) * range;
    }

    // Uniformly sample an arithmetic operator.
    private static Operator randomOperator(Random random) {
        Operator[] operators = Operator.values();
        return operators[random.nextInt(operators.length)];
    }

    // Evaluate fitness for every individual.
    private static void evaluatePopulation(List<Individual> population, Dataset dataset) {
        for (Individual individual : population) {
            Metrics metrics = evaluate(individual, dataset);
            individual.fitness = metrics.accuracy;
            individual.accuracy = metrics.accuracy;
        }
    }

    // Create the next generation using elitism, crossover, and mutation.
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
                child.root = trimToDepth(child.root, MAX_OFFSPRING_DEPTH, dataset, random);

            next.add(child);
        }

        return next;
    }

    // Tournament selection biased to higher fitness.
    private static Individual tournamentSelect(List<Individual> population, Random random) {
        Individual best = null;
        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            Individual candidate = population.get(random.nextInt(population.size()));
            if (best == null || candidate.fitness > best.fitness)
                best = candidate;
        }
        return best.deepCopy();
    }

    //subtree crossover between two parents.
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

    // Collect every node with its path for swap/mutation operations.
    private static List<NodePath> collectNodes(Node node, List<Boolean> path, List<NodePath> result) {
        result.add(new NodePath(node, new ArrayList<>(path)));
        if (node instanceof OperatorNode) {
            OperatorNode operator = (OperatorNode) node;
            path.add(Boolean.TRUE);
            collectNodes(operator.left, path, result);
            path.remove(path.size() - 1);
            path.add(Boolean.FALSE);
            collectNodes(operator.right, path, result);
            path.remove(path.size() - 1);
        }
        return result;
    }

    // replace the subtree found at the given path.
    private static Node replaceAtPath(Node current, List<Boolean> path, Node replacement) {
        if (path.isEmpty())
            return replacement;
        if (!(current instanceof OperatorNode))
            return current.deepCopy();

        OperatorNode operator = (OperatorNode) current;
        boolean goLeft = path.get(0);
        List<Boolean> remainder = path.subList(1, path.size());
        Node newLeft = operator.left;
        Node newRight = operator.right;

        if (goLeft)
            newLeft = replaceAtPath(operator.left, remainder, replacement);
        else
            newRight = replaceAtPath(operator.right, remainder, replacement);

        return new OperatorNode(operator.operator, newLeft, newRight);
    }

    //Point mutation on an operator or terminal node.
    private static void mutate(Individual individual, Dataset dataset, Random random) {
        List<NodePath> nodes = collectNodes(individual.root, new ArrayList<>(), new ArrayList<>());
        if (nodes.isEmpty())
            return;

        NodePath target = nodes.get(random.nextInt(nodes.size()));
        Node mutated = target.node.deepCopy();

        if (mutated instanceof OperatorNode) {
            OperatorNode operator = (OperatorNode) mutated;
            operator.operator = randomOperator(random);
            mutated = operator;
        } else if (mutated instanceof FeatureNode) {
            FeatureNode feature = (FeatureNode) mutated;
            if (random.nextDouble() < 0.3)
                mutated = new ConstantNode(randomConstant(dataset, random));
            else
                feature.featureIndex = random.nextInt(dataset.featureCount);
        } else if (mutated instanceof ConstantNode) {
            if (random.nextDouble() < 0.3)
                mutated = new FeatureNode(random.nextInt(dataset.featureCount));
            else
                ((ConstantNode) mutated).value = randomConstant(dataset, random);
        }

        if (target.path.isEmpty())
            individual.root = mutated;
        else
            individual.root = replaceAtPath(individual.root, target.path, mutated);
    }

    // Enforce the maximum depth after genetic operators.
    private static Node trimToDepth(Node node, int maxDepth, Dataset dataset, Random random) {
        return trimToDepth(node, maxDepth, 1, dataset, random);
    }

    // Recursively trim deeper subtrees to terminals.
    private static Node trimToDepth(Node node, int maxDepth, int currentDepth, Dataset dataset, Random random) {
        if (!(node instanceof OperatorNode))
            return node.deepCopy();
        if (currentDepth >= maxDepth)
            return randomTerminal(dataset, random);

        OperatorNode operator = (OperatorNode) node;
        Node left = trimToDepth(operator.left, maxDepth, currentDepth + 1, dataset, random);
        Node right = trimToDepth(operator.right, maxDepth, currentDepth + 1, dataset, random);
        return new OperatorNode(operator.operator, left, right);
    }

    // evaluate accuracy and classification metrics for a given model.
    private static Metrics evaluate(Individual individual, Dataset dataset) {
        Metrics metrics = new Metrics();
        metrics.confusion = new int[2][2];

        int correct = 0;
        for (Instance instance : dataset.instances) {
            double rawScore = individual.root.evaluate(instance);
            double probability = sigmoid(rawScore);
            int prediction = probability >= 0.5 ? POSITIVE_CLASS : 0;
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

    // pretty-print the confusion matrix for quick inspection
    private static void printConfusionMatrix(int[][] confusion) {
        System.out.println("Confusion matrix [actual x predicted]:");
        System.out.println("            predicted 0   predicted 1");
        System.out.printf(Locale.US, "actual 0    %10d   %10d%n", confusion[0][0], confusion[0][1]);
        System.out.printf(Locale.US, "actual 1    %10d   %10d%n", confusion[1][0], confusion[1][1]);
    }

    // protected division to satisfy closure.
    private static double safeDivide(double numerator, double denominator) {
        if (Math.abs(denominator) < 1e-6)
            return numerator;
        return numerator / denominator;
    }

    // Squash raw GP output into a probability for classification.
    private static double sigmoid(double value) {
        if (value >= 0) {
            double exp = Math.exp(-value);
            return 1.0 / (1.0 + exp);
        }
        double exp = Math.exp(value);
        return exp / (1.0 + exp);
    }

    // Load the dataset and compute basic feature ranges.
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

        for (int i = 1; i < lines.size(); i++) {
            String[] parts = splitCsvLine(lines.get(i));
            if (parts.length < featureCount + 1)
                continue;

            int label = parseIntSafe(parts[0]);
            int[] features = new int[featureCount];
            for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
                int value = parseIntSafe(parts[featureIndex + 1]);
                features[featureIndex] = value;
                minValues[featureIndex] = Math.min(minValues[featureIndex], value);
                maxValues[featureIndex] = Math.max(maxValues[featureIndex], value);
            }
            instances.add(new Instance(label, features));
        }

        return new Dataset(featureNames, instances, featureCount, minValues, maxValues);
    }

    // Normalize CSV lines by trimming blanks and folding split rows
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

    //Simple CSV splitter (no quoted commas expected).
    private static String[] splitCsvLine(String line) {
        return line.split(",", -1);
    }

    //quick header check to catch missing column names.
    private static boolean isNumeric(String value) {
        try {
            Integer.parseInt(value.trim());
            return true;
        } catch (NumberFormatException ex) {
            return false;
        }
    }

    // Parse ints with safe fallback for blanks.
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

    // Resolve the first existing file from the candidates list
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

    // Persist the best individual for later testing.
    private static void saveModel(Individual best) throws IOException {
        File modelFile = new File(MODEL_FILE);
        File parent = modelFile.getParentFile();
        if (parent != null && !parent.exists())
            parent.mkdirs();
        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(modelFile))) {
            output.writeObject(best);
        }
    }

    // Load the trained GP model from disk
    private static Individual loadModel(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream input = new ObjectInputStream(new FileInputStream(file))) {
            return (Individual) input.readObject();
        }
    }
}
