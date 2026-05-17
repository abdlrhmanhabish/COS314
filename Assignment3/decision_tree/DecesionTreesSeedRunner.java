import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

public class DecesionTreesSeedRunner {

    private static final int START_SEED = 1;
    private static final int END_SEED = 30;
    private static final String TRAIN_FILE = "../Breast_train.csv";
    private static final String TEST_FILE = "../Breast_test.csv";
    private static final String RESULTS_DIR = "results";
    private static final String CSV_FILE = RESULTS_DIR + File.separator + "seed_results.csv";
    private static final String BEST_RUN_LOG = RESULTS_DIR + File.separator + "best_run_output.log";
    private static final String RESULT_FILE = RESULTS_DIR + File.separator + "result.txt";

    private static class RunResult {
        final int seed;
        final double trainAccuracy;
        final double testAccuracy;
        final double fMeasure;
        final double runtime;

        RunResult(int seed, double trainAccuracy, double testAccuracy, double fMeasure, double runtime) {
            this.seed = seed;
            this.trainAccuracy = trainAccuracy;
            this.testAccuracy = testAccuracy;
            this.fMeasure = fMeasure;
            this.runtime = runtime;
        }
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Files.createDirectories(Path.of(RESULTS_DIR));

        List<RunResult> results = new ArrayList<>();
        for (int seed = START_SEED; seed <= END_SEED; seed++) {
            System.out.println("Running seed " + seed + "...");
            List<String> output = runDecisionTree(seed, null);
            RunResult result = parseOutput(seed, output);
            results.add(result);
            System.out.printf(Locale.US, "Seed %d done. Test acc=%.4f%n", seed, result.testAccuracy);
        }

        writeCsv(results);

        RunResult best = results.stream()
                .max(Comparator.comparingDouble((RunResult r) -> r.testAccuracy)
                        .thenComparingDouble(r -> r.trainAccuracy))
                .orElse(null);

        if (best == null) {
            System.err.println("No results generated.");
            return;
        }

        System.out.println("Best seed: " + best.seed);

        List<String> bestOutput = runDecisionTree(best.seed, Path.of(BEST_RUN_LOG));
        RunResult bestFromLog = parseOutput(best.seed, bestOutput);

        writeSummary(bestFromLog, results);
    }

    private static List<String> runDecisionTree(int seed, Path logPath) throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder("java", "-cp", ".", "DecisionTreeGP");
        builder.redirectErrorStream(true);
        Process process = builder.start();

        try (BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8))) {
            writer.write("train");
            writer.newLine();
            writer.write(Integer.toString(seed));
            writer.newLine();
            writer.write(TRAIN_FILE);
            writer.newLine();
            writer.write(TEST_FILE);
            writer.newLine();
        }

        List<String> output = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.add(line);
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            System.err.println("DecisionTreeGP exited with code " + exitCode + " for seed " + seed);
        }

        if (logPath != null) {
            Files.write(logPath, output, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING);
        }

        return output;
    }

    private static RunResult parseOutput(int seed, List<String> output) {
        double trainAccuracy = findMetric(output, "training accuracy");
        double testAccuracy = findMetric(output, "test accuracy");
        double fMeasure = findMetric(output, "f-measure");
        double runtime = findMetric(output, "runtime");
        return new RunResult(seed, trainAccuracy, testAccuracy, fMeasure, runtime);
    }

    private static double findMetric(List<String> output, String token) {
        for (String line : output) {
            String lowered = line.toLowerCase(Locale.ROOT);
            if (lowered.contains(token)) {
                return parseValueAfterColon(line);
            }
        }
        return 0.0;
    }

    private static double parseValueAfterColon(String line) {
        int idx = line.indexOf(':');
        if (idx == -1) {
            return 0.0;
        }
        String value = line.substring(idx + 1).replace("%", "").replace("s", "").trim();
        if (value.isEmpty()) {
            return 0.0;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException ex) {
            return 0.0;
        }
    }

    private static void writeCsv(List<RunResult> results) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("seed,train_accuracy,test_accuracy,f_measure,runtime");
        for (RunResult result : results) {
            lines.add(String.format(Locale.US, "%d,%.4f,%.4f,%.4f,%.4f", result.seed, result.trainAccuracy,
                    result.testAccuracy, result.fMeasure, result.runtime));
        }
        Files.write(Path.of(CSV_FILE), lines, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
    }

    private static void writeSummary(RunResult best, List<RunResult> allResults) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("Winning Seed Value: " + best.seed);
        lines.add("");
        lines.add("Final Classification Performance (Table 2):");
        lines.add(String.format(Locale.US, "Training Accuracy (%%): %.4f%%", best.trainAccuracy));
        lines.add(String.format(Locale.US, "Test Accuracy (%%): %.4f%%", best.testAccuracy));
        lines.add(String.format(Locale.US, "F-measure: %.4f", best.fMeasure));
        lines.add(String.format(Locale.US, "Runtime: %.4f s", best.runtime));
        lines.add("");
        lines.add("Test accuracies for all 30 runs:");
        for (RunResult result : allResults) {
            lines.add(String.format(Locale.US, "%d: %.4f%%", result.seed, result.testAccuracy));
        }

        Path modelText = Path.of("decision_tree_gp.model.txt");
        if (Files.exists(modelText)) {
            lines.add("");
            lines.add("Best model tree (seed " + best.seed + "):");
            lines.add("");
            lines.addAll(Files.readAllLines(modelText, StandardCharsets.UTF_8));
        }

        Files.write(Path.of(RESULT_FILE), lines, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
    }
}