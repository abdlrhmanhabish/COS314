
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
        System.out.println("\n" + "=".repeat(100));
        System.out.printf("%-25s | %-10s | %-10s | %-15s | %-15s\n","Problem Instance", "Algorithm", "Seed", "Best Solution", "Runtime (s)");
        System.out.println("-".repeat(100));

        for(File file : files){
            Data data = loadData(file); // function that converts files into data objects
            if (data == null) {
                System.out.println("Failed to load instance from file: " + file.getName());
                continue;
            }

            long startTLS = System.currentTimeMillis();
            double LTSResult = LTS(data, seed);
            long endTLS = System.currentTimeMillis();
            double timeILS = (endTLS - startTLS) / 1000.0;
            System.out.printf("%-25s | %-10s | %-15.2f | %-15.3f\n", file.getName(), "ILS", LTSResult, timeILS);

            long startGA = System.currentTimeMillis();
            double GAResult = GA(data, seed);
            long endGA = System.currentTimeMillis();
            double timeGA = (endGA - startGA) / 1000.0;
            System.out.printf("%-25s | %-10s | %-15.2f | %-15.3f\n", "", "GA", GAResult, timeGA);
            System.out.println("-".repeat(73));
        }
        scanner.close();
    }

    //stubbed out functions for algos
    static double LTS(Data data, long seed){
        return 0;
    }

    static double GA(Data data, long seed){
        return 0;
    }

    // converts file data to data objs
    static Data loadData(File file){
        try (Scanner scanner = new Scanner(file)) {
            scanner.useLocale(Locale.US); // same fix in main
            Data data = new Data();
            data.n = scanner.nextInt();
            data.capacity = scanner.nextDouble();
            data.values = new double[data.n];
            data.weights = new double[data.n];
            for(int i = 0; i < data.n; i++){
                data.values[i] = scanner.nextDouble();
                data.weights[i] = scanner.nextDouble();
            }
            scanner.close();
            return data;
        } 
        catch (Exception e) {
            System.err.println("Error reading " + file.getName() + ": " + e.getMessage());
            return null;
        }
    }
}
