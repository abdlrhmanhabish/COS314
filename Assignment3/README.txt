COS314 Assignment 3 - Genetic Programming

Team Members:
    1. Hamdaan Mirza, u24898008
    2. Abdelrahman Ahmed, u24898008
    3. Abhay Rooplall, u24568792

Options to run Decision Trees GP:
    A. Run the JAR:
        cd decision_tree
        java -jar DecisionTreeGP.jar

    B. Compile and run from source:
        cd decision_tree
        javac DecisionTreeGP.java
        java DecisionTreeGP

    C. 30-seed runner (outputs to decision_tree/results/):
        cd decision_tree
        javac DecisionTreeGP.java SeedRunner.java
        java SeedRunner

Options to run Arithmetic GP:
    A. Run the JAR:
        cd arithmetic
        java -jar ArithmeticGP.jar

    B. Compile and run from source:
        cd arithmetic
        javac ArithmeticGP.java
        java ArithmeticGP

    C. 30-seed runner (outputs to arithmetic/results/):
        cd arithmetic
        javac ArithmeticGP.java ArithmeticSeedRunner.java
        java ArithmeticSeedRunner

Program Usage and Inputs

Both programs are interactive and will request all parameters needed to run.

To train the model:
	Enter mode: train
    Enter seed value: numeric seed, for example 45
    The program automatically uses Breast_train.csv and Breast_test.csv.

Note: The program outputs the best tree/expression and metrics per generation, evaluates the
final model, and saves it as decision_tree_gp.model or arithmetic_gp.model.

To test an existing model:
	Enter mode: test
    The program automatically uses the saved model and Breast_test.csv.

Note: This loads the pre-trained model and outputs the final test accuracy, F-measure, and
confusion matrix.

30 Independent Runs Requirement (Both Algorithms)

For each algorithm, you must perform 30 independent runs using unique seed values, record
the best-performing run, and use that run for demo day.

Decision Tree GP:
	Use the Java seed runner in decision_tree/ (SeedRunner.java) to execute 30 runs and record
	the best seed automatically.

Arithmetic GP:
	Use the Java seed runner in arithmetic/ (ArithmeticSeedRunner.java) to execute 30 runs and
	record the best seed automatically.

3. Demo Replication (Winning Seed)

To replicate the best classification performance reported in the submission, run in train
mode using the following seed:
	Best Seed Value: 28

4. File Structure Requirements

Ensure the following files are present with this structure:
	Breast_train.csv
	Breast_test.csv
	decision_tree/DecisionTreeGP.jar (or .java / .class files)
	decision_tree/SeedRunner.java (optional Java-only 30-seed runner)
	arithmetic/ArithmeticGP.jar (or .java / .class files)
	arithmetic/ArithmeticSeedRunner.java (optional Java-only 30-seed runner)