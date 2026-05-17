COS314 Assignment 3 - Genetic Programming

Team Members:
    1. Hamdaan Mirza, u24898008
    2. Abdelrahman Ahmed, u24898008
    3. Abhay Rooplall, u24568792

Options to verify the best seed for Arithmetic GP:
    A. Run the JAR:
        cd arithmetic
        java -jar ArithmeticSeedRunner.jar

    B. Compile and run from source:
        cd arithmetic
        javac ArithmeticGP.java ArithmeticSeedRunner.java
        java ArithmeticSeedRunner

 
 Options to verify the best seed for Decision Trees GP:
     A. Run the JAR:
        cd decision_tree
        java -jar DecesionTreesSeedRunner.jar

    B. Compile and run from source:
        cd decision_tree
        javac DecisionTreeGP.java DecesionTreesSeedRunner.java
        java DecesionTreesSeedRunner


Options to run Arithmetic GP (best seed = 26):
    A. Run the JAR:
        cd arithmetic
        java -jar ArithmeticGP.jar

    B. Compile and run from source:
        cd arithmetic
        javac ArithmeticGP.java
        java ArithmeticGP

Options to run Decision Trees GP (best seed = 19):
    A. Run the JAR:
        cd decision_tree
        java -jar DecisionTreeGP.jar

    B. Compile and run from source:
        cd decision_tree
        javac DecisionTreeGP.java
        java DecisionTreeGP

Notes:
    1. When you run the seed-runner program, it runs the GP with 30 unique seeds, identifies the winning seed value,
    and saves output in a results folder.
    2. When you train a programs, it outputs the best tree/expression and metrics per generation, evaluates the
    final model, and saves it as decision_tree_gp.model or arithmetic_gp.model.
    3. When you test a program, it loads the pre-trained model and outputs the final test accuracy, F-measure, and
    confusion matrix.