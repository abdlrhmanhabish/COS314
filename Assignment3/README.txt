---

# README: COS314 Assignment Three - Genetic Programming

**Team Members:** Hamdaan Mirza [Student Number], [Team Member Names/Numbers]

**Language Used:** Java

---

## 1. Execution Instructions (How to Run)

As per the assignment requirements, this program is designed to run via the command line and does not require an IDE or external libraries.

### Option A: Running the Executable JAR (Recommended)

1. Open your terminal or command prompt.
2. Navigate to the directory containing `DecisionTreeGP.jar`.
3. Run the following command:
```bash
java -jar DecisionTreeGP.jar

```



### Option B: Compiling and Running from Source

1. Open your terminal and navigate to the directory containing `DecisionTreeGP.java`.
2. Compile the code:
```bash
javac DecisionTreeGP.java

```


3. Run the compiled class:
```bash
java DecisionTreeGP

```



---

## 2. Program Usage & Inputs

The program will interactively request the necessary parameters to run.

### To Train the Model:

When you run the program, follow the on-screen prompts:

* **Enter mode:** Type `train`
* **Enter seed value:** [Enter a numeric seed, e.g., `45`]
* **Enter training filepath:** [Provide the path, e.g., `Breast_train.csv`]
* **Enter test filepath:** [Provide the path, e.g., `Breast_test.csv`]

> **Note:** The program will output the best tree and metrics per generation, evaluate the final model, and save it as `decision_tree_gp.model`.

### To Test an Existing Model:

* **Enter mode:** Type `test`
* **Enter test filepath:** [Provide the path, e.g., `Breast_test.csv`]
* **Enter model filepath:** [Provide the path, e.g., `decision_tree_gp.model`]

> **Note:** This will load the pre-trained tree and output the final test accuracy, F-measure, and confusion matrix.

---

## 3. Demo Replication (Winning Seed)

To replicate our best classification performance as reported in our final submission document, please run the program in **train** mode using the following seed:

* **Best Seed Value:** `28`

---

## 4. File Structure Requirements

Ensure the following files are in the same directory when running the program:

* `Breast_train.csv` (Attached dataset)
* `Breast_test.csv` (Attached dataset)
* `DecisionTreeGP.jar` (or `.java` / `.class` files)