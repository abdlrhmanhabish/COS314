COS314 Assignment 2

Team Members
-Hamdaan Mirza       u24631494
-Abdelrahman Ahmed   u24898008
-Abhay Rooplall      u24568792 

The executable JAR is already included.
Steps to run: 
1. Open a terminal in the folder containing Knapsack.jar.
2. Make sure the folder Knapsack Instances is in the same directory as Knapsack.jar.
3. Run: java -jar Knapsack.jar
4. When prompted, enter seed value. We have used seed value 45

If u need to recomplie and make sure its working:
1. Compile:
javac Knapsack.java
2. Create manifest:
printf 'Main-Class: Knapsack\n' > MANIFEST.MF
3. Build executable JAR:
jar cfm Knapsack.jar MANIFEST.MF Knapsack.class 'Knapsack$Data.class'