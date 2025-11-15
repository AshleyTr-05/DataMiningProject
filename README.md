# 📘 **README.md**

````markdown
# 🧠 Data Mining Project (Java + Weka)

This project uses **Java**, **Weka**, and **VS Code** to perform data preprocessing and machine-learning classification.  
Follow this guide to set up the environment and run the project correctly.

---

# 🚀 Project Setup (Everyone Must Follow This)

## 1. Install Required Tools

### 🔧 Java JDK
Install **Java JDK 11 or 17**  
Check your version:

```bash
java -version
````

### 🔧 VS Code

Install **Visual Studio Code**.

### 🔧 Java Extensions for VS Code

Inside VS Code, go to:

```
Extensions → search “Extension Pack for Java” → Install
```

---

## 2. Clone the Project from GitHub

1. Open **VS Code**
2. Press `Ctrl + Shift + P`
3. Type **Git: Clone**
4. Paste the repository URL:

```
https://github.com/yourname/data-mining-project.git
```

5. Open the cloned folder in VS Code when prompted.

---

## 3. Verify Project Structure

After cloning, your folder should look like this:

```
FinalProject/
│
├─ lib/
│   └─ weka.jar
│
├─ datasets/
│   └─ heart_disease.csv
│
└─ src/
    └─ Preprocessor.java
```

If you don’t see these files, you cloned incorrectly.

---

## 4. Compile the Preprocessor

1. Open **Terminal** in VS Code
2. Make sure the terminal path ends with the project folder
3. Run the commands below:

```bat
mkdir bin
javac -cp "lib\weka.jar;src" -d bin src\Preprocessor.java
```

This compiles the Java code and places `.class` files in the `bin/` directory.

---

## 5. Run the Preprocessor

```bat
java -cp "bin;lib\weka.jar" Preprocessor
```

If everything works, you should see output like:

```
=== Preprocessing done ===
Input CSV  : datasets/heart_disease.csv
Output ARFF: datasets/heart_disease_preprocessed.arff
Instances  : 303
Attributes : 14
First 3 rows:
  ...
```

This means:

✔ Weka is working
✔ Java is working
✔ Project is set up correctly
✔ Preprocessing successful

---

# 📂 Project Workflow for Team Members

Each team member will work on their own Java file inside `src/`:

* `Preprocessor.java` – Giang
* `J48Classifier.java` – Member A
* `ImprovedModel.java` – Member B
* `Evaluator.java` – Member C

To compile any file:

```bat
javac -cp "lib\weka.jar;src" -d bin src\FileName.java
```

To run it:

```bat
java -cp "bin;lib\weka.jar" FileName
```

---

# ✔ Notes

* Do **NOT** delete `weka.jar`
* Do **NOT** move files out of `src/`, `lib/`, or `datasets/`
* Make sure all file paths match exactly (Windows is sensitive)

---

# 🎉 You're Ready to Work!

If you have questions, check the terminal output or ask in the group chat.
Happy coding and good luck with the project! 🚀

```
