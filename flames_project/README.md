# 🔥 FLAMES Relationship Calculator (Python)

## 📌 Project Overview
The FLAMES Relationship Calculator is a simple Python program that predicts the relationship between two people based on their names. The program uses the classic **FLAMES game algorithm**, which removes common letters from two names and calculates the relationship result.

FLAMES stands for:
- **F** – Friends  
- **L** – Love  
- **A** – Affection  
- **M** – Marriage  
- **E** – Enemy  
- **S** – Siblings  

The program takes two names as input, removes common characters between them, counts the remaining letters, and determines the relationship using the FLAMES logic.

---

## ❗ Problem Statement
People often play the FLAMES game for fun to predict relationships between two names. Manually performing the FLAMES process can be time-consuming. This project automates the process using Python.

---

## 💡 Solution
This program:
1. Takes two names as input from the user.
2. Removes common characters between the two names.
3. Counts the remaining letters.
4. Uses the FLAMES algorithm to determine the relationship.
5. Displays the final relationship result.

---

## ⚙️ Technologies Used
- 🐍 **Python**
- Basic **string manipulation**
- **Lists and loops**
- Conditional statements

---

## 🧠 How the Program Works
1. The user enters two names.
2. The program converts names to lowercase and removes spaces.
3. Common letters between the two names are removed.
4. The remaining letters are counted.
5. The count is used to eliminate letters from the FLAMES list.
6. The final remaining letter determines the relationship.

---

## 🔄 Algorithm
1. Convert both names to lowercase.
2. Remove spaces from the names.
3. Convert names into lists of characters.
4. Remove common characters from both lists.
5. Count the remaining characters.
6. Use the count to iterate through the FLAMES list.
7. Remove elements until only one relationship remains.

---
