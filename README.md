# Electoral Districting Analysis & Gerrymandering Visualization  
**A computational study of political districting strategies and their impact on electoral fairness**  

---

## 📁 Project Structure

### `districting_methods.py`  
Core module implementing electoral districting algorithms:
- **Fair districting** – Maximizing proportional representation and/or compactness
- **Partisan gerrymandering** – Simulating biased district configurations

### `/gerrymandering_visualisation`  
Django web application for interactive analysis of the case study
- **District mapping**
- **Comparative scenarios**:
  - Fair vs. gerrymandered configurations
  - Fragmented vs. united opposition strategies (Poland 2023 parlimentary elections)

### To turn on the server - run `run.bat` file

### `run.ipynb`  
Jupyter notebook for executing districting algorithms with customizable parameters

### `show_results.ipynb`
Jupyter notebook for visualizing and analyzing results from districting algorithms

---

## 📌 Case Study: 2023 Polish Parliamentary Elections
Analyzes real election data under two configurations:
1. **Default scenario**:  
   Opposition parties run separately (KO, Lewica, Trzecia Droga)
2. **Coalition scenario**:  
   United "Pakt Senacki" list for democratic opposition

---