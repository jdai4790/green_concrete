# 🧱 Green Concrete Project – Setup Guide

## ✅ 1. Extract the zip file
Unzip the folder to any local directory, e.g.:
```
C:\Users\<your_name>\Documents\green_concrete
```

## ✅ 2. Create and activate your virtual environment

### **PowerShell (recommended on Windows)**
```powershell
cd "C:\Users\<your_name>\Documents\green_concrete"
python -m venv venv
& ".\venv\Scripts\Activate.ps1"
pip install -r requirements.txt
```

### **Command Prompt (cmd)**
```cmd
cd "C:\Users\<your_name>\Documents\green_concrete"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

> 💡 Make sure you have Python 3.13.6 (or newer) installed.

---

## ✅ 3. Run the program
Once everything is installed, run any script from the `scripts/` folder, e.g.:
```powershell
python scripts\linear_regression.py
```

Outputs (plots, metrics, Excel files) will appear in the `outputs/` folder.

---

## ✅ 4. Project Layout
```
green_concrete/
│
├── data/
├── outputs/
├── scripts/
├── requirements.txt
└── README.md
```

---

## ⚙️ 5. Notes
- The `venv/` folder is **not included** in the zip — each user must create their own local venv.
- After installing new packages, update the requirements file using:
  ```powershell
  pip freeze > requirements.txt
  ```
- To deactivate the environment:
  ```powershell
  deactivate
  ```

---

Developed by Jeffrey Dai  
University of Sydney – CIVL4022 Thesis (2025)
