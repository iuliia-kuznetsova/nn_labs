# nn_labs
Labs created while studying neural networks

---

## Setup: Virtual Environment & Dependencies

### 1. Create the virtual environment
Run once from the project root:
```powershell
cd C:\Users\iulii\Documents\Labs\nn_labs
python -m venv .venv
```

### 2. Activate it
```powershell
.venv\Scripts\Activate.ps1
```

Your prompt will change to show `(.venv)` — that means it's active. Every `python` and `pip` command now uses the isolated environment.

> If you get a permissions error, run this first (once per machine):
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install dependencies
```powershell
pip install -r nn_basics\labs\requirements.txt
```

### 4. Register the environment as a Jupyter kernel
So notebooks can use it:
```powershell
pip install ipykernel
python -m ipykernel install --user --name=nn_labs --display-name "Python (nn_labs)"
```

After this, open any notebook → click the kernel selector (top-right) → choose **Python (nn_labs)**.

---

## Everyday Workflow

```powershell
# Start of session — activate
.venv\Scripts\Activate.ps1

# End of session — deactivate
deactivate
```

### Useful commands inside the venv

| Command | What it does |
|---|---|
| `pip list` | Show all installed packages |
| `pip freeze > requirements.txt` | Save current state to requirements file |
| `pip install <package>` | Add a new package |
