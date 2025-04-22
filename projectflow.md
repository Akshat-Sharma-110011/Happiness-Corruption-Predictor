# ✅ Step-by-Step Executable Workflow in PyCharm

## 🔧 1. Project Setup
- Open PyCharm and create a new project (or open your existing project).
- Place your template.py file in the root directory.
- Run template.py from PyCharm to generate the initial project structure.

## 📦 2. Configure setup.py and pyproject.toml
- Create and update these files to define your local packages and project metadata.
- In PyCharm, ensure the directory containing your local modules has an __init__.py file.
- Reference: crashcourse.txt for exact format.

## 🐍 3. Virtual Environment & Dependencies
In terminal (inside PyCharm):
```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
```

Back in PyCharm:
- Go to Settings > Project > Python Interpreter and set interpreter to vehicle.
- Create requirements.txt and install packages:
```bash
pip install -r requirements.txt
```
- Run `pip list` in PyCharm terminal to verify installation.

## ☁️ 4. MongoDB Atlas Integration
- Setup cluster, user, and IP access (as you mentioned).
- Save your connection string securely.
- In PyCharm terminal:
```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/"
```
- On Windows:
```powershell
$env:MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/"
```
- Use .env file or os.environ.get("MONGODB_URL") in code to access it securely.
- Test connection in mongoDB_demo.ipynb using Jupyter notebook inside PyCharm (install Jupyter plugin if needed).

## 📝 5. Logging and Exception Handling
- Create logger.py and exception.py, test them using demo.py in PyCharm.
- Run via right-click > Run demo.

## 🧪 6. Data Ingestion (Modular Coding in PyCharm)
Create the following modules in PyCharm's src/ directory:
- constants/__init__.py
- configuration/mongo_db_connection.py
- data_access/proj1_data.py
- entity/config_entity.py
- entity/artifact_entity.py
- components/data_ingestion.py

Ensure each component works independently. Use PyCharm debugger to test each step.
- Run demo.py to validate data ingestion.
- Keep MongoDB URL setup using terminal (or environment variable file).

## 🔍 7. Data Validation, Transformation, Model Trainer
Add:
- utils/main_utils.py
- config/schema.yaml
- Extend entity/estimator.py

Implement and test each:
- data_validation.py
- data_transformation.py
- model_trainer.py

Use the PyCharm test runner or execute files in scripts/ directory for step-by-step validation.

## 🛢️ 8. AWS Setup (for Model Evaluation & Pusher)
Store AWS credentials using .env or set them in PyCharm's terminal:
```bash
export AWS_ACCESS_KEY_ID="XXX"
export AWS_SECRET_ACCESS_KEY="XXX"
```

Update constants/__init__.py with:
```python
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
MODEL_BUCKET_NAME = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"
```

Add:
- configuration/aws_connection.py
- aws_storage/ and entity/s3_estimator.py

## ✅ 9. Model Evaluation & Model Pusher
- Follow the same module structure and testing method as in Data Ingestion.
- Use breakpoints & PyCharm debugging for each class/method in your pipeline.

## 🚀 10. Prediction Pipeline & App
- Create app.py, prediction_pipeline/, and static/, templates/ folders.
- Run Flask app inside PyCharm or terminal:
```bash
python app.py
```
- Access via localhost:5000.

## 🔄 11. Docker & CI/CD
- Create Dockerfile, .dockerignore, .github/workflows/aws.yaml
- Push code to GitHub.
- Connect GitHub secrets for:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - AWS_DEFAULT_REGION
  - ECR_REPO
- PyCharm tip: Use the Version Control panel to push and track commits easily.

## 🖥️ 12. EC2 Deployment
- SSH into EC2 from PyCharm or terminal.
- Follow Docker installation and GitHub runner setup.
- Run ./run.sh to connect EC2 as a GitHub runner.

## 🌐 13. Launch App on EC2
- Expose port 5080 from EC2 security settings.
- Run Docker container and access your app via http://<public-ip>:5080.

## ✅ Tips for Using PyCharm Efficiently
- Use Python Console for quick testing.
- Use Run Configurations to automate running of specific files.
- Enable env file support in PyCharm:
  - Run > Edit Configurations > Environment variables > Load from .env file -n 