from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


def run_dvc_command(command):
    import subprocess
    import sys
    import os
    import site

    pkgs = {
        "dvc": "dvc",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        "numpy": "numpy",
        "mlflow": "mlflow",
        "hydra": "hydra-core",
        "hydra_plugins.hydra_optuna_sweeper": "hydra-optuna-sweeper",
        "joblib": "joblib",
        "optuna": "optuna",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn"
    }
    to_install = []
    for mod, pkg in pkgs.items():
        try:
            __import__(mod)
        except ImportError:
            to_install.append(pkg)
    
    if to_install:
        print(f"Required packages {to_install} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + to_install + ["--user"])
        if site.getusersitepackages() not in sys.path:
            sys.path.append(site.getusersitepackages())
    
    env = os.environ.copy()
    user_site = site.getusersitepackages()
    user_bin = os.path.join(os.path.expanduser("~"), ".local", "bin")
    
    if user_bin not in env.get('PATH', ''):
        env['PATH'] = f"{user_bin}:{env.get('PATH', '')}"
    if user_site not in env.get('PYTHONPATH', ''):
        env['PYTHONPATH'] = f"{user_site}:{env.get('PYTHONPATH', '')}"
    
    env['MLFLOW_TRACKING_URI'] = 'sqlite:////opt/airflow/mlflow.db'

    db_path = '/opt/airflow/mlflow.db'
    if os.path.isdir(db_path):
        import shutil
        try: shutil.rmtree(db_path)
        except Exception: pass

    env['MLFLOW_TRACKING_URI'] = 'sqlite:////opt/airflow/mlflow.db'

    if os.path.exists(db_path):
        try:
            check_db = subprocess.run(
                [sys.executable, "-c", f"import mlflow; mlflow.set_tracking_uri('sqlite:////{db_path}'); mlflow.search_experiments()"],
                env=env, capture_output=True, text=True
            )
            if "alembic.util.exc.CommandError" in check_db.stderr or "ResolutionError" in check_db.stderr:
                print(f"Detected incompatible MLflow database. Resetting for fresh start...")
                os.rename(db_path, f"{db_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        except Exception:
            pass

        if os.path.exists(db_path):
            try: os.chmod(db_path, 0o666)
            except Exception: pass

            for suffix in ['-journal', '-shm', '-wal']:
                fpath = db_path + suffix
                if os.path.exists(fpath):
                    try: os.remove(fpath)
                    except Exception: pass

    mlruns_dir = '/opt/airflow/mlruns'
    if not os.path.exists(mlruns_dir):
        try: os.makedirs(mlruns_dir, mode=0o777, exist_ok=True)
        except Exception: pass
    else:
        try: os.chmod(mlruns_dir, 0o777)
        except Exception: pass
    
    result = subprocess.run(
        [sys.executable, "-m", "dvc"] + command.split(),
        cwd='/opt/airflow',
        env=env,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"DVC command '{command}' failed with code {result.returncode}")


def check_data_exists():
    import os
    if not os.path.exists('/opt/airflow/data/raw/weatherAUS.csv'):
        raise Exception("Data file not found")
    print("Data exists")


def check_model_performance():
    import json
    try:
        with open('/opt/airflow/metrics.json', 'r') as f:
            metrics = json.load(f)
        roc_auc = metrics.get('roc_auc', 0)
        if roc_auc >= 0.80:
            return 'register_model'
    except Exception:
        pass
    return 'stop_pipeline'


def register_model_task():
    import subprocess
    import sys
    import site

    try:
        import mlflow
    except ImportError:
        print("MLflow not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow", "scikit-learn", "--user"])
        if site.getusersitepackages() not in sys.path:
            sys.path.append(site.getusersitepackages())

    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow_db = 'sqlite:////opt/airflow/mlflow.db'
    client = MlflowClient(tracking_uri=mlflow_db)
    experiment = client.get_experiment_by_name("weather_prediction_hpo")
    if not experiment:
        return
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )
    if not runs:
        return
    run_id = runs[0].info.run_id
    model_name = "WeatherModel"
    mlflow.set_tracking_uri(mlflow_db)
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        model_name
    )
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_training_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    sensor_task = PythonOperator(
        task_id='sensor_task',
        python_callable=check_data_exists,
    )

    prepare_task = PythonOperator(
        task_id='prepare_task',
        python_callable=run_dvc_command,
        op_args=['repro prepare'],
    )

    train_task = PythonOperator(
        task_id='train_task',
        python_callable=run_dvc_command,
        op_args=['repro train'],
    )

    evaluate_task = BranchPythonOperator(
        task_id='evaluate_task',
        python_callable=check_model_performance,
    )

    register_model = PythonOperator(
        task_id='register_model',
        python_callable=register_model_task,
    )

    stop_pipeline = BashOperator(
        task_id='stop_pipeline',
        bash_command='echo "Model performance below threshold."',
    )

    sensor_task >> prepare_task >> train_task >> evaluate_task
    evaluate_task >> [register_model, stop_pipeline]
