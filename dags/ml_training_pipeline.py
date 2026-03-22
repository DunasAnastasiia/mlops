from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import json
def check_model_performance():
    with open('/opt/airflow/metrics.json', 'r') as f:
        metrics = json.load(f)
    roc_auc = metrics.get('roc_auc', 0)
    if roc_auc >= 0.80:
        return 'register_model'
    return 'stop_pipeline'

def register_model_task():
    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri='sqlite:////opt/airflow/mlflow.db')
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
    mlflow.set_tracking_uri('sqlite:////opt/airflow/mlflow.db')
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

    sensor_task = BashOperator(
        task_id='sensor_task',
        bash_command='dvc status data/raw/weatherAUS.csv.dvc',
    )

    prepare_task = BashOperator(
        task_id='prepare_task',
        bash_command='dvc repro prepare',
    )

    train_task = BashOperator(
        task_id='train_task',
        bash_command='dvc repro train',
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
