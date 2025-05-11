from prefect import flow, task
from pipeline.phase4_generating_embeddings_tasks import generate_and_store_embeddings

@task(retries=3, retry_delay_seconds=2)
def run_generate_embeddings():
    generate_and_store_embeddings()

@flow
def phase4_flow():
    run_generate_embeddings()

if __name__ == "__main__":
    phase4_flow()