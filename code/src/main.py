from pipeline.full_pipeline import run_all_combinations_pipeline
#from utils.helpers import notify

if __name__ == "__main__":
    run_all_combinations_pipeline(n_jobs=-1)
    #notify("All code finished", "https://ntfy.sh/your_topic")  # Example for optional notification upon completion
