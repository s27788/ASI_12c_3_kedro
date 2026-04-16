import os
from dotenv import load_dotenv
import wandb

load_dotenv()

print("ENTITY:", os.getenv("WANDB_ENTITY"))
print("PROJECT:", os.getenv("WANDB_PROJECT"))
print("KEY:", "OK" if os.getenv("WANDB_API_KEY") else "BRAK")

run = wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY")
)

wandb.log({"test_metric": 123})

wandb.finish()

print("DONE - sprawdź wandb.ai")