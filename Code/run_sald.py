# run_sald.py
from sald import sald_distill
from train_eval import evaluate_on_synth
from config import OUT_DIR
import os

if __name__ == "__main__":
    synth_path = sald_distill()
    print("Evaluating distilled dataset ...")
    evaluate_on_synth(synth_path, epochs=30)
