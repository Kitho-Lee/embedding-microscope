import argparse
import json
from pathlib import Path
from .main import Experiment
from .engine import run_experiment

def main():
    parser=argparse.ArgumentParser(description="Re-run exported experiment config")
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",type=Path,default=Path("result.json"))
    args=parser.parse_args()
    source=json.loads(args.input.read_text())
    config=Experiment.model_validate(source.get("config",source))
    args.output.write_text(json.dumps(run_experiment(config.model_dump()),ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
