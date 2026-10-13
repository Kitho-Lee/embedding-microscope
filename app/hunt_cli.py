"""Run a corpus-specific search and optionally emit desired-behavior tests."""
import argparse
import json
from pathlib import Path
from .main import Hunt
from .hunt import search, regression_bundle

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,default=Path('hunt-result.json'))
    parser.add_argument('--regression-dir',type=Path)
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    config=Hunt.model_validate(data.get('config',data))
    result=search(config.model_dump())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if args.regression_dir and result['best']:
        bundle=regression_bundle(result)
        args.regression_dir.mkdir(parents=True,exist_ok=True)
        (args.regression_dir/'case.json').write_text(json.dumps(bundle['case'],ensure_ascii=False,indent=2))
        (args.regression_dir/'test_regression.py').write_text(bundle['pytest'])
        (args.regression_dir/'README.txt').write_text(bundle['instructions'])
    print(f"Evaluated {result['evaluated']}/{result['search_space']} combinations; {result['failure_count']} failures")
if __name__=='__main__':main()
