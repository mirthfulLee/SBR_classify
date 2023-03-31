import sys
import argparse

from allennlp.commands import main, parse_args

batch = "64"
cuda = "-1"

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    "TreeVul_IR/out_treevul/model.tar.gz",
    "temp/sub_test_samples_compressed.csv",
    "--output-file", "temp/sub_test_metrics.json",
    "--batch-size", batch,
    "--cuda-device", cuda,
    "--include-package", "TreeVul_IR"
]

main()