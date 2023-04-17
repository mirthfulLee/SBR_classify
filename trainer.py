import sys
import argparse

from allennlp.commands import main, parse_args

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    "model_config/config_treevul.json",
    "-s", "model/weighted_treevul",
    "--include-package", "TreeVul_IR",
    "--force"
]

main()