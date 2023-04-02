import sys
import argparse

from allennlp.commands import main, parse_args

batch = "64"
cuda = "2"

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    "TreeVul_IR/config_sbrtree.json",
    "-s", "TreeVul_IR/sbr_treevul",
    "--include-package", "TreeVul_IR",
    "--force"
]

main()