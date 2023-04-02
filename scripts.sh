


# TreeVul_IR
allennlp train TreeVul_IR/config_tree.json -s TreeVul_IR/out_treevul --include-package TreeVul_IR
allennlp evaluate TreeVul_IR/out_treevul/model.tar.gz data/test_samples_compressed.csv --output-file TreeVul_IR/out_treevul/test_metrics.json --batch-size 512 --cuda-device 3 --include-package TreeVul_IR
allennlp evaluate TreeVul_IR/my_treevul/model.tar.gz data/test_samples_compressed.csv --output-file TreeVul_IR/my_treevul/test_metrics.json --predictions-output-file TreeVul_IR/my_treevul/test_results.json --batch-size 512 --cuda-device 2 --include-package TreeVul_IR
allennlp evaluate TreeVul_IR/weighted_treevul/model.tar.gz data/test_samples_compressed.csv --output-file TreeVul_IR/weighted_treevul/test_metrics.json --predictions-output-file TreeVul_IR/weighted_treevul/test_results.json --batch-size 512 --cuda-device 2 --include-package TreeVul_IR