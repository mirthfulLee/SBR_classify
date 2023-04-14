


# TreeVul_IR
allennlp train TreeVul_IR/config_tree.json -s TreeVul_IR/out_treevul --include-package TreeVul_IR
allennlp evaluate model/my_treevul/model.tar.gz data1/test_samples_compressed.csv --output-file model/my_treevul/test_metrics.json  --batch-size 512 --cuda-device 2 --include-package TreeVul_IR --overrides '{"model.root_thres": 0.8}'
allennlp evaluate model/textcnn/model.tar.gz data1/test_samples_compressed.csv --output-file model/textcnn/test_metrics.json --batch-size 32 --cuda-device 3 --include-package TextCNN --overrides '{"model.thres": 0.89}'
allennlp evaluate model/MODEL_NAME/model.tar.gz data1/test_samples_compressed.csv --output-file model/MODEL_NAME/test_metrics.json --predictions-output-file model/MODEL_NAME/test_results.json --batch-size 512 --cuda-device 2 --include-package MODEL_NAME