export BERT_BASE_DIR=/home/herla/notebooks/text-analysis/uncased_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/home/herla/notebooks/text-analysis/uncased_L-12_H-768_A-12/bert_model.ckpt

STARTTIME=$(date +%s)

python3 run_classifier.py --task_name=MRPC --do_train=true --do_eval=true --data_dir=./data --use_gpu=True --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --do_lower_case=True --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=./bert_output/

ENDTIME=$(date +%s)
secs=$(($ENDTIME - $STARTTIME))
printf 'Elapsed Time %dh:%dm:%ds\n' $(($secs/3600)) $(($secs%3600/60)) $(($secs%60))