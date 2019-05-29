export BERT_BASE_DIR=/home/herla/notebooks/text-analysis/uncased_L-12_H-768_A-12
export TRAINED_CLASSIFIER=./bert_output/model.ckpt-3348
#export TRAINED_CLASSIFIER=/home/herla/notebooks/text-analysis/uncased_L-12_H-768_A-12/bert_model.ckpt
STARTTIME=$(date +%s)

python3 run_classifier.py \
--task_name=MRPC \
--use_gpu=True \
--do_predict=true \
--data_dir=./data \
--do_lower_case=True \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$TRAINED_CLASSIFIER \
--do_lower_case=True \
--max_seq_length=128 \
--output_dir=./bert_output/

ENDTIME=$(date +%s)
secs=$(($ENDTIME - $STARTTIME))
printf 'Elapsed Time %dh:%dm:%ds\n' $(($secs/3600)) $(($secs%3600/60)) $(($secs%60))