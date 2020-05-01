export STORAGE_BUCKET=gs://bertje-us-central-1
export TPU_NAME=w-de-vries-21

python3 run-pretraining.py \
  --bert_config_file=${STORAGE_BUCKET}/bertje/bert_config.json \
  --input_file="${STORAGE_BUCKET}/tf_examples_*.tfrecord" \
  --output_dir=${STORAGE_BUCKET}/pretraining_output \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --do_train=True \
  --do_eval=True \
  --train_batch_size=256 \
  --eval_batch_size=32 \
  --learning_rate=1e-4 \
  --num_train_steps=1000000 \
  --num_warmup_steps=10000 \
  --save_checkpoints_steps=10000 \
  --iterations_per_loop=10000 \
  --max_eval_steps=10000 \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --num_tpu_cores=8
