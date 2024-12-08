#!/bin/bash
domains=('books' 'dvd' 'electronics' 'kitchen')

export CUDA_VISIBLE_DEVICES=3
data='../data/amazon/'
output='../run_out/amazon/'

for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            echo "####################### ${src_domain}===>>>${tar_domain}#######################:"
            python -B run_clue_base.py \
                --data_dir "${data}${src_domain}-${tar_domain}" --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_batch_size 24 --num_train_epochs 8 --learning_rate 3e-5 --max_seq_length 512 \
                --retrieval_num 20 --eval_logging_steps 100 --eval_batch_size 200 \
                --bert_model 'bert_base' --seed 42 --do_train --do_eval --evaluate_during_training 
            printf "\n\n\n"

        fi
    done
done




