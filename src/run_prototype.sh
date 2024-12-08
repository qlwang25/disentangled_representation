#!/bin/bash
domains=('dvd' 'books' 'electronics' 'kitchen')

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
            python -B run_prototype.py \
                --data_dir "${data}${src_domain}-${tar_domain}" --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_batch_size 15 --num_train_epochs 10 --learning_rate 2e-5 --max_seq_length 512 \
                --eval_logging_steps 100 --eval_batch_size 200 --labeled_num 1600 \
                --bert_model 'bert_base' --do_train --do_eval --evaluate_during_training 
            printf "\n\n\n"

        fi
    done
done




