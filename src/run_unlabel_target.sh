#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

domains=('dvd' 'books' 'electronics')

data='../data/amazon/'
output='../response/amazon/'
for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            echo "####################### ${src_domain}===>>>${tar_domain}#######################:"
            python -B run_unlabel_target.py \
                --data_dir "${data}${src_domain}-${tar_domain}" --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_batch_size 16 --num_train_epochs 8 --learning_rate 3e-5 --max_seq_length 512 \
                --eval_logging_steps 100 --eval_batch_size 200 \
                --bert_model 'bert_base' --do_train --do_eval
            printf "\n\n\n"
        fi
    done
done




