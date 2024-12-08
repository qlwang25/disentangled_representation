#!/bin/bash
domains=('dvd' 'books' 'electronics' 'kitchen')

export CUDA_VISIBLE_DEVICES=0
data='../data/amazon/'
output='../run_out/amazon/'

for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            echo "####################### ${src_domain}===>>>${tar_domain} 1200 #######################:"
            python -B run_base_sa.py --task_type 'sa' \
                --data_dir "${data}${src_domain}-${tar_domain}" --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_batch_size 10 --num_train_epochs 8 --learning_rate 2e-5 --max_seq_length 512  --labeled_num 1600 \
                --bert_model 'bert_base' --seed 42 --do_train --do_eval --evaluate_during_training 
            printf "\n\n\n"

            echo "####################### ${src_domain}===>>>${tar_domain} 3000 #######################:"
            python -B run_base_sa.py --task_type 'sa' \
                --data_dir "${data}${src_domain}-${tar_domain}" --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_batch_size 10 --num_train_epochs 8 --learning_rate 2e-5 --max_seq_length 512  --unlabeled_num 3000 \
                --bert_model 'bert_base' --seed 42 --do_train --do_eval --evaluate_during_training 
            printf "\n\n\n"

        fi
    done
done




