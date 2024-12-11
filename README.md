## init whole project
If you want to use GPU and CUDA 11.8, you should run the code in the new environment.

If you want to use CPU, you should run the code in the environment of environment.yml

## download the data

run  `python download_and_extract.py`;

## execute the train_embedding python

run `nohup python train_embedding.py > output.log 2>&1 &`

## best performance:

 python social_lstm_model.py --epochs 20 --learning_rate 0.005 --enable_mean_pooling false --model 1 --warmup_steps 200 --dropout 0.2 --final_layer_social  --final_dense

 python social_lstm_model.py --epochs 20 --learning_rate 0.005 --enable_mean_pooling false --model 1 --warmup_steps 200 --dropout 0.2 --final_layer_social  --final_dense --embedding_self_trained true


 python social_lstm_model.py --epochs 20 --learning_rate 0.005 --enable_mean_pooling false --model 1 --warmup_steps 200 --dropout 0.2 --final_layer_social  --final_dense --include_meta

python train_embedding.py --loss_type 2 --embedding_type 2 --negative_sample 5

python train_embedding.py --loss_type 2 --embedding_type 3 --negative_sample 5