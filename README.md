## init whole project
If you want to use GPU and CUDA 11.8, you should run the code in the new environment.

If you want to use CPU, you should run the code in the environment of environment.yml

## download the data

run  `python download_and_extract.py`;

## execute the train_embedding python

run `nohup python train_embedding.py > output.log 2>&1 &`

## best performance:

Overall best val:  (0.7330634218193111, 6, 100)

python social_lstm_model.py --epochs 20 --learning_rate 0.002 --enable_mean_pooling false --model 1 --warmup_steps 200 --dropout 0.2 --final_layer_social  --final_dense --enable_scheduler --include_text


python social_lstm_model.py --epochs 20 --learning_rate 0.005 --enable_mean_pooling false --model 1 --warmup_steps 200 --dropout 0.2 --final_layer_social  --final_dense --enable_scheduler --include_text



python social_lstm_model.py --epochs 20 --learning_rate 0.05 --enable_mean_pooling false --model 7 --warmup_steps 200 --dropout 0.5  --final_layer_social  --final_dense --enable_scheduler


python social_lstm_model.py --epochs 20 --learning_rate 0.001 --enable_mean_pooling false --model 7 --warmup_steps 200 --dropout 0.5  --final_layer_social  --final_dense --enable_scheduler







python train_embedding.py --loss_type 2 --embedding_type 2 --negative_sample 5

python train_embedding.py --loss_type 3 --embedding_type 3 --negative_sample 5