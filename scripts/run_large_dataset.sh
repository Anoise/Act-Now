## M
online_learning='full'
root_path=datasets/large/Milano
itr=1
method=large
ns=(1 )
bszs=(1 )
datasets=(Milano)
# when you run FreTS or PatchTST, it may case out-of-memory, plase set small batch size.
models=(Lade Informer Autoformer FEDformer Periodformer PSLD FourierGNN DLinear Transformer)
lens=(24 48 72 )
for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for dataset in ${datasets[*]}; do
for model in ${models[*]}; do
for len in ${lens[*]}; do
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --is_training 1 \
    --method $method \
    --root_path $root_path \
    --n_inner $n \
    --test_bsz $bsz \
    --data $dataset\
    --seq_len 36 \
    --pred_len $len \
    --itr $itr \
    --train_epochs 1 \
    --learning_rate 1e-3 \
    --online_learning $online_learning \
    --n_node 10000 \
    --n_part 25 \
    --batch_size 64 \
    --sample_freq 1 \
    --online_valid 0 \
    --model $model \
    >'Results/'$dataset'_'$method'_'$model'_'$len.log
done
done
done
done
done










