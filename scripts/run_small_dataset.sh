## M
root_path=datasets/small/
itr=1
method=small
ns=(1 )
bszs=(1 )
datasets=(ETTh2 ETTm1 WTH ECL Traffic)
# when you run FreTS or PatchTST, it may case out-of-memory, plase set small batch size.
models=(Lade Informer Autoformer FEDformer Periodformer PSLD FourierGNN DLinear Transformer) 
lens=(24 48 72 )
for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for dataset in ${datasets[*]}; do
for model in ${models[*]}; do
for len in ${lens[*]}; do
CUDA_VISIBLE_DEVICES=0 python -u main.py --method $method --root_path $root_path --n_inner $n --test_bsz $bsz --data $dataset --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $itr --train_epochs 10 --learning_rate 1e-3 --online_learning 'full' --model $model >'Results/'$dataset'_'$method'_'$model'_'$len.log
done
done
done
done
done