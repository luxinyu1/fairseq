if [ "$1" != "no-preprocess" ]; then
    
    # learn BPE

    cd ../fastBPE

    echo "Learning BPE..."
    ./fast learnbpe 50000 ../datasets/wikilarge/wikilarge.train.complex ../datasets/wikilarge/wikilarge.train.simple > ../transformer/bpecodes50000

    echo "Applying BPE..."

    TASK=wikilarge
    
    cd ../transformer
    mkdir ${TASK}
    cd ../fastBPE

    for split in 'train' 'test' 'valid'; do
        for type in 'complex' 'simple'; do
            ./fast applybpe ../transformer/${TASK}/${split}.bpe.${type} ../datasets/wikilarge/wikilarge.${split}.${type} ../transformer/bpecodes50000
        done
    done

    cd ../transformer
    
    # preprocess
    fairseq-preprocess \
      --source-lang "complex" \
      --target-lang "simple" \
      --trainpref "./${TASK}/train.bpe" \
      --testpref "./${TASK}/test.bpe" \
      --validpref "./${TASK}/valid.bpe"  \
      --destdir "./${TASK}-bin/" \
      --bpe fastbpe \
      --workers 60
    
fi

# training

# CUDA_VISIBLE_DEVICES=0 fairseq-train ./${TASK}-bin/ \
#     --task translation  \
#     --bpe fastbpe \
#     --source-lang complex --target-lang simple \
#     --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 2048 \
#     --arch transformer --save-dir ./checkpoints/

CUDA_VISIBLE_DEVICES=0 fairseq-train ${TASK}-bin/ \
    --max-tokens 2048 \
    --task translation \
    --source-lang complex --target-lang simple \
    --arch transformer \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 20000 --warmup-updates 5000 \
    --memory-efficient-fp16 --update-freq 1 \
    --skip-invalid-size-inputs-valid-test