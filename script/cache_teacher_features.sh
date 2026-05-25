cd src
python -m training.cache_teacher_features \
    --output-cache ../cache/teacher_features.pt \
    --teacher-model ViT-B-16 \
    --teacher-pretrained ../pretrained_models/vit_b_16.pt \
    --train-data "path/to/cc3m_train.csv,path/to/cc12m.csv" \
    --data-root "path/to/cc3m/images/,path/to/cc12m/images/" \
    --dataset-type csv \
    --csv-img-key filepath \
    --csv-caption-key title \
    --batch-size 128 \
    --workers 8 \
    --device cuda \
    --verbose
