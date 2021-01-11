#!bin/bash

python src/style_transfer.py \
    --content_image_pth data/content_images/tuebingen_neckarfront.jpg \
    --style_image_pth data/style_images/gogh2.jpg \
    --image_height 512 \
    --image_width 512 \
    --from_noise True \
    --content_weight 1 \
    --style_weight 0 \
    --content_layer_name conv3_2 \
    --style_layer_names "conv1_1" "conv2_1" "conv3_1" "conv4_1" "conv5_1" \
    --iter_count 5 \
    --optimizer "lbfgs"