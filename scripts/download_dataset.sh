#!/bin/bash

# For testing locally
FLICKER_8K_TEXT="1sIxT8WrW21vaQvUY3BLGnnmAY-ocZhpO"
FLICKER_8K_IMAGE="176wGCHHp2DpoDblsliEkX4fTpfQUbZOq"

gdown "$FLICKER_8K_TEXT" -O "datasets/download_ds_file.zip" &
gdown "$FLICKER_8K_IMAGE" -O "datasets/download_image_file.zip" &
