#!/usr/bin/env bash

gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

echo "Downloading dataset..."
gdrive_download 131dl5KAnJ6f1xeZu9x5zyalotykDDdGJ icons.zip
gdrive_download 1lD9vwkSZ7gNHg3vBwRIPCjDgeP-3P8UX transformed_svgs.zip
gdrive_download 1wSTmMs_kxHFAkSo2ewC_hbuC44oI5Bdg transformed_pngs.zip

echo "Download done. Unzipping..."
unzip icons.zip -d icons
unzip transformed_svgs.zip
unzip transformed_pngs.zip -d pngs
echo "Done."
