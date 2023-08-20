#!/bin/bash

echo "Downloading model.tar.gz..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15XaHPwHSjOPe1ZLubWKmK6StF7KKPJtH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15XaHPwHSjOPe1ZLubWKmK6StF7KKPJtH" -O model.tar.gz && rm -rf /tmp/cookies.txt

echo "Extracting model.tar.gz..."
tar -xzf model.tar.gz
rm model.tar.gz

echo "Downloading eval.tar.gz..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17D7yCfB5WrhQKTkj-Q8m-tIFuYXUVoCA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17D7yCfB5WrhQKTkj-Q8m-tIFuYXUVoCA" -O eval.tar.gz && rm -rf /tmp/cookies.txt

echo "Extracting eval.tar.gz..."
tar -xzf eval.tar.gz
rm eval.tar.gz