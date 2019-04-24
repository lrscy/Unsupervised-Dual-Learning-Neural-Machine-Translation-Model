#!/bin/bash

languageA="chinese"
languageB="english"
abbrA="zh"
abbrB="en"
nmt_vocab_sizeA=30000
nmt_vocab_sizeB=30000

# preprocess original data
echo "Preprocessing"
cd data/ori/
python data_preprocess.py --languageA ${languageA} --languageB ${languageB}
cd ../..

echo "Copying data"
if [ ! -d "lm/data" ]; then
   mkdir "lm/data"
fi
cp -r data/new/ lm/data/

if [ ! -d "nmt/data" ]; then
   mkdir "nmt/data"
fi
cp -r data/new/ nmt/data/

# train language models
echo "Training language model"

if [ ! -d "lm/models" ]; then
   mkdir "lm/models"
fi

cd lm/
python main.py --data data/new/${languageA} --cuda --save models/${languageA}.pt
python main.py --data data/new/${languageB} --cuda --save models/${languageB}.pt

cp data/new/${languageA}/dict.pkl models/dict.${abbrA}.pkl
cp data/new/${languageB}/dict.pkl models/dict.${abbrB}.pkl
cd ..

# train nmt
echo "Training warm-up nmt"

cd nmt/
echo "Generating vocabs"
cd data/new
cd ${languageA}
mv train.txt train.${abbrA}
mv valid.txt valid.${abbrA}
mv test.txt  test.${abbrA}
cd ..
cd ${languageB}
mv train.txt train.${abbrB}
mv valid.txt valid.${abbrB}
mv test.txt  test.${abbrB}
cd ..
mv ${languageA}/* .
mv ${languageB}/* .
rm -r ${languageA}
rm -r ${languageB}
cd ../..

python vocab.py --src_vocab_size ${nmt_vocab_sizeA} \
                --tgt_vocab_size ${nmt_vocab_sizeB} \
                --train_src data/new/train.$abbrA \
                --train_tgt data/new/train.$abbrB \
                --output vocab.$abbrA$abbrB.bin
mv vocab.$abbrA$abbrB.bin data/new/

python vocab.py --src_vocab_size ${nmt_vocab_sizeB} \
                --tgt_vocab_size ${nmt_vocab_sizeA} \
                --train_src data/new/train.$abbrB \
                --train_tgt data/new/train.$abbrA \
                --output vocab.$abbrB$abbrA.bin
mv vocab.$abbrB$abbrA.bin data/new/

echo "Training nmts"
./scripts/train-small.sh $abbrA $abbrB $abbrA$abbrB
./scripts/train-small.sh $abbrB $abbrA $abbrB$abbrA

if [ ! -d "nmt/models" ]; then
   mkdir "nmt/models"
fi

mv model.$abbrA$abbrB.* models/
mv model.$abbrB$abbrA.* models/
cd ..

# train dual-learning model
echo "Training dual-learning model"
./train-dual.sh

if [ ! -d "models" ]; then
   mkdir "models"
fi

mv modelA.* models/
mv modelB.* models/

# test
echo "Test"
cd nmt/
if [ ! -d "results" ]; then
   mkdir "results"
fi

echo "Test on translation model $languageA to $languageB"
echo "Seq2seq model"
./scripts/test.sh data/new/test.$abbrA data/new/test.$abbrB models/model.$abbrA$abbrB.bin results/translated_${languageB}_seq2seq_model_test.txt
./scripts/multi-bleu.perl data/new/test.$abbrB < results/translated_${languageB}_seq2seq_model_test.txt

echo "Dual-learning model"
./scripts/test.sh data/new/test.$abbrA data/new/test.$abbrB ../models/modelA.iter800.bin results/translated_${languageB}_dual_model_test.txt
./scripts/multi-bleu.perl data/new/test.$abbrB < results/translated_${languageB}_dual_model_test.txt

echo "Test on translation model $languageB to $languageA"
echo "Seq2seq model"
./scripts/test.sh data/new/test.$abbrB data/new/test.$abbrA models/model.$abbrB$abbrA.bin results/translated_${languageA}_seq2seq_model_test.txt
./scripts/multi-bleu.perl data/new/test.$abbrA < results/translated_${languageA}_seq2seq_model_test.txt

echo "Dual-learning model"
./scripts/test.sh data/new/test.$abbrB data/new/test.$abbrA ../models/modelB.iter800.bin results/translated_${languageA}_dual_model_test.txt
./scripts/multi-bleu.perl data/new/test.$abbrA < results/translated_${languageA}_dual_model_test.txt
