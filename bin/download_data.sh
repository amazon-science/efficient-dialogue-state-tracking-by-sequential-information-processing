# =============================================
# Download data related to the MultiWoz dataset
# =============================================


set -e


mkdir -p data/raw
cd data/raw


# MultiWoz 2.1 and 2.2
git clone https://github.com/budzianowski/multiwoz.git
unzip multiwoz/data/MultiWOZ_2.1.zip
mv MultiWOZ_2.1 multiwoz_21
mv multiwoz/data/MultiWOZ_2.2 multiwoz_22
wget https://raw.githubusercontent.com/facebookresearch/Zero-Shot-DST/main/T5DST/utils/slot_description.json -O ./multiwoz_21/slot_description.json
rm -f multiwoz_21/.DS_Store
rm -f multiwoz_22/.DS_Store
rm -rf __MACOSX
rm -rf MultiWOZ_2.1.zip
rm -rf multiwoz
rm -f multiwoz_21/*_db.json multiwoz_21/*.md multiwoz_21/*README* multiwoz_21/.README.swp


# MultiWoz 2.3
wget https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip
unzip MultiWOZ2_3.zip
rm -rf MultiWOZ2_3.zip
mv MultiWOZ2_3 multiwoz_23
rm -f multiwoz_23/.DS_Store


# MultiWoz 2.4
wget https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip
unzip MULTIWOZ2.4.zip
rm -rf MULTIWOZ2.4.zip
mv MULTIWOZ2.4 multiwoz_24
rm -f multiwoz_24/.DS_Store


# Schema-Guided Dataset
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git
mkdir --parents ./schema_guided_dialogue
for split in train dev test
do
    mv ./dstc8-schema-guided-dialogue/$split ./schema_guided_dialogue/$split
done
rm -rf ./dstc8-schema-guided-dialogue
