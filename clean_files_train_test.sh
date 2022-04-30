source_clean_directory="/home/dadyatlova_1/dataset/main/common_voice/no_reverb_clean_wav"
source_directory="/home/dadyatlova_1/dataset/main/no_reverb_50h"

TRAIN_DIR="$source_directory/train_noisy"
VAL_DIR="$source_directory/val_noisy"
TEST_DIR="$source_directory/test_noisy"

declare -a TRAIN_ARRAY=($(ls $TRAIN_DIR))
read -d'\n' TRAIN_ARRAY < <(printf '%s\n' "${TRAIN_ARRAY[@]}"|tac)
declare -a VAL_ARRAY=($(ls $VAL_DIR))
read -d'\n' VAL_ARRAY < <(printf '%s\n' "${VAL_ARRAY[@]}"|tac)
declare -a TEST_ARRAY=($(ls $TEST_DIR))
read -d'\n' TEST_ARRAY < <(printf '%s\n' "${TEST_ARRAY[@]}"|tac)

mkdir -p "$source_directory/train_clean"
mkdir -p "$source_directory/val_clean"
mkdir -p "$source_directory/test_clean"

for name in ${TRAIN_ARRAY[@]}; do
  cp "${source_clean_directory}/${name}" "$source_directory/train_clean"
done

for name in ${VAL_ARRAY[@]}; do
  cp "${source_clean_directory}/${name}" "$source_directory/val_clean"
done

for name in ${TEST_ARRAY[@]}; do
  cp "${source_clean_directory}/${name}" "$source_directory/test_clean"
done

echo "Files were copied successfully"
