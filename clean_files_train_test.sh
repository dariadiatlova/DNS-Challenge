source_directory="/home/dadyatlova_1/russian_speech_denoiser/DNS-Challenge/datasets/training_set_jan14_min35_10_40h"

TRAIN_DIR="$source_directory/noisy_train"
VAL_DIR="$source_directory/noisy_val"
TEST_DIR="$source_directory/noisy_test"

declare -a TRAIN_ARRAY=($(ls $TRAIN_DIR))
read -d'\n' TRAIN_ARRAY < <(printf '%s\n' "${TRAIN_ARRAY[@]}"|tac)
declare -a VAL_ARRAY=($(ls $VAL_DIR))
read -d'\n' VAL_ARRAY < <(printf '%s\n' "${VAL_ARRAY[@]}"|tac)
declare -a TEST_ARRAY=($(ls $TEST_DIR))
read -d'\n' TEST_ARRAY < <(printf '%s\n' "${TEST_ARRAY[@]}"|tac)

mkdir "$source_directory/clean_train"
mkdir "$source_directory/clean_val"
mkdir "$source_directory/clean_test"

for name in ${TRAIN_ARRAY[@]}; do
  i=${name##*_}
  cp "${source_directory}/clean/clean_fileid_${i}" "$source_directory/clean_train"
done

for name in ${VAL_ARRAY[@]}; do
  i=${name##*_}
  cp "${source_directory}/clean/clean_fileid_${i}" "$source_directory/clean_val"
done

for name in ${TEST_ARRAY[@]}; do
  i=${name##*_}
  cp "${source_directory}/clean/clean_fileid_${i}" "$source_directory/clean_test"
done

echo "Files were copied successfully"
