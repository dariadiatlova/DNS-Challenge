# Configuration file for train val test split

# - train_rate: percent of files to use for training
# - val_rate: percent of files to use for validation
# - test_rate: percent of files to use for test

# - source_directory: directory with noisy files used for division
# - train_directory: directory with noisy wav files to use for train
# - val_directory: directory with noisy wav files to use for validation
# - test_directory: directory with noisy wav files to use for testing

train_rate=70
val_rate=10
test_rate=20

source_directory="/home/dadyatlova_1/russian_speech_denoiser/DNS-Challenge/datasets/training_set_jan14_min35_10_40h/noisy"
train_directory="/home/dadyatlova_1/russian_speech_denoiser/DNS-Challenge/datasets/training_set_jan14_min35_10_40h/noisy_train"
val_directory="/home/dadyatlova_1/russian_speech_denoiser/DNS-Challenge/datasets/training_set_jan14_min35_10_40h/noisy_val"
test_directory="/home/dadyatlova_1/russian_speech_denoiser/DNS-Challenge/datasets/training_set_jan14_min35_10_40h/noisy_test"

N=$(ls $source_directory | wc -l)

TEN_PERCENT=$((N / 100))
TRAIN_RATE=$((TEN_PERCENT * train_rate))
VAL_RATE=$((TEN_PERCENT * val_rate + TRAIN_RATE))
TEST_RATE=$((TEN_PERCENT * test_rate + TRAIN_RATE + VAL_RATE))

declare -a LIST=($(ls $source_directory))
COUNTER=0

mkdir $train_directory
mkdir $val_directory
mkdir $test_directory

for i in ${LIST[@]}; do
  n=${i##*_}
  path="${source_directory}/${i}"
  if (($COUNTER < $TRAIN_RATE)); then cp $path "$train_directory/noisy_$n";
  elif (($COUNTER < $VAL_RATE)); then cp $path "$val_directory/noisy_$n";
  else cp $path "$test_directory/noisy_$n"
  fi
  ((COUNTER++))
done

echo "Files were moved successfully"
