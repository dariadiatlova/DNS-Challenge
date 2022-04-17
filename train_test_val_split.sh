# Configuration file for train val test split

# - train_rate: percent of files to use for training
# - val_rate: percent of files to use for validation
# - test_rate: percent of files to use for test

# - noisy_source_directory: directory with noisy files used for division
# - clean_source_directory: directory with clean files used for division

# - train_noisy_directory: directory with noisy wav files to use for train
# - val_noisy_directory: directory with noisy wav files to use for validation
# - test_noisy_directory: directory with noisy wav files to use for testing

# - train_clean_directory: directory with clean wav files to use for train
# - val_clean_directory: directory with clean wav files to use for validation
# - test_clean_directory: directory with clean wav files to use for testing

train_rate=70
val_rate=10
test_rate=20

noisy_source_directory="/home/dadyatlova_1/dataset/main/data_120_hours/noisy_wav"
clean_source_directory="/home/dadyatlova_1/dataset/main/data_120_hours/clean_wav"

train_noisy_directory="/home/dadyatlova_1/dataset/main/data_120_hours/noisy_train"
val_noisy_directory="/home/dadyatlova_1/dataset/main/data_120_hours/noisy_val"
test_noisy_directory="/home/dadyatlova_1/dataset/main/data_120_hours/noisy_test"

train_clean_directory="/home/dadyatlova_1/dataset/main/data_120_hours/clean_train"
val_clean_directory="/home/dadyatlova_1/dataset/main/data_120_hours/clean_val"
test_clean_directory="/home/dadyatlova_1/dataset/main/data_120_hours/clean_test"

N=$(ls $noisy_source_directory | wc -l)

TEN_PERCENT=$((N / 100))
TRAIN_RATE=$((TEN_PERCENT * train_rate))
VAL_RATE=$((TEN_PERCENT * val_rate + TRAIN_RATE))
TEST_RATE=$((TEN_PERCENT * test_rate + TRAIN_RATE + VAL_RATE))

declare -a LIST=($(ls $noisy_source_directory))
LIST=( $(shuf -e "${LIST[@]}") )

mkdir -p $train_noisy_directory $val_noisy_directory $test_noisy_directory $train_clean_directory $val_clean_directory $test_clean_directory

COUNTER=0
for i in ${LIST[@]}; do
  base_name=$(basename ${i})
  noisy_path="${noisy_source_directory}/${base_name}"
  clean_path="${clean_source_directory}/${base_name}"

  if (($COUNTER < $TRAIN_RATE)); then
    cp $noisy_path "$train_noisy_directory/$base_name" && cp $clean_path "$train_clean_directory/$base_name";
  elif (($COUNTER < $VAL_RATE)); then
    cp $noisy_path "$val_noisy_directory/$base_name" && cp $clean_path "$val_clean_directory/$base_name";
  else
    cp $noisy_path "$test_noisy_directory/$base_name" && cp $clean_path "$test_clean_directory/$base_name"
  fi
  ((COUNTER++))

done

echo "Files were moved successfully"
