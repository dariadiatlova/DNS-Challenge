cd /home/dadyatlova_1/dataset/main
mkdir data_120_hours

source_txt1_directory="youtube/txt"
source_txt2_directory="youtube/fake_txt"
source_txt3_directory="common_voice/txt"
source_txt4_directory="common_voice/fake_noised_txt"

source_clean1_directory="youtube/clean_wav"
source_clean2_directory="youtube/fake_clean_wav"
source_clean3_directory="common_voice/clean_wav"
source_clean4_directory="common_voice/fake_clean_wav"

source_noisy1_directory="youtube/noised_wav"
source_noisy2_directory="youtube/fake_noised_wav"
source_noisy3_directory="common_voice/noised_wav"
source_noisy4_directory="common_voice/fake_noised_wav"

target_txt_directory="data_120_hours/txt"
target_clean_directory="data_120_hours/clean_wav"
target_noisy_directory="data_120_hours/noisy_wav"

mkdir -p $target_txt_directory $target_clean_directory $target_noisy_directory

# copy txt files
find $source_txt1_directory -name '*.txt' -exec cp {} $target_txt_directory \;
find $source_txt2_directory -name '*.txt' -exec cp {} $target_txt_directory \;
find $source_txt3_directory -name '*.txt' -exec cp {} $target_txt_directory \;
find $source_txt4_directory -name '*.txt' -exec cp {} $target_txt_directory \;

# copy clean audios
find $source_clean1_directory -name '*.wav' -exec cp {} $target_clean_directory \;
find $source_clean2_directory -name '*.wav' -exec cp {} $target_clean_directory \;
find $source_clean3_directory -name '*.wav' -exec cp {} $target_clean_directory \;
find $source_clean4_directory -name '*.wav' -exec cp {} $target_clean_directory \;

# copy noisy files
find $source_noisy1_directory -name '*.wav' -exec cp {} $target_noisy_directory \;
find $source_noisy2_directory -name '*.wav' -exec cp {} $target_noisy_directory \;
find $source_noisy3_directory -name '*.wav' -exec cp {} $target_noisy_directory \;
find $source_noisy4_directory -name '*.wav' -exec cp {} $target_noisy_directory \;
