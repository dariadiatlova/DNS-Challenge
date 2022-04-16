import argparse
import configparser as CP
import os
import random
import re
from pathlib import Path
from typing import Dict, Optional, List

import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from tqdm import trange

from audiolib import audioread, audiowrite, segmental_snr_mixer, normalize

MAXTRIES = 50
MAXFILELEN = 100

np.random.seed(5)
random.seed(5)


def get_file_names(tsv_filepath: str, root_path: str):
    """
    Function takes the path the tsv file and returns the column with filenames, changed from .mp3 to .wav.
    :param tsv_filepath: str
    :param root_path: str path to the root directory with the extracted filenames
    :return: ndarray
    """
    df = pd.read_csv(tsv_filepath, sep='\t')
    mp3_filenames = np.array(df.path)
    pattern = re.compile(".*(?=.mp3)")
    wav_filenames = []
    for filename in mp3_filenames:
        if pattern.search(filename) is not None:
            wav_filenames.append(root_path + pattern.search(filename).group() + ".wav")
    return np.array(wav_filenames)


def _audio_activity_check(filename) -> Optional[bool]:
    """
    Function take path to th audio file, loads it and return True if file is not empty and readable and None otherwise.
    :param filename:
    :return:
    """
    try:
        print(filename)
        y, sr = librosa.load(filename, sr=48000)
        if np.max(abs(y)) == 0:
            return
        else:
            return True
    except Exception:
        # return None if for any reason could not read the audio file
        return


def _write_txt(string, filename):
    """
    Function takes a  string and the path ending with '.txt' and writes the string to the given path.
    :param string: stacked text to write to the file.
    :param filename: path to write the transcript.
    :return: None
    """
    with open(filename, 'w') as f:
        f.write(string)
        f.close()
    return


def _get_sentence(wav_filename: str, df: pd.DataFrame) -> str:
    """
    Function takes wav filename takes its name (all before wav) and returns filename.txt
    :param wav_filename: string of format <filename>.wav
    :return: txt_file_name: string of format <filename>.txt
    """
    pattern = re.compile(".*(?=.wav)")
    if pattern.search(wav_filename) is not None:
        wav_filename = re.compile("[^\/]+$").search(wav_filename).group()
        mp3_file_name = pattern.search(wav_filename).group() + ".mp3"
        sentence = df[df.index == mp3_file_name].sentence[0].lower()
        return sentence
    return " "


def _get_reverb(params: Dict):
    """
    Function generates a reverberation to add to the audio.
    :param params: dictionary with config parameters
    :return: np.array
    """
    rir_index = random.randint(0, len(params['myrir']) - 1)

    my_rir = os.path.normpath(params['myrir'][rir_index])
    (fs_rir, samples_rir) = wavfile.read(my_rir)

    my_channel = int(params['mychannel'][rir_index])

    if samples_rir.ndim == 1:
        samples_rir_ch = np.array(samples_rir)

    elif my_channel > 1:
        samples_rir_ch = samples_rir[:, my_channel - 1]
    else:
        samples_rir_ch = samples_rir[:, my_channel - 1]
    return samples_rir_ch


def add_pyreverb(clean_speech, rir):
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[:clean_speech.shape[0]]
    return reverb_speech


def get_noise_audiofile(params: Dict):
    """
    Function generates random filename with noise audio file.
    :param params: dictionary with config.
    :return: Tuple[np.ndarray, str] noise audio and domain from which the audio was picked.
    """
    foldernames = list(params["noise_filenames_dict"].keys())
    folder_picked = random.choice(foldernames)
    filename = random.choice(params["noise_filenames_dict"][folder_picked])
    noise = audioread(filename)[0]
    return noise, folder_picked


def gen_new_audio(filenames: List[str], params: Dict, df : pd.DataFrame):
    """
    Function takes names of wav files, add them one to another and saves new audio, generated txt.
    :param filenames: List of absolute wav files to add to each other.
    :param params: Dictionary with parameters config.
    :return:
    """
    text_to_add = []
    fs_output = params['fs']
    output_audio = np.zeros(1).astype(np.float32)
    remaining_audio_length = int(params['audio_length'] * params['fs']) - 1
    silence = np.zeros(params['fs'] // 5).astype(np.float32)
    for audiopath in filenames:
        # add new audio-file till the target length will be reached
        if remaining_audio_length > 0:
            input_audio, fs_input = audioread(audiopath)

            if fs_input != fs_output:
                input_audio = librosa.resample(input_audio, fs_input, fs_output)

            if len(input_audio) < remaining_audio_length:
                text_to_add.append(_get_sentence(audiopath, df))
                output_audio = np.concatenate([output_audio, input_audio])

                remaining_audio_length -= input_audio.shape[0]
                if len(silence) > remaining_audio_length:
                    silence = silence[:remaining_audio_length]
                output_audio = np.concatenate([output_audio, silence])
                remaining_audio_length -= silence.shape[0]

            else:
                padding = np.zeros(remaining_audio_length)
                output_audio = np.concatenate([padding[:len(padding) // 2], output_audio, padding[len(padding) // 2:]])
                remaining_audio_length -= padding.shape[0]

    if remaining_audio_length > 0:
        padding = np.zeros(remaining_audio_length)
        output_audio = np.concatenate([padding[:len(padding) // 2], output_audio, padding[len(padding) // 2:]])

    assert len(output_audio) == int(params['audio_length'] * params['fs']), f"Output audio with size: " \
                                                                            f"{len(output_audio)} does not match target" \
                                                                            f"length of {params['audio_length']}."
    new_txt = " ".join(text_to_add)
    return output_audio, new_txt


def main_gen(params: Dict):
    """
    Function calls gen_audio() to generate the audio signals and verifies that they meet
    the requirements, and writes the files to storage.
    :param params: dict with parameters
    :return: None
    """
    clean_file_names = params['cleanfilenames']
    files_to_generate = params['num_files']
    file_indices = list(range(params["num_cleanfiles"]))
    df = pd.read_csv(params["csv_path"], sep="\t").set_index("path")

    for j in trange(files_to_generate):
        # each time use 6 audio files to generate audio as 1 audio ~ 1 sec len
        indices_to_use = random.choices(file_indices, k=6)

        # check that all sampled files are not empty
        for i, idx in enumerate(indices_to_use):
            if _audio_activity_check(clean_file_names[idx]):
                pass
            else:
                while True:
                    idx = random.choice(file_indices)
                    if _audio_activity_check(clean_file_names[idx]):
                        indices_to_use[i] = idx
                        break

        clean_audio, new_txt = gen_new_audio(np.array(clean_file_names)[indices_to_use], params, df)
        clean_audio = clean_audio / (max(abs(clean_audio)) + np.finfo(float).eps)
        clean_audio = normalize(clean_audio)
        _write_txt(new_txt, params["transcripts_destination"] + f"/t_fake_noisy{j}.txt")

        # add reverberation to clean generated audio and writes in to file
        samples_rir_ch = _get_reverb(params)
        clean_audio = add_pyreverb(clean_audio, samples_rir_ch)

        audiowrite(params["clean_destination"] + f"/{j}_clean.wav", clean_audio)
        audiowrite(params["fake_noisy_destination"] + f"/{j}_fake_noisy.wav", clean_audio)
    return


def main_body():
    parser = argparse.ArgumentParser()

    # configurations: read youtube_noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument('--cfg', default='cv_fake_noisyspeech_synthesizer.cfg',
                        help='Read cv_fake_noisyspeech_synthesizer.cfg for all the details')
    parser.add_argument('--cfg_str', type=str, default='noisy_speech')
    args = parser.parse_args()

    params = dict()
    params['args'] = args
    cfgpath = Path(__file__).parent.parent / f"configs/{args.cfg}"
    assert os.path.exists(cfgpath), f'No configuration file as [{cfgpath}]'

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params['cfg'] = cfg._sections[args.cfg_str]
    cfg = params['cfg']

    # paths
    params["transcripts_destination"] = cfg["transcripts_destination"]
    params["clean_destination"] = cfg["clean_destination"]
    params["fake_noisy_destination"] = cfg["fake_noisy_destination"]
    params["csv_path"] = cfg["csv_path"]

    # audio params
    params['fs'] = int(cfg['sampling_rate'])
    params['audioformat'] = cfg['audioformat']
    params['audio_length'] = float(cfg['audio_length'])
    params['silence_length'] = float(cfg['silence_length'])
    params['total_hours'] = float(cfg['total_hours'])

    # rir
    params['rir_choice'] = int(cfg['rir_choice'])
    params['lower_t60'] = float(cfg['lower_t60'])
    params['upper_t60'] = float(cfg['upper_t60'])
    params['rir_table_csv'] = str(cfg['rir_table_csv'])

    # rir
    temp = pd.read_csv(params['rir_table_csv'], skiprows=[1], sep=',', header=None,
                       names=['wavfile', 'channel', 'T60_WB', 'C50_WB', 'isRealRIR'])
    temp.keys()

    rir_wav = temp['wavfile'][1:]  # 115413
    rir_channel = temp['channel'][1:]
    rir_t60 = temp['T60_WB'][1:]
    rir_isreal = temp['isRealRIR'][1:]

    rir_wav2 = [w.replace('\\', '/') for w in rir_wav]
    rir_channel2 = [w for w in rir_channel]
    rir_t60_2 = [w for w in rir_t60]
    rir_isreal2 = [w for w in rir_isreal]

    lower_t60 = params['lower_t60']
    upper_t60 = params['upper_t60']

    real_indices = [i for i, x in enumerate(rir_isreal2) if x == "1"]

    chosen_i = []
    for i in real_indices:
        if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
            chosen_i.append(i)

    myrir = [rir_wav2[i] for i in chosen_i]
    mychannel = [rir_channel2[i] for i in chosen_i]
    myt60 = [rir_t60_2[i] for i in chosen_i]
    params['myrir'] = myrir
    params['mychannel'] = mychannel
    params['myt60'] = myt60

    # num of files
    params['num_files'] = int((params['total_hours'] * 60 * 60) / params['audio_length'])
    print(f"Number of files to be synthesized: {params['num_files']}")

    cleanfilenames = get_file_names(cfg["csv_path"], cfg["root_path"])
    params['cleanfilenames'] = cleanfilenames
    params['num_cleanfiles'] = len(params['cleanfilenames'])

    main_gen(params)


if __name__ == '__main__':

    main_body()
