from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datasets import audio
import os
import numpy as np 
from hparams import hparams
from tacotron.utils.utils import mulaw_quantize


def build_from_path(input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=4):
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split('|')
				wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(parts[0]))
				text = parts[2]
				futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, index, wav_path, text)))
				index += 1

	return [future.result() for future in futures if future.result() is not None]


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text):

	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path)
	except :
		print('file {} present in csv not in folder'.format(
			wav_path))
		return None

	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	if hparams.trim_silence:
		wav = audio.trim_silence(wav)

	out = mulaw_quantize(wav, hparams.quantize_channels)

	start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
	wav = wav[start: end]
	out = out[start: end]

	constant_values = mulaw_quantize(0, hparams.quantize_channels)
	out_dtype = np.int16

	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	linear_spectrogram = audio.linearspectrogram(wav).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1] 

	assert linear_frames == mel_frames

	l, r = audio.pad_lr(wav, hparams.fft_size, audio.get_hop_size())

	out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	time_steps = len(out)
	assert time_steps >= mel_frames * audio.get_hop_size()

	out = out[:mel_frames * audio.get_hop_size()]
	assert time_steps % audio.get_hop_size() == 0

	audio_filename = 'speech-audio-{:05d}.npy'.format(index)
	mel_filename = 'speech-mel-{:05d}.npy'.format(index)
	linear_filename = 'speech-linear-{:05d}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)