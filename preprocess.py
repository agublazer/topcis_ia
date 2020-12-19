import os
from tqdm import tqdm
from datasets import preprocessor
from hparams import hparams


def preprocess(n_jobs, input_folders, out_dir):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)
	metadata = preprocessor.build_from_path(input_folders, mel_dir, linear_dir, wav_dir, n_jobs)
	write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))


def run_preprocess(path, n_jobs):
	input_folders = path + '/LJSpeech-1.1'
	output_folder = path + 'training_data'

	preprocess(n_jobs, input_folders, output_folder)


def main():
	run_preprocess('', 4)


if __name__ == '__main__':
	main()
