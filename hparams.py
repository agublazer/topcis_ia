import tensorflow as tf
import numpy as np


# Default hyperparameters
hparams = tf.contrib.training.HParams(
	cleaners='english_cleaners',

	# Audio
	num_mels=80,
	num_freq=513,
	rescale=True,
	rescaling_max=0.999,
	trim_silence=True,

	# Mel spectrogram
	fft_size=1024,
	hop_size=256,
	sample_rate=22050,  # 22050 Hz (corresponding to ljspeech dataset)
	frame_shift_ms=None,

	signal_normalization=True,
	allow_clipping_in_normalization=True,
	symmetric_mels=True,  # Whether to scale the data to be symmetric around 0
	max_abs_value=4., 

	# Limits
	min_level_db=- 100,
	ref_level_db=20,
	fmin=125,
	fmax=7600,

	# Tacotron
	outputs_per_step=1,
	stop_at_any=True,  # Determines whether the decoder should stop when predicting <stop> to any frame or to all of them

	embedding_dim=512,

	enc_conv_num_layers=3,
	enc_conv_kernel_size=(5, ),
	enc_conv_channels=512,
	encoder_lstm_units=256,

	smoothing=False,  # Whether to smooth the attention normalization function
	attention_dim=128,  # dimension of attention space
	attention_filters=32,  # number of attention convolution filters
	attention_kernel=(31, ),  # kernel size of attention convolution
	cumulative_weights=True,  # Whether to cumulate (sum) all previous attention weights or simply feed previous weights

	prenet_layers=[256, 256],  # number of layers and number of units of prenet
	decoder_layers=2,  # number of decoder lstm layers
	decoder_lstm_units=1024,
	max_iters=2500,

	postnet_num_layers=5,  # number of postnet convolutional layers
	postnet_kernel_size=(5, ),  # size of postnet convolution filters for each layer
	postnet_channels=512,  # number of postnet convolution filters for each layer

	mask_encoder=False,
	impute_finished=False,  # Whether to use loss mask for padded sequences
	mask_finished=False, 

	predict_linear=False,

	quantize_channels=256,

	silence_threshold=2,

	# Mixture of logistic distributions:
	log_scale_min=float(np.log(1e-14)),

	# Tacotron Training
	tacotron_batch_size=32,  # number of training samples on each training steps
	tacotron_reg_weight=1e-6,  # regularization weight (for l2 regularization)
	tacotron_scale_regularization=True,

	tacotron_decay_learning_rate=True,
	tacotron_start_decay=50000,
	tacotron_decay_steps=50000,
	tacotron_decay_rate=0.4,
	tacotron_initial_learning_rate=1e-3,
	tacotron_final_learning_rate=1e-5,

	tacotron_zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
	tacotron_dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet

	tacotron_teacher_forcing_ratio=1.,
)
