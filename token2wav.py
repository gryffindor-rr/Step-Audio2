import io

import torch
import torchaudio
import s3tokenizer
import onnxruntime
import numpy as np
import soundfile as sf

import torchaudio.compliance.kaldi as kaldi
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
from hyperpyyaml import load_hyperpyyaml

def fade_in_out(fade_in_mel:torch.Tensor, fade_out_mel:torch.Tensor, window:torch.Tensor):
    """perform fade_in_out in tensor style
    """
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = \
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel


class Token2wav():

    def __init__(self, model_path, float16=False, stream_context_window=100):
        self.float16 = float16
        self.stream_context_window = stream_context_window

        self.audio_tokenizer = s3tokenizer.load_model(f"{model_path}/speech_tokenizer_v2_25hz.onnx").cuda().eval()

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])

        with open(f"{model_path}/flow.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
            self.flow = configs['flow']
        if float16:
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{model_path}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

        self.cache = {}

        # stream conf
        self.mel_cache_len = 8  # hard-coded, 160ms
        self.source_cache_len = int(self.mel_cache_len * 480)   # 50hz mel -> 24kHz wave
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).cuda()

        # hifigan cache
        self.hift_cache_dict = {}


    def _prepare_prompt(self, prompt_wav):
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(self.spk_model.run(
            None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
        )[0], device='cuda')

        audio, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).cuda()
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device='cuda')
        prompt_mels = torch.nn.functional.pad(prompt_mels, (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]), mode='replicate')
        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def __call__(self, generated_speech_tokens, prompt_wav):
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device='cuda')
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device='cuda')

        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(generated_speech_tokens, generated_speech_tokens_lens,
                prompt_speech_tokens, prompt_speech_tokens_lens,
                prompt_mels, prompt_mels_lens, spk_emb, 10)

        wav, _ = self.hift(speech_feat=mel)
        output = io.BytesIO()
        wav_np = wav.cpu().numpy().T
        sf.write(output, wav_np, 24000, format='WAV')

        return output.getvalue()

    def set_stream_cache(self, prompt_wav, preallocate_to_max=False, max_chunk_size=50):
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]
        self.stream_cache = self.flow.setup_cache(
            torch.cat([prompt_speech_tokens, prompt_speech_tokens[:, :3]], dim=1),
            prompt_mels, spk_emb, n_timesteps=10)

        # Track absolute position for conformer (for positional encoding)
        # This is independent of cache size when using sliding window
        self.conformer_offset = prompt_mels.shape[1]

        # Pre-allocate cache to maximum size for constant memory usage
        if preallocate_to_max:
            # Calculate ceiling: prompt + context_window + frames_per_chunk
            frames_per_chunk = (max_chunk_size - 3) * 2
            cache_ceiling = prompt_mels.shape[1] + self.stream_context_window + frames_per_chunk

            # Pad estimator attention cache to ceiling
            current_est_size = self.stream_cache['estimator_att_cache'].shape[4]
            if current_est_size < cache_ceiling:
                pad_size = cache_ceiling - current_est_size
                # Pad with zeros at the end
                padding = torch.zeros(
                    *self.stream_cache['estimator_att_cache'].shape[:4],
                    pad_size,
                    self.stream_cache['estimator_att_cache'].shape[5],
                    device='cuda',
                    dtype=self.stream_cache['estimator_att_cache'].dtype
                )
                self.stream_cache['estimator_att_cache'] = torch.cat([
                    self.stream_cache['estimator_att_cache'], padding
                ], dim=4)

            # Pad conformer attention cache to ceiling
            current_conf_size = self.stream_cache['conformer_att_cache'].shape[3]
            if current_conf_size < cache_ceiling:
                pad_size = cache_ceiling - current_conf_size
                # Pad with zeros at the end
                padding = torch.zeros(
                    *self.stream_cache['conformer_att_cache'].shape[:3],
                    pad_size,
                    self.stream_cache['conformer_att_cache'].shape[4],
                    device='cuda',
                    dtype=self.stream_cache['conformer_att_cache'].dtype
                )
                self.stream_cache['conformer_att_cache'] = torch.cat([
                    self.stream_cache['conformer_att_cache'], padding
                ], dim=3)

            # Track actual valid size vs allocated size
            self.cache_valid_size = prompt_mels.shape[1]
            self.cache_allocated_size = cache_ceiling
            print(f"Pre-allocated cache to ceiling size: {cache_ceiling} frames")
            print(f"  Estimator: {self.stream_cache['estimator_att_cache'].shape}")
            print(f"  Conformer: {self.stream_cache['conformer_att_cache'].shape}")

        # hift cache
        self.hift_cache_dict = dict(
            mel = torch.zeros(1, prompt_mels.shape[2], 0, device='cuda'),
            source = torch.zeros(1, 1, 0, device='cuda'),
            speech = torch.zeros(1, 0, device='cuda'),
        )


    def stream(self, generated_speech_tokens, prompt_wav, last_chunk=False):
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device='cuda')
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device='cuda')

        if self.stream_cache is None:
            raise ValueError("stream_cache is not set")

        # For pre-allocated cache, pass the valid size
        cache_valid_size = getattr(self, 'cache_valid_size', None)

        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            chunk_mel, self.stream_cache = self.flow.inference_chunk(
                token=generated_speech_tokens,
                spk=spk_emb,
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=10,
                conformer_offset=self.conformer_offset,
                cache_valid_size=cache_valid_size,
            )
        # Update conformer offset based on new cache size
        # Cache structure: dimension 3 is in mel-space (after repeat/upsample)
        current_conformer_len = self.stream_cache['conformer_att_cache'].shape[3]
        current_estimator_len = self.stream_cache['estimator_att_cache'].shape[4]
        self.conformer_offset = current_conformer_len

        # Check if using pre-allocated cache
        if hasattr(self, 'cache_allocated_size'):
            # Pre-allocated mode: maintain constant allocated size
            conformer_threshold = prompt_mels.shape[1] + self.stream_context_window

            # Determine actual valid size (may have grown beyond allocated)
            actual_valid_size = max(current_conformer_len, current_estimator_len)

            if actual_valid_size > conformer_threshold:
                print(f"    [Sliding window + resize] valid_size {actual_valid_size} -> {conformer_threshold}, target allocated: {self.cache_allocated_size}")

                # Sliding window: keep prompt + last context_window frames, then pad to allocated size
                # Estimator cache
                new_estimator = torch.cat([
                    self.stream_cache['estimator_att_cache'][:, :, :, :, :prompt_mels.shape[1]],  # prompt: 386
                    self.stream_cache['estimator_att_cache'][:, :, :, :, actual_valid_size - self.stream_context_window:actual_valid_size],  # recent: 16
                ], dim=4)

                # Pad to allocated size
                if new_estimator.shape[4] < self.cache_allocated_size:
                    pad_size = self.cache_allocated_size - new_estimator.shape[4]
                    est_padding = torch.zeros(
                        *new_estimator.shape[:4],
                        pad_size,
                        new_estimator.shape[5],
                        device='cuda',
                        dtype=new_estimator.dtype
                    )
                    new_estimator = torch.cat([new_estimator, est_padding], dim=4)

                self.stream_cache['estimator_att_cache'] = new_estimator

                # Conformer cache
                new_conformer = torch.cat([
                    self.stream_cache['conformer_att_cache'][:, :, :, :prompt_mels.shape[1]],  # prompt: 386
                    self.stream_cache['conformer_att_cache'][:, :, :, actual_valid_size - self.stream_context_window:min(actual_valid_size, current_conformer_len)],  # recent
                ], dim=3)

                # Pad to allocated size
                if new_conformer.shape[3] < self.cache_allocated_size:
                    pad_size = self.cache_allocated_size - new_conformer.shape[3]
                    conf_padding = torch.zeros(
                        *new_conformer.shape[:3],
                        pad_size,
                        new_conformer.shape[4],
                        device='cuda',
                        dtype=new_conformer.dtype
                    )
                    new_conformer = torch.cat([new_conformer, conf_padding], dim=3)

                self.stream_cache['conformer_att_cache'] = new_conformer

                # Update valid size
                self.cache_valid_size = conformer_threshold
            else:
                # Not yet at threshold, just pad to allocated size
                if current_estimator_len < self.cache_allocated_size:
                    pad_size = self.cache_allocated_size - current_estimator_len
                    est_padding = torch.zeros(
                        *self.stream_cache['estimator_att_cache'].shape[:4],
                        pad_size,
                        self.stream_cache['estimator_att_cache'].shape[5],
                        device='cuda',
                        dtype=self.stream_cache['estimator_att_cache'].dtype
                    )
                    self.stream_cache['estimator_att_cache'] = torch.cat([
                        self.stream_cache['estimator_att_cache'], est_padding
                    ], dim=4)

                if current_conformer_len < self.cache_allocated_size:
                    pad_size = self.cache_allocated_size - current_conformer_len
                    conf_padding = torch.zeros(
                        *self.stream_cache['conformer_att_cache'].shape[:3],
                        pad_size,
                        self.stream_cache['conformer_att_cache'].shape[4],
                        device='cuda',
                        dtype=self.stream_cache['conformer_att_cache'].dtype
                    )
                    self.stream_cache['conformer_att_cache'] = torch.cat([
                        self.stream_cache['conformer_att_cache'], conf_padding
                    ], dim=3)

                self.cache_valid_size = actual_valid_size
        else:
            # Original mode: dynamic trim
            # Trim estimator attention cache to control memory
            if self.stream_cache['estimator_att_cache'].shape[4] > (prompt_mels.shape[1] + self.stream_context_window):
                self.stream_cache['estimator_att_cache'] = torch.cat([
                    self.stream_cache['estimator_att_cache'][:, :, :, :, :prompt_mels.shape[1]],
                    self.stream_cache['estimator_att_cache'][:, :, :, :, -self.stream_context_window:],
                ], dim=4)

            # Trim conformer attention cache to control memory (with explicit offset tracking)
            conformer_threshold = prompt_mels.shape[1] + self.stream_context_window
            if current_conformer_len > conformer_threshold:
                print(f"    [Trimming conformer cache] {current_conformer_len} -> {conformer_threshold} (offset={self.conformer_offset})")

                # Both cache parts have the same dimension 3 size (mel-space)
                # Trim first conformer cache (pre-upsampling, repeated 2x to mel-space)
                # Keep: prompt + last context_window frames (in mel-space)
                num_encoder_layers = len(self.flow.encoder.encoders)
                trimmed_conformer1 = torch.cat([
                    self.stream_cache['conformer_att_cache'][:num_encoder_layers, :, :, :prompt_mels.shape[1], :],
                    self.stream_cache['conformer_att_cache'][:num_encoder_layers, :, :, current_conformer_len - self.stream_context_window:, :],
                ], dim=3)

                # Trim second conformer cache (post-upsampling, naturally in mel-space)
                # Keep: prompt + last context_window frames (in mel-space)
                trimmed_conformer2 = torch.cat([
                    self.stream_cache['conformer_att_cache'][num_encoder_layers:, :, :, :prompt_mels.shape[1], :],
                    self.stream_cache['conformer_att_cache'][num_encoder_layers:, :, :, current_conformer_len - self.stream_context_window:, :],
                ], dim=3)

                # Combine trimmed caches
                self.stream_cache['conformer_att_cache'] = torch.cat([trimmed_conformer1, trimmed_conformer2], dim=0)

        # vocoder cache
        hift_cache_mel = self.hift_cache_dict['mel']
        hift_cache_source = self.hift_cache_dict['source']
        hift_cache_speech = self.hift_cache_dict['speech']
        mel = torch.concat([hift_cache_mel, chunk_mel], dim=2)

        speech, source = self.hift(mel, hift_cache_source)

        # overlap speech smooth
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        # update vocoder cache
        self.hift_cache_dict = dict(
            mel = mel[..., -self.mel_cache_len:].clone().detach(),
            source = source[:, :, -self.source_cache_len:].clone().detach(),
            speech = speech[:, -self.source_cache_len:].clone().detach(),
        )
        if not last_chunk:
            speech = speech[:, :-self.source_cache_len]

        wav_np = speech.cpu().numpy()
        # Clip to [-1, 1] to avoid overflow, then scale to int16
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype('<i2')  # 16-bit little-endian PCM
        pcm_bytes = wav_int16.tobytes()
        return pcm_bytes

if __name__ == '__main__':
    # stream_context_window: number of recent mel frames to keep (default=100)
    # Use smaller values (e.g., 20) for lower memory, larger values for better quality
    token2wav = Token2wav('/cache/zhanglei/models/Step-Audio-2-mini/token2wav', stream_context_window=16)

    tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]

    prompt_wav = 'assets/default_male.wav'

    # Non-streaming version
    print("Running non-streaming inference...")
    audio = token2wav(tokens, prompt_wav)
    output_path_non_stream = 'assets/give_me_a_brief_introduction_to_the_great_wall.wav'
    with open(output_path_non_stream, 'wb') as f:
        f.write(audio)
    print(f"Non-streaming audio saved to {output_path_non_stream}")

    # Streaming version
    print("\nRunning streaming inference...")
    output_path_stream = 'assets/give_me_a_brief_introduction_to_the_great_wall_stream.wav'

    # Split tokens into chunks for streaming
    chunk_size = 25

    # Initialize stream cache with pre-allocation
    # Set preallocate_to_max=True to use constant memory
    token2wav.set_stream_cache(prompt_wav, preallocate_to_max=True, max_chunk_size=chunk_size)

    # Process chunks
    pcm_chunks = []

    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        is_last = (i + chunk_size >= len(tokens))
        pcm_bytes = token2wav.stream(chunk, prompt_wav, last_chunk=is_last)
        pcm_chunks.append(pcm_bytes)

        # Print cache sizes
        chunk_num = i//chunk_size + 1
        total_chunks = (len(tokens) + chunk_size - 1)//chunk_size
        print(f"\nChunk {chunk_num}/{total_chunks}:")
        print(f"  PCM bytes: {len(pcm_bytes)} bytes")

        # Stream cache info
        if token2wav.stream_cache is not None and 'estimator_att_cache' in token2wav.stream_cache:
            est_cache_shape = token2wav.stream_cache['estimator_att_cache'].shape
            est_cache_size = token2wav.stream_cache['estimator_att_cache'].numel() * token2wav.stream_cache['estimator_att_cache'].element_size()
            print(f"  Estimator attention cache shape: {est_cache_shape}, size: {est_cache_size / 1024 / 1024:.2f} MB")

            conf_cache_shape = token2wav.stream_cache['conformer_att_cache'].shape
            conf_cache_size = token2wav.stream_cache['conformer_att_cache'].numel() * token2wav.stream_cache['conformer_att_cache'].element_size()
            conf_cache_actual = conf_cache_shape[3]
            print(f"  Conformer attention cache shape: {conf_cache_shape}, size: {conf_cache_size / 1024 / 1024:.2f} MB")
            print(f"  Conformer offset: {token2wav.conformer_offset}, cache actual size (mel-space): {conf_cache_actual}")

    # Combine all PCM chunks
    all_pcm = b''.join(pcm_chunks)

    # Convert PCM bytes back to numpy array and save as WAV
    pcm_array = np.frombuffer(all_pcm, dtype='<i2')
    audio_float = pcm_array.astype(np.float32) / 32767.0

    # Save as WAV file
    sf.write(output_path_stream, audio_float, 24000, format='WAV')
    print(f"Streaming audio saved to {output_path_stream}")

    print("\nBoth versions completed successfully!")
