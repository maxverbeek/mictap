pkgs: {
  whisper = pkgs.fetchurl {
    url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin";
    hash = "sha256-OUIhcJzVrR9AxG5gMcphvOiJMebgiMGIKUxtWlX/p+I=";
  };
  vad = pkgs.fetchurl {
    url = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";
    hash = "sha256-KZQNmNQrkfvQXOSJ8+z3xy8KQvAn5IdZGaKPtMBOos8=";
  };
  segmentation = "${
    pkgs.fetchzip {
      url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
      hash = "sha256-hqaCTZJKZp6IHxYzgVBd9Bss6wC1qg+edB/v10BT1tA=";
    }
  }/model.onnx";
  # "recongition" is upstream's spelling of the release tag.
  embedding = pkgs.fetchurl {
    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";
    hash = "sha256-qjz8FpY6EFhqk5P1A11ta1fpjTWLNH+AwqML9PAM66I=";
  };
}
