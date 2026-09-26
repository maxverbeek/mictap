{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs, ... }:
    let
      forAllSystems =
        f: nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (s: f nixpkgs.legacyPackages.${s});
    in
    {
      nixosModules = rec {
        server = import ./nix/server.nix self;
        recorder = import ./nix/recorder.nix self;
        default.imports = [
          server
          recorder
        ];
      };

      packages = forAllSystems (pkgs: {
        default = pkgs.rustPlatform.buildRustPackage {
          pname = "mictap";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          buildInputs = [ pkgs.sherpa-onnx ];
        };
      });

      devShells = forAllSystems (pkgs: {
        default =
          let
            models = import ./nix/models.nix pkgs;
          in
          pkgs.mkShell {
            packages = with pkgs; [
              cargo
              rustc
              clippy
              rustfmt
              rust-analyzer
              whisper-cpp
              ffmpeg-headless
            ];
            buildInputs = [ pkgs.sherpa-onnx ];
            MICTAP_WHISPER_MODEL = models.whisper;
            MICTAP_VAD_MODEL = models.vad;
            MICTAP_SEG_MODEL = models.segmentation;
            MICTAP_EMB_MODEL = models.embedding;
          };
      });
    };
}
