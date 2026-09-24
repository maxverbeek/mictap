{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems =
        f: nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (s: f nixpkgs.legacyPackages.${s});
    in
    {
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
        default = pkgs.mkShell {
          packages = with pkgs; [ cargo rustc clippy rustfmt rust-analyzer ];
          buildInputs = [ pkgs.sherpa-onnx ];
        };
      });
    };
}
