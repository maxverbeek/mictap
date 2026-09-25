self:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.mictap.server;
  inherit (lib) mkOption types;
  port = lib.last (lib.splitString ":" cfg.listen);
  tools = with pkgs; [
    whisper-cpp
    sherpa-onnx
    ffmpeg
  ];
in
{
  options.services.mictap.server = {
    enable = lib.mkEnableOption "the mictap transcription server";

    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalExpression "mictap.packages.\${system}.default";
      description = "The mictap package.";
    };

    listen = mkOption {
      type = types.str;
      default = "127.0.0.1:8765";
      example = "0.0.0.0:8765";
      description = ''
        Address and port to listen on. The API has no login: listen only where
        trusted clients can reach it (loopback, a VPN such as Tailscale), and
        don't open the port in the firewall.
      '';
    };

    url = mkOption {
      type = types.str;
      default = "http://${config.networking.hostName}:${port}";
      defaultText = lib.literalExpression ''"http://''${networking.hostName}:<port of listen>"'';
      description = "How browsers reach this server; the base of the timestamp links in transcripts.";
    };

    outputDir = mkOption {
      type = types.path;
      example = "/srv/notes/mictap";
      description = ''
        Folder the transcripts are written to, usually inside a synced notes
        vault. The service runs in a chroot and this folder is the only one it
        can see outside its own state.
      '';
    };

    user = mkOption {
      type = types.str;
      default = "mictap";
      description = ''
        User the server runs as. Set it to the user that owns the synced vault
        (the sync or WebDAV daemon's user) so both can overwrite each other's
        files. The user "mictap" is created when left at the default.
      '';
    };

    group = mkOption {
      type = types.str;
      default = "mictap";
      description = "Group the server runs as. Created when left at the default.";
    };

    models = {
      whisper = mkOption {
        type = types.path;
        default = pkgs.fetchurl {
          url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin";
          hash = "sha256-OUIhcJzVrR9AxG5gMcphvOiJMebgiMGIKUxtWlX/p+I=";
        };
        defaultText = "ggml-large-v3-turbo-q5_0.bin";
        description = "whisper.cpp model (ggml).";
      };
      vad = mkOption {
        type = types.path;
        default = pkgs.fetchurl {
          url = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";
          hash = "sha256-KZQNmNQrkfvQXOSJ8+z3xy8KQvAn5IdZGaKPtMBOos8=";
        };
        defaultText = "ggml-silero-v5.1.2.bin";
        description = "Voice activity detection model (ggml silero).";
      };
      segmentation = mkOption {
        type = types.path;
        default = "${
          pkgs.fetchzip {
            url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
            hash = "sha256-hqaCTZJKZp6IHxYzgVBd9Bss6wC1qg+edB/v10BT1tA=";
          }
        }/model.onnx";
        defaultText = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx";
        description = "Speaker segmentation model (sherpa-onnx).";
      };
      embedding = mkOption {
        type = types.path;
        # "recongition" is upstream's spelling of the release tag.
        default = pkgs.fetchurl {
          url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";
          hash = "sha256-qjz8FpY6EFhqk5P1A11ta1fpjTWLNH+AwqML9PAM66I=";
        };
        defaultText = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";
        description = "Speaker embedding model (sherpa-onnx).";
      };
    };

    settings = mkOption {
      type = types.attrsOf (
        types.oneOf [
          types.str
          types.int
          types.float
        ]
      );
      default = { };
      example = {
        MICTAP_CLUSTER_THRESHOLD = 0.9;
        MICTAP_MATCH_THRESHOLD = 0.75;
      };
      description = "Extra environment variables, for the tunables listed in the README.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = lib.versionAtLeast config.systemd.package.version "257";
        message = "services.mictap.server needs systemd >= 257 (PrivatePIDs).";
      }
    ];

    users.users = lib.mkIf (cfg.user == "mictap") {
      mictap = {
        isSystemUser = true;
        inherit (cfg) group;
      };
    };
    users.groups = lib.mkIf (cfg.group == "mictap") { mictap = { }; };

    systemd.tmpfiles.rules = [ "d ${cfg.outputDir} 0750 ${cfg.user} ${cfg.group} -" ];

    systemd.services.mictap-server = {
      description = "mictap transcription server";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      path = tools;
      environment = {
        MICTAP_LISTEN = cfg.listen;
        MICTAP_URL = cfg.url;
        MICTAP_VAULT = cfg.outputDir;
        MICTAP_WHISPER_MODEL = "${cfg.models.whisper}";
        MICTAP_VAD_MODEL = "${cfg.models.vad}";
        MICTAP_SEG_MODEL = "${cfg.models.segmentation}";
        MICTAP_EMB_MODEL = "${cfg.models.embedding}";
        # There is no /etc in the chroot.
        TZ = if config.time.timeZone == null then "UTC" else config.time.timeZone;
        TZDIR = "${pkgs.tzdata}/share/zoneinfo";
      }
      // lib.mapAttrs (_: toString) cfg.settings;

      # A chroot holding only the listed store paths; outputDir and the state
      # dir are bind-mounted in. Nothing else on the host is visible.
      confinement = {
        enable = true;
        packages = tools ++ [
          pkgs.tzdata
          cfg.models.whisper
          cfg.models.vad
          cfg.models.segmentation
          cfg.models.embedding
        ];
      };

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/mictap-server";
        User = cfg.user;
        Group = cfg.group;
        BindPaths = [ cfg.outputDir ];
        StateDirectory = "mictap";
        StateDirectoryMode = "0750";
        UMask = "0027";
        Restart = "on-failure";
        RestartSec = 5;

        # The user may be shared with a sync daemon: keep this service from
        # seeing or ptracing that daemon's processes.
        PrivatePIDs = true;
        ProtectProc = "invisible";
        SystemCallFilter = [ "@system-service" ];
        NoNewPrivileges = true;

        PrivateDevices = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [
          "AF_INET"
          "AF_INET6"
          "AF_UNIX"
        ];
        RestrictNamespaces = true;
        LockPersonality = true;

        # Transcription takes every core; stay out of other services' way.
        Nice = 19;
        CPUWeight = 20;
      };
    };
  };
}
