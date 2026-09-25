self:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.mictap.recorder;
  inherit (lib) mkOption types;
in
{
  options.services.mictap.recorder = {
    enable = lib.mkEnableOption "the mictap recorder, a user service in the graphical session";

    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalExpression "mictap.packages.\${system}.default";
      description = "The mictap package.";
    };

    server = mkOption {
      type = types.str;
      example = "http://myserver:8765";
      description = "URL of the mictap server recordings are uploaded to.";
    };

    allowlist = mkOption {
      type = types.listOf types.str;
      default = [
        "zen"
        "chromium"
        "chrome"
        "zoom"
        "slack"
      ];
      example = [
        "firefox"
        "teams"
      ];
      description = ''
        Apps whose microphone use starts a recording. Matched
        case-insensitively as substrings of PipeWire's application name and
        binary.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];

    systemd.user.services.mictap = {
      description = "mictap meeting recorder";
      after = [ "graphical-session.target" ];
      partOf = [ "graphical-session.target" ];
      wantedBy = [ "graphical-session.target" ];
      path = [
        pkgs.pipewire
        pkgs.libnotify
      ];
      environment = {
        MICTAP_SERVER = cfg.server;
        MICTAP_ALLOWLIST = lib.concatStringsSep "," cfg.allowlist;
      };
      serviceConfig = {
        ExecStart = "${cfg.package}/bin/mictap daemon";
        Restart = "on-failure";
        RestartSec = 2;
      };
    };
  };
}
