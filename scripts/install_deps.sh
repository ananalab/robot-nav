#!/usr/bin/env bash
# Installe tout ce qu'il faut pour compiler robot_nav (macOS ou Linux).
#
#   scripts/install_deps.sh            # installation normale (brew ou apt)
#   scripts/install_deps.sh --local    # Linux sans droits admin (ex. salles PPTI) :
#                                      # compile yaml-cpp, Eigen et OpenCV dans ~/.local
#
# Ce script reprend exactement ce qui a été installé sur la machine de développement.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_MODE=0
[[ "${1:-}" == "--local" ]] && LOCAL_MODE=1

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

install_python_venv() {
  say "Environnement Python pour tracer les logs (.venv : matplotlib, numpy)"
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install --quiet --upgrade pip
  "$ROOT/.venv/bin/pip" install --quiet matplotlib numpy
}

install_arduino_core() {
  say "Cœur Arduino AVR (compilation du firmware pour la carte UNO)"
  arduino-cli core update-index
  arduino-cli core install arduino:avr
}

install_macos() {
  # Sur Mac Apple Silicon, on utilise le Homebrew natif (/opt/homebrew) et pas
  # celui de /usr/local qui tourne sous Rosetta (bibliothèques x86_64 incompatibles).
  local brew_bin=/opt/homebrew/bin/brew
  [[ "$(uname -m)" == "x86_64" ]] && brew_bin=/usr/local/bin/brew
  if [[ ! -x "$brew_bin" ]]; then
    say "Homebrew absent : installation (https://brew.sh)"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
  xcode-select -p >/dev/null 2>&1 || { say "Outils en ligne de commande Xcode"; xcode-select --install; }
  say "Paquets Homebrew : cmake ninja pkg-config opencv eigen yaml-cpp arduino-cli python"
  arch -"$(uname -m)" "$brew_bin" install cmake ninja pkg-config opencv eigen yaml-cpp arduino-cli python
  install_arduino_core
  install_python_venv
}

install_linux_apt() {
  say "Paquets apt (demande le mot de passe administrateur)"
  sudo apt-get update
  sudo apt-get install -y build-essential git cmake ninja-build pkg-config \
    libeigen3-dev libyaml-cpp-dev python3-venv curl
  # OpenCV de la distribution seulement s'il est assez récent (nouvelle API ArUco).
  local cv_version
  cv_version="$(apt-cache policy libopencv-dev | awk '/Candidat|Candidate/ {print $2}')"
  if dpkg --compare-versions "${cv_version%%+*}" ge 4.7; then
    sudo apt-get install -y libopencv-dev
  else
    say "OpenCV de la distribution trop ancien (${cv_version}) : compilation locale"
    build_opencv_local
  fi
  sudo usermod -aG dialout "$USER"
  echo "Accès au port série : déconnectez-vous puis reconnectez-vous une fois."
  install_arduino_cli_local
  install_arduino_core
  install_python_venv
}

# ----- installations sans droits admin, dans ~/.local -------------------------
PREFIX="$HOME/.local"

build_cmake_project() {  # build_cmake_project <url-archive> <nom> [options cmake...]
  local url="$1" name="$2"; shift 2
  local work; work="$(mktemp -d)"
  curl -fsSL "$url" | tar -xz -C "$work" --strip-components=1
  cmake -S "$work" -B "$work/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" "$@"
  cmake --build "$work/build"
  cmake --install "$work/build"
  rm -rf "$work"
  echo "$name installé dans $PREFIX"
}

build_opencv_local() {
  say "Compilation d'OpenCV 4.10 dans $PREFIX (20 à 40 minutes)"
  build_cmake_project https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz opencv \
    -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,calib3d,features2d,objdetect \
    -DWITH_FFMPEG=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_python3=OFF -DBUILD_opencv_apps=OFF
}

install_arduino_cli_local() {
  command -v arduino-cli >/dev/null && return
  say "arduino-cli (binaire officiel) dans $PREFIX/bin"
  mkdir -p "$PREFIX/bin"
  curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR="$PREFIX/bin" sh
}

install_linux_local() {
  for tool in g++ cmake curl; do
    command -v "$tool" >/dev/null || { echo "Il manque $tool : demandez-le à l'administrateur."; exit 1; }
  done
  command -v ninja >/dev/null || { python3 -m pip install --user ninja; }
  build_cmake_project https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz eigen \
    -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
  build_cmake_project https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz yaml-cpp \
    -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF
  build_opencv_local
  install_python_venv
  echo
  echo "Ajoutez à votre ~/.bashrc :  export CMAKE_PREFIX_PATH=\$HOME/.local:\$CMAKE_PREFIX_PATH"
  echo "                              export LD_LIBRARY_PATH=\$HOME/.local/lib:\$LD_LIBRARY_PATH"
}

case "$(uname -s)" in
  Darwin) install_macos ;;
  Linux)
    if [[ $LOCAL_MODE -eq 1 ]]; then install_linux_local; else install_linux_apt; fi ;;
  *) echo "Système non géré (macOS ou Linux seulement)."; exit 1 ;;
esac

say "Terminé. Vérifiez avec : scripts/check_env"
