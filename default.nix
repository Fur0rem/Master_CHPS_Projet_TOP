# Fetch from the unstable channel
let pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};

in pkgs.mkShell {
  buildInputs = with pkgs; [
    clang-tools
    gcc
    clang
    re2
    coreutils-prefixed
    cmake
    mpich
    mpi
    linuxPackages_latest.perf
    typst
    hwloc
    python3
      (python3.withPackages(ps: with ps; [matplotlib]))
    ];
  NIX_ENFORCE_NO_NATIVE=0; # Enable native compilation

  # Environment variables suggested by the lab subject
  OMP_PROC_BIND="true";
  OMP_PLACES="cores";
}
