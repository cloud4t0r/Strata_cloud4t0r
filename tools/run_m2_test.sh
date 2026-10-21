#!/bin/sh
# Un run de test du moteur MiniMax-M2, un seul a la fois et avec un delai maxi.
#
# Pourquoi : deux instances du moteur sur les memes cartes ont_epuisse la VRAM de la carte de 16 Go et
# l'ont fait figer (2026-10-21, ce machine).  Un run bloque doit etre tue, pas laisser la machine figee.
#
#   tools/run_m2_test.sh mono [options...]     une carte
#   tools/run_m2_test.sh split [options...]   les deux cartes (--layer-split N)
#
# Variables d'environnement :
#   STRATA_BUILD_DIR    le repertoire du build (build-hip par defaut)
#   STRATA_M2_MODEL     le fichier GGUF
#   STRATA_M2_TIMEOUT   le delai maxi en secondes (900 par defaut)
#   STRATA_M2_EXTRA     des options de plus pour le moteur (ex. --layer-split 31)
set -u

mode=${1:-mono}
[ $# -gt 0 ] && shift

BUILD=${STRATA_BUILD_DIR:-build-hip}
MODEL=${STRATA_M2_MODEL:-/mnt/gygabyte/brunets/m2/MiniMaxAI_MiniMax-M2-IQ2_XXS-00001-of-00002.gguf}
TIMEOUT=${STRATA_M2_TIMEOUT:-900}
EXTRA=${STRATA_M2_EXTRA:-}
LOG=${STRATA_M2_LOG:-/tmp/opencode/m2_$(date +%H%M%S)_$mode.log}

mkdir -p "$(dirname "$LOG")"

# une seule instance a la fois
if pgrep -x strata-m2 > /dev/null 2>&1; then
    echo "run_m2_test: un strata-m2 tourne deja (pid $(pgrep -x strata-m2 | tr '\n' ' ')) : rien ne part"
    exit 3
fi

case "$mode" in
    mono)  DEVICES=1 ;;
    split) DEVICES=1,0 ;;
    *) echo "run_m2_test: mode inconnu ($mode) : mono ou split"; exit 2 ;;
esac

cd "$BUILD" || exit 2
export HIP_VISIBLE_DEVICES=$DEVICES
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}

# timeout : un run bloque est tue (et l'etat des cartes note), pas laisse en plan
timeout -k 20 "$TIMEOUT" \
    ./strata-m2 --model "$MODEL" --max-context 4096 \
    --tokens "200034,517,5505,300,5969,355" --n 64 --low-ram $EXTRA "$@" > "$LOG" 2>&1
rc=$?

if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    echo "run_m2_test: le run a depasse ${TIMEOUT}s, il est tue (log $LOG)"
    echo "run_m2_test: l'etat des cartes avant de reessayer :"
    rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[|Used|Total" | head -12
    exit 124
fi

echo "run_m2_test: fin (code $rc, log $LOG)"
grep -E "^gen:|^strata-m2: hybrid experts" "$LOG" | tail -2
exit $rc
