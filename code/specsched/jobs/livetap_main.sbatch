#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-livetap-main
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# vLLM main dev install (fork branch feat/spec-live-tap): unit test (ring + GPU record) and the e2e tap test
export W=/fs/scratch/PCS0296/vllmdev
export TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache XDG_CACHE_HOME=$W/xdg HF_HOME=/fs/scratch/PCS0296/specsched/hf
export VLLM_CACHE_ROOT=$W/vllmcache TORCHINDUCTOR_CACHE_DIR=$W/inductor TRITON_CACHE_DIR=$W/triton FLASHINFER_WORKSPACE_BASE=$W/flashinfer
mkdir -p $VLLM_CACHE_ROOT $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $FLASHINFER_WORKSPACE_BASE
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=WARNING CC=gcc CXX=g++
module unload intel 2>/dev/null; module load cuda/13.2.1 2>/dev/null || module load cuda/12.9.1
cd $W/vllm && source .venv/bin/activate
set -x
git fetch -q origin +feat/spec-live-tap:refs/remotes/origin/feat/spec-live-tap && git checkout -q -B feat/spec-live-tap origin/feat/spec-live-tap && git reset -q --hard origin/feat/spec-live-tap && git log -1 --format="%h %s"
python -c "import vllm; print(vllm.__version__)"
export VLLM_TEST_TARGET=/fs/scratch/PCS0296/bursar/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
export VLLM_TEST_DRAFT=/fs/scratch/PCS0296/specsched/models/qwen3-8b-eagle3
python -m pytest -q -x tests/v1/worker/test_live_tap.py -p no:cacheprovider -W ignore 2>&1 | grep -E "passed|failed|error|Error" | tail -3
python -m pytest -q -x -s tests/v1/e2e/spec_decode/eagle/test_live_tap.py -p no:cacheprovider --basetemp=$W/pytest_tap -W ignore 2>&1 | grep -v -i "warning\|INFO " | tail -40
python -m pytest -q -x -s tests/v1/e2e/spec_decode/eagle/test_set_speculation.py -p no:cacheprovider --basetemp=$W/pytest_sw -W ignore > $W/pytest_sw.log 2>&1; grep -E "passed|failed" $W/pytest_sw.log | tail -2; grep -n -E "ERROR .*(Error|assert|raise|File \"/fs)" $W/pytest_sw.log | grep -v "EngineDeadError\|core_client\|_base.py\|run_method\|_contextlib" | tail -14 | cut -c1-260
echo "=== STEP done"
