#!/bash
# SFT 前期准备 v2：三路并行 + 阿里云镜像（实测 867KB/s vs 清华 310~455KB/s）
# 详见 docs/SFT_MILESTONES.md MS0 / docs/SFT_STATUS.md
#
# 背景：
#  - huggingface.co 本机不可达，权重走 ModelScope
#  - LLaMA-Factory 必须源码安装（template: minicpm5 未进 PyPI v0.9.5）
#  - 训练/服务双环境隔离（transformers 版本冲突，官方 cookbook 明示）
#  - pip 防卡死：连接超时 60s、失败重试 10 次；已下载缓存保留
set -x

CONDA=/opt/anaconda3
source $CONDA/etc/profile.d/conda.sh
ALIYUN="https://mirrors.aliyun.com/pypi/web/simple"
PIP_OPTS="-i $ALIYUN --retries 10 --timeout 60"
LOGS=$HOME/codes/Agent_MFF_Early_Warning/logs
SERVE=$HOME/.conda/envs/mff_sft_serve
TRAIN=$HOME/.conda/envs/mff_sft

# ---------- 0. 服务环境若不存在则创建，先装小依赖 modelscope（供权重下载用） ----------
[ -d $SERVE ] || conda create -n mff_sft_serve python=3.11 -y
$SERVE/bin/pip install $PIP_OPTS modelscope

# ---------- 三路并行 ----------
# A. 服务环境：vLLM
( $SERVE/bin/pip install $PIP_OPTS "vllm>=0.21"
  echo "JOB-A-EXIT:$?" ) > $LOGS/sft_prep_serve.log 2>&1 &

# B. 训练环境：conda + LLaMA-Factory 源码版 + deepspeed
( conda create -n mff_sft python=3.10 -y
  [ -d $HOME/codes/LLaMA-Factory ] || \
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git $HOME/codes/LLaMA-Factory
  cd $HOME/codes/LLaMA-Factory
  $TRAIN/bin/pip install $PIP_OPTS -e ".[torch,metrics]" deepspeed
  echo "JOB-B-EXIT:$?" ) > $LOGS/sft_prep_train.log 2>&1 &

# C. 权重下载：ModelScope（5GB）
( mkdir -p $HOME/models
  $SERVE/bin/modelscope download \
    --model OpenBMB/MiniCPM5-2B \
    --local_dir $HOME/models/MiniCPM5-2B
  echo "JOB-C-EXIT:$?" ) > $LOGS/sft_prep_weights.log 2>&1 &

wait
echo "==== SFT PREP DONE $(date) ===="
grep -h "EXIT:" $LOGS/sft_prep_{serve,train,weights}.log
