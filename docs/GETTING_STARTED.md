# Getting Started

This page provides basic tutorials about the usage of MRDet.
For installation instructions, please see [INSTALL.md](INSTALL.md).


### Test MRDet

- [x] single GPU testing
- [x] multiple GPU testing
- [x] visualize detection results

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test_dota.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--outdir ${RESULT_DIR}] [--out ${RESULT_FILE}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--outdir ${RESULT_DIR}] [--out ${RESULT_FILE}]
```

Optional arguments:
- `RESULT_DIR`: directory of the output results in pickle format and .txt format. 
- `RESULT_FILE`: the filename of the output results in pickle format.
- `--show`: If specified, detection results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing. 

Examples:

Assume that you have already downloaded the checkpoints to `checkpoints/`.

1. Test MRDet and show the results.

```shell
python tools/test_dota.py configs/mrdet/mrdet_r101_fpn_2x_dota.py \
    checkpoints/mrdet_r101_fpn_2x_dota.pth \
    --outdir eval/mrdet_r101_fpn_2x_dota \
    --out dota_test.pkl \
    --show
```

2. Test MRDet with 8 GPUs.

```shell
./tools/dist_test.sh configs/mrdet/mrdet_r101_fpn_2x_dota.py \
    checkpoints/mrdet_r101_fpn_2x_dota.pth \
    8 --outdir eval/mrdet_r101_fpn_2x_dota --out dota_test.pkl
```
3. Obtain the result in .txt format
```shell
python tools/parse_results_dota.py --config configs/mrdet/mrdet_r101_fpn_2x_dota.py \
    --outdir eval/mrdet_r101_fpn_2x_dota --pkl_file dota_test.pkl --type OBB
```

## Train MRDet

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.


## Useful tools

### Analyze logs

You can plot loss/mAP curves given a training log file. Run `pip install seaborn` first to install the dependency.

![loss curve image](../demo/loss_curve.png)

```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the classification loss of some run.

```shell
python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

```shell
python tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_reg --out losses.pdf
```
