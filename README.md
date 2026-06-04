# jjj

课程视频智能剪辑项目（步骤 1~5 + 标注训练优化闭环）。

## 快速入口

- 详细使用文档：`README_操作说明.md`
- 全流程运行：`python run_all.py`
- 手动标注后训练：`python train.py --annotation_dir D:\video\annotations\`
- 预计算训练特征缓存：`python precompute_features.py`
- 自动调参：`python auto_tune.py --video <视频路径> --annotation <标注路径> --output <结果JSON>`
