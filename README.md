# README


This is the code for 《融合行为词的罪名预测多任务学习模型》




## Setup

### Requirements

```bash
conda create --name AC python=3.8
```
### Datasets


Data format:
```json
{"token": "2014年4月6日凌晨，被告人顾震臣与被害人陈莉在本市虹口区密云路弄号室内，因感情纠纷发生争执。顾震臣用拳头猛击陈莉脸部，并致其头部撞到墙上。后因陈莉状况恶化，当日晚上被顾震臣等人送医救治。4月9日凌晨，顾震臣在家人陪同下至公安机关自首。同日被害人陈莉在华山医院经抢救无效死亡。案发后经鉴定，被害人陈丽系生前受钝性外力作用头部致颅脑损伤而死亡。",
 "entities": [[43, 44], [54, 55], [62, 62], [92, 93], [116, 117], [136, 137], [151, 151]],
 "accusation": ["故意伤害罪"]
}
```
### Customized data
If you want to use your own data, please organize your data line like the following way, the data folder should 
have the following files
```text
data/
    - accusation_type.json
    - train.json
    - dev.json
    - test.json
```
`

## Example

### Train


```bash
python identifier.py train --config configs/example.conf
```

Note: You should edit this "gpu_queue" in `main.py` according to the actual number of GPUs. 


