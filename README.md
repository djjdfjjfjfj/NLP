要求

Python: 3.9或更高版本
Python包：使用pip安装所需库：

pip install -r requirements.txt


命令用法说明

命令结构

python script.py -c <category> [-s] | -l [-o]

参数说明

必选参数（二选一）
`-c <category>` 或 `--category <category>` → 指定要查询的任务类别（例如，"Translation"、"Summarization"等）
`-l` 或 `--list` → 列出所有可用的任务类别

可选参数
`-s` 或 `--save` → 将查询结果保存到CSV文件
`-o` 或 `--offline` → 离线模式，使用本地预处理的数据文件（无需联网）

示例用法

！！！推荐使用离线模式，离线模式为我后期改进功能，不需要国外网络，如需使用在线功能，请确保连接国外网络并确保已安装ChromeDriver，并在脚本中的CHROMEDRIVER_PATH变量中正确设置路径！！！！

先执行cd NLP-main

1. 列出所有可用的任务类别（离线模式）

python script.py -l -o


2. 查询Translation（翻译）类别的数据（离线模式）

python script.py -c "Translation" -o


3. 查询Translation（翻译）类别的数据并保存到CSV文件（离线模式）

python script.py -c "Translation" -s -o

！！！推荐使用离线模式，离线模式为我后期改进功能，不需要国外网络，如需使用在线功能，请确保连接国外网络并确保已安装ChromeDriver，并在脚本中的CHROMEDRIVER_PATH变量中正确设置路径！！！！


4. 列出所有可用的任务类别

python script.py -l


5. 查询Translation（翻译）类别的数据

python script.py -c "Translation"


6. 查询Translation（翻译）类别的数据并保存到CSV文件

python script.py -c "Translation" -s


注意事项

1. 任务类别名称区分大小写，请确保输入正确
2. 离线模式需要`all_datasets_by_task_Updated.csv`文件存在于当前目录
3. 保存的CSV文件将以任务类别名称命名（例如，Translation_offline.csv）



评估工具
在命令行中运行脚本：

python evals.py --file <CSV文件> [--eval <指标>] [--save] [--offline]


	-f 或 --file → 指定要评估的CSV文件
	-e 或 --eval → 指定一个评估指标
	-e all 或 --eval all → 评估所有可用指标
    -s 或 --save  → 将结果保存到CSV文件
	-o 或 --offline → 离线模式，跳过Hugging Face API调用（无需联网）
    
示例：

python evals.py --file Summarization.csv --eval all --save
python evals.py --file Summarization.csv --eval popularity_level --save
python evals.py --file Translation_offline.csv --eval all --save --offline




