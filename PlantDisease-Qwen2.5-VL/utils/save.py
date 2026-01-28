'''
$lhm 251023
保存简报和完整报告到本地
'''
import sys, os
import time
from utils import CONFIG_AND_SETTINGS, LOGGER

def briefing2file(str_list, file_type='.txt'):
    file_dir = CONFIG_AND_SETTINGS['briefings_dir']
    os.makedirs(file_dir, exist_ok=True)

    file_path = os.path.join(file_dir, f"briefing_{time.strftime('%y%m%d%H%M', time.localtime())}{file_type}")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in str_list:
                f.write(str(line) + '\n')
    except Exception as e:
        LOGGER.error(f"\n简报写入本地失败：{e}\n")
    else:
        LOGGER.info(f"\n简报已保存到{os.path.abspath(file_path)}")

def fullreport2file(prompt_list, answer_list, file_type='.txt'):
    file_dir = CONFIG_AND_SETTINGS['fullreports_dir']
    os.makedirs(file_dir, exist_ok=True)

    file_path = os.path.join(file_dir, f"fullreport_{time.strftime('%y%m%d%H%M', time.localtime())}{file_type}")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, prompt in enumerate(prompt_list):
                f.write("·提示词：\n")
                f.write(str(prompt) + '\n\n')

                f.write("·QwenIA：\n")
                f.write(str(answer_list[i]) + '\n\n')

    except Exception as e:
        LOGGER.error(f"\n完整报告写入本地失败：{e}\n")
    else:
        LOGGER.info(f"\n完整报告已保存到{os.path.abspath(file_path)}")

# Debug Only
if __name__ == '__main__':
    pass