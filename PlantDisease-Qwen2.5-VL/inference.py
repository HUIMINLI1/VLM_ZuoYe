'''
$lhm 251019
'''
import sys, os
from solutions.llama_server import chat, briefing, build_img_message, build_text_message
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIG_AND_SETTINGS, SERVER_CONFIG, LOGGER
from utils.monitor import performance_monitor, wait_for_server
from utils.img_handler import handle_files

# Debug Only
IMG_PATH = ["assets/1_1.png", "assets/1_2.png"]

@performance_monitor()
def main():

    wait_for_server(port=SERVER_CONFIG['PORT'])
    messages = CONFIG_AND_SETTINGS['raw_messages']
    file_paths = []
    LOGGER.info("QwenIA初始化完成，在提示词中键入'--h'(help)获取帮助。")

    while True:
        print("\n------QwenIA Standby😎------")

        img_path_input = input("图像路径：")
        text_input = input("询问任何问题：")

        query = f"<image>{img_path_input}<image> {text_input}"

        if '--f' in query.lower():
            img_path_input, text_input = img_path_input.replace('--f', ''), text_input.replace('--f', '')
            file_path_input = input("文档路径：")
            query +=  f" <file>{file_path_input}<file>"
        else: file_path_input=""

        if '--q' in query.lower():
            print("\n------QwenIA Exiting🤐------")
            break
        elif '--c' in query.lower():
            LOGGER.info("已取消。")
            continue
        elif '--h' in query.lower():
            print("\n------QwenIA Help🤓------")
            print(" --请分别键入图像路径和文本提示词。例如：\n"
                  "     图像路径：path/to/image，或path/to/image1, path/to/image2\n"
                  "     询问任何问题：Give a detailed caption of the image.\n"
                  " --模式介绍：\n"
                 f"     --简报生成：**文本提示词留空**将自动启用该模式。模型会根据输入的图像生成1份简报，保存在{CONFIG_AND_SETTINGS['briefings_dir']}中，保存路径可在配置文件中修改。\n"
                  "     --对话模式：在终端界面与模型进行常规的对话交流。使用WebUI服务会禁用知识库检索功能。\n\n"
                  " --上传知识库文档：在提示词中键入'--f'(file)后触发。路径格式与图像路径相同。"
                  " --取消本次已经键入的提示词：在提示词中键入'--c'(cancel)。\n"
                  " --退出程序：在提示词中键入'--q'(quit)。llama-server（如果使用）需要手动关闭。\n"
                  " --中止生成：按下'Ctrl+C'。\n")
            continue

        # 附件目前仅对对话模式生效。如果没有文本提示，附件输入不会被处理。
        # 与图像相同，如果有新的文件输入，则会替换掉旧的文件输入。已经检索出的内容作为历史消息不会清空。
        # 优先检查附件输入，这样如果附件出问题不会对messages变量做任何改变
        if file_path_input:
            try:
                file_paths = handle_files([file_path_input])
            except Exception as e: LOGGER.error(e); continue

        # 如果有新的图像输入，则会替换掉旧的图像输入，这对所有模式都是一样的。如果想处理多个图像，则作为列表一次性输入进来。
        if img_path_input:
            try:
                img_path_input = handle_files([img_path_input])

                for img_path in img_path_input:
                    messages = build_img_message(messages, img_path, clean=True)
            except Exception as e: LOGGER.error(e); continue

        # 文本输入对应了对话模式。这里clean=False意味着历史聊天内容不会被删除，token会积累。
        if text_input:
            messages = build_text_message(messages, text_input, clean=False)
        elif not img_path_input: continue # 没有任何输入

        print("\n------QwenIA Running🤔------")

        # $wxy: To Debug, comment try...except statement.
        #       To override KeyboardInterrupt, uncomment it.
        # try:
        if not text_input:
            if len(img_path_input) > 2:
                LOGGER.error("简报模式支持最多2张图像输入")
                continue
            # clean一次messages内容，简报模式不需要历史消息。
            messages = [messages[0], messages[-1]]
            messages[-1]['content'] = [
                content for content in messages[-1]['content']
                if content["type"] == "image_url"
            ]
            briefing(messages, img_path_input, show_process=CONFIG_AND_SETTINGS['briefing_process'])
        else:
            # TODO: 检查一下token数是否超限。因为llama-server多模态推理时不会启用ctx_shift
            # messages = keep_m_tokens(messages) 
            chat(messages, img_path_input, file_paths)

        # except (Exception, KeyboardInterrupt) as e:
        #     if isinstance(e, KeyboardInterrupt):
        #         LOGGER.info("'\n'已停止。")
        #     else:
        #         LOGGER.critical(f"崩溃：{e}，请重试。\n")

if __name__ == "__main__":
    main()
