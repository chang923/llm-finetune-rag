import os

def load_key(key_name="DEEPSEEK_API_KEY"):
    """
    根据传入的 key_name，读取同目录下 {key_name}.txt 文件的内容，返回字符串。
    如果文件不存在或读取失败，抛出异常。
    """
    abspath = os.path.abspath(__file__)
    script_dir = os.path.dirname(abspath)

    filename = f"{key_name}.txt"
    file_path = os.path.join(script_dir, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if not key:
            raise ValueError(f"{filename} 文件内容为空")
        return key
    except FileNotFoundError:
        raise FileNotFoundError(f"API Key 文件不存在: {file_path}")
    except Exception as e:
        raise RuntimeError(f"读取 API Key 时出错: {e}")