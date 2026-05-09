import logging
import os

logger = logging.getLogger(__name__)

# 停用词文件默认路径：data/stopwords.txt（相对于项目根目录）
_DEFAULT_STOPWORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "stopwords.txt"
)


def load_stopwords(file_path=None):
    """加载中文停用词表"""
    if file_path is None:
        file_path = _DEFAULT_STOPWORDS_PATH
    try:
        with open(file_path, "r", encoding="utf8") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        logger.warning(f"Stopwords file not found: {file_path}, using defaults")
        return {"的", "了", "在", "是", "我", "你", "他", "她", "它", "们", "就", "都", "而", "及", "与"}
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            return set(line.strip() for line in f if line.strip())
