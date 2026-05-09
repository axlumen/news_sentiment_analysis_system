import re
from datetime import datetime


class TextCleaner:
    @staticmethod
    def clean_text(text):
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^一-龥a-zA-Z0-9，。！？；：""''（）【】《》、·\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def parse_date(s):
        if not s:
            return None
        try:
            for fmt in ["%Y年%m月%d日 %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                        "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    return datetime.strptime(s.strip(), fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None
