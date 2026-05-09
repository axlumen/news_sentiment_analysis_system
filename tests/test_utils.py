"""工具函数单元测试"""
import unittest
from datetime import datetime

from utils.common import (
    generate_hash,
    safe_get,
    chunk_list,
    remove_none_values,
    format_datetime,
    truncate_string,
    parse_int,
    parse_float,
    ensure_list,
)


class TestUtils(unittest.TestCase):
    """工具函数测试类"""
    
    def test_generate_hash(self):
        """测试MD5哈希生成"""
        result = generate_hash("test")
        self.assertEqual(len(result), 32)
        self.assertEqual(result, "098f6bcd4621d373cade4e832627b4f6")
    
    def test_safe_get(self):
        """测试安全获取字典值"""
        data = {"key1": "value1", "key2": 123}
        self.assertEqual(safe_get(data, "key1"), "value1")
        self.assertEqual(safe_get(data, "key3", "default"), "default")
        self.assertIsNone(safe_get(data, "key3"))
    
    def test_chunk_list(self):
        """测试列表分块"""
        items = [1, 2, 3, 4, 5, 6, 7]
        chunks = chunk_list(items, 3)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], [1, 2, 3])
        self.assertEqual(chunks[1], [4, 5, 6])
        self.assertEqual(chunks[2], [7])
    
    def test_remove_none_values(self):
        """测试移除None值"""
        data = {"a": 1, "b": None, "c": "test", "d": None}
        result = remove_none_values(data)
        self.assertEqual(len(result), 2)
        self.assertIn("a", result)
        self.assertIn("c", result)
        self.assertNotIn("b", result)
    
    def test_format_datetime(self):
        """测试日期时间格式化"""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = format_datetime(dt)
        self.assertIsInstance(result, str)
        self.assertIn("2024-01-15", result)
        self.assertIsNone(format_datetime(None))
    
    def test_truncate_string(self):
        """测试字符串截断"""
        text = "Hello, World!"
        self.assertEqual(truncate_string(text, 20), text)
        self.assertEqual(truncate_string(text, 5), "Hello...")
    
    def test_parse_int(self):
        """测试整数解析"""
        self.assertEqual(parse_int("123"), 123)
        self.assertEqual(parse_int("abc", 456), 456)
        self.assertEqual(parse_int(None), 0)
    
    def test_parse_float(self):
        """测试浮点数解析"""
        self.assertEqual(parse_float("3.14"), 3.14)
        self.assertEqual(parse_float("abc", 2.5), 2.5)
        self.assertEqual(parse_float(None), 0.0)
    
    def test_ensure_list(self):
        """测试列表确保"""
        self.assertEqual(ensure_list(None), [])
        self.assertEqual(ensure_list([1, 2, 3]), [1, 2, 3])
        self.assertEqual(ensure_list("single"), ["single"])


if __name__ == "__main__":
    unittest.main()