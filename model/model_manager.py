"""模型管理器 - 单例模式管理情感分析模型"""
import os
import pickle
import logging
import threading
from typing import Optional, Dict, Any

import torch

from model.bilstm_attention import SentimentModel

logger = logging.getLogger(__name__)

# 模型配置
MODEL_CONFIG = {
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'max_length': 512,
    'num_classes': 3,
    'dropout_rate': 0.5,
}

# 标签映射
LABEL_MAP_REVERSE = {0: "negative", 1: "neutral", 2: "positive"}


class ModelManager:
    """
    模型管理器 - 线程安全的单例模式
    
    确保模型只被加载一次，避免重复加载带来的性能开销
    使用双重检查锁定保证线程安全
    """
    
    _instance = None
    _model = None
    _vocab = None
    _device = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def model(self):
        """获取模型实例"""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def vocab(self):
        """获取词表"""
        if self._vocab is None:
            self._load_vocab()
        return self._vocab
    
    @property
    def device(self):
        """获取计算设备"""
        if self._device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._device
    
    def _load_vocab(self, vocab_path: Optional[str] = None):
        """
        加载词表
        
        参数：
            vocab_path: 词表文件路径，默认为默认路径
        """
        if vocab_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vocab_path = os.path.join(base_dir, "model", "dataset", "processed", "vocab.pkl")
        
        try:
            with open(vocab_path, "rb") as f:
                vocab_obj = pickle.load(f)
            
            if isinstance(vocab_obj, dict) and "vocab" in vocab_obj:
                self._vocab = vocab_obj["vocab"]
            else:
                self._vocab = vocab_obj
            
            logger.info(f"词表加载成功，大小：{len(self._vocab)}")
        except Exception as e:
            logger.error(f"词表加载失败：{e}")
            self._vocab = {"<PAD>": 0, "<UNK>": 1}
    
    def _load_model(self, model_path: Optional[str] = None):
        """
        加载模型
        
        参数：
            model_path: 模型文件路径，默认为默认路径
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "model", "checkpoints", "best_model.pth")
        
        # 确保词表已加载
        if self._vocab is None:
            self._load_vocab()
        
        try:
            vocab_size = len(self._vocab)
            self._model = SentimentModel(
                vocab_size=vocab_size,
                embedding_dim=MODEL_CONFIG['embedding_dim'],
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                output_dim=MODEL_CONFIG['num_classes'],
                n_layers=MODEL_CONFIG['num_layers'],
                dropout=MODEL_CONFIG['dropout_rate']
            ).to(self.device)
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["state_dict"])
                else:
                    self._model.load_state_dict(checkpoint)
            else:
                self._model.load_state_dict(checkpoint)
            
            self._model.eval()
            logger.info(f"模型加载成功：{model_path}")
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            self._model = None
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        预测单条文本的情感
        
        参数：
            text: 待预测的文本
        
        返回：
            预测结果字典，包含情感标签和概率
        """
        from data.predict_sentiment import preprocess_text, text_to_indices
        
        if self._model is None or self._vocab is None:
            logger.error("模型或词表未加载")
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.3333,
                "negative_prob": 0.3333,
                "neutral_prob": 0.3334,
                "positive_prob": 0.3333
            }
        
        try:
            words = preprocess_text(text)
            indices = text_to_indices(words, self._vocab)
            x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits, _ = self._model(x)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            pred_label_idx = int(prob.argmax())
            pred_label = LABEL_MAP_REVERSE[pred_label_idx]
            
            return {
                "sentiment": pred_label,
                "sentiment_score": float(round(prob[pred_label_idx], 4)),
                "negative_prob": float(round(prob[0], 4)),
                "neutral_prob": float(round(prob[1], 4)),
                "positive_prob": float(round(prob[2], 4))
            }
        except Exception as e:
            logger.error(f"预测失败：{e}")
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.3333,
                "negative_prob": 0.3333,
                "neutral_prob": 0.3334,
                "positive_prob": 0.3333
            }
    
    def predict_batch(self, texts: list) -> list:
        """
        批量预测文本情感
        
        参数：
            texts: 待预测的文本列表
        
        返回：
            预测结果列表
        """
        from data.predict_sentiment import preprocess_text, text_to_indices
        
        if self._model is None or self._vocab is None:
            logger.error("模型或词表未加载")
            return [self.predict(text) for text in texts]
        
        results = []
        for text in texts:
            results.append(self.predict(text))
        
        return results
    
    def reload_model(self, model_path: Optional[str] = None, vocab_path: Optional[str] = None):
        """
        重新加载模型和词表
        
        参数：
            model_path: 模型文件路径
            vocab_path: 词表文件路径
        """
        self._model = None
        self._vocab = None
        self._load_vocab(vocab_path)
        self._load_model(model_path)
        logger.info("模型已重新加载")


# 创建全局模型管理器实例
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """
    获取模型管理器实例
    
    返回：
        ModelManager实例
    """
    return model_manager