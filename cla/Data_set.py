import numpy as np
from torch.utils.data import Dataset

from macls.data_utils.audio import AudioSegment
from macls.data_utils.augmentor.augmentation import AugmentationPipeline
from macls.utils.logger import setup_logger
import math
logger = setup_logger(__name__)

class Data_set(Dataset):
    def __init__(self,
                 data_list_path,
                 do_vad=False,
                 max_duration=1,
                 min_duration=0.5,
                 augmentation_config='{}',
                 mode='train',
                 sample_rate=16000,
                 use_dB_normalization=False,
                 target_dB=-20):
        """音频数据加载器

        Args:
            data_list_path: 包含音频路径和标签的数据列表文件的路径
            do_vad: 是否对音频进行语音活动检测（VAD）来裁剪静音部分
            max_duration: 最长的音频长度，大于这个长度会裁剪掉
            min_duration: 过滤最短的音频长度
            augmentation_config: 用于指定音频增强的配置
            mode: 数据集模式。在训练模式下，数据集可能会进行一些数据增强的预处理
            sample_rate: 采样率
            use_dB_normalization: 是否对音频进行音量归一化
            target_dB: 音量归一化的大小
        """
        super().__init__()
        self.do_vad = do_vad
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self._augmentation_pipeline = AugmentationPipeline(augmentation_config=augmentation_config)
        # 获取数据列表
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
            self.lines = self.lines[1:]
    def __getitem__(self, idx):
        # 分割音频路径和标签
        audio_path, label = self.lines[idx].replace('\n', '').split(',')
        # 读取音频
        audio_segment = AudioSegment.from_file(audio_path)
        # 裁剪静音
        # if self.do_vad:
        #     audio_segment.vad(top_db=10)
        # 数据太短不利于训练
        # if self.mode == 'train':
        #     if audio_segment.duration < self.min_duration:
        #         return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
        if audio_segment.duration < self.max_duration:
            for i in range(math.floor(np.log2(self.max_duration/audio_segment.duration))+1):
                audio_segment = AudioSegment.concatenate(audio_segment, audio_segment)

        # 重采样
        if audio_segment.sample_rate != self._target_sample_rate:
            audio_segment.resample(self._target_sample_rate)
        # decibel normalization
        # if self._use_dB_normalization:
        #     audio_segment.normalize(target_db=self._target_dB)
        # 裁剪需要的数据
        audio_segment.crop(duration=self.max_duration, mode=self.mode)
        # 音频增强
        self._augmentation_pipeline.transform_audio(audio_segment)
        return np.array(audio_segment.samples, dtype=np.float32), np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)



