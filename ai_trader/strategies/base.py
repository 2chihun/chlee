"""매매 전략 기본 인터페이스"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """매매 시그널"""
    type: SignalType
    stock_code: str
    price: int
    quantity: int = 0
    stop_loss: int = 0
    take_profit: int = 0
    confidence: float = 0.0  # 0.0 ~ 1.0
    reason: str = ""
    strategy_name: str = ""


class BaseStrategy(ABC):
    """전략 기본 클래스

    모든 매매 전략은 이 클래스를 상속하고
    analyze() 와 generate_signal() 을 구현해야 합니다.
    """

    def __init__(self, name: str, params: Optional[dict] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터를 분석하여 지표와 시그널 컬럼을 추가합니다.

        Args:
            df: OHLCV DataFrame

        Returns:
            분석 결과가 추가된 DataFrame
        """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[dict] = None) -> Signal:
        """가장 최근 데이터 기반으로 매매 시그널을 생성합니다.

        Args:
            df: analyze()를 거친 DataFrame
            current_position: 현재 보유 포지션 정보 (없으면 None)

        Returns:
            Signal 객체
        """

    def get_params(self) -> dict:
        """전략 파라미터를 반환합니다."""
        return self.params.copy()

    def set_params(self, params: dict):
        """전략 파라미터를 설정합니다."""
        self.params.update(params)
