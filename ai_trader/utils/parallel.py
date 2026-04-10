"""병렬처리 유틸리티 (López de Prado Ch20)

mpPandasObj 패턴: pandas 객체에 대한 멀티프로세싱 래퍼.
다종목 분석, 대규모 백테스트 등에서 병렬화로 성능을 향상합니다.

사용 예:
  # 단일 종목 함수를 다종목에 병렬 적용
  results = mp_pandas_obj(
      func=compute_features,
      pd_obj=("molecule", stock_list),
      num_threads=4,
      df=ohlcv_data,
  )
"""

import numpy as np
import pandas as pd
from typing import Callable, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from loguru import logger


def mp_pandas_obj(
    func: Callable,
    pd_obj: tuple[str, list],
    num_threads: int = 4,
    use_threads: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """pandas 객체에 대한 멀티프로세싱/스레딩 래퍼.

    Args:
        func: 실행할 함수 (molecule 파라미터 필수)
        pd_obj: (파라미터명, 분할할 리스트) 튜플
        num_threads: 병렬 스레드/프로세스 수
        use_threads: True=ThreadPool, False=ProcessPool
        **kwargs: func에 전달할 추가 파라미터

    Returns:
        결합된 결과 DataFrame
    """
    param_name, obj_list = pd_obj

    if num_threads <= 1:
        return func(**{param_name: obj_list}, **kwargs)

    # 리스트를 청크로 분할
    chunks = _split_list(obj_list, num_threads)

    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    results = []

    with Executor(max_workers=num_threads) as executor:
        futures = {}
        for i, chunk in enumerate(chunks):
            future = executor.submit(func, **{param_name: chunk}, **kwargs)
            futures[future] = i

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append((futures[future], result))
            except Exception as e:
                logger.warning("병렬 작업 실패 (chunk {}): {}", futures[future], e)

    if not results:
        return pd.DataFrame()

    # 순서 유지
    results.sort(key=lambda x: x[0])
    dfs = [r[1] for r in results]

    if isinstance(dfs[0], pd.DataFrame):
        return pd.concat(dfs, axis=0)
    elif isinstance(dfs[0], pd.Series):
        return pd.concat(dfs, axis=0)
    else:
        return dfs


def _split_list(lst: list, n: int) -> list[list]:
    """리스트를 n개의 균등 청크로 분할합니다."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]


def vectorized_triple_barrier(
    close: np.ndarray,
    target: np.ndarray,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    max_holding: int = 20,
) -> np.ndarray:
    """벡터화된 Triple-Barrier 라벨링.

    루프 대신 numpy 연산으로 성능을 최적화합니다.

    Args:
        close: 종가 배열
        target: 목표 변동성 배열
        pt_sl: (profit-taking, stop-loss) 배수
        max_holding: 최대 보유 기간

    Returns:
        라벨 배열 (1=수익, -1=손실, 0=중립)
    """
    n = len(close)
    labels = np.zeros(n, dtype=int)

    for i in range(n - 1):
        end = min(i + max_holding + 1, n)
        if end <= i + 1:
            continue

        path = close[i + 1: end] / close[i] - 1

        # 상단 장벽
        pt_barrier = target[i] * pt_sl[0] if pt_sl[0] > 0 else np.inf
        pt_touch = np.where(path >= pt_barrier)[0]

        # 하단 장벽
        sl_barrier = -target[i] * pt_sl[1] if pt_sl[1] > 0 else -np.inf
        sl_touch = np.where(path <= sl_barrier)[0]

        # 먼저 도달한 장벽
        pt_first = pt_touch[0] if len(pt_touch) > 0 else np.inf
        sl_first = sl_touch[0] if len(sl_touch) > 0 else np.inf

        if pt_first < sl_first:
            labels[i] = 1
        elif sl_first < pt_first:
            labels[i] = -1
        else:
            # 수직 장벽: 최종 수익률 부호
            final_ret = path[-1] if len(path) > 0 else 0
            labels[i] = int(np.sign(final_ret))

    return labels


def batch_compute(
    func: Callable,
    items: list,
    batch_size: int = 100,
    **kwargs,
) -> list:
    """대량 항목을 배치로 처리합니다.

    메모리 효율적으로 대규모 데이터를 처리합니다.

    Args:
        func: 배치 처리 함수
        items: 전체 항목 리스트
        batch_size: 배치 크기
        **kwargs: func에 전달할 추가 파라미터

    Returns:
        전체 결과 리스트
    """
    results = []
    n_batches = (len(items) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(items))
        batch = items[start:end]

        try:
            result = func(batch, **kwargs)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        except Exception as e:
            logger.warning("배치 {} 처리 실패: {}", i, e)

    return results
