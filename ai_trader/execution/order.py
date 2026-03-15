"""한국투자증권 API 주문 실행 모듈"""

import datetime as dt
from typing import Optional
from dataclasses import dataclass

import requests
from loguru import logger

from config.settings import KISConfig
from data.collector import KISAuth


@dataclass
class OrderResult:
    """주문 결과"""
    success: bool
    order_id: str = ""
    stock_code: str = ""
    side: str = ""
    price: int = 0
    quantity: int = 0
    message: str = ""
    executed_at: Optional[dt.datetime] = None


class KISOrderExecutor:
    """한국투자증권 주문 실행기"""

    # TR-ID (모의투자) - 통합주문 API (KRX+NXT 지원)
    TR_BUY_PAPER = "VTTC0012U"
    TR_SELL_PAPER = "VTTC0011U"
    # TR-ID (실전)
    TR_BUY_LIVE = "TTTC0012U"
    TR_SELL_LIVE = "TTTC0011U"
    # 주문 조회
    TR_ORDER_INQUIRY_PAPER = "VTTC8001R"
    TR_ORDER_INQUIRY_LIVE = "TTTC8001R"
    # 잔고 조회
    TR_BALANCE_PAPER = "VTTC8434R"
    TR_BALANCE_LIVE = "TTTC8434R"

    def __init__(self, auth: KISAuth):
        self.auth = auth
        self.config = auth.config
        self.base_url = auth.config.base_url

    def _get_tr_id(self, action: str) -> str:
        mapping = {
            "buy": self.TR_BUY_PAPER if self.config.is_paper else self.TR_BUY_LIVE,
            "sell": self.TR_SELL_PAPER if self.config.is_paper else self.TR_SELL_LIVE,
            "inquiry": self.TR_ORDER_INQUIRY_PAPER if self.config.is_paper else self.TR_ORDER_INQUIRY_LIVE,
            "balance": self.TR_BALANCE_PAPER if self.config.is_paper else self.TR_BALANCE_LIVE,
        }
        return mapping[action]

    def buy(
        self,
        stock_code: str,
        quantity: int,
        price: int = 0,
        order_type: str = "00",
    ) -> OrderResult:
        """매수 주문

        Args:
            stock_code: 종목코드
            quantity: 주문 수량
            price: 주문 가격 (0이면 시장가)
            order_type: 주문 유형 (00=지정가, 01=시장가, 02=조건부지정가)
        """
        if price == 0:
            order_type = "01"

        tr_id = self._get_tr_id("buy")
        cano = self.config.account_no.replace("-", "")

        body = {
            "CANO": cano[:8],
            "ACNT_PRDT_CD": cano[8:] if len(cano) > 8 else "01",
            "PDNO": stock_code,
            "ORD_DVSN": order_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "",
            "CNDT_PRIC": "",
        }

        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            headers = self.auth.get_headers(tr_id)
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") == "0":
                output = data.get("output", {})
                order_id = output.get("ODNO", "")
                logger.info(
                    "매수 주문 성공: {} {} {}주 @{}원 (주문번호: {})",
                    stock_code, "지정가" if price > 0 else "시장가",
                    quantity, price, order_id,
                )
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    stock_code=stock_code,
                    side="BUY",
                    price=price,
                    quantity=quantity,
                    message=data.get("msg1", ""),
                    executed_at=dt.datetime.now(),
                )
            else:
                msg = data.get("msg1", "알 수 없는 오류")
                logger.error("매수 주문 실패: {} - {}", stock_code, msg)
                return OrderResult(success=False, stock_code=stock_code, side="BUY", message=msg)

        except Exception as e:
            logger.error("매수 주문 예외: {} - {}", stock_code, e)
            return OrderResult(success=False, stock_code=stock_code, side="BUY", message=str(e))

    def sell(
        self,
        stock_code: str,
        quantity: int,
        price: int = 0,
        order_type: str = "00",
    ) -> OrderResult:
        """매도 주문"""
        if price == 0:
            order_type = "01"

        tr_id = self._get_tr_id("sell")
        cano = self.config.account_no.replace("-", "")

        body = {
            "CANO": cano[:8],
            "ACNT_PRDT_CD": cano[8:] if len(cano) > 8 else "01",
            "PDNO": stock_code,
            "ORD_DVSN": order_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "01",
            "CNDT_PRIC": "",
        }

        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            headers = self.auth.get_headers(tr_id)
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") == "0":
                output = data.get("output", {})
                order_id = output.get("ODNO", "")
                logger.info(
                    "매도 주문 성공: {} {}주 @{}원 (주문번호: {})",
                    stock_code, quantity, price, order_id,
                )
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    stock_code=stock_code,
                    side="SELL",
                    price=price,
                    quantity=quantity,
                    message=data.get("msg1", ""),
                    executed_at=dt.datetime.now(),
                )
            else:
                msg = data.get("msg1", "알 수 없는 오류")
                logger.error("매도 주문 실패: {} - {}", stock_code, msg)
                return OrderResult(success=False, stock_code=stock_code, side="SELL", message=msg)

        except Exception as e:
            logger.error("매도 주문 예외: {} - {}", stock_code, e)
            return OrderResult(success=False, stock_code=stock_code, side="SELL", message=str(e))

    def get_balance(self) -> dict:
        """계좌 잔고를 조회합니다."""
        tr_id = self._get_tr_id("balance")
        cano = self.config.account_no.replace("-", "")

        params = {
            "CANO": cano[:8],
            "ACNT_PRDT_CD": cano[8:] if len(cano) > 8 else "01",
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
            headers = self.auth.get_headers(tr_id)
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") != "0":
                return {"error": data.get("msg1", "조회 실패")}

            holdings = []
            for item in data.get("output1", []):
                if int(item.get("hldg_qty", "0")) > 0:
                    holdings.append({
                        "stock_code": item["pdno"],
                        "stock_name": item["prdt_name"],
                        "quantity": int(item["hldg_qty"]),
                        "avg_price": int(float(item["pchs_avg_pric"])),
                        "current_price": int(item["prpr"]),
                        "pnl": int(item["evlu_pfls_amt"]),
                        "pnl_pct": float(item["evlu_pfls_rt"]),
                    })

            summary = data.get("output2", [{}])[0] if data.get("output2") else {}
            return {
                "holdings": holdings,
                "total_evaluation": int(summary.get("tot_evlu_amt", "0")),
                "total_pnl": int(summary.get("evlu_pfls_smtl_amt", "0")),
                "available_cash": int(summary.get("dnca_tot_amt", "0")),
            }

        except Exception as e:
            logger.error("잔고 조회 예외: {}", e)
            return {"error": str(e)}

    def get_order_history(self, start_date: str = "", end_date: str = "") -> list:
        """주문 체결 내역을 조회합니다."""
        tr_id = self._get_tr_id("inquiry")
        cano = self.config.account_no.replace("-", "")

        if not start_date:
            start_date = dt.datetime.now().strftime("%Y%m%d")
        if not end_date:
            end_date = start_date

        params = {
            "CANO": cano[:8],
            "ACNT_PRDT_CD": cano[8:] if len(cano) > 8 else "01",
            "INQR_STRT_DT": start_date,
            "INQR_END_DT": end_date,
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "01",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
            headers = self.auth.get_headers(tr_id)
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") != "0":
                return []

            orders = []
            for item in data.get("output1", []):
                orders.append({
                    "order_id": item.get("odno", ""),
                    "stock_code": item.get("pdno", ""),
                    "stock_name": item.get("prdt_name", ""),
                    "side": "BUY" if item.get("sll_buy_dvsn_cd") == "02" else "SELL",
                    "order_qty": int(item.get("ord_qty", "0")),
                    "exec_qty": int(item.get("tot_ccld_qty", "0")),
                    "exec_price": int(float(item.get("avg_prvs", "0"))),
                    "order_time": item.get("ord_tmd", ""),
                })
            return orders

        except Exception as e:
            logger.error("주문 내역 조회 예외: {}", e)
            return []
