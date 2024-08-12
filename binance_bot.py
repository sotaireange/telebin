import os
os.environ['TZ'] = 'UTC'
from binance.client import Client
from binance.enums import *
from datetime import datetime
from decimal import Decimal
from typing import Union
import time
import warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd
import ta
import numpy as np
import logging
import asyncio
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message


class AlgoBot():
    def __init__(self,data):
        self.data=data
        self.config()
        self.client = self.create_client()
        self.set_qty_prc()
        try:
            self.client.futures_change_position_mode(symbol=self.coin, dualSidePosition="false")
        except:
            pass


    def config(self):
        self.API_KEY = self.data["api"]
        self.API_SECRET_KEY = self.data["secret"]
        self.coin = self.data["coin"]
        self.time_frame = f"{self.data["time_frame"]}m"
        self.leverage = int(self.data["leverage"])
        self.testnet=False
        self.qty_step = 0.0
        self.prc_step = 0.0
        self.key_value = int(self.data['key_value'])
        self.atr_period=int(self.data['atr'])
        self.tp=int(self.data['tp'])
        self.sl=int(self.data['sl'])
        self.usd=float(self.data['bal'])


    def heikin_ashi(self, df):
        ha_df = df.copy()
        ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        ha_df['Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
        ha_df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        ha_df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        return ha_df

    def signal(self,df):
        src = df['Close']
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=self.atr_period).average_true_range()
        nLoss = self.key_value * df['ATR']
        df['xATRTrailingStop'] = np.nan
        df['pos'] = 0
        for i in range(1, len(df)):
            prev_xATR = df['xATRTrailingStop'].iloc[i-1]
            curr_src = src.iloc[i]
            prev_src = src.iloc[i-1]
            curr_nLoss = nLoss.iloc[i]

            if curr_src > prev_xATR and prev_src > prev_xATR:
                df.loc[df.index[i], 'xATRTrailingStop'] = max(prev_xATR, curr_src - curr_nLoss)
            elif curr_src < prev_xATR and prev_src < prev_xATR:
                df.loc[df.index[i], 'xATRTrailingStop'] = min(prev_xATR, curr_src + curr_nLoss)
            elif curr_src > prev_xATR:
                df.loc[df.index[i], 'xATRTrailingStop'] = curr_src - curr_nLoss
            else:
                df.loc[df.index[i], 'xATRTrailingStop'] = curr_src + curr_nLoss

            if prev_src < prev_xATR and curr_src > prev_xATR:
                df.loc[df.index[i], 'pos'] = 1
            elif prev_src > prev_xATR and curr_src < prev_xATR:
                df.loc[df.index[i], 'pos'] = -1
            else:
                df.loc[df.index[i], 'pos'] = df['pos'].iloc[i-1]
        df['EMA'] = ta.trend.EMAIndicator(close=src, window=1).ema_indicator()

        df['above'] = (df['EMA'].shift(1) < df['xATRTrailingStop'].shift(1)) & (df['EMA'] > df['xATRTrailingStop'])
        df['below'] = (df['xATRTrailingStop'].shift(1) < df['EMA'].shift(1)) & (df['xATRTrailingStop'] > df['EMA'])
        conditions = [
            (src > df['xATRTrailingStop']) & df['above'],
            (src < df['xATRTrailingStop']) & df['below']
        ]
        df['signal']=np.select(conditions, [1,-1],default=0)
        return df['signal'].iloc[-1]

    def round_step_size(self, quantity: Union[float, Decimal], step_size: Union[float, Decimal]) -> float:
        quantity = Decimal(str(quantity))
        return float(quantity - quantity % Decimal(str(step_size)))

    def get_tp_sl_price(self,price,buy):
        try:
            if buy:
                tp_price = self.round_step_size(price * (1 + float(self.tp) / 100), self.prc_step)
                sl_price = self.round_step_size(price * (1 - float(self.sl) / 100), self.prc_step)
            else:
                tp_price = self.round_step_size(price * (1 - float(self.tp) / 100), self.prc_step)
                sl_price = self.round_step_size(price * (1 + float(self.sl) / 100), self.prc_step)

            if self.tp==0: tp_price=0
            if self.sl==0: sl_price=0

            return {'tp_price': tp_price,
                    'sl_price': sl_price}
        except Exception as e :
            logging.error(f"Error get tp_sl price \n{e}")


    def create_client(self):
        client = Client(self.API_KEY, self.API_SECRET_KEY, testnet=self.testnet)
        return client

    def set_qty_prc(self):
        info = self.client.futures_exchange_info()
        symbol=(pd.DataFrame(info['symbols']).set_index('symbol')).loc[self.coin]
        self.qty_step = float(symbol['filters'][2]['stepSize'])
        self.prc_step = float(symbol['filters'][0]['tickSize'])

    def get_last_price(self):
        info = self.client.futures_symbol_ticker(symbol=self.coin)
        return float(info['price'])

    def get_data(self):
        flag = True
        while flag:
            try:
                klines = self.client.futures_klines(symbol=self.coin, interval=self.time_frame, limit=100)
                break
            except Exception as e:
                time.sleep(2)
                print(e, "line 123")
                self.client = self.create_client()
        data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data.set_index('Open time', inplace=True)


        data = data.astype({
            'Open': 'float',
            'High': 'float',
            'Low': 'float',
            'Close': 'float',
            'Volume': 'float'
        })
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        return data

    def position(self):
        while True:
            try:
                position = self.client.futures_position_information(symbol=self.coin)[0]
                return position
            except Exception as e:
                time.sleep(2)
                print(f'{e} line 192')
                self.client = self.create_client()



    def set_leverage(self):
        try:
            self.client.futures_change_leverage(symbol=self.coin, leverage=self.leverage)
        except Exception as e:
            pass


    def place_order(self, buy):
        usd_for_order = self.usd
        try:
            side = SIDE_BUY if buy else SIDE_SELL
            price = self.get_last_price()
            qnt = float(self.round_step_size(float(usd_for_order) / float(price) * float(self.leverage), self.qty_step))

            order = self.client.futures_create_order(
                symbol=self.coin,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=qnt
            )
            return order
        except Exception as e:
            logging.exception("Не удалось исполнить ордер")
            time.sleep(2)
            return False


    def cancel_order(self, qnt, buy):
        while True:
            try:
                side = SIDE_BUY if buy else SIDE_SELL
                if qnt > 0:
                    self.client.futures_create_order(
                        symbol=self.coin,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=qnt
                    )
                return
            except Exception as e:
                time.sleep(10)
                print(e, "line 214")
                self.client = self.create_client()

    def take_stop(self,buy):
        try:
            position=self.position()
            price=float(position['entryPrice'])
            qnt=abs(float(position['positionAmt']))
            prices=self.get_tp_sl_price(price,buy)
            tp_price=prices['tp_price']
            sl_price=prices['sl_price']
            side = SIDE_SELL if buy else SIDE_BUY

            if tp_price:
                self.client.futures_create_order(
                    symbol=self.coin,
                    side=side,
                    type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                    quantity=qnt,
                    stopPrice=tp_price,
                    timeInForce="GTE_GTC",
                    closePosition=True
                )

            if sl_price:
                self.client.futures_create_order(
                    symbol=self.coin,
                    side=side,
                    type=FUTURE_ORDER_TYPE_STOP_MARKET,
                    quantity=qnt,
                    stopPrice=sl_price,
                    timeInForce="GTE_GTC",
                    closePosition=True
                )

        except Exception as e:
            logging.error(f"Error when take stop \n {e}")


    def open_order(self, buy):
        try:
            position = self.position()
            qnt = abs(float(position['positionAmt']))
            if qnt > 0:
                self.cancel_order(qnt, buy)
            self.set_leverage()

            order=self.place_order(buy)
            if order:
                self.take_stop(buy)
                return True
        except Exception as e:
            logging.error(f'Ошибка при открытии ордера \n {e}\n line 229')
            return False

    async def start_trade(self,state:FSMContext,message : Message):
        last_date = None
        side=0
        state_cur=await state.get_state()
        while state_cur=='Main:RUN':
            try:
                df = self.get_data()
                if df.empty:
                    continue
                if last_date != df.index[-1]:
                    sgnl = self.signal(df)
                    if sgnl == 1 and side!=1:
                        flag=self.open_order(buy=True)
                        if flag:
                            await message.answer(text=f"Long {self.coin} По цене {df['Close'][-1]} \nВремя: {df.index[-1]}")
                            side=1

                    elif sgnl == -1 and side !=-1:
                        flag=self.open_order(buy=False)
                        if flag:
                            await message.answer(text=(f"Short {self.coin} по цене {df['Close'][-1]} \nВремя: {df.index[-1]}"))
                            side=-1
                    last_date = df.index[-1]
                state_cur=await state.get_state()
                await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"Ошибка {self.coin}\n 252 LINE {e}")

