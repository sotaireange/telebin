import os
os.environ['TZ'] = 'UTC'
from pybit.unified_trading import HTTP
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
        self.session=self.create_session()
        self.set_qty_prc()
        try:
            self.session.switch_position_mode(category="linear", coin="USDT", mode=0)
        except:
            pass

    def set_qty_prc(self):
        info = self.session.get_instruments_info(category='linear', symbol=self.coin)
        self.qty_step = float(info['result']['list'][0]['lotSizeFilter']['qtyStep'])
        self.prc_step = float(info['result']['list'][0]['priceFilter']['tickSize'])

    def config(self):
        self.API_KEY = self.data["api"]
        self.API_SECRET_KEY = self.data["secret"]
        self.coin = self.data["coin"]
        self.time_frame = int(self.data["time_frame"])
        self.leverage = int(self.data["leverage"])
        self.config_testnet=False
        self.qty_step = 0.0
        self.prc_step = 0.0
        self.key_value = int(self.data['key_value'])
        self.atr_period=int(self.data['atr'])
        self.tp_percent=int(self.data['tp'])
        self.sl_percent=int(self.data['sl'])
        self.usd=float(self.data['bal'])

    def heikin_ashi(self,df):
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
        return df['signal'].iloc[-2]


    def create_session(self):
        session = HTTP(
            testnet=self.config_testnet,
            api_key=self.API_KEY,
            api_secret=self.API_SECRET_KEY,
        )
        return session



    def get_last_price(self):
        info = self.session.get_tickers(category='linear', symbol=self.coin)
        return float(info['result']['list'][0]['lastPrice'])

    def get_data(self):
        flag = True
        while flag:
            try:
                klines = self.session.get_kline(category='linear', symbol=self.coin, interval=self.time_frame, limit=100)['result']['list']
                flag = False
            except Exception as e:
                time.sleep(10)
                print(e,"line 123")
                self.session = self.create_session()
        date_ = []
        open_ = []
        close_ = []
        high_ = []
        low_ = []
        volume_ = []
        for kln in klines:
            date_.append(datetime.fromtimestamp(int(kln[0]) / 1000))
            open_.append(float(kln[1]))
            high_.append(float(kln[2]))
            low_.append(float(kln[3]))
            close_.append(float(kln[4]))
            volume_.append(float(kln[5]))
        data = pd.DataFrame({'Open': open_, 'High': high_, 'Low': low_, 'Close': close_, 'Volume': volume_})
        data.index.name = 'Date'
        data.index = date_
        return data[::-1]
    #КОНЕЦ РАБОТЫ СО СИГНАЛОМ


    def round_step_size(self,quantity: Union[float, Decimal], step_size: Union[float, Decimal]) -> float:
        quantity = Decimal(str(quantity))
        return float(quantity - quantity % Decimal(str(step_size)))

    def take_stop(self,buy=False):
        price = self.get_last_price()
        qnt = float(self.round_step_size(float(self.usd) / float(price) * float(self.leverage), self.qty_step))
        if buy:
            tp_price = self.round_step_size(price * (1 + float(self.tp_percent) / 100), self.prc_step)
            sl_price=self.round_step_size(price * (1 - float(self.sl_percent) / 100), self.prc_step)
        else:
            tp_price = self.round_step_size(price * (1 - float(self.tp_percent) / 100), self.prc_step)
            sl_price=self.round_step_size(price * (1 + float(self.sl_percent) / 100), self.prc_step)


        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=self.coin,
                tpslMode="Partial",
                takeProfit=tp_price,
                tpOrderType="Market",
                tpSize=str(qnt),

                stopLoss=sl_price,
                slOrderType="Market",
                slSize=str(qnt),
                positionIdx=0,
            )

            return
        except Exception as e:
            pass
    def place_order(self,buy):
        #usd_for_order=self.get_balance()
        usd_for_order=self.usd
        try:
            if buy:
                side='Buy'
            else:
                side='Sell'
            price = self.get_last_price()
            qnt = float(self.round_step_size(float(usd_for_order) / float(price) * float(self.leverage), self.qty_step))
            id_ = self.session.place_order(
                category='linear',
                symbol=self.coin,
                side=side,
                orderType="Market",
                qty=str(qnt),
                positionIdx=0
            )['result']['orderId']
            return True
        except Exception as e:
            logging.error("Не удалось исполнить ордер")
            time.sleep(2)
            return False


    def set_leverage(self):
        try:
            self.session.set_leverage(category='linear',
                                      symbol=self.coin,
                                      buyLeverage=str(self.leverage),
                                      sellLeverage=str(self.leverage))
        except Exception as e:
            pass


    def position(self):
        while True:
            try:
                position = self.session.get_positions(category='linear', symbol=self.coin)['result']['list'][0]
                return position
            except Exception as e:
                time.sleep(2)
                logging.error(f'{e} line 192')
                self.session = self.create_session()

    def cancel_order(self,qnt,buy):
        while True:
            try:
                if buy:
                    side='Buy'
                else:
                    side='Sell'
                if qnt>0:
                    id_ = self.session.place_order(
                        category='linear',
                        symbol=self.coin,
                        side=side,
                        orderType="Market",
                        qty=str(qnt),
                        positionIdx=0
                    )['result']['orderId']
                return
            except Exception as e:
                time.sleep(2)
                logging.error(f'{e} line 214')
                self.session = self.create_session()

    def open_order(self,buy):
        try:
            position=self.position()

            if float(position['size'])>0:
                #print(f"CANCEL ORDER {position['size']}")
                qnt=float(position['size'])
                self.cancel_order(qnt,buy)
            self.set_leverage()
            id=self.place_order(buy)
            if id:
                return True
            else:
                return False
        except Exception as e:
            logging.error(f'{e} line 233')

            return False

    async def start_trade(self,state:FSMContext,message : Message):
        self.last_date = '2024'
        state_cur=await state.get_state()
        while state_cur=='Main:RUN':
            try:
                df = self.get_data()
                if df.empty:
                    await asyncio.sleep(5)
                    continue
                if self.last_date!=df.index[-1]:
                    sgnl = self.signal(df)
                    if sgnl==1:
                        #print(self.position()['side'])
                        if self.position()['side']=="Buy":
                            continue
                        flag=self.open_order(buy=True)
                        if flag:
                            self.take_stop(True)
                            await message.answer(text=f"Покупка {self.coin} По цене {df['Close'][-1]} \nВремя: {self.last_date}")
                            self.last_date = df.index[-1]

                    elif sgnl==-1:
                        #print(self.position()['side'])
                        if self.position()['side']=="Sell":
                            continue
                        flag=self.open_order(buy=False)
                        if flag:
                            self.take_stop(False)
                            await message.answer(text=(f"Продажа {self.coin} по цене {df['Close'][-1]} \nВремя: {self.last_date}"))
                            self.last_date = df.index[-1]
                await asyncio.sleep(10)
                state_cur= await state.get_state()
            except Exception as e:
                logging.error(f"Ошибка {self.coin}\n 252 LINE")



